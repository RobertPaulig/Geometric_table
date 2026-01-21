from __future__ import annotations

import csv
import json
import math
import os
import platform
import random
import statistics
import time
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
from scipy.linalg import eigvals_banded, eigh_tridiagonal

import hetero1a
from hetero2.integration.adaptive import adaptive_approximate_on_grid
from hetero2.integration.metrics import curve_checksum_sha256
from hetero2.integration.types import AdaptiveIntegrationConfig
from hetero2.physics_operator import DOS_ENERGY_MARGIN_SIGMAS_DEFAULT, compute_dos_curve, load_atoms_db_v1


_Z_TO_SYMBOL = {6: "C", 7: "N", 8: "O"}


@dataclass(frozen=True, slots=True)
class PolymerScaleFixture:
    fixture_id: str
    smiles: str
    types_z: tuple[int, ...]
    n_atoms: int
    n_hetero: int
    motif: str
    seed: int


@dataclass(frozen=True, slots=True)
class P5Config:
    n_atoms_bins: tuple[int, ...]
    samples_per_bin: int
    seed: int
    curve_id: str
    energy_points: int
    dos_eta: float
    potential_scale_gamma: float
    edge_weight_mode: str
    integrator_eps_abs: float
    integrator_eps_rel: float
    integrator_subdomains_max: int
    integrator_poly_degree_max: int
    integrator_quad_order_max: int
    integrator_eval_budget_max: int
    integrator_split_criterion: str
    overhead_region_n_max: int
    gate_n_min: int
    correctness_gate_rate: float
    min_scale_samples: int
    speedup_gate_break_even: float
    speedup_gate_strong: float


def _sha256_file(path: Path) -> str:
    import hashlib

    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def _compute_file_infos(out_dir: Path, *, skip_names: set[str] | None = None) -> list[dict[str, object]]:
    skip = skip_names or set()
    infos: list[dict[str, object]] = []
    for path in sorted(out_dir.rglob("*")):
        if path.is_dir():
            continue
        if path.name in skip:
            continue
        rel = path.relative_to(out_dir).as_posix()
        infos.append({"path": f"./{rel}", "size_bytes": int(path.stat().st_size), "sha256": _sha256_file(path)})
    return infos


def _write_checksums(out_dir: Path, file_infos: list[dict[str, object]]) -> None:
    lines: list[str] = []
    for info in file_infos:
        sha = str(info.get("sha256") or "")
        rel = str(info.get("path") or "").lstrip("./")
        if not sha or not rel:
            continue
        lines.append(f"{sha}  {rel}")
    (out_dir / "checksums.sha256").write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def _write_zip_pack(out_dir: Path, *, zip_name: str = "evidence_pack.zip") -> None:
    zip_path = out_dir / zip_name
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(out_dir.rglob("*")):
            if path.is_dir():
                continue
            if path.name == zip_name:
                continue
            zf.write(path, path.relative_to(out_dir).as_posix())


def _write_manifest(out_dir: Path, *, config: dict[str, object], files: list[dict[str, object]]) -> None:
    payload: dict[str, object] = {
        "tool_version": getattr(hetero1a, "__version__", None),
        "git_sha": os.environ.get("GITHUB_SHA") or None,
        "python_version": platform.python_version(),
        "timestamp_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "config": dict(config),
        "files": sorted(list(files), key=lambda x: str(x.get("path", ""))),
    }
    (out_dir / "manifest.json").write_text(json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def _detect_git_sha() -> str:
    sha = str(os.environ.get("GITHUB_SHA") or "").strip()
    if sha:
        return sha

    try:
        import subprocess

        repo_root = Path(__file__).resolve().parents[2]
        out = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo_root, stderr=subprocess.DEVNULL, text=True)
        sha = str(out or "").strip()
        if sha:
            return sha
    except Exception:
        pass

    return "UNKNOWN"


def _median(values: Iterable[float]) -> float:
    vals = [float(x) for x in values if math.isfinite(float(x))]
    if not vals:
        return float("nan")
    return float(statistics.median(vals))


def _percentile(values: Iterable[float], *, q: float) -> float:
    vals = sorted(float(x) for x in values if math.isfinite(float(x)))
    if not vals:
        return float("nan")
    if q <= 0:
        return float(vals[0])
    if q >= 100:
        return float(vals[-1])
    pos = (len(vals) - 1) * (float(q) / 100.0)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return float(vals[lo])
    frac = pos - lo
    return float(vals[lo] * (1.0 - frac) + vals[hi] * frac)


def _auto_energy_grid(*, eigenvalues: np.ndarray, energy_points: int, eta: float) -> np.ndarray:
    vals = np.asarray(eigenvalues, dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return np.array([], dtype=float)
    n = int(energy_points)
    if n < 2:
        raise ValueError("energy_points must be >= 2")
    margin = float(DOS_ENERGY_MARGIN_SIGMAS_DEFAULT) * float(eta)
    e_min = float(np.min(vals)) - margin
    e_max = float(np.max(vals)) + margin
    if not math.isfinite(e_min) or not math.isfinite(e_max) or e_max <= e_min:
        return np.array([], dtype=float)
    return np.linspace(e_min, e_max, n, dtype=float)


def generate_polymer_scale_fixtures(
    *,
    n_atoms_bins: Sequence[int],
    samples_per_bin: int,
    seed: int,
) -> list[PolymerScaleFixture]:
    bins = [int(x) for x in n_atoms_bins]
    if not bins or any(n < 2 for n in bins):
        raise ValueError("n_atoms_bins must contain ints >= 2")
    per_bin = int(samples_per_bin)
    if per_bin < 1:
        raise ValueError("samples_per_bin must be >= 1")

    motifs: list[tuple[str, tuple[int, ...]]] = [
        ("CCOCN", (6, 6, 8, 6, 7)),
        ("CCNOC", (6, 6, 7, 8, 6)),
        ("COCCN", (6, 8, 6, 6, 7)),
        ("CNCCO", (6, 7, 6, 6, 8)),
    ]
    fixtures: list[PolymerScaleFixture] = []
    for n_atoms in bins:
        for idx in range(per_bin):
            sample_seed = int(seed) ^ (n_atoms * 1009) ^ (idx * 7919)
            rng = random.Random(sample_seed)
            motif_name, motif = motifs[idx % len(motifs)]
            offset = int(rng.randrange(len(motif)))
            types = [int(motif[(i + offset) % len(motif)]) for i in range(int(n_atoms))]

            # Break symmetry deterministically while keeping the molecule chemically trivial (single bonds).
            types[0] = 6
            types[-1] = 6
            if len(types) >= 4:
                pos = 1 + (idx * 7) % (len(types) - 2)
                # Flip between O/N to make asymmetry robust.
                types[pos] = 8 if types[pos] != 8 else 7

            if all(z == 6 for z in types):
                # Should never happen due to motif design, but keep a hard guardrail.
                types[len(types) // 2] = 8

            n_hetero = int(sum(1 for z in types if int(z) != 6))
            smiles = "".join(_Z_TO_SYMBOL.get(int(z), "C") for z in types)
            fixture_id = f"poly_n{int(n_atoms)}_s{idx:02d}"
            fixtures.append(
                PolymerScaleFixture(
                    fixture_id=str(fixture_id),
                    smiles=str(smiles),
                    types_z=tuple(int(z) for z in types),
                    n_atoms=int(n_atoms),
                    n_hetero=int(n_hetero),
                    motif=str(motif_name),
                    seed=int(sample_seed),
                )
            )
    return fixtures


def generate_ring_scale_fixtures(
    *,
    n_atoms_bins: Sequence[int],
    samples_per_bin: int,
    seed: int,
) -> list[PolymerScaleFixture]:
    bins = [int(x) for x in n_atoms_bins]
    if not bins or any(n < 3 for n in bins):
        raise ValueError("n_atoms_bins must contain ints >= 3 for ring-suite")
    per_bin = int(samples_per_bin)
    if per_bin < 1:
        raise ValueError("samples_per_bin must be >= 1")

    motifs: list[tuple[str, tuple[int, ...]]] = [
        ("COCN", (6, 8, 6, 7)),
        ("CNOC", (6, 7, 8, 6)),
        ("CCON", (6, 6, 8, 7)),
        ("NCCO", (7, 6, 6, 8)),
    ]
    fixtures: list[PolymerScaleFixture] = []
    for n_atoms in bins:
        for idx in range(per_bin):
            sample_seed = int(seed) ^ (n_atoms * 1009) ^ (idx * 104729) ^ 0xC0FFEE
            rng = random.Random(sample_seed)
            motif_name, motif = motifs[idx % len(motifs)]
            offset = int(rng.randrange(len(motif)))
            types = [int(motif[(i + offset) % len(motif)]) for i in range(int(n_atoms))]

            # Break symmetry deterministically (avoid rotationally symmetric patterns).
            types[0] = 6
            types[-1] = 6
            if len(types) >= 6:
                pos_a = 1 + (idx * 7) % (len(types) - 2)
                pos_b = 1 + (idx * 13) % (len(types) - 2)
                if pos_b == pos_a:
                    pos_b = 1 + ((pos_b + 3) % (len(types) - 2))
                types[pos_a] = 8 if types[pos_a] != 8 else 7
                types[pos_b] = 7 if types[pos_b] != 7 else 8

            if not any(int(z) == 7 for z in types):
                types[len(types) // 3] = 7
            if not any(int(z) == 8 for z in types):
                types[(2 * len(types)) // 3] = 8

            n_hetero = int(sum(1 for z in types if int(z) != 6))
            parts = [_Z_TO_SYMBOL.get(int(types[0]), "C") + "1"]
            if len(types) > 2:
                parts.extend(_Z_TO_SYMBOL.get(int(z), "C") for z in types[1:-1])
            parts.append(_Z_TO_SYMBOL.get(int(types[-1]), "C") + "1")
            smiles = "".join(parts)
            fixture_id = f"ring_n{int(n_atoms)}_s{idx:02d}"
            fixtures.append(
                PolymerScaleFixture(
                    fixture_id=str(fixture_id),
                    smiles=str(smiles),
                    types_z=tuple(int(z) for z in types),
                    n_atoms=int(n_atoms),
                    n_hetero=int(n_hetero),
                    motif=str(motif_name),
                    seed=int(sample_seed),
                )
            )
    return fixtures


def _chain_edge_weights(
    *,
    types_z: Sequence[int],
    edge_weight_mode: str,
    alpha: float = 0.5,
) -> np.ndarray:
    n = int(len(types_z))
    if n < 2:
        return np.array([], dtype=float)
    mode = str(edge_weight_mode)
    if mode not in {"unweighted", "bond_order", "bond_order_delta_chi"}:
        raise ValueError("edge_weight_mode must be one of: unweighted, bond_order, bond_order_delta_chi")

    if mode in {"unweighted", "bond_order"}:
        return np.ones((n - 1,), dtype=float)

    atoms_db = load_atoms_db_v1()
    chi_by_z = {int(k): float(v) for k, v in atoms_db.chi_by_atomic_num.items()}
    w = np.ones((n - 1,), dtype=float)
    for i in range(n - 1):
        chi_i = float(chi_by_z[int(types_z[i])])
        chi_j = float(chi_by_z[int(types_z[i + 1])])
        w[i] = 1.0 + float(alpha) * abs(float(chi_i) - float(chi_j))
    return w


def _ring_edge_weights(
    *,
    types_z: Sequence[int],
    edge_weight_mode: str,
    alpha: float = 0.5,
) -> np.ndarray:
    n = int(len(types_z))
    if n < 3:
        raise ValueError("ring topology requires n_atoms >= 3")
    mode = str(edge_weight_mode)
    if mode not in {"unweighted", "bond_order", "bond_order_delta_chi"}:
        raise ValueError("edge_weight_mode must be one of: unweighted, bond_order, bond_order_delta_chi")

    if mode in {"unweighted", "bond_order"}:
        return np.ones((n,), dtype=float)

    atoms_db = load_atoms_db_v1()
    chi_by_z = {int(k): float(v) for k, v in atoms_db.chi_by_atomic_num.items()}
    w = np.ones((n,), dtype=float)
    for i in range(n):
        j = (i + 1) % n
        chi_i = float(chi_by_z[int(types_z[i])])
        chi_j = float(chi_by_z[int(types_z[j])])
        w[i] = 1.0 + float(alpha) * abs(float(chi_i) - float(chi_j))
    return w


def _ring_banded_permutation(n: int) -> list[int]:
    n = int(n)
    if n < 0:
        raise ValueError("n must be >= 0")
    if n <= 2:
        return list(range(n))
    perm: list[int] = [0, 1]
    lo = 2
    hi = n - 1
    take_hi = True
    while len(perm) < n:
        if take_hi:
            perm.append(int(hi))
            hi -= 1
        else:
            perm.append(int(lo))
            lo += 1
        take_hi = not take_hi
    return perm


def build_h_banded_ring(
    *,
    types_z: Sequence[int],
    edge_weight_mode: str,
    potential_scale_gamma: float,
) -> tuple[np.ndarray, np.ndarray]:
    atoms_db = load_atoms_db_v1()
    eps_by_z = {int(k): float(v) for k, v in atoms_db.potential_by_atomic_num.items()}
    for z in types_z:
        if int(z) not in eps_by_z:
            raise ValueError(f"Missing epsilon for Z={int(z)} in atoms_db_v1.json")

    n = int(len(types_z))
    w = _ring_edge_weights(types_z=types_z, edge_weight_mode=str(edge_weight_mode))
    deg = np.zeros((n,), dtype=float)
    for i in range(n):
        deg[i] = float(w[(i - 1) % n] + w[i])

    v0 = np.array([float(eps_by_z[int(z)]) for z in types_z], dtype=float)
    diag = deg + float(potential_scale_gamma) * v0

    perm = _ring_banded_permutation(n)
    pos = {int(old): int(new) for new, old in enumerate(perm)}
    ab = np.zeros((3, n), dtype=float)  # lower banded (diag + 2 subdiagonals), scipy.linalg.eigvals_banded format
    ab[0, :] = np.asarray(diag, dtype=float)[perm]

    for i in range(n):
        j = (i + 1) % n
        a = int(pos[int(i)])
        b = int(pos[int(j)])
        row = a if a >= b else b
        col = b if a >= b else a
        offset = int(row - col)
        if offset > 2:
            raise ValueError("ring band permutation failed (bandwidth > 2)")
        # lower banded storage: ab[offset, col] = A[row, col] for row >= col
        ab[offset, col] = -float(w[i])

    return np.asarray(ab, dtype=float), np.asarray(v0, dtype=float)


def build_h_tridiagonal_chain(
    *,
    types_z: Sequence[int],
    edge_weight_mode: str,
    potential_scale_gamma: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    atoms_db = load_atoms_db_v1()
    eps_by_z = {int(k): float(v) for k, v in atoms_db.potential_by_atomic_num.items()}
    for z in types_z:
        if int(z) not in eps_by_z:
            raise ValueError(f"Missing epsilon for Z={int(z)} in atoms_db_v1.json")

    n = int(len(types_z))
    w = _chain_edge_weights(types_z=types_z, edge_weight_mode=str(edge_weight_mode))
    deg = np.zeros((n,), dtype=float)
    if n >= 2:
        deg[0] = float(w[0])
        deg[-1] = float(w[-1])
        for i in range(1, n - 1):
            deg[i] = float(w[i - 1] + w[i])

    v0 = np.array([float(eps_by_z[int(z)]) for z in types_z], dtype=float)
    diag = deg + float(potential_scale_gamma) * v0
    off = -np.asarray(w, dtype=float)
    return np.asarray(diag, dtype=float), np.asarray(off, dtype=float), np.asarray(v0, dtype=float)


def compute_p5_speedup_rows(
    *,
    fixtures_by_family: Mapping[str, Sequence[PolymerScaleFixture]],
    cfg: P5Config,
) -> tuple[
    list[dict[str, object]],
    dict[str, object],
    list[dict[str, object]],
    list[dict[str, object]],
    list[dict[str, object]],
]:
    adaptive_cfg = AdaptiveIntegrationConfig(
        eps_abs=float(cfg.integrator_eps_abs),
        eps_rel=float(cfg.integrator_eps_rel),
        subdomains_max=int(cfg.integrator_subdomains_max),
        poly_degree_max=int(cfg.integrator_poly_degree_max),
        quad_order_max=int(cfg.integrator_quad_order_max),
        eval_budget_max=int(cfg.integrator_eval_budget_max),
        split_criterion=str(cfg.integrator_split_criterion),
    )

    adaptive_trace_rows: list[dict[str, object]] = []
    adaptive_summary_rows: list[dict[str, object]] = []
    sample_rows: list[dict[str, object]] = []
    timing_sample_rows: list[dict[str, object]] = []
    preferred = ["polymer", "ring"]
    ordered_families = [f for f in preferred if f in fixtures_by_family]
    ordered_families.extend(sorted(str(k) for k in fixtures_by_family.keys() if str(k) not in set(ordered_families)))

    for family in ordered_families:
        topo = str(family)
        for fx in fixtures_by_family.get(topo, []):
            t_build0 = time.perf_counter()
            if topo == "polymer":
                diag, off, v0 = build_h_tridiagonal_chain(
                    types_z=fx.types_z,
                    edge_weight_mode=str(cfg.edge_weight_mode),
                    potential_scale_gamma=float(cfg.potential_scale_gamma),
                )
                eigvals = eigh_tridiagonal(diag, off, eigvals_only=True, check_finite=False)
            elif topo == "ring":
                ab, v0 = build_h_banded_ring(
                    types_z=fx.types_z,
                    edge_weight_mode=str(cfg.edge_weight_mode),
                    potential_scale_gamma=float(cfg.potential_scale_gamma),
                )
                eigvals = eigvals_banded(ab, lower=True, check_finite=False)
            else:
                raise ValueError(f"Unknown topology family: {topo}")

            grid = _auto_energy_grid(eigenvalues=eigvals, energy_points=int(cfg.energy_points), eta=float(cfg.dos_eta))
            if grid.size == 0:
                raise ValueError("Energy grid is empty (invalid eigenvalues range)")
            e_min = float(np.min(grid)) if grid.size else float("nan")
            e_max = float(np.max(grid)) if grid.size else float("nan")
            build_operator_ms = float((time.perf_counter() - t_build0) * 1000.0)

            def _f(x: np.ndarray) -> np.ndarray:
                return compute_dos_curve(eigenvalues=eigvals, energy_grid=np.asarray(x, dtype=float), eta=float(cfg.dos_eta))

            t0 = time.perf_counter()
            baseline_values = _f(grid)
            baseline_ms = float((time.perf_counter() - t0) * 1000.0)

            tol_scale = float(np.max(np.abs(np.asarray(baseline_values, dtype=float)))) if baseline_values.size else 0.0
            res = adaptive_approximate_on_grid(
                f=_f,
                energy_grid=grid,
                cfg=adaptive_cfg,
                tol_scale=tol_scale,
                baseline_values=np.asarray(baseline_values, dtype=float),
            )
            adaptive_ms = float(res.summary.get("walltime_ms_total", float("nan")))
            adaptive_eval_ms = float(res.summary.get("eval_walltime_ms_total", float("nan")))
            adaptive_overhead_ms = float(res.summary.get("overhead_walltime_ms_total", float("nan")))
            adaptive_evals_total = int(res.summary.get("evals_total", 0) or 0)
            segments_used = int(res.summary.get("segments_used", 0) or 0)
            cache_hit_rate = float(res.summary.get("cache_hit_rate", float("nan")))
            adaptive_verdict_row = str(res.summary.get("verdict", ""))

            base_arr = np.asarray(baseline_values, dtype=float)
            adapt_arr = np.asarray(res.values, dtype=float)
            base_value = float(np.max(np.abs(base_arr))) if base_arr.size else float("nan")
            adapt_value = float(np.max(np.abs(adapt_arr))) if adapt_arr.size else float("nan")
            abs_err = float(np.max(np.abs(adapt_arr - base_arr))) if base_arr.size and adapt_arr.size else float("nan")
            denom = abs(float(base_value)) + 1e-12
            rel_err = float(abs_err / denom) if math.isfinite(float(abs_err)) else float("nan")
            tol = (
                float(cfg.integrator_eps_abs) + float(cfg.integrator_eps_rel) * abs(float(base_value))
                if math.isfinite(float(base_value))
                else float("nan")
            )
            correctness_pass = bool(math.isfinite(float(abs_err)) and math.isfinite(float(tol)) and float(abs_err) <= float(tol))

            baseline_points = int(grid.size)
            eval_ratio = (
                float(baseline_points) / float(adaptive_evals_total)
                if baseline_points > 0 and adaptive_evals_total > 0
                else float("nan")
            )
            speedup = float(baseline_ms / adaptive_ms) if adaptive_ms and adaptive_ms > 0.0 and math.isfinite(adaptive_ms) else float("nan")

            for tr in res.trace:
                adaptive_trace_rows.append(
                    {
                        "family": str(topo),
                        "fixture_id": str(fx.fixture_id),
                        "curve_id": str(cfg.curve_id),
                        "n_atoms": int(fx.n_atoms),
                        "segment_id": int(tr.segment_id),
                        "E_left": float(tr.e_left),
                        "E_right": float(tr.e_right),
                        "n_probe_points": int(tr.n_probe_points),
                        "poly_degree": int(tr.poly_degree),
                        "quad_order": int(tr.quad_order),
                        "n_function_evals": int(tr.n_function_evals),
                        "error_est": float(tr.error_est),
                        "walltime_ms_segment": float(tr.walltime_ms_segment),
                        "split_reason": str(tr.split_reason),
                    }
                )

            adaptive_summary_rows.append(
                {
                    "family": str(topo),
                    "fixture_id": str(fx.fixture_id),
                    "curve_id": str(cfg.curve_id),
                    "n_atoms": int(fx.n_atoms),
                    "baseline_points": int(baseline_points),
                    "adaptive_evals_total": int(adaptive_evals_total),
                    "eval_ratio": float(eval_ratio),
                    "segments_used": int(segments_used),
                    "cache_hit_rate": float(cache_hit_rate),
                    "adaptive_verdict_row": str(adaptive_verdict_row),
                    "energy_min": float(e_min),
                    "energy_max": float(e_max),
                }
            )

            sample_rows.append(
                {
                    "family": str(topo),
                    "row_kind": "sample",
                    "id": str(fx.fixture_id),
                    "smiles": str(fx.smiles),
                    "n_atoms": int(fx.n_atoms),
                    "n_hetero": int(fx.n_hetero),
                    "motif": str(fx.motif),
                    "seed": int(fx.seed),
                    "curve_id": str(cfg.curve_id),
                    "edge_weight_mode": str(cfg.edge_weight_mode),
                    "potential_scale_gamma": float(cfg.potential_scale_gamma),
                    "dos_eta": float(cfg.dos_eta),
                    "energy_points": int(baseline_points),
                    "energy_min": float(e_min),
                    "energy_max": float(e_max),
                    "baseline_walltime_ms": float(baseline_ms),
                    "adaptive_walltime_ms": float(adaptive_ms),
                    "speedup": float(speedup),
                    "baseline_points": int(baseline_points),
                    "adaptive_evals_total": int(adaptive_evals_total),
                    "eval_ratio": float(eval_ratio),
                    "segments_used": int(segments_used),
                    "cache_hit_rate": float(cache_hit_rate),
                    "adaptive_verdict_row": str(adaptive_verdict_row),
                    "baseline_value": float(base_value),
                    "adaptive_value": float(adapt_value),
                    "abs_err": float(abs_err),
                    "rel_err": float(rel_err),
                    "tol": float(tol),
                    "correctness_pass": bool(correctness_pass),
                    "eigs_checksum": str(
                        curve_checksum_sha256(energy_grid=list(range(int(eigvals.size))), values=[float(x) for x in eigvals.tolist()])
                    ),
                    "v0_checksum": str(curve_checksum_sha256(energy_grid=list(range(int(v0.size))), values=[float(x) for x in v0.tolist()])),
                }
            )

            dos_ldos_eval_ms = float(baseline_ms + adaptive_eval_ms) if math.isfinite(adaptive_eval_ms) else float("nan")
            integration_logic_ms = float(adaptive_overhead_ms) if math.isfinite(adaptive_overhead_ms) else float("nan")
            total_ms_no_io = (
                float(build_operator_ms + dos_ldos_eval_ms + integration_logic_ms)
                if math.isfinite(dos_ldos_eval_ms) and math.isfinite(integration_logic_ms)
                else float("nan")
            )
            timing_sample_rows.append(
                {
                    "family": str(topo),
                    "row_kind": "sample",
                    "fixture_id": str(fx.fixture_id),
                    "n_atoms": int(fx.n_atoms),
                    "build_operator_ms": float(build_operator_ms),
                    "dos_ldos_eval_ms": float(dos_ldos_eval_ms),
                    "integration_logic_ms": float(integration_logic_ms),
                    "io_ms": 0.0,
                    "total_ms": float(total_ms_no_io),
                    "baseline_walltime_ms": float(baseline_ms),
                    "adaptive_walltime_ms": float(adaptive_ms),
                    "adaptive_eval_walltime_ms": float(adaptive_eval_ms),
                    "adaptive_overhead_walltime_ms": float(adaptive_overhead_ms),
                    "baseline_points": int(baseline_points),
                    "adaptive_evals_total": int(adaptive_evals_total),
                    "speedup": float(speedup),
                    "eval_ratio": float(eval_ratio),
                }
            )

    # Aggregates per N.
    by_n: dict[int, list[dict[str, object]]] = {}
    for row in sample_rows:
        n = int(row.get("n_atoms", 0) or 0)
        by_n.setdefault(n, []).append(row)

    bin_rows: list[dict[str, object]] = []
    for n in sorted(by_n.keys()):
        rows = by_n[n]
        speeds = [float(r.get("speedup", float("nan"))) for r in rows]
        eval_ratios = [float(r.get("eval_ratio", float("nan"))) for r in rows]
        pass_flags = [bool(r.get("correctness_pass", False)) for r in rows]
        n_samples = int(len(rows))
        bin_rows.append(
            {
                "row_kind": "bin",
                "n_atoms_bin": int(n),
                "n_samples": int(n_samples),
                "median_speedup": float(_median(speeds)),
                "median_eval_ratio": float(_median(eval_ratios)),
                "pass_rate": float(sum(1 for x in pass_flags if x) / n_samples) if n_samples > 0 else float("nan"),
                "p95_speedup": float(_percentile(speeds, q=95.0)),
            }
        )

    # Verdict logic (P5).
    max_n = max(int(x) for x in by_n.keys())
    max_bin = next((b for b in bin_rows if int(b.get("n_atoms_bin", -1)) == int(max_n)), None)
    max_bin_speed = float(max_bin.get("median_speedup")) if isinstance(max_bin, dict) else float("nan")
    max_bin_pass_rate = float(max_bin.get("pass_rate")) if isinstance(max_bin, dict) else float("nan")

    break_even_n = None
    for b in bin_rows:
        n = int(b.get("n_atoms_bin", 0) or 0)
        if n < int(cfg.gate_n_min):
            continue
        ms = float(b.get("median_speedup", float("nan")))
        if math.isfinite(ms) and ms >= float(cfg.speedup_gate_break_even):
            break_even_n = int(n)
            break

    scale_speedup_verdict = "FAIL_SPEEDUP_AT_SCALE"
    if math.isfinite(max_bin_pass_rate) and max_bin_pass_rate < 1.0:
        scale_speedup_verdict = "FAIL_CORRECTNESS_AT_SCALE"
    elif math.isfinite(max_bin_speed) and max_bin_speed >= float(cfg.speedup_gate_strong):
        scale_speedup_verdict = "PASS_SPEEDUP"
    elif math.isfinite(max_bin_speed) and max_bin_speed >= float(cfg.speedup_gate_break_even):
        scale_speedup_verdict = "PASS_BREAK_EVEN"

    energy_mins = [float(r.get("energy_min", float("nan"))) for r in sample_rows]
    energy_maxs = [float(r.get("energy_max", float("nan"))) for r in sample_rows]
    energy_min_global = float(min(x for x in energy_mins if math.isfinite(float(x)))) if any(math.isfinite(float(x)) for x in energy_mins) else float("nan")
    energy_max_global = float(max(x for x in energy_maxs if math.isfinite(float(x)))) if any(math.isfinite(float(x)) for x in energy_maxs) else float("nan")

    scale_samples = [r for r in sample_rows if int(r.get("n_atoms", 0) or 0) >= int(cfg.gate_n_min)]
    n_scale = int(len(scale_samples))
    pass_rate_at_scale = float(sum(1 for r in scale_samples if bool(r.get("correctness_pass", False))) / n_scale) if n_scale > 0 else float("nan")
    speedup_median_at_scale = float(_median([float(r.get("speedup", float("nan"))) for r in scale_samples]))
    eval_ratio_median_at_scale = float(_median([float(r.get("eval_ratio", float("nan"))) for r in scale_samples]))

    correctness_gate_rate = float(cfg.correctness_gate_rate)
    integrator_correctness_verdict = "PASS_CORRECTNESS_AT_SCALE" if math.isfinite(pass_rate_at_scale) and pass_rate_at_scale >= correctness_gate_rate else "FAIL_CORRECTNESS_AT_SCALE"

    integrator_speedup_verdict = "NOT_VALID_DUE_TO_CORRECTNESS"
    integrator_verdict_reason = f"FAIL: correctness_verdict={integrator_correctness_verdict}"
    if integrator_correctness_verdict == "PASS_CORRECTNESS_AT_SCALE":
        if n_scale < int(cfg.min_scale_samples):
            integrator_speedup_verdict = "INCONCLUSIVE_NOT_ENOUGH_SCALE_SAMPLES"
            integrator_verdict_reason = f"INCONCLUSIVE: |S|={n_scale} < min_scale_samples={int(cfg.min_scale_samples)} (N>=gate_n_min={int(cfg.gate_n_min)})"
        else:
            gate_speedup = float(cfg.speedup_gate_break_even)  # break-even by default (1.0)
            if math.isfinite(speedup_median_at_scale) and speedup_median_at_scale >= gate_speedup:
                integrator_speedup_verdict = "PASS_SPEEDUP_AT_SCALE"
                integrator_verdict_reason = f"PASS: correctness ok at scale; speedup_median_at_scale={speedup_median_at_scale} >= {gate_speedup}"
            else:
                integrator_speedup_verdict = "FAIL_SPEEDUP_AT_SCALE"
                integrator_verdict_reason = f"FAIL: correctness ok at scale; speedup_median_at_scale={speedup_median_at_scale} < {gate_speedup}"
    else:
        integrator_speedup_verdict = "NOT_VALID_DUE_TO_CORRECTNESS"
        integrator_verdict_reason = (
            f"NOT_VALID: correctness_pass_rate_at_scale={pass_rate_at_scale} < correctness_gate_rate={correctness_gate_rate} (N>=gate_n_min={int(cfg.gate_n_min)})"
        )

    valid_flags = []
    for r in sample_rows:
        baseline_ms = float(r.get("baseline_walltime_ms", float("nan")))
        adaptive_ms = float(r.get("adaptive_walltime_ms", float("nan")))
        baseline_points = int(r.get("baseline_points", 0) or 0)
        adaptive_evals = int(r.get("adaptive_evals_total", 0) or 0)
        abs_err = float(r.get("abs_err", float("nan")))
        tol = float(r.get("tol", float("nan")))
        valid_flags.append(
            bool(
                math.isfinite(baseline_ms)
                and math.isfinite(adaptive_ms)
                and adaptive_ms > 0.0
                and baseline_points > 0
                and adaptive_evals > 0
                and math.isfinite(abs_err)
                and math.isfinite(tol)
            )
        )
    integrator_valid_row_fraction = float(sum(1 for x in valid_flags if x) / len(valid_flags)) if valid_flags else float("nan")

    topology_families = [f for f in ["polymer", "ring"] if f in fixtures_by_family]
    topology_gate_n_min = int(cfg.gate_n_min)

    def _family_scale_stats(fam: str) -> tuple[float, float, int]:
        fam_scale = [
            r
            for r in sample_rows
            if str(r.get("family") or "") == str(fam) and int(r.get("n_atoms", 0) or 0) >= int(topology_gate_n_min)
        ]
        n_fam = int(len(fam_scale))
        pass_rate_fam = float(sum(1 for r in fam_scale if bool(r.get("correctness_pass", False))) / n_fam) if n_fam > 0 else float("nan")
        med_speed = float(_median([float(r.get("speedup", float("nan"))) for r in fam_scale]))
        return float(med_speed), float(pass_rate_fam), int(n_fam)

    poly_speed_med, poly_pass_rate, poly_n_scale = _family_scale_stats("polymer")
    ring_speed_med, ring_pass_rate, ring_n_scale = _family_scale_stats("ring")

    poly_speed_verdict = "NOT_VALID_DUE_TO_CORRECTNESS"
    ring_speed_verdict = "NOT_VALID_DUE_TO_CORRECTNESS"
    if str(integrator_correctness_verdict) == "PASS_CORRECTNESS_AT_SCALE":
        if math.isfinite(poly_pass_rate) and poly_pass_rate < correctness_gate_rate:
            poly_speed_verdict = "FAIL_CORRECTNESS_AT_SCALE"
        elif poly_n_scale < int(cfg.min_scale_samples):
            poly_speed_verdict = "INCONCLUSIVE_NOT_ENOUGH_SCALE_SAMPLES"
        elif math.isfinite(poly_speed_med) and poly_speed_med >= float(cfg.speedup_gate_break_even):
            poly_speed_verdict = "PASS_SPEEDUP_AT_SCALE"
        else:
            poly_speed_verdict = "FAIL_SPEEDUP_AT_SCALE"

        if math.isfinite(ring_pass_rate) and ring_pass_rate < correctness_gate_rate:
            ring_speed_verdict = "FAIL_CORRECTNESS_AT_SCALE"
        elif ring_n_scale < int(cfg.min_scale_samples):
            ring_speed_verdict = "INCONCLUSIVE_NOT_ENOUGH_SCALE_SAMPLES"
        elif math.isfinite(ring_speed_med) and ring_speed_med >= float(cfg.speedup_gate_break_even):
            ring_speed_verdict = "PASS_SPEEDUP_AT_SCALE"
        else:
            ring_speed_verdict = "FAIL_SPEEDUP_AT_SCALE"

    topology_hardness_verdict = "NOT_VALID_DUE_TO_CORRECTNESS"
    topology_hardness_reason = f"NOT_VALID: integrator_correctness_verdict={integrator_correctness_verdict}"
    if str(integrator_correctness_verdict) == "PASS_CORRECTNESS_AT_SCALE":
        poly_ok = str(poly_speed_verdict) == "PASS_SPEEDUP_AT_SCALE"
        ring_ok = str(ring_speed_verdict) == "PASS_SPEEDUP_AT_SCALE"
        if poly_ok and ring_ok:
            topology_hardness_verdict = "SUCCESS_TOPOLOGY_ROBUST"
        elif poly_ok and not ring_ok:
            topology_hardness_verdict = "ILLUSION_CONFIRMED_TOPOLOGY_DEPENDENT"
        else:
            topology_hardness_verdict = "NO_SPEEDUP_YET"
        topology_hardness_reason = (
            f"polymer(verdict={poly_speed_verdict}, median={poly_speed_med}) "
            f"ring(verdict={ring_speed_verdict}, median={ring_speed_med}) "
            f"gate_n_min={topology_gate_n_min}"
        )

    metadata: dict[str, object] = {
        "schema_version": "hetero2_scale_speedup_metadata.v1",
        "law_ref": {
            "contract_path": "docs/contracts/INTEGRATION_SCALE_CONTRACT.md",
            "contract_commit": _detect_git_sha(),
            "contract_version": "p5.1.v1",
        },
        "curve_id": str(cfg.curve_id),
        "edge_weight_mode": str(cfg.edge_weight_mode),
        "potential_scale_gamma": float(cfg.potential_scale_gamma),
        "potential_unit_model": "dimensionless",
        "integrator_mode": "both",
        "integrator_energy_min": float(energy_min_global),
        "integrator_energy_max": float(energy_max_global),
        "integrator_energy_points": int(cfg.energy_points),
        "integrator_eta": float(cfg.dos_eta),
        "integrator_eps_abs": float(cfg.integrator_eps_abs),
        "integrator_eps_rel": float(cfg.integrator_eps_rel),
        "integrator_subdomains_max": int(cfg.integrator_subdomains_max),
        "integrator_poly_degree_max": int(cfg.integrator_poly_degree_max),
        "integrator_quad_order_max": int(cfg.integrator_quad_order_max),
        "integrator_eval_budget_max": int(cfg.integrator_eval_budget_max),
        "integrator_split_criterion": str(cfg.integrator_split_criterion),
        "gate_n_min": int(cfg.gate_n_min),
        "correctness_gate_rate": float(cfg.correctness_gate_rate),
        "min_scale_samples": int(cfg.min_scale_samples),
        "integrator_valid_row_fraction": float(integrator_valid_row_fraction),
        "integrator_correctness_pass_rate_at_scale": float(pass_rate_at_scale),
        "integrator_speedup_median_at_scale": float(speedup_median_at_scale),
        "integrator_eval_ratio_median_at_scale": float(eval_ratio_median_at_scale),
        "integrator_correctness_verdict": str(integrator_correctness_verdict),
        "integrator_speedup_verdict": str(integrator_speedup_verdict),
        "integrator_verdict_reason": str(integrator_verdict_reason),
        "scale_n_atoms_min": int(min(int(x) for x in by_n.keys())),
        "scale_n_atoms_max": int(max_n),
        "scale_overhead_region_n_max": int(cfg.overhead_region_n_max),
        "scale_gate_n_min": int(cfg.gate_n_min),
        "scale_speedup_gate_break_even": float(cfg.speedup_gate_break_even),
        "scale_speedup_gate_strong": float(cfg.speedup_gate_strong),
        "scale_break_even_n_estimate": int(break_even_n) if break_even_n is not None else None,
        "scale_speedup_median_at_maxN": float(max_bin_speed),
        "scale_speedup_verdict": str(scale_speedup_verdict),
        "scale_bins": bin_rows,
        "topology_families": list(topology_families),
        "topology_gate_n_min": int(topology_gate_n_min),
        "speedup_median_at_scale_polymer": float(poly_speed_med),
        "speedup_median_at_scale_ring": float(ring_speed_med),
        "speedup_verdict_at_scale_polymer": str(poly_speed_verdict),
        "speedup_verdict_at_scale_ring": str(ring_speed_verdict),
        "topology_hardness_verdict": str(topology_hardness_verdict),
        "topology_hardness_reason": str(topology_hardness_reason),
    }

    return sample_rows + bin_rows, metadata, adaptive_trace_rows, adaptive_summary_rows, timing_sample_rows


def write_p5_evidence_pack(
    *,
    out_dir: Path,
    cfg: P5Config,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)

    polymer_fixtures = generate_polymer_scale_fixtures(n_atoms_bins=cfg.n_atoms_bins, samples_per_bin=cfg.samples_per_bin, seed=cfg.seed)
    ring_fixtures = generate_ring_scale_fixtures(n_atoms_bins=cfg.n_atoms_bins, samples_per_bin=cfg.samples_per_bin, seed=cfg.seed)

    polymer_csv = out_dir / "fixtures_polymer_scale.csv"
    polymer_csv.write_text("", encoding="utf-8")
    with polymer_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["id", "smiles", "n_atoms", "n_hetero", "motif", "seed"])
        w.writeheader()
        for fx in polymer_fixtures:
            w.writerow(
                {
                    "id": fx.fixture_id,
                    "smiles": fx.smiles,
                    "n_atoms": int(fx.n_atoms),
                    "n_hetero": int(fx.n_hetero),
                    "motif": fx.motif,
                    "seed": int(fx.seed),
                }
            )

    ring_csv = out_dir / "fixtures_ring_scale.csv"
    ring_csv.write_text("", encoding="utf-8")
    with ring_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["id", "smiles", "n_atoms", "n_hetero", "motif", "seed"])
        w.writeheader()
        for fx in ring_fixtures:
            w.writerow(
                {
                    "id": fx.fixture_id,
                    "smiles": fx.smiles,
                    "n_atoms": int(fx.n_atoms),
                    "n_hetero": int(fx.n_hetero),
                    "motif": fx.motif,
                    "seed": int(fx.seed),
                }
            )

    t0 = time.perf_counter()
    rows, metadata, adaptive_trace_rows, adaptive_summary_rows, timing_sample_rows = compute_p5_speedup_rows(
        fixtures_by_family={"polymer": polymer_fixtures, "ring": ring_fixtures},
        cfg=cfg,
    )
    runtime_s = float(time.perf_counter() - t0)

    speedup_vs_n_csv = out_dir / "speedup_vs_n.csv"
    speedup_vs_n_by_family_csv = out_dir / "speedup_vs_n_by_family.csv"
    speedup_vs_n_md = out_dir / "speedup_vs_n.md"
    compare_csv = out_dir / "integration_compare.csv"
    speed_profile_csv = out_dir / "integration_speed_profile.csv"
    adaptive_trace_csv = out_dir / "adaptive_integration_trace.csv"
    adaptive_summary_json = out_dir / "adaptive_integration_summary.json"
    timing_breakdown_csv = out_dir / "timing_breakdown.csv"
    timing_breakdown_by_family_csv = out_dir / "timing_breakdown_by_family.csv"
    summary_csv = out_dir / "summary.csv"
    summary_metadata_json = out_dir / "summary_metadata.json"
    metrics_json = out_dir / "metrics.json"
    index_md = out_dir / "index.md"

    io_t0 = time.perf_counter()

    fieldnames = sorted({k for row in rows for k in row.keys()})
    speedup_vs_n_csv.write_text("", encoding="utf-8")
    with speedup_vs_n_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            w.writerow(row)

    by_family_n: dict[tuple[str, int], list[dict[str, object]]] = {}
    for row in rows:
        if row.get("row_kind") != "sample":
            continue
        fam = str(row.get("family") or "")
        if not fam:
            continue
        n = int(row.get("n_atoms", 0) or 0)
        by_family_n.setdefault((fam, n), []).append(row)

    family_bin_rows: list[dict[str, object]] = []
    for fam in ["polymer", "ring"]:
        ns = sorted({n for (f, n) in by_family_n.keys() if f == fam})
        for n in ns:
            rows_n = by_family_n.get((fam, int(n)), [])
            n_samples = int(len(rows_n))
            speeds = [float(r.get("speedup", float("nan"))) for r in rows_n]
            eval_ratios = [float(r.get("eval_ratio", float("nan"))) for r in rows_n]
            pass_flags = [bool(r.get("correctness_pass", False)) for r in rows_n]
            pass_rate = float(sum(1 for x in pass_flags if x) / n_samples) if n_samples > 0 else float("nan")
            family_bin_rows.append(
                {
                    "family": str(fam),
                    "n_atoms": int(n),
                    "n_samples": int(n_samples),
                    "median_speedup": float(_median(speeds)),
                    "median_eval_ratio": float(_median(eval_ratios)),
                    "correctness_pass_rate": float(pass_rate),
                }
            )

    speedup_vs_n_by_family_csv.write_text("", encoding="utf-8")
    with speedup_vs_n_by_family_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "family",
                "n_atoms",
                "n_samples",
                "median_speedup",
                "median_eval_ratio",
                "correctness_pass_rate",
            ],
        )
        w.writeheader()
        for row in family_bin_rows:
            w.writerow(row)

    speedup_vs_n_md.write_text(
        "\n".join(
            [
                "# Large-Scale Speedup Proof (P5)",
                "",
                "This evidence pack measures baseline vs adaptive DOS curve approximation runtime as a function of N_atoms.",
                "",
                f"- curve_id: {cfg.curve_id}",
                f"- n_atoms_bins: {list(cfg.n_atoms_bins)}",
                f"- samples_per_bin: {cfg.samples_per_bin}",
                f"- seed: {cfg.seed}",
                f"- energy_points: {cfg.energy_points}",
                f"- dos_eta: {cfg.dos_eta}",
                f"- edge_weight_mode: {cfg.edge_weight_mode}",
                f"- potential_scale_gamma: {cfg.potential_scale_gamma} (dimensionless)",
                f"- integrator_eps_abs: {cfg.integrator_eps_abs}",
                f"- integrator_eps_rel: {cfg.integrator_eps_rel}",
                f"- integrator_subdomains_max: {cfg.integrator_subdomains_max}",
                f"- integrator_poly_degree_max: {cfg.integrator_poly_degree_max}",
                f"- integrator_quad_order_max: {cfg.integrator_quad_order_max}",
                f"- integrator_eval_budget_max: {cfg.integrator_eval_budget_max}",
                f"- scale_overhead_region_n_max: {cfg.overhead_region_n_max}",
                f"- scale_gate_n_min: {cfg.gate_n_min}",
                "",
                "Outputs:",
                "- fixtures_polymer_scale.csv (input fixtures)",
                "- fixtures_ring_scale.csv (input fixtures; ring-suite)",
                "- speedup_vs_n.csv (samples + bin aggregates)",
                "- speedup_vs_n_by_family.csv (bin aggregates per topology family)",
                "- timing_breakdown.csv (cost decomposition; samples + bin aggregates)",
                "- timing_breakdown_by_family.csv (cost decomposition; bin aggregates per topology family)",
                "- integration_compare.csv (baseline vs adaptive correctness + timing)",
                "- integration_speed_profile.csv (eval ratio + speedup per sample)",
                "- adaptive_integration_trace.csv (segment trace; audit)",
                "- adaptive_integration_summary.json (per-sample integration summary; audit)",
                "- summary_metadata.json (scale verdict + config)",
                "",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    # Integration audit artifacts (P5.1): compare + speed profile + adaptive trace/summary.
    compare_fields = [
        "fixture_id",
        "curve_id",
        "baseline_value",
        "adaptive_value",
        "abs_err",
        "rel_err",
        "pass_tolerance",
        "baseline_walltime_ms",
        "adaptive_walltime_ms",
        "speedup",
        "adaptive_verdict_row",
    ]
    compare_csv.write_text("", encoding="utf-8")
    with compare_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=compare_fields)
        w.writeheader()
        for row in rows:
            if row.get("row_kind") != "sample":
                continue
            w.writerow(
                {
                    "fixture_id": row.get("id", ""),
                    "curve_id": row.get("curve_id", ""),
                    "baseline_value": row.get("baseline_value", ""),
                    "adaptive_value": row.get("adaptive_value", ""),
                    "abs_err": row.get("abs_err", ""),
                    "rel_err": row.get("rel_err", ""),
                    "pass_tolerance": row.get("correctness_pass", ""),
                    "baseline_walltime_ms": row.get("baseline_walltime_ms", ""),
                    "adaptive_walltime_ms": row.get("adaptive_walltime_ms", ""),
                    "speedup": row.get("speedup", ""),
                    "adaptive_verdict_row": row.get("adaptive_verdict_row", ""),
                }
            )

    speed_fields = [
        "fixture_id",
        "curve_id",
        "baseline_points",
        "adaptive_evals_total",
        "eval_ratio",
        "baseline_walltime_ms",
        "adaptive_walltime_ms",
        "speedup",
        "cache_hit_rate",
        "segments_used",
        "adaptive_verdict_row",
        "row_verdict",
    ]
    speed_profile_csv.write_text("", encoding="utf-8")
    with speed_profile_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=speed_fields)
        w.writeheader()
        for row in rows:
            if row.get("row_kind") != "sample":
                continue
            ok = bool(row.get("correctness_pass", False)) and str(row.get("adaptive_verdict_row", "")) not in {"FAIL_NUMERICAL"}
            w.writerow(
                {
                    "fixture_id": row.get("id", ""),
                    "curve_id": row.get("curve_id", ""),
                    "baseline_points": row.get("baseline_points", ""),
                    "adaptive_evals_total": row.get("adaptive_evals_total", ""),
                    "eval_ratio": row.get("eval_ratio", ""),
                    "baseline_walltime_ms": row.get("baseline_walltime_ms", ""),
                    "adaptive_walltime_ms": row.get("adaptive_walltime_ms", ""),
                    "speedup": row.get("speedup", ""),
                    "cache_hit_rate": row.get("cache_hit_rate", ""),
                    "segments_used": row.get("segments_used", ""),
                    "adaptive_verdict_row": row.get("adaptive_verdict_row", ""),
                    "row_verdict": "OK" if ok else "ERROR",
                }
            )

    adaptive_trace_fields = [
        "fixture_id",
        "curve_id",
        "n_atoms",
        "segment_id",
        "E_left",
        "E_right",
        "n_probe_points",
        "poly_degree",
        "quad_order",
        "n_function_evals",
        "error_est",
        "walltime_ms_segment",
        "split_reason",
    ]
    adaptive_trace_csv.write_text("", encoding="utf-8")
    with adaptive_trace_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=adaptive_trace_fields)
        w.writeheader()
        for tr in adaptive_trace_rows:
            w.writerow({k: tr.get(k, "") for k in adaptive_trace_fields})

    adaptive_summary_payload: dict[str, object] = {
        "schema_version": "hetero2_adaptive_integration_summary.v1",
        "curve_id": str(cfg.curve_id),
        "integrator_mode": "both",
        "samples": adaptive_summary_rows,
    }
    adaptive_summary_json.write_text(
        json.dumps(adaptive_summary_payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )

    # Summary rows: only per-sample results, in a stable schema for quick inspection.
    summary_fields = [
        "id",
        "status",
        "reason",
        "n_atoms",
        "n_hetero",
        "baseline_walltime_ms",
        "adaptive_walltime_ms",
        "speedup",
        "eval_ratio",
        "correctness_pass",
    ]
    summary_csv.write_text("", encoding="utf-8")
    with summary_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=summary_fields)
        w.writeheader()
        for row in rows:
            if row.get("row_kind") != "sample":
                continue
            ok = bool(row.get("correctness_pass", False))
            status = "OK" if ok else "ERROR"
            w.writerow(
                {
                    "id": row.get("id", ""),
                    "status": status,
                    "reason": "" if ok else "correctness_failed",
                    "n_atoms": row.get("n_atoms", ""),
                    "n_hetero": row.get("n_hetero", ""),
                    "baseline_walltime_ms": row.get("baseline_walltime_ms", ""),
                    "adaptive_walltime_ms": row.get("adaptive_walltime_ms", ""),
                    "speedup": row.get("speedup", ""),
                    "eval_ratio": row.get("eval_ratio", ""),
                    "correctness_pass": row.get("correctness_pass", ""),
                }
            )

    io_ms_total_estimate = float((time.perf_counter() - io_t0) * 1000.0)
    n_timing_samples = int(sum(1 for r in timing_sample_rows if r.get("row_kind") == "sample"))
    io_ms_per_sample_estimate = float(io_ms_total_estimate / n_timing_samples) if n_timing_samples > 0 else 0.0

    for r in timing_sample_rows:
        if r.get("row_kind") != "sample":
            continue
        r["io_ms"] = float(io_ms_per_sample_estimate)
        b = float(r.get("build_operator_ms", float("nan")))
        d = float(r.get("dos_ldos_eval_ms", float("nan")))
        i = float(r.get("integration_logic_ms", float("nan")))
        io = float(r.get("io_ms", float("nan")))
        r["total_ms"] = float(b + d + i + io) if all(math.isfinite(x) for x in [b, d, i, io]) else float("nan")

    timing_by_n: dict[int, list[dict[str, object]]] = {}
    for r in timing_sample_rows:
        if r.get("row_kind") != "sample":
            continue
        n = int(r.get("n_atoms", 0) or 0)
        timing_by_n.setdefault(n, []).append(r)

    timing_bin_rows: list[dict[str, object]] = []
    for n in sorted(timing_by_n.keys()):
        rows_n = timing_by_n[n]
        timing_bin_rows.append(
            {
                "row_kind": "bin",
                "n_atoms_bin": int(n),
                "n_samples": int(len(rows_n)),
                "median_build_operator_ms": float(_median([float(x.get("build_operator_ms", float("nan"))) for x in rows_n])),
                "median_dos_ldos_eval_ms": float(_median([float(x.get("dos_ldos_eval_ms", float("nan"))) for x in rows_n])),
                "median_integration_logic_ms": float(_median([float(x.get("integration_logic_ms", float("nan"))) for x in rows_n])),
                "median_io_ms": float(_median([float(x.get("io_ms", float("nan"))) for x in rows_n])),
                "median_total_ms": float(_median([float(x.get("total_ms", float("nan"))) for x in rows_n])),
            }
        )

    timing_rows = timing_sample_rows + timing_bin_rows
    timing_fields = sorted({k for row in timing_rows for k in row.keys()})
    timing_breakdown_csv.write_text("", encoding="utf-8")
    with timing_breakdown_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=timing_fields)
        w.writeheader()
        for r in timing_rows:
            w.writerow(r)

    # P5.5: per-family timing aggregates (bin-level) to compare polymer vs ring cost profiles.
    timing_by_family_n: dict[tuple[str, int], list[dict[str, object]]] = {}
    for r in timing_sample_rows:
        if r.get("row_kind") != "sample":
            continue
        fam = str(r.get("family") or "")
        if not fam:
            continue
        n = int(r.get("n_atoms", 0) or 0)
        timing_by_family_n.setdefault((fam, n), []).append(r)

    timing_family_bin_rows: list[dict[str, object]] = []
    for fam in ["polymer", "ring"]:
        ns = sorted({n for (f, n) in timing_by_family_n.keys() if f == fam})
        for n in ns:
            rows_n = timing_by_family_n.get((fam, int(n)), [])
            timing_family_bin_rows.append(
                {
                    "family": str(fam),
                    "n_atoms": int(n),
                    "n_samples": int(len(rows_n)),
                    "median_build_operator_ms": float(_median([float(x.get("build_operator_ms", float("nan"))) for x in rows_n])),
                    "median_dos_ldos_eval_ms": float(_median([float(x.get("dos_ldos_eval_ms", float("nan"))) for x in rows_n])),
                    "median_integration_logic_ms": float(_median([float(x.get("integration_logic_ms", float("nan"))) for x in rows_n])),
                    "median_io_ms": float(_median([float(x.get("io_ms", float("nan"))) for x in rows_n])),
                    "median_total_ms": float(_median([float(x.get("total_ms", float("nan"))) for x in rows_n])),
                }
            )

    timing_breakdown_by_family_csv.write_text("", encoding="utf-8")
    with timing_breakdown_by_family_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "family",
                "n_atoms",
                "n_samples",
                "median_build_operator_ms",
                "median_dos_ldos_eval_ms",
                "median_integration_logic_ms",
                "median_io_ms",
                "median_total_ms",
            ],
        )
        w.writeheader()
        for row in timing_family_bin_rows:
            w.writerow(row)

    scale_timing_samples = [r for r in timing_sample_rows if int(r.get("n_atoms", 0) or 0) >= int(cfg.gate_n_min)]
    build_med = float(_median([float(r.get("build_operator_ms", float("nan"))) for r in scale_timing_samples]))
    dos_med = float(_median([float(r.get("dos_ldos_eval_ms", float("nan"))) for r in scale_timing_samples]))
    integ_med = float(_median([float(r.get("integration_logic_ms", float("nan"))) for r in scale_timing_samples]))
    io_med = float(_median([float(r.get("io_ms", float("nan"))) for r in scale_timing_samples]))
    total_med = float(_median([float(r.get("total_ms", float("nan"))) for r in scale_timing_samples]))

    # P5.5: per-family scale medians and a simple "ring cost gap" verdict to localize where ring loses.
    scale_by_family: dict[str, list[dict[str, object]]] = {}
    for fam in ["polymer", "ring"]:
        scale_by_family[fam] = [
            r
            for r in timing_sample_rows
            if r.get("row_kind") == "sample"
            and str(r.get("family") or "") == fam
            and int(r.get("n_atoms", 0) or 0) >= int(cfg.gate_n_min)
        ]

    def _median_key(rows: list[dict[str, object]], key: str) -> float:
        return float(_median([float(r.get(key, float("nan"))) for r in rows]))

    poly_rows = scale_by_family.get("polymer", [])
    ring_rows = scale_by_family.get("ring", [])
    poly_n = int(len(poly_rows))
    ring_n = int(len(ring_rows))

    poly_build = _median_key(poly_rows, "build_operator_ms")
    poly_dos = _median_key(poly_rows, "dos_ldos_eval_ms")
    poly_integ = _median_key(poly_rows, "integration_logic_ms")
    poly_io = _median_key(poly_rows, "io_ms")
    poly_total = _median_key(poly_rows, "total_ms")

    ring_build = _median_key(ring_rows, "build_operator_ms")
    ring_dos = _median_key(ring_rows, "dos_ldos_eval_ms")
    ring_integ = _median_key(ring_rows, "integration_logic_ms")
    ring_io = _median_key(ring_rows, "io_ms")
    ring_total = _median_key(ring_rows, "total_ms")

    def _ratio(num: float, den: float) -> float:
        return float(num / den) if math.isfinite(num) and math.isfinite(den) and den > 0.0 else float("nan")

    ratio_build = _ratio(ring_build, poly_build)
    ratio_dos = _ratio(ring_dos, poly_dos)
    ratio_integ = _ratio(ring_integ, poly_integ)
    ratio_io = _ratio(ring_io, poly_io)
    ratio_total = _ratio(ring_total, poly_total)

    gap_verdict = "INCONCLUSIVE_NOT_ENOUGH_SCALE_SAMPLES"
    if poly_n >= int(cfg.min_scale_samples) and ring_n >= int(cfg.min_scale_samples) and math.isfinite(ratio_total):
        if ratio_total <= 1.0:
            gap_verdict = "RING_NOT_SLOWER_THAN_POLYMER"
        else:
            top = max(
                (ratio_build, "RING_SLOWER_DUE_TO_BUILD_OPERATOR"),
                (ratio_dos, "RING_SLOWER_DUE_TO_DOS_LDOS_EVAL"),
                (ratio_integ, "RING_SLOWER_DUE_TO_INTEGRATION_LOGIC"),
                (ratio_io, "RING_SLOWER_DUE_TO_IO"),
                key=lambda x: float(x[0]) if math.isfinite(float(x[0])) else float("-inf"),
            )
            gap_verdict = str(top[1]) if math.isfinite(float(top[0])) else "RING_SLOWER_MIXED"

    gap_reason = (
        f"scale_n_min={int(cfg.gate_n_min)} polymer_n={poly_n} ring_n={ring_n} "
        f"ratio_total={ratio_total} ratio_build={ratio_build} ratio_dos={ratio_dos} ratio_integration_logic={ratio_integ} ratio_io={ratio_io}"
    )

    metadata["cost_median_build_operator_ms_at_scale_polymer"] = float(poly_build)
    metadata["cost_median_dos_ldos_eval_ms_at_scale_polymer"] = float(poly_dos)
    metadata["cost_median_integration_logic_ms_at_scale_polymer"] = float(poly_integ)
    metadata["cost_median_io_ms_at_scale_polymer_estimate"] = float(poly_io)
    metadata["cost_median_total_ms_at_scale_polymer_estimate"] = float(poly_total)
    metadata["cost_scale_samples_polymer"] = int(poly_n)

    metadata["cost_median_build_operator_ms_at_scale_ring"] = float(ring_build)
    metadata["cost_median_dos_ldos_eval_ms_at_scale_ring"] = float(ring_dos)
    metadata["cost_median_integration_logic_ms_at_scale_ring"] = float(ring_integ)
    metadata["cost_median_io_ms_at_scale_ring_estimate"] = float(ring_io)
    metadata["cost_median_total_ms_at_scale_ring_estimate"] = float(ring_total)
    metadata["cost_scale_samples_ring"] = int(ring_n)

    metadata["cost_ratio_ring_vs_polymer_build_operator_ms_at_scale"] = float(ratio_build)
    metadata["cost_ratio_ring_vs_polymer_dos_ldos_eval_ms_at_scale"] = float(ratio_dos)
    metadata["cost_ratio_ring_vs_polymer_integration_logic_ms_at_scale"] = float(ratio_integ)
    metadata["cost_ratio_ring_vs_polymer_io_ms_at_scale"] = float(ratio_io)
    metadata["cost_ratio_ring_vs_polymer_total_ms_at_scale_estimate"] = float(ratio_total)

    metadata["topology_ring_cost_gap_verdict_at_scale"] = str(gap_verdict)
    metadata["topology_ring_cost_gap_reason_at_scale"] = str(gap_reason)

    # P5.6: ring speedup law (ring-suite KPI at scale; correctness gates speed).
    ring_scale_speed_rows = [
        r
        for r in rows
        if r.get("row_kind") == "sample"
        and str(r.get("family") or "") == "ring"
        and int(r.get("n_atoms", 0) or 0) >= int(cfg.gate_n_min)
    ]
    ring_n_speed = int(len(ring_scale_speed_rows))
    ring_pass_rate_speed = (
        float(sum(1 for r in ring_scale_speed_rows if bool(r.get("correctness_pass", False))) / ring_n_speed)
        if ring_n_speed > 0
        else float("nan")
    )
    ring_speed_med_speed = float(_median([float(r.get("speedup", float("nan"))) for r in ring_scale_speed_rows]))
    ring_eval_ratio_med_speed = float(_median([float(r.get("eval_ratio", float("nan"))) for r in ring_scale_speed_rows]))

    gate_speedup = float(cfg.speedup_gate_break_even)
    poly_speed_med = float(metadata.get("speedup_median_at_scale_polymer", float("nan")))

    ring_speedup_verdict = "NO_SPEEDUP_YET"
    if ring_n_speed > 0 and math.isfinite(ring_pass_rate_speed) and ring_pass_rate_speed < float(cfg.correctness_gate_rate):
        ring_speedup_verdict = "NOT_VALID_DUE_TO_CORRECTNESS"
    elif ring_n_speed < int(cfg.min_scale_samples):
        ring_speedup_verdict = "NO_SPEEDUP_YET"
    elif math.isfinite(ring_speed_med_speed) and ring_speed_med_speed >= gate_speedup:
        ring_speedup_verdict = "PASS_RING_SPEEDUP_AT_SCALE"
    elif (
        math.isfinite(ring_speed_med_speed)
        and ring_speed_med_speed < gate_speedup
        and math.isfinite(poly_speed_med)
        and poly_speed_med >= gate_speedup
    ):
        ring_speedup_verdict = "FAIL_RING_SPEEDUP_AT_SCALE"
    else:
        ring_speedup_verdict = "NO_SPEEDUP_YET"

    ring_speedup_reason = (
        f"gate_n_min={int(cfg.gate_n_min)} min_scale_samples={int(cfg.min_scale_samples)} ring_n={ring_n_speed} "
        f"correctness_gate_rate={float(cfg.correctness_gate_rate)} ring_pass_rate={ring_pass_rate_speed} "
        f"ring_median_speedup={ring_speed_med_speed} polymer_median_speedup={poly_speed_med} gate_speedup={gate_speedup} "
        f"ring_cost_gap_verdict={metadata.get('topology_ring_cost_gap_verdict_at_scale')}"
    )

    metadata["ring_speedup_median_at_scale"] = float(ring_speed_med_speed)
    metadata["ring_eval_ratio_median_at_scale"] = float(ring_eval_ratio_med_speed)
    metadata["ring_correctness_pass_rate_at_scale"] = float(ring_pass_rate_speed)
    metadata["ring_speedup_verdict_at_scale"] = str(ring_speedup_verdict)
    metadata["ring_speedup_verdict_reason_at_scale"] = str(ring_speedup_reason)

    denom_kpi = float(dos_med + integ_med + io_med) if all(math.isfinite(x) for x in [dos_med, integ_med, io_med]) else float("nan")
    share_dos = float(dos_med / denom_kpi) if math.isfinite(denom_kpi) and denom_kpi > 0.0 else float("nan")
    share_integ = float(integ_med / denom_kpi) if math.isfinite(denom_kpi) and denom_kpi > 0.0 else float("nan")
    share_io = float(io_med / denom_kpi) if math.isfinite(denom_kpi) and denom_kpi > 0.0 else float("nan")

    bottleneck_verdict = "MIXED"
    if math.isfinite(share_dos) and math.isfinite(share_integ) and math.isfinite(share_io):
        top = max((share_dos, "BOTTLENECK_IS_DOS_LDOS"), (share_integ, "BOTTLENECK_IS_INTEGRATOR"), (share_io, "BOTTLENECK_IS_IO"), key=lambda x: x[0])
        bottleneck_verdict = str(top[1]) if float(top[0]) >= 0.6 else "MIXED"

    denom_total = float(build_med + dos_med + integ_med + io_med) if all(math.isfinite(x) for x in [build_med, dos_med, integ_med, io_med]) else float("nan")
    frac_build = float(build_med / denom_total) if math.isfinite(denom_total) and denom_total > 0.0 else float("nan")
    frac_dos = float(dos_med / denom_total) if math.isfinite(denom_total) and denom_total > 0.0 else float("nan")
    frac_integ = float(integ_med / denom_total) if math.isfinite(denom_total) and denom_total > 0.0 else float("nan")
    frac_io = float(io_med / denom_total) if math.isfinite(denom_total) and denom_total > 0.0 else float("nan")

    metadata["cost_bottleneck_verdict_at_scale"] = str(bottleneck_verdict)
    metadata["cost_median_build_operator_ms_at_scale"] = float(build_med)
    metadata["cost_median_dos_ldos_eval_ms_at_scale"] = float(dos_med)
    metadata["cost_median_integration_logic_ms_at_scale"] = float(integ_med)
    metadata["cost_median_io_ms_at_scale_estimate"] = float(io_med)
    metadata["cost_median_total_ms_at_scale_estimate"] = float(total_med)
    metadata["cost_fraction_build_operator_at_scale_estimate"] = float(frac_build)
    metadata["cost_fraction_dos_ldos_eval_at_scale_estimate"] = float(frac_dos)
    metadata["cost_fraction_integration_logic_at_scale_estimate"] = float(frac_integ)
    metadata["cost_fraction_io_at_scale_estimate"] = float(frac_io)
    metadata["cost_share_dos_ldos_eval_among_kpi_at_scale_estimate"] = float(share_dos)
    metadata["cost_share_integration_logic_among_kpi_at_scale_estimate"] = float(share_integ)
    metadata["cost_share_io_among_kpi_at_scale_estimate"] = float(share_io)
    metadata["cost_io_walltime_ms_total_estimate"] = float(io_ms_total_estimate)
    metadata["cost_io_walltime_ms_per_sample_estimate"] = float(io_ms_per_sample_estimate)

    # P5.3 KPI: compare integration_logic_ms at scale against a pinned baseline ("before") and
    # record a simple verdict for the optimization step.
    integration_logic_before_ms_at_scale = 4.544935500007341  # P5.2 truth r3 baseline (registry-grade)
    before_override = str(os.environ.get("P5_COST_INTEGRATION_LOGIC_MS_AT_SCALE_BEFORE") or "").strip()
    if before_override:
        try:
            integration_logic_before_ms_at_scale = float(before_override)
        except ValueError:
            pass

    integration_logic_after_ms_at_scale = float(integ_med)
    integration_logic_speedup_at_scale = (
        float(integration_logic_before_ms_at_scale) / float(integration_logic_after_ms_at_scale)
        if math.isfinite(float(integration_logic_before_ms_at_scale))
        and math.isfinite(float(integration_logic_after_ms_at_scale))
        and float(integration_logic_after_ms_at_scale) > 0.0
        else float("nan")
    )

    n_scale_timing = int(len(scale_timing_samples))
    correctness_ok = str(metadata.get("integrator_correctness_verdict") or "") == "PASS_CORRECTNESS_AT_SCALE"
    cost_opt_verdict = "INCONCLUSIVE"
    if correctness_ok and n_scale_timing >= int(cfg.min_scale_samples) and math.isfinite(float(integration_logic_speedup_at_scale)):
        cost_opt_verdict = "PASS" if float(integration_logic_speedup_at_scale) >= 1.0 else "FAIL"

    metadata["cost_median_integration_logic_ms_at_scale_before"] = float(integration_logic_before_ms_at_scale)
    metadata["cost_median_integration_logic_ms_at_scale_after"] = float(integration_logic_after_ms_at_scale)
    metadata["cost_integration_logic_speedup_at_scale"] = float(integration_logic_speedup_at_scale)
    metadata["cost_integration_logic_opt_verdict_at_scale"] = str(cost_opt_verdict)

    summary_metadata_json.write_text(json.dumps(metadata, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")

    # Metrics: keep the minimal "counts" gate compatible with existing publish workflows.
    n_ok = int(sum(1 for row in rows if row.get("row_kind") == "sample" and bool(row.get("correctness_pass", False))))
    n_err = int(sum(1 for row in rows if row.get("row_kind") == "sample" and not bool(row.get("correctness_pass", False))))
    metrics_payload = {
        "schema_version": "hetero2_metrics_scale.v1",
        "counts": {"OK": int(n_ok), "SKIP": 0, "ERROR": int(n_err)},
        "runtime_s_total": float(runtime_s),
        "scale": {
            "scale_n_atoms_min": metadata.get("scale_n_atoms_min"),
            "scale_n_atoms_max": metadata.get("scale_n_atoms_max"),
            "scale_break_even_n_estimate": metadata.get("scale_break_even_n_estimate"),
            "scale_speedup_median_at_maxN": metadata.get("scale_speedup_median_at_maxN"),
            "scale_speedup_verdict": metadata.get("scale_speedup_verdict"),
        },
    }
    metrics_json.write_text(json.dumps(metrics_payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")

    index_md.write_text(
        "\n".join(
            [
                "# Evidence Index",
                "",
                "## P5 Large-Scale Proof",
                "",
                "- fixtures: ./fixtures_polymer_scale.csv",
                "- fixtures (ring-suite): ./fixtures_ring_scale.csv",
                "- speedup vs N: ./speedup_vs_n.csv",
                "- speedup vs N (by family): ./speedup_vs_n_by_family.csv",
                "- report: ./speedup_vs_n.md",
                "- timing breakdown: ./timing_breakdown.csv",
                "- timing breakdown (by family): ./timing_breakdown_by_family.csv",
                "- integration compare: ./integration_compare.csv",
                "- integration speed profile: ./integration_speed_profile.csv",
                "- adaptive trace: ./adaptive_integration_trace.csv",
                "- adaptive integration summary: ./adaptive_integration_summary.json",
                "- metadata: ./summary_metadata.json",
                "- metrics: ./metrics.json",
                "",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    file_infos = _compute_file_infos(out_dir, skip_names={"manifest.json", "checksums.sha256", "evidence_pack.zip"})
    _write_manifest(
        out_dir,
        config={
            "p5_config": {
                "n_atoms_bins": list(cfg.n_atoms_bins),
                "samples_per_bin": int(cfg.samples_per_bin),
                "seed": int(cfg.seed),
                "curve_id": str(cfg.curve_id),
                "energy_points": int(cfg.energy_points),
                "dos_eta": float(cfg.dos_eta),
                "edge_weight_mode": str(cfg.edge_weight_mode),
                "potential_scale_gamma": float(cfg.potential_scale_gamma),
                "integrator_eps_abs": float(cfg.integrator_eps_abs),
                "integrator_eps_rel": float(cfg.integrator_eps_rel),
                "integrator_subdomains_max": int(cfg.integrator_subdomains_max),
                "integrator_poly_degree_max": int(cfg.integrator_poly_degree_max),
                "integrator_quad_order_max": int(cfg.integrator_quad_order_max),
                "integrator_eval_budget_max": int(cfg.integrator_eval_budget_max),
                "integrator_split_criterion": str(cfg.integrator_split_criterion),
                "overhead_region_n_max": int(cfg.overhead_region_n_max),
                "gate_n_min": int(cfg.gate_n_min),
                "correctness_gate_rate": float(cfg.correctness_gate_rate),
                "min_scale_samples": int(cfg.min_scale_samples),
            }
        },
        files=file_infos,
    )
    file_infos_final = _compute_file_infos(out_dir, skip_names={"checksums.sha256", "evidence_pack.zip"})
    _write_checksums(out_dir, file_infos_final)
    _write_zip_pack(out_dir, zip_name="evidence_pack.zip")
    return out_dir / "evidence_pack.zip"
