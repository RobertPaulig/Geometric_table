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
from typing import Iterable, Sequence

import numpy as np
from scipy.linalg import eigh_tridiagonal

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
    fixtures: Sequence[PolymerScaleFixture],
    cfg: P5Config,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    adaptive_cfg = AdaptiveIntegrationConfig(
        eps_abs=float(cfg.integrator_eps_abs),
        eps_rel=float(cfg.integrator_eps_rel),
        subdomains_max=int(cfg.integrator_subdomains_max),
        poly_degree_max=int(cfg.integrator_poly_degree_max),
        quad_order_max=int(cfg.integrator_quad_order_max),
        eval_budget_max=int(cfg.integrator_eval_budget_max),
        split_criterion=str(cfg.integrator_split_criterion),
    )

    sample_rows: list[dict[str, object]] = []
    for fx in fixtures:
        diag, off, v0 = build_h_tridiagonal_chain(
            types_z=fx.types_z,
            edge_weight_mode=str(cfg.edge_weight_mode),
            potential_scale_gamma=float(cfg.potential_scale_gamma),
        )
        eigvals = eigh_tridiagonal(diag, off, eigvals_only=True, check_finite=False)
        grid = _auto_energy_grid(eigenvalues=eigvals, energy_points=int(cfg.energy_points), eta=float(cfg.dos_eta))
        if grid.size == 0:
            raise ValueError("Energy grid is empty (invalid eigenvalues range)")

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
        adaptive_evals_total = int(res.summary.get("evals_total", 0) or 0)
        segments_used = int(res.summary.get("segments_used", 0) or 0)
        cache_hit_rate = float(res.summary.get("cache_hit_rate", float("nan")))
        adaptive_verdict_row = str(res.summary.get("verdict", ""))

        base_arr = np.asarray(baseline_values, dtype=float)
        adapt_arr = np.asarray(res.values, dtype=float)
        base_value = float(np.max(np.abs(base_arr))) if base_arr.size else float("nan")
        abs_err = float(np.max(np.abs(adapt_arr - base_arr))) if base_arr.size and adapt_arr.size else float("nan")
        tol = float(cfg.integrator_eps_abs) + float(cfg.integrator_eps_rel) * abs(float(base_value)) if math.isfinite(float(base_value)) else float("nan")
        correctness_pass = bool(math.isfinite(float(abs_err)) and math.isfinite(float(tol)) and float(abs_err) <= float(tol))

        baseline_points = int(grid.size)
        eval_ratio = float(adaptive_evals_total) / float(baseline_points) if baseline_points > 0 else float("nan")
        speedup = float(baseline_ms / adaptive_ms) if adaptive_ms and adaptive_ms > 0.0 and math.isfinite(adaptive_ms) else float("nan")

        sample_rows.append(
            {
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
                "baseline_walltime_ms": float(baseline_ms),
                "adaptive_walltime_ms": float(adaptive_ms),
                "speedup": float(speedup),
                "baseline_points": int(baseline_points),
                "adaptive_evals_total": int(adaptive_evals_total),
                "eval_ratio": float(eval_ratio),
                "segments_used": int(segments_used),
                "cache_hit_rate": float(cache_hit_rate),
                "adaptive_verdict_row": str(adaptive_verdict_row),
                "abs_err": float(abs_err),
                "tol": float(tol),
                "correctness_pass": bool(correctness_pass),
                "eigs_checksum": str(curve_checksum_sha256(energy_grid=list(range(int(eigvals.size))), values=[float(x) for x in eigvals.tolist()])),
                "v0_checksum": str(curve_checksum_sha256(energy_grid=list(range(int(v0.size))), values=[float(x) for x in v0.tolist()])),
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

    metadata: dict[str, object] = {
        "schema_version": "hetero2_scale_speedup_metadata.v1",
        "curve_id": str(cfg.curve_id),
        "edge_weight_mode": str(cfg.edge_weight_mode),
        "potential_scale_gamma": float(cfg.potential_scale_gamma),
        "potential_unit_model": "dimensionless",
        "integrator_mode": "both",
        "integrator_eps_abs": float(cfg.integrator_eps_abs),
        "integrator_eps_rel": float(cfg.integrator_eps_rel),
        "integrator_subdomains_max": int(cfg.integrator_subdomains_max),
        "integrator_poly_degree_max": int(cfg.integrator_poly_degree_max),
        "integrator_quad_order_max": int(cfg.integrator_quad_order_max),
        "integrator_eval_budget_max": int(cfg.integrator_eval_budget_max),
        "integrator_split_criterion": str(cfg.integrator_split_criterion),
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
    }

    return sample_rows + bin_rows, metadata


def write_p5_evidence_pack(
    *,
    out_dir: Path,
    cfg: P5Config,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)

    fixtures = generate_polymer_scale_fixtures(n_atoms_bins=cfg.n_atoms_bins, samples_per_bin=cfg.samples_per_bin, seed=cfg.seed)

    fixtures_csv = out_dir / "fixtures_polymer_scale.csv"
    fixtures_csv.write_text("", encoding="utf-8")
    with fixtures_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["id", "smiles", "n_atoms", "n_hetero", "motif", "seed"])
        w.writeheader()
        for fx in fixtures:
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
    rows, metadata = compute_p5_speedup_rows(fixtures=fixtures, cfg=cfg)
    runtime_s = float(time.perf_counter() - t0)

    speedup_vs_n_csv = out_dir / "speedup_vs_n.csv"
    speedup_vs_n_md = out_dir / "speedup_vs_n.md"
    summary_csv = out_dir / "summary.csv"
    summary_metadata_json = out_dir / "summary_metadata.json"
    metrics_json = out_dir / "metrics.json"
    index_md = out_dir / "index.md"

    fieldnames = sorted({k for row in rows for k in row.keys()})
    speedup_vs_n_csv.write_text("", encoding="utf-8")
    with speedup_vs_n_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
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
                "- speedup_vs_n.csv (samples + bin aggregates)",
                "- summary_metadata.json (scale verdict + config)",
                "",
            ]
        )
        + "\n",
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
                "- speedup vs N: ./speedup_vs_n.csv",
                "- report: ./speedup_vs_n.md",
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
            }
        },
        files=file_infos,
    )
    file_infos_final = _compute_file_infos(out_dir, skip_names={"checksums.sha256", "evidence_pack.zip"})
    _write_checksums(out_dir, file_infos_final)
    _write_zip_pack(out_dir, zip_name="evidence_pack.zip")
    return out_dir / "evidence_pack.zip"
