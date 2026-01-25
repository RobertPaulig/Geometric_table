from __future__ import annotations

"""
ACCURACY-A4.0 - Signed/Orientable Edge Observable runner (opt-in).

SoT contract:
  docs/specs/accuracy_a4_0_signed_edge_observable.md

Key constraints (enforced here):
  - variant=B (fixed): B_ij := Re(exp(-i * theta_ij) * K_ij)
  - edge-only scoring: S_edge := sum_{(i,j) in E} w_ij * B_ij
  - 0 DOF: no kappa, no parameter grids, no nested selection
  - graph-only, LOOCV by group_id (evaluation-only; no training)
"""

import argparse
import csv
import json
import math
import os
import platform
import random
import shutil
import statistics
import subprocess
import sys
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from hetero2.chemgraph import ChemGraph
from hetero2.phase_channel import magnetic_laplacian, normalize_flux_phi, phase_matrix_flux_on_cycles, sssr_cycles_from_mol
from hetero2.physics_operator import AtomsDbV1, load_atoms_db_v1


class AccuracyA40Error(ValueError):
    pass


DEFAULT_INPUT_CSV = Path("data/accuracy/isomer_truth.v1.csv")
DEFAULT_OUT_DIR = Path("out_accuracy_a1_isomers_a4_0")

# Contract: variant=B uses phase-channel link phases; we keep the same fixed Φ baseline as A3.4/A3.5.
PHI_FIXED = float(normalize_flux_phi(math.pi / 2.0))

# Contract: 0 DOF; keep the same kernel settings as the established pipeline (A3.4/A3.5).
HEAT_TAU = 1.0

EDGE_WEIGHT_MODE = "bond_order_delta_chi"
EDGE_AROMATIC_MULT = 0.0
EDGE_DELTA_CHI_ALPHA = 1.0

EDGE_TOPK = 10


def _sha256_bytes(data: bytes) -> str:
    import hashlib

    h = hashlib.sha256()
    h.update(data)
    return h.hexdigest()


def _sha256_file(path: Path) -> str:
    import hashlib

    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def _sha256_text_normalized(path: Path) -> str:
    text = path.read_text(encoding="utf-8")
    normalized = "\n".join(text.splitlines()) + ("\n" if text else "")
    return _sha256_bytes(normalized.encode("utf-8"))


def _detect_git_sha() -> str:
    sha = str(os.environ.get("GITHUB_SHA") or "").strip()
    if sha:
        return sha
    try:
        repo_root = Path(__file__).resolve().parents[1]
        out = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo_root, stderr=subprocess.DEVNULL, text=True)
        sha = str(out or "").strip()
        if sha:
            return sha
    except Exception:
        pass
    return "UNKNOWN"


def _require_columns(fieldnames: list[str] | None, required: Iterable[str]) -> None:
    if not fieldnames:
        raise AccuracyA40Error("missing CSV header")
    missing = [c for c in required if c not in fieldnames]
    if missing:
        raise AccuracyA40Error(f"missing required columns: {missing}")


def _rankdata(values: list[float]) -> list[float]:
    indexed = sorted((float(v), i) for i, v in enumerate(values))
    ranks: list[float] = [0.0] * len(values)
    i = 0
    n = len(indexed)
    while i < n:
        j = i + 1
        while j < n and indexed[j][0] == indexed[i][0]:
            j += 1
        avg_rank = (float(i + 1) + float(j)) / 2.0
        for k in range(i, j):
            ranks[indexed[k][1]] = float(avg_rank)
        i = j
    return ranks


def _pearson_corr(x: list[float], y: list[float]) -> float:
    if len(x) != len(y) or len(x) < 2:
        return float("nan")
    mx = float(statistics.fmean(x))
    my = float(statistics.fmean(y))
    sx = math.sqrt(sum((xi - mx) ** 2 for xi in x))
    sy = math.sqrt(sum((yi - my) ** 2 for yi in y))
    if sx == 0.0 or sy == 0.0:
        return float("nan")
    cov = sum((xi - mx) * (yi - my) for xi, yi in zip(x, y))
    return float(cov / (sx * sy))


def _spearman_corr(x: list[float], y: list[float]) -> float:
    if len(x) != len(y) or len(x) < 2:
        return float("nan")
    rx = _rankdata(x)
    ry = _rankdata(y)
    return _pearson_corr(rx, ry)


def _pairwise_order_accuracy(truth: list[float], pred: list[float]) -> tuple[int, int, float]:
    if len(truth) != len(pred) or len(truth) < 2:
        return 0, 0, float("nan")
    correct = 0
    total = 0
    n = len(truth)
    for i in range(n):
        for j in range(i + 1, n):
            dt = float(truth[j]) - float(truth[i])
            if dt == 0.0:
                continue
            dp = float(pred[j]) - float(pred[i])
            total += 1
            if dp == 0.0:
                continue
            if dt * dp > 0.0:
                correct += 1
    acc = float(correct) / float(total) if total else float("nan")
    return correct, total, acc


def _attach_group_rel_and_ranks(records: list[dict[str, object]]) -> None:
    by_group: dict[str, list[dict[str, object]]] = {}
    for r in records:
        by_group.setdefault(str(r.get("group_id") or ""), []).append(r)

    for gid, group in by_group.items():
        pred_raw = [float(rr["pred_raw"]) for rr in group]
        truth_rel = [float(rr["truth_rel_kcalmol"]) for rr in group]

        pred_min = float(min(pred_raw)) if pred_raw else 0.0
        for rr in group:
            rr["group_id"] = str(gid)
            rr["pred_rel"] = float(float(rr["pred_raw"]) - pred_min)

        # 1 = best (lowest)
        pred_ranks = _rankdata(pred_raw)
        truth_ranks = _rankdata(truth_rel)
        for rr, pr, tr in zip(group, pred_ranks, truth_ranks):
            rr["pred_rank"] = float(pr)
            rr["truth_rank"] = float(tr)


def _compute_group_metrics(records: list[dict[str, object]], *, pred_rel_key: str) -> dict[str, dict[str, object]]:
    by_group: dict[str, list[dict[str, object]]] = {}
    for r in records:
        gid = str(r.get("group_id") or "")
        by_group.setdefault(gid, []).append(r)

    out: dict[str, dict[str, object]] = {}
    for gid, group in by_group.items():
        group_sorted = sorted(group, key=lambda rr: str(rr.get("id")))
        truth = [float(rr["truth_rel_kcalmol"]) for rr in group_sorted]
        pred = [float(rr[pred_rel_key]) for rr in group_sorted]
        spearman = _spearman_corr(pred, truth)
        correct, total, acc = _pairwise_order_accuracy(truth, pred)

        best_pred_idx = int(np.argmin(np.asarray(pred, dtype=float)))
        truth_min = float(min(truth))
        truth_best = {str(group_sorted[i]["id"]) for i, t in enumerate(truth) if float(t) == truth_min}
        pred_best = str(group_sorted[best_pred_idx]["id"])
        top1 = 1.0 if pred_best in truth_best else 0.0

        out[gid] = {
            "group_id": str(gid),
            "n": int(len(group_sorted)),
            "spearman_pred_vs_truth": float(spearman),
            "pairwise_order_accuracy": float(acc),
            "pairwise_correct": int(correct),
            "pairwise_total": int(total),
            "top1_accuracy": float(top1),
            "pred_best_id": pred_best,
            "truth_best_ids": sorted(truth_best),
        }
    return out


def _aggregate_loocv_metrics(group_metrics: dict[str, dict[str, object]]) -> dict[str, object]:
    spearman_vals: list[float] = []
    top1_vals: list[float] = []
    negative_groups: list[str] = []
    pairwise_correct = 0
    pairwise_total = 0
    groups_total = int(len(group_metrics))

    for gid, gm in group_metrics.items():
        s = float(gm.get("spearman_pred_vs_truth", float("nan")))
        if math.isfinite(s):
            spearman_vals.append(float(s))
            if float(s) < 0.0:
                negative_groups.append(str(gid))
        top1_vals.append(float(gm.get("top1_accuracy", float("nan"))))
        pairwise_correct += int(gm.get("pairwise_correct", 0) or 0)
        pairwise_total += int(gm.get("pairwise_total", 0) or 0)

    mean_spearman = float(statistics.fmean(spearman_vals)) if spearman_vals else float("nan")
    median_spearman = float(statistics.median(spearman_vals)) if spearman_vals else float("nan")
    top1_mean = float(statistics.fmean(top1_vals)) if top1_vals else float("nan")
    pairwise_acc = float(pairwise_correct) / float(pairwise_total) if pairwise_total else float("nan")

    return {
        "groups_total": int(groups_total),
        "mean_spearman_by_group": float(mean_spearman),
        "median_spearman_by_group": float(median_spearman),
        "pairwise_order_accuracy_overall": float(pairwise_acc),
        "pairwise_correct": int(pairwise_correct),
        "pairwise_total": int(pairwise_total),
        "top1_accuracy_mean": float(top1_mean),
        "num_groups_spearman_negative": int(len(negative_groups)),
        "negative_spearman_groups": sorted(negative_groups),
    }


def _worst_groups_by_spearman(group_metrics: dict[str, dict[str, object]], *, n: int = 3) -> list[dict[str, object]]:
    scored: list[tuple[float, str, dict[str, object]]] = []
    for gid, gm in group_metrics.items():
        s = float(gm.get("spearman_pred_vs_truth", float("nan")))
        scored.append((s, str(gid), dict(gm)))
    scored.sort(key=lambda x: (float("inf") if not math.isfinite(float(x[0])) else float(x[0]), x[1]))
    out: list[dict[str, object]] = []
    for s, gid, gm in scored[: int(n)]:
        row = dict(gm)
        row["group_id"] = str(gid)
        row["spearman_pred_vs_truth"] = float(s)
        out.append(row)
    return out


def _ring_edge_counts(cycles: Sequence[Sequence[int]]) -> tuple[int, int, int]:
    ring_count = 0
    edge_counts: dict[frozenset[int], int] = {}
    for cyc in cycles:
        atoms = [int(x) for x in cyc]
        m = len(atoms)
        if m < 3:
            continue
        ring_count += 1
        for k in range(m):
            a = atoms[k]
            b = atoms[(k + 1) % m]
            e = frozenset((a, b))
            edge_counts[e] = int(edge_counts.get(e, 0)) + 1
    n_ring_edges = int(len(edge_counts))
    n_shared_edges = int(sum(1 for c in edge_counts.values() if int(c) > 1))
    return int(ring_count), int(n_ring_edges), int(n_shared_edges)


def _build_weight_adjacency(
    bonds: Sequence[Sequence[object]],
    *,
    n: int,
    atoms_db: AtomsDbV1,
    types: Sequence[int],
    mode: str,
    aromatic_multiplier: float,
    delta_chi_alpha: float,
) -> np.ndarray:
    if mode not in {"bond_order", "bond_order_delta_chi"}:
        raise AccuracyA40Error(f"invalid edge_weight_mode: {mode}")
    am = float(aromatic_multiplier)
    alpha = float(delta_chi_alpha)

    chi: np.ndarray | None = None
    if mode == "bond_order_delta_chi":
        missing = sorted({int(z) for z in types if int(z) not in atoms_db.chi_by_atomic_num})
        if missing:
            raise AccuracyA40Error(f"missing atoms_db chi for Z={missing}")
        chi = np.asarray([float(atoms_db.chi_by_atomic_num[int(z)]) for z in types], dtype=float)

    w_adj = np.zeros((int(n), int(n)), dtype=float)
    for i, j, bond_order, aromatic in bonds:
        a, b = int(i), int(j)
        if a == b:
            continue
        w = float(bond_order)
        if int(aromatic):
            w = w * (1.0 + am)
        if chi is not None:
            w = w * (1.0 + alpha * float(abs(float(chi[a]) - float(chi[b]))))
        w_adj[a, b] = w
        w_adj[b, a] = w
    return w_adj


def _phase_operator_for_row(
    *,
    weights: np.ndarray,
    cycles_heavy: Sequence[Sequence[int]],
    flux_phi: float,
) -> tuple[np.ndarray, np.ndarray]:
    n = int(weights.shape[0])
    A = phase_matrix_flux_on_cycles(n=n, cycles=cycles_heavy, phi=float(flux_phi))
    L_phase = magnetic_laplacian(weights=weights, A=A)
    return np.asarray(L_phase, dtype=np.complex128), np.asarray(A, dtype=float)


def _heat_kernel_matrix_shifted_trace_normalized_hermitian(
    H: np.ndarray, *, tau: float
) -> tuple[np.ndarray, float, float, float]:
    """
    K = exp(-tau * (H - lambda_min I)) / trace(exp(-tau * (H - lambda_min I))).
    Returns: (K, trace_heat, lambda_min, lambda_max).
    """
    tau_val = float(tau)
    if tau_val <= 0.0 or not math.isfinite(tau_val):
        raise AccuracyA40Error("heat_tau must be > 0 and finite")

    H0 = np.asarray(H)
    if H0.ndim != 2 or H0.shape[0] != H0.shape[1] or H0.shape[0] == 0:
        raise AccuracyA40Error("H must be non-empty square")

    eigvals, eigvecs = np.linalg.eigh(H0)
    eigvals = np.asarray(eigvals).reshape(-1)
    eigvecs = np.asarray(eigvecs)
    if eigvals.size == 0:
        raise AccuracyA40Error("H must be non-empty")

    lambda_min = float(np.real(eigvals[0]))
    lambda_max = float(np.real(eigvals[-1]))
    weights = np.exp(-tau_val * (np.real(eigvals) - float(lambda_min)))
    trace_heat = float(np.sum(weights))
    if not math.isfinite(trace_heat) or trace_heat <= 0.0:
        raise AccuracyA40Error("invalid trace_heat in kernel")
    w_norm = weights / float(trace_heat)

    K = (eigvecs * w_norm) @ np.conj(eigvecs).T
    return np.asarray(K, dtype=np.complex128), float(trace_heat), float(lambda_min), float(lambda_max)


@dataclass(frozen=True)
class _Row:
    mid: str
    gid: str
    smiles: str
    truth_rel: float
    n_heavy_atoms: int
    types: tuple[int, ...]
    bonds: tuple[tuple[int, int, float, int], ...]
    cycles_heavy: tuple[tuple[int, ...], ...]
    n_rings: int
    n_ring_edges: int
    n_shared_ring_edges: int


def _heavy_mapping_from_mol(mol) -> tuple[list[int], dict[int, int]]:
    heavy: list[int] = []
    mapping: dict[int, int] = {}
    for idx, atom in enumerate(mol.GetAtoms()):
        if int(atom.GetAtomicNum()) > 1:
            mapping[int(idx)] = int(len(heavy))
            heavy.append(int(idx))
    return heavy, mapping


def _load_rows(input_csv: Path) -> list[_Row]:
    reader = csv.DictReader(input_csv.read_text(encoding="utf-8").splitlines())
    _require_columns(reader.fieldnames, required=["id", "group_id", "smiles", "energy_rel_kcalmol"])
    raw_rows = [dict(r) for r in reader]
    if not raw_rows:
        raise AccuracyA40Error("input_csv has no data rows")

    rows: list[_Row] = []
    for r in raw_rows:
        mid = str(r.get("id") or "").strip()
        gid = str(r.get("group_id") or "").strip()
        smiles = str(r.get("smiles") or "").strip()
        if not mid or not gid or not smiles:
            raise AccuracyA40Error(f"invalid row (id/group_id/smiles required): {r}")

        truth_rel = float(str(r.get("energy_rel_kcalmol") or "").strip())

        cg = ChemGraph(smiles)
        mol = cg.mol
        n_heavy_atoms = int(cg.n_heavy_atoms())
        types = tuple(int(z) for z in cg.heavy_atom_types())
        heavy, heavy_mapping = _heavy_mapping_from_mol(mol)
        if len(heavy) != n_heavy_atoms:
            raise AccuracyA40Error("unexpected heavy atom count mismatch")

        bonds_set: set[tuple[int, int, float, int]] = set()
        for bond in mol.GetBonds():
            u = int(bond.GetBeginAtomIdx())
            v = int(bond.GetEndAtomIdx())
            if u not in heavy_mapping or v not in heavy_mapping:
                continue
            i = int(heavy_mapping[u])
            j = int(heavy_mapping[v])
            if i == j:
                continue
            if i > j:
                i, j = j, i
            bo = float(bond.GetBondTypeAsDouble())
            aromatic = 1 if bool(bond.GetIsAromatic()) else 0
            bonds_set.add((int(i), int(j), float(bo), int(aromatic)))

        bonds = tuple(sorted(bonds_set))

        cycles_full = sssr_cycles_from_mol(mol)
        cycles_heavy: list[tuple[int, ...]] = []
        for cyc in cycles_full:
            mapped: list[int] = []
            for a in cyc:
                if int(a) not in heavy_mapping:
                    raise AccuracyA40Error("unexpected non-heavy atom in SSSR cycle")
                mapped.append(int(heavy_mapping[int(a)]))
            cycles_heavy.append(tuple(mapped))

        n_rings, n_ring_edges, n_shared = _ring_edge_counts(cycles_heavy)

        rows.append(
            _Row(
                mid=str(mid),
                gid=str(gid),
                smiles=str(smiles),
                truth_rel=float(truth_rel),
                n_heavy_atoms=int(n_heavy_atoms),
                types=tuple(types),
                bonds=tuple(bonds),
                cycles_heavy=tuple(cycles_heavy),
                n_rings=int(n_rings),
                n_ring_edges=int(n_ring_edges),
                n_shared_ring_edges=int(n_shared),
            )
        )
    return rows


def _compute_edge_score_for_row(
    row: _Row,
    *,
    atoms_db: AtomsDbV1,
) -> tuple[dict[str, object], dict[str, object], list[dict[str, object]]]:
    w_adj = _build_weight_adjacency(
        row.bonds,
        n=int(row.n_heavy_atoms),
        atoms_db=atoms_db,
        types=row.types,
        mode=str(EDGE_WEIGHT_MODE),
        aromatic_multiplier=float(EDGE_AROMATIC_MULT),
        delta_chi_alpha=float(EDGE_DELTA_CHI_ALPHA),
    )

    lap_phase, A = _phase_operator_for_row(weights=w_adj, cycles_heavy=row.cycles_heavy, flux_phi=float(PHI_FIXED))
    K, trace_heat, lambda_min, lambda_max = _heat_kernel_matrix_shifted_trace_normalized_hermitian(
        np.asarray(lap_phase, dtype=np.complex128), tau=float(HEAT_TAU)
    )

    edge_rows: list[dict[str, object]] = []
    s_edge = 0.0
    sum_abs = 0.0
    for i, j, bond_order, aromatic in row.bonds:
        a = int(i)
        b = int(j)
        if a == b:
            continue
        w = float(w_adj[a, b])
        theta = float(A[a, b])
        kij = complex(K[a, b])
        qij = complex(np.exp(-1j * theta) * kij)
        bij = float(np.real(qij))
        contrib = float(w * bij)
        s_edge += float(contrib)
        sum_abs += float(abs(contrib))

        edge_rows.append(
            {
                "group_id": str(row.gid),
                "id": str(row.mid),
                "smiles": str(row.smiles),
                "edge_u": int(a),
                "edge_v": int(b),
                "bond_order": float(bond_order),
                "bond_is_aromatic": int(aromatic),
                "w_ij": float(w),
                "theta_ij": float(theta),
                "K_ij_re": float(np.real(kij)),
                "K_ij_im": float(np.imag(kij)),
                "q_ij_re": float(np.real(qij)),
                "q_ij_im": float(np.imag(qij)),
                "B_ij": float(bij),
                "edge_contrib": float(contrib),
                "abs_edge_contrib": float(abs(contrib)),
            }
        )

    edge_rows_sorted = sorted(edge_rows, key=lambda r: float(r.get("abs_edge_contrib") or 0.0), reverse=True)

    pred_rec: dict[str, object] = {
        "id": str(row.mid),
        "group_id": str(row.gid),
        "smiles": str(row.smiles),
        "truth_rel_kcalmol": float(row.truth_rel),
        "pred_raw": float(s_edge),
        "variant": "B",
        "phi_fixed": float(PHI_FIXED),
    }
    mol_rec: dict[str, object] = {
        "id": str(row.mid),
        "group_id": str(row.gid),
        "S_edge": float(s_edge),
        "sum_abs_edge_contrib": float(sum_abs),
        "num_edges": int(len(row.bonds)),
        "n_heavy_atoms": int(row.n_heavy_atoms),
        "n_rings": int(row.n_rings),
        "n_ring_edges": int(row.n_ring_edges),
        "n_shared_ring_edges": int(row.n_shared_ring_edges),
        "trace_heat": float(trace_heat),
        "lambda_min": float(lambda_min),
        "lambda_max": float(lambda_max),
    }
    return pred_rec, mol_rec, edge_rows_sorted


def _compute_file_infos(out_dir: Path, *, skip_names: set[str]) -> list[dict[str, object]]:
    infos: list[dict[str, object]] = []
    for path in sorted(out_dir.rglob("*")):
        if path.is_dir():
            continue
        if path.name in skip_names:
            continue
        rel = path.relative_to(out_dir).as_posix()
        infos.append({"path": f"./{rel}", "size_bytes": int(path.stat().st_size), "sha256": _sha256_file(path)})
    return infos


def _write_manifest(out_dir: Path, *, config: dict[str, object], files: list[dict[str, object]]) -> None:
    payload = {
        "schema_version": "manifest.v1",
        "created_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "config": config,
        "files": files,
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )


def _write_provenance(out_dir: Path, *, payload: dict[str, object]) -> None:
    (out_dir / "provenance.json").write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )


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


def _write_outputs_a4_0(
    *,
    input_csv: Path,
    out_dir: Path,
    experiment_id: str,
    seed: int,
    all_pred_records: list[dict[str, object]],
    fold_rows: list[dict[str, object]],
    group_metrics_all: dict[str, dict[str, object]],
    metrics_test: dict[str, object],
    worst_groups: list[dict[str, object]],
    edge_score_records: list[dict[str, object]],
    edge_contrib_topk_records: list[dict[str, object]],
    evidence_groups: list[str],
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    predictions_path = out_dir / "predictions.csv"
    with predictions_path.open("w", encoding="utf-8", newline="") as f:
        fieldnames = [
            "fold_id",
            "id",
            "group_id",
            "smiles",
            "truth_rel_kcalmol",
            "truth_rank",
            "pred_raw",
            "pred_rel",
            "pred_rank",
            "variant",
            "phi_fixed",
        ]
        w = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
        w.writeheader()
        for rec in sorted(
            all_pred_records, key=lambda rr: (int(rr.get("fold_id") or 0), str(rr.get("group_id")), str(rr.get("id")))
        ):
            w.writerow({k: rec.get(k, "") for k in fieldnames})

    summary_path = out_dir / "summary.csv"
    shutil.copyfile(predictions_path, summary_path)

    fold_metrics_path = out_dir / "fold_metrics.csv"
    with fold_metrics_path.open("w", encoding="utf-8", newline="") as f:
        fieldnames = [
            "fold_id",
            "test_group_id",
            "group_id",
            "variant",
            "phi_fixed",
            "n",
            "spearman_pred_vs_truth",
            "pairwise_order_accuracy",
            "pairwise_correct",
            "pairwise_total",
            "top1_accuracy",
            "pred_best_id",
            "truth_best_ids",
        ]
        w = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
        w.writeheader()
        for rec in sorted(fold_rows, key=lambda rr: int(rr.get("fold_id") or 0)):
            row = {k: rec.get(k, "") for k in fieldnames}
            row["truth_best_ids"] = json.dumps(list(rec.get("truth_best_ids") or []), ensure_ascii=False)
            w.writerow(row)

    group_metrics_path = out_dir / "group_metrics.csv"
    with group_metrics_path.open("w", encoding="utf-8", newline="") as f:
        fieldnames = [
            "group_id",
            "n",
            "spearman_pred_vs_truth",
            "pairwise_order_accuracy",
            "pairwise_correct",
            "pairwise_total",
            "top1_accuracy",
            "pred_best_id",
            "truth_best_ids",
        ]
        w = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
        w.writeheader()
        for gid, gm in sorted(group_metrics_all.items(), key=lambda x: str(x[0])):
            row = {k: gm.get(k, "") for k in fieldnames}
            row["truth_best_ids"] = json.dumps(list(gm.get("truth_best_ids") or []), ensure_ascii=False)
            w.writerow(row)

    edge_score_path = out_dir / "edge_score_by_molecule.csv"
    with edge_score_path.open("w", encoding="utf-8", newline="") as f:
        fieldnames = [
            "fold_id",
            "id",
            "group_id",
            "S_edge",
            "sum_abs_edge_contrib",
            "num_edges",
            "n_heavy_atoms",
            "n_rings",
            "n_ring_edges",
            "n_shared_ring_edges",
            "trace_heat",
            "lambda_min",
            "lambda_max",
        ]
        w = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
        w.writeheader()
        for rec in sorted(edge_score_records, key=lambda rr: (int(rr.get("fold_id") or 0), str(rr.get("group_id")), str(rr.get("id")))):
            w.writerow({k: rec.get(k, "") for k in fieldnames})

    edge_topk_path = out_dir / "edge_contrib_topk.csv"
    with edge_topk_path.open("w", encoding="utf-8", newline="") as f:
        fieldnames = [
            "fold_id",
            "group_id",
            "id",
            "edge_rank",
            "edge_u",
            "edge_v",
            "bond_order",
            "bond_is_aromatic",
            "w_ij",
            "theta_ij",
            "K_ij_re",
            "K_ij_im",
            "q_ij_re",
            "q_ij_im",
            "B_ij",
            "edge_contrib",
            "abs_edge_contrib",
        ]
        w = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
        w.writeheader()
        for rec in edge_contrib_topk_records:
            w.writerow({k: rec.get(k, "") for k in fieldnames})

    kpi_verdict = "PASS" if int(metrics_test.get("num_groups_spearman_negative") or 0) == 0 else "FAIL"
    metrics_payload: dict[str, object] = {
        "schema_version": "accuracy_a1_isomers_a4_0.v1",
        "experiment_id": str(experiment_id),
        "variant": "B",
        "phi_fixed": float(PHI_FIXED),
        "heat_tau": float(HEAT_TAU),
        "edge_weight_mode": str(EDGE_WEIGHT_MODE),
        "kpi": {
            "verdict": str(kpi_verdict),
            "must_num_groups_spearman_negative_test": 0,
            "num_groups_spearman_negative_test": int(metrics_test.get("num_groups_spearman_negative") or 0),
            "negative_spearman_groups_test": list(metrics_test.get("negative_spearman_groups") or []),
        },
        "metrics_loocv_test_functional_only": dict(metrics_test),
        "worst_groups_test": list(worst_groups),
        "evidence_groups_for_edge_topk": list(evidence_groups),
        "files": {
            "predictions_csv": "predictions.csv",
            "fold_metrics_csv": "fold_metrics.csv",
            "group_metrics_csv": "group_metrics.csv",
            "edge_score_by_molecule_csv": "edge_score_by_molecule.csv",
            "edge_contrib_topk_csv": "edge_contrib_topk.csv",
            "metrics_json": "metrics.json",
            "provenance_json": "provenance.json",
            "manifest_json": "manifest.json",
            "checksums_sha256": "checksums.sha256",
            "index_md": "index.md",
            "summary_csv": "summary.csv",
        },
    }
    (out_dir / "metrics.json").write_text(
        json.dumps(metrics_payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )

    index_lines = [
        f"# {experiment_id} (Isomers) A4.0 signed/orientable edge observable (edge-only)",
        "",
        "LOOCV (by group_id) metrics (functional_only):",
        f"- mean_spearman_by_group: {metrics_test.get('mean_spearman_by_group')}",
        f"- median_spearman_by_group: {metrics_test.get('median_spearman_by_group')}",
        f"- pairwise_order_accuracy_overall: {metrics_test.get('pairwise_order_accuracy_overall')} ({metrics_test.get('pairwise_correct')}/{metrics_test.get('pairwise_total')})",
        f"- top1_accuracy_mean: {metrics_test.get('top1_accuracy_mean')}",
        f"- num_groups_spearman_negative_test: {metrics_test.get('num_groups_spearman_negative')}",
        "",
        f"KPI verdict: {kpi_verdict}",
        "",
        "Evidence groups for edge top-k:",
        "```json",
        json.dumps(list(evidence_groups), ensure_ascii=False, sort_keys=True, indent=2),
        "```",
        "",
    ]
    (out_dir / "index.md").write_text("\n".join(index_lines) + "\n", encoding="utf-8")

    repo_root = Path(__file__).resolve().parents[1]
    extra_files: list[tuple[Path, Path]] = [
        (Path("data/accuracy/isomer_truth.v1.csv"), input_csv),
        (Path("docs/contracts/isomer_truth.v1.md"), repo_root / "docs/contracts/isomer_truth.v1.md"),
        (
            Path("data/accuracy/raw/dft_golden_isomers_v2_spice2_0_1.csv"),
            repo_root / "data/accuracy/raw/dft_golden_isomers_v2_spice2_0_1.csv",
        ),
        (
            Path("data/accuracy/raw/dft_golden_isomers_v2_spice2_0_1.csv.sha256"),
            repo_root / "data/accuracy/raw/dft_golden_isomers_v2_spice2_0_1.csv.sha256",
        ),
        (Path("data/atoms_db_v1.json"), repo_root / "data/atoms_db_v1.json"),
        (Path("docs/specs/accuracy_a4_0_signed_edge_observable.md"), repo_root / "docs/specs/accuracy_a4_0_signed_edge_observable.md"),
    ]
    for rel_dst, src in extra_files:
        if not src.exists():
            continue
        dst = out_dir / rel_dst
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dst)

    provenance: dict[str, object] = {
        "experiment_id": str(experiment_id),
        "git_sha": _detect_git_sha(),
        "source_sha_main": _detect_git_sha(),
        "python_version": platform.python_version(),
        "timestamp_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "command": " ".join([Path(sys.argv[0]).name] + sys.argv[1:]),
        "input_csv": input_csv.as_posix(),
        "input_sha256_normalized": _sha256_text_normalized(input_csv),
        "variant": "B",
        "phi_fixed": float(PHI_FIXED),
        "heat_tau": float(HEAT_TAU),
        "edge_weight_mode": str(EDGE_WEIGHT_MODE),
        "seed": int(seed),
        "kpi_verdict": str(kpi_verdict),
        "num_groups_spearman_negative_test": int(metrics_test.get("num_groups_spearman_negative") or 0),
        "negative_spearman_groups_test": list(metrics_test.get("negative_spearman_groups") or []),
    }
    _write_provenance(out_dir, payload=provenance)

    config_for_manifest = {
        "experiment_id": str(experiment_id),
        "input_csv": input_csv.as_posix(),
        "input_sha256_normalized": _sha256_text_normalized(input_csv),
        "variant": "B",
        "phi_fixed": float(PHI_FIXED),
        "heat_tau": float(HEAT_TAU),
        "seed": int(seed),
    }
    file_infos = _compute_file_infos(out_dir, skip_names={"manifest.json", "checksums.sha256", "evidence_pack.zip"})
    manifest_files = list(file_infos)
    manifest_files.append({"path": "./manifest.json", "size_bytes": None, "sha256": None})
    _write_manifest(out_dir, config=config_for_manifest, files=manifest_files)
    file_infos_final = _compute_file_infos(out_dir, skip_names={"checksums.sha256", "evidence_pack.zip"})
    _write_checksums(out_dir, file_infos_final)
    _write_zip_pack(out_dir, zip_name="evidence_pack.zip")


def run_a4_0(*, input_csv: Path, out_dir: Path, seed: int, experiment_id: str) -> None:
    atoms_db = load_atoms_db_v1()

    rows = _load_rows(input_csv)
    rows_sorted = sorted(rows, key=lambda r: (str(r.gid), str(r.mid)))
    group_ids = sorted({r.gid for r in rows_sorted})
    if len(group_ids) < 2:
        raise AccuracyA40Error("need at least 2 groups for LOOCV")

    rng = random.Random(int(seed))
    fold_order = list(group_ids)
    rng.shuffle(fold_order)

    all_pred_records: list[dict[str, object]] = []
    edge_score_records: list[dict[str, object]] = []
    edge_contrib_topk_all: list[dict[str, object]] = []
    fold_rows: list[dict[str, object]] = []

    for fold_id, test_gid in enumerate(fold_order, start=1):
        test_rows = [r for r in rows_sorted if str(r.gid) == str(test_gid)]
        if not test_rows:
            raise AccuracyA40Error("empty test group in LOOCV")

        test_recs: list[dict[str, object]] = []
        for r in test_rows:
            pred_rec, mol_rec, edge_rows_sorted = _compute_edge_score_for_row(r, atoms_db=atoms_db)
            pred_rec["fold_id"] = int(fold_id)
            mol_rec["fold_id"] = int(fold_id)
            test_recs.append(pred_rec)
            edge_score_records.append(mol_rec)

            for k, er in enumerate(edge_rows_sorted[: int(EDGE_TOPK)], start=1):
                rr = dict(er)
                rr["fold_id"] = int(fold_id)
                rr["edge_rank"] = int(k)
                edge_contrib_topk_all.append(rr)

        _attach_group_rel_and_ranks(test_recs)
        all_pred_records.extend(test_recs)

        gm_test = _compute_group_metrics(test_recs, pred_rel_key="pred_rel").get(str(test_gid))
        if gm_test is None:
            raise AccuracyA40Error("missing group metrics for test group")
        fold_rows.append(
            {
                "fold_id": int(fold_id),
                "test_group_id": str(test_gid),
                "group_id": str(test_gid),
                "variant": "B",
                "phi_fixed": float(PHI_FIXED),
                **{k: gm_test.get(k, "") for k in gm_test.keys()},
            }
        )

    group_metrics_all = _compute_group_metrics(all_pred_records, pred_rel_key="pred_rel")
    metrics_test = _aggregate_loocv_metrics(group_metrics_all)
    worst_groups = _worst_groups_by_spearman(group_metrics_all, n=3)

    negative_groups = [str(g) for g in metrics_test.get("negative_spearman_groups") or []]
    if negative_groups:
        evidence_groups = list(negative_groups)
    else:
        evidence_groups = [str(r.get("group_id") or "") for r in worst_groups]

    edge_contrib_topk_records = [r for r in edge_contrib_topk_all if str(r.get("group_id") or "") in set(evidence_groups)]

    _write_outputs_a4_0(
        input_csv=input_csv,
        out_dir=out_dir,
        experiment_id=experiment_id,
        seed=seed,
        all_pred_records=all_pred_records,
        fold_rows=fold_rows,
        group_metrics_all=group_metrics_all,
        metrics_test=metrics_test,
        worst_groups=worst_groups,
        edge_score_records=edge_score_records,
        edge_contrib_topk_records=edge_contrib_topk_records,
        evidence_groups=evidence_groups,
    )


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="ACCURACY-A4.0 signed/orientable edge observable (edge-only) runner (opt-in).")
    p.add_argument("--experiment_id", type=str, default="ACCURACY-A4.0", help="Experiment identifier for outputs.")
    p.add_argument("--input_csv", type=Path, default=DEFAULT_INPUT_CSV, help="Canonical isomer truth CSV.")
    p.add_argument("--out_dir", type=Path, default=DEFAULT_OUT_DIR, help="Output directory.")
    p.add_argument("--seed", type=int, default=0, help="Seed for fold order shuffling.")
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_arg_parser().parse_args(list(argv) if argv is not None else None)
    run_a4_0(
        input_csv=Path(args.input_csv),
        out_dir=Path(args.out_dir),
        seed=int(args.seed),
        experiment_id=str(args.experiment_id),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
