from __future__ import annotations

"""
ACCURACY-A3.5 - Edge-Coherence Condensate runner (opt-in).

Contract SoT:
  docs/specs/accuracy_a3_5_edge_coherence_condensate.md
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


class AccuracyA35Error(ValueError):
    pass


DEFAULT_INPUT_CSV = Path("data/accuracy/isomer_truth.v1.csv")
DEFAULT_OUT_DIR = Path("out_accuracy_a1_isomers_a3_5")

PHI_FIXED = float(normalize_flux_phi(math.pi / 2.0))
KAPPA_CANDIDATES = [0.0, 0.25, 0.5, 1.0]
KAPPA_SWEEP_FILE = "kappa_sweep_test.csv"


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
        raise AccuracyA35Error("missing CSV header")
    missing = [c for c in required if c not in fieldnames]
    if missing:
        raise AccuracyA35Error(f"missing required columns: {missing}")


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


def _heavy_atom_mapping(mol) -> tuple[list[int], dict[int, int]]:
    heavy: list[int] = []
    mapping: dict[int, int] = {}
    for idx, atom in enumerate(mol.GetAtoms()):
        if int(atom.GetAtomicNum()) > 1:
            mapping[int(idx)] = len(heavy)
            heavy.append(int(idx))
    return heavy, mapping


def _heavy_bonds_with_attrs(mol, mapping: dict[int, int]) -> tuple[tuple[int, int, float, int], ...]:
    edges: set[tuple[int, int, float, int]] = set()
    for bond in mol.GetBonds():
        u = int(bond.GetBeginAtomIdx())
        v = int(bond.GetEndAtomIdx())
        if u in mapping and v in mapping:
            i = int(mapping[u])
            j = int(mapping[v])
            if i == j:
                continue
            if i > j:
                i, j = j, i
            bo = float(bond.GetBondTypeAsDouble())
            arom = 1 if bool(bond.GetIsAromatic()) else 0
            edges.add((i, j, bo, int(arom)))
    return tuple(sorted(edges))


def _node_features(mol, heavy: list[int]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    degree: list[float] = []
    valence: list[float] = []
    aromatic: list[float] = []
    formal_charge: list[float] = []
    in_ring: list[float] = []
    for idx in heavy:
        atom = mol.GetAtomWithIdx(int(idx))
        degree.append(float(atom.GetTotalDegree()))
        valence.append(float(atom.GetTotalValence()))
        aromatic.append(1.0 if bool(atom.GetIsAromatic()) else 0.0)
        formal_charge.append(float(atom.GetFormalCharge()))
        in_ring.append(1.0 if bool(atom.IsInRing()) else 0.0)
    return (
        np.asarray(degree, dtype=float),
        np.asarray(valence, dtype=float),
        np.asarray(aromatic, dtype=float),
        np.asarray(formal_charge, dtype=float),
        np.asarray(in_ring, dtype=float),
    )


def _cycles_heavy_from_mol_sssr(*, mol, heavy_mapping: dict[int, int]) -> tuple[tuple[int, ...], ...]:
    cycles_full = sssr_cycles_from_mol(mol)
    cycles_heavy: list[tuple[int, ...]] = []
    for cyc in cycles_full:
        mapped: list[int] = []
        for a in cyc:
            if int(a) not in heavy_mapping:
                raise AccuracyA35Error("unexpected non-heavy atom in SSSR cycle")
            mapped.append(int(heavy_mapping[int(a)]))
        cycles_heavy.append(tuple(mapped))
    return tuple(cycles_heavy)


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
        raise AccuracyA35Error(f"invalid edge_weight_mode: {mode}")
    am = float(aromatic_multiplier)
    alpha = float(delta_chi_alpha)

    chi: np.ndarray | None = None
    if mode == "bond_order_delta_chi":
        missing = sorted({int(z) for z in types if int(z) not in atoms_db.chi_by_atomic_num})
        if missing:
            raise AccuracyA35Error(f"missing atoms_db chi for Z={missing}")
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


def _laplacian_from_adjacency(adj: np.ndarray) -> np.ndarray:
    deg = np.diag(np.asarray(adj, dtype=float).sum(axis=1))
    return deg - np.asarray(adj, dtype=float)


def _entropy(p: np.ndarray) -> float:
    p0 = np.asarray(p, dtype=float).reshape(-1)
    if p0.size == 0:
        return 0.0
    return float(-np.sum(p0 * np.log(p0 + 1e-300)))


def _build_base_potential(
    row: "_Row",
    *,
    atoms_db: AtomsDbV1,
    gamma: float,
    potential_variant: str,
) -> np.ndarray:
    eps_by_z = atoms_db.potential_by_atomic_num
    missing_eps = sorted({int(z) for z in row.types if int(z) not in eps_by_z})
    if missing_eps:
        raise AccuracyA35Error(f"missing atoms_db epsilon for Z={missing_eps}")
    epsilon = np.asarray([float(eps_by_z[int(z)]) for z in row.types], dtype=float)

    variant = str(potential_variant)
    if variant != "epsilon_z":
        raise AccuracyA35Error(f"invalid potential_variant: {variant}")

    return np.asarray(float(gamma) * epsilon, dtype=float)


def _mass_vector(types: Sequence[int], *, mode: str) -> np.ndarray:
    mz = str(mode)
    z = np.asarray([float(int(t)) for t in types], dtype=float)
    if mz == "Z":
        return z
    if mz == "sqrtZ":
        return np.sqrt(z)
    raise AccuracyA35Error(f"invalid mass_mode: {mz}")


def _heat_kernel_matrix_shifted_trace_normalized_hermitian(
    H: np.ndarray, *, tau: float
) -> tuple[np.ndarray, float, float, float]:
    """
    K = exp(-tau * (H - lambda_min I)) / trace(exp(-tau * (H - lambda_min I))).
    Returns: (K, trace_heat, lambda_min, lambda_max).
    """
    tau_val = float(tau)
    if tau_val <= 0.0 or not math.isfinite(tau_val):
        raise AccuracyA35Error("heat_tau must be > 0 and finite")

    H0 = np.asarray(H)
    if H0.ndim != 2 or H0.shape[0] != H0.shape[1] or H0.shape[0] == 0:
        raise AccuracyA35Error("H must be non-empty square")

    eigvals, eigvecs = np.linalg.eigh(H0)
    eigvals = np.asarray(eigvals).reshape(-1)
    eigvecs = np.asarray(eigvecs)
    if eigvals.size == 0:
        raise AccuracyA35Error("H must be non-empty")

    lambda_min = float(np.real(eigvals[0]))
    lambda_max = float(np.real(eigvals[-1]))
    weights = np.exp(-tau_val * (np.real(eigvals) - float(lambda_min)))
    trace_heat = float(np.sum(weights))
    if not math.isfinite(trace_heat) or trace_heat <= 0.0:
        raise AccuracyA35Error("invalid trace_heat in rho-law")
    w_norm = weights / float(trace_heat)

    K = (eigvecs * w_norm) @ np.conj(eigvecs).T
    return np.asarray(K, dtype=np.complex128), float(trace_heat), float(lambda_min), float(lambda_max)


def _rho_realness_and_normalize(
    rho_raw: np.ndarray,
    *,
    rho_floor: float,
    rho_imag_tol: float = 1e-12,
) -> tuple[np.ndarray, float, float, bool, float, float]:
    """
    Fixator #2 (rho):
      - require rho_imag_max < tol
      - rho := Re(rho_raw)
      - apply floor before -log (tracked separately by caller)
      - hard normalize rho /= sum(rho)
    Returns:
      (rho_normed, rho_sum, rho_imag_max, rho_renorm_applied, rho_renorm_delta, rho_floor_rate)
    """
    rho0 = np.asarray(rho_raw)
    if rho0.ndim != 1:
        rho0 = rho0.reshape(-1)

    rho_imag_max = float(np.max(np.abs(np.imag(rho0)))) if rho0.size else 0.0
    if not math.isfinite(rho_imag_max) or rho_imag_max >= float(rho_imag_tol):
        raise AccuracyA35Error(f"rho_complex_violation: rho_imag_max={rho_imag_max}")

    rho_r = np.asarray(np.real(rho0), dtype=float).reshape(-1)
    if rho_r.size == 0:
        raise AccuracyA35Error("rho must be non-empty")

    rho_floor_val = float(rho_floor)
    if rho_floor_val < 0.0 or not math.isfinite(rho_floor_val):
        raise AccuracyA35Error("rho_floor must be >= 0 and finite")

    if rho_floor_val > 0.0:
        floor_mask = rho_r < rho_floor_val
        floor_rate = float(np.mean(floor_mask)) if floor_mask.size else 0.0
        rho_r = np.maximum(rho_r, rho_floor_val)
    else:
        floor_rate = 0.0

    rho_sum = float(np.sum(rho_r))
    if not math.isfinite(rho_sum) or rho_sum <= 0.0:
        raise AccuracyA35Error("invalid rho_sum in rho normalization")

    rho_renorm_delta = float(rho_sum - 1.0)
    rho_sum_tol = 1e-8
    rho_renorm_applied = bool(abs(rho_renorm_delta) > rho_sum_tol)
    rho_n = np.asarray(rho_r / float(rho_sum), dtype=float)
    return rho_n, float(rho_sum), float(rho_imag_max), bool(rho_renorm_applied), float(rho_renorm_delta), float(floor_rate)


def _phi_from_rho(*, rho: np.ndarray, phi_eps: float, gauge_fix_phi_mean: bool) -> np.ndarray:
    eps = float(phi_eps)
    if eps <= 0.0 or not math.isfinite(eps):
        raise AccuracyA35Error("phi_eps must be > 0 and finite")
    phi = -np.log(np.asarray(rho, dtype=float).reshape(-1) + eps)
    if gauge_fix_phi_mean:
        phi = phi - float(np.mean(phi))
    return np.asarray(phi, dtype=float)


def _c_from_heat_kernel_edges(
    *,
    K: np.ndarray,
    bonds: Sequence[Sequence[object]],
) -> tuple[np.ndarray, float, float, float, float]:
    """
    A3.5 edge-coherence definition (SoT):
      c_ij := |K_ij|  for (i,j) in E
      C_i := sum_{j~i} c_ij
      c_sum := sum(C)
      c_norm := C / c_sum (or 0-vector if c_sum==0)

    Returns:
      (c_norm, c_sum, c_edge_sum, c_entropy, c_norm_entropy)
    Where:
      c_edge_sum = sum_{(i,j) in E} |K_ij|
      c_entropy = entropy(normalized edge magnitudes)  (0 if c_edge_sum==0)
      c_norm_entropy = entropy(c_norm) (0 if c_sum==0)
    """
    K0 = np.asarray(K, dtype=np.complex128)
    if K0.ndim != 2 or K0.shape[0] != K0.shape[1]:
        raise AccuracyA35Error("K must be square")
    n = int(K0.shape[0])

    c_node = np.zeros((n,), dtype=float)
    c_edges: list[float] = []
    for i, j, *_rest in bonds:
        a = int(i)
        b = int(j)
        if a == b:
            continue
        c_edge = float(abs(K0[a, b]))
        if not math.isfinite(c_edge):
            raise AccuracyA35Error("non-finite c_edge")
        c_edges.append(float(c_edge))
        c_node[a] += float(c_edge)
        c_node[b] += float(c_edge)

    c_edge_sum = float(sum(c_edges))
    if c_edge_sum > 0.0:
        c_entropy = _entropy(np.asarray(c_edges, dtype=float) / float(c_edge_sum))
    else:
        c_entropy = 0.0

    c_sum = float(np.sum(c_node))
    if c_sum > 0.0:
        c_norm = np.asarray(c_node / float(c_sum), dtype=float)
        c_norm_entropy = _entropy(c_norm)
    else:
        c_norm = np.zeros_like(c_node, dtype=float)
        c_norm_entropy = 0.0

    return np.asarray(c_norm, dtype=float), float(c_sum), float(c_edge_sum), float(c_entropy), float(c_norm_entropy)


def _mix_rho_and_c(
    *,
    rho: np.ndarray,
    c_norm: np.ndarray,
    kappa: float,
) -> tuple[np.ndarray, bool, float]:
    """
    rho_eff := (1-kappa)*rho + kappa*c_norm
    Returns:
      (rho_eff, rho_eff_renorm_applied, rho_eff_sum)
    """
    k = float(kappa)
    if k < 0.0 or k > 1.0 or not math.isfinite(k):
        raise AccuracyA35Error("kappa must be in [0,1] and finite")

    rho0 = np.asarray(rho, dtype=float).reshape(-1)
    c0 = np.asarray(c_norm, dtype=float).reshape(-1)
    if rho0.size == 0 or c0.size != rho0.size:
        raise AccuracyA35Error("rho and c_norm must have same non-zero length")

    rho_eff = (1.0 - k) * rho0 + k * c0
    rho_eff_sum = float(np.sum(rho_eff))
    if not math.isfinite(rho_eff_sum) or rho_eff_sum <= 0.0:
        raise AccuracyA35Error("invalid rho_eff_sum")
    rho_sum_tol = 1e-8
    renorm_applied = bool(abs(rho_eff_sum - 1.0) > rho_sum_tol)
    if renorm_applied:
        rho_eff = np.asarray(rho_eff / float(rho_eff_sum), dtype=float)

    return np.asarray(rho_eff, dtype=float), bool(renorm_applied), float(rho_eff_sum)


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


@dataclass(frozen=True)
class _Row:
    mid: str
    gid: str
    smiles: str
    truth_rel: float
    n_heavy_atoms: int
    types: tuple[int, ...]
    bonds: tuple[tuple[int, int, float, int], ...]
    node_degree: np.ndarray
    node_valence: np.ndarray
    node_aromatic: np.ndarray
    node_formal_charge: np.ndarray
    node_in_ring: np.ndarray
    cycles_heavy: tuple[tuple[int, ...], ...]
    n_rings: int
    n_ring_edges: int
    n_shared_ring_edges: int


def _load_rows(input_csv: Path) -> list[_Row]:
    reader = csv.DictReader(input_csv.read_text(encoding="utf-8").splitlines())
    _require_columns(reader.fieldnames, required=["id", "group_id", "smiles", "energy_rel_kcalmol"])
    raw_rows = [dict(r) for r in reader]
    if not raw_rows:
        raise AccuracyA35Error("input_csv has no data rows")

    rows: list[_Row] = []
    for r in raw_rows:
        mid = str(r.get("id") or "").strip()
        gid = str(r.get("group_id") or "").strip()
        smiles = str(r.get("smiles") or "").strip()
        if not mid or not gid or not smiles:
            raise AccuracyA35Error(f"invalid row (id/group_id/smiles required): {r}")
        truth_rel = float(str(r.get("energy_rel_kcalmol") or "").strip())

        g = ChemGraph(smiles=smiles)
        mol = g.mol
        heavy, mapping = _heavy_atom_mapping(mol)
        types = tuple(int(mol.GetAtomWithIdx(int(idx)).GetAtomicNum()) for idx in heavy)
        bonds = _heavy_bonds_with_attrs(mol, mapping)
        degree, valence, aromatic, formal_charge, in_ring = _node_features(mol, heavy)

        cycles_heavy = _cycles_heavy_from_mol_sssr(mol=mol, heavy_mapping=mapping)
        n_rings, n_ring_edges, n_shared = _ring_edge_counts(cycles_heavy)

        rows.append(
            _Row(
                mid=mid,
                gid=gid,
                smiles=smiles,
                truth_rel=float(truth_rel),
                n_heavy_atoms=int(len(heavy)),
                types=types,
                bonds=bonds,
                node_degree=np.asarray(degree, dtype=float),
                node_valence=np.asarray(valence, dtype=float),
                node_aromatic=np.asarray(aromatic, dtype=float),
                node_formal_charge=np.asarray(formal_charge, dtype=float),
                node_in_ring=np.asarray(in_ring, dtype=float),
                cycles_heavy=cycles_heavy,
                n_rings=int(n_rings),
                n_ring_edges=int(n_ring_edges),
                n_shared_ring_edges=int(n_shared),
            )
        )
    return rows


def _attach_pred_rel(records: list[dict[str, object]]) -> None:
    by_group: dict[str, list[dict[str, object]]] = {}
    for r in records:
        by_group.setdefault(str(r.get("group_id") or ""), []).append(r)

    for gid, group in by_group.items():
        vals = [float(rr["pred_raw"]) for rr in group]
        mn = float(min(vals)) if vals else 0.0
        for rr in group:
            rr["pred_rel"] = float(float(rr["pred_raw"]) - mn)
            rr["group_id"] = str(gid)


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


@dataclass(frozen=True)
class _A35Config:
    gamma: float = 0.28
    potential_variant: str = "epsilon_z"
    edge_weight_mode: str = "bond_order_delta_chi"
    edge_aromatic_mult: float = 0.0
    edge_delta_chi_alpha: float = 1.0

    heat_tau: float = 1.0
    phi_eps: float = 1e-6
    rho_floor: float = 0.0

    sc_iters: int = 3
    sc_eta: float = 0.1
    sc_damping: float = 0.5
    sc_clip: float = 0.5
    sc_max_backtracks: int = 3
    gauge_fix_phi_mean: bool = True

    coef_alpha: float = 1.0
    coef_beta: float = 1.0
    coef_gamma: float = 1.0
    mass_mode: str = "Z"


_ROW_STATIC_CACHE: dict[str, dict[str, object]] = {}


def _get_row_static(row: _Row, *, atoms_db: AtomsDbV1, cfg: _A35Config) -> dict[str, object]:
    key = str(row.mid)
    cached = _ROW_STATIC_CACHE.get(key)
    if cached is not None:
        return cached

    w_adj = _build_weight_adjacency(
        row.bonds,
        n=int(row.n_heavy_atoms),
        atoms_db=atoms_db,
        types=row.types,
        mode=str(cfg.edge_weight_mode),
        aromatic_multiplier=float(cfg.edge_aromatic_mult),
        delta_chi_alpha=float(cfg.edge_delta_chi_alpha),
    )
    lap_base = _laplacian_from_adjacency(w_adj)
    v0 = _build_base_potential(row, atoms_db=atoms_db, gamma=float(cfg.gamma), potential_variant=str(cfg.potential_variant))
    mvec = _mass_vector(row.types, mode=str(cfg.mass_mode))

    lap_phase, _A = _phase_operator_for_row(weights=w_adj, cycles_heavy=row.cycles_heavy, flux_phi=float(PHI_FIXED))
    K, trace_heat, lambda_min, lambda_max = _heat_kernel_matrix_shifted_trace_normalized_hermitian(
        np.asarray(lap_phase, dtype=np.complex128), tau=float(cfg.heat_tau)
    )
    rho_diag = np.diag(np.asarray(K, dtype=np.complex128))
    rho, rho_sum, rho_imag_max, rho_renorm_applied, rho_renorm_delta, rho_floor_rate = _rho_realness_and_normalize(
        rho_diag, rho_floor=float(cfg.rho_floor)
    )
    rho_entropy = _entropy(rho)

    c_norm, c_sum, c_edge_sum, c_entropy, c_norm_entropy = _c_from_heat_kernel_edges(K=K, bonds=row.bonds)

    cached = {
        "w_adj": np.asarray(w_adj, dtype=float),
        "lap_base": np.asarray(lap_base, dtype=float),
        "v0": np.asarray(v0, dtype=float),
        "mvec": np.asarray(mvec, dtype=float),
        "rho": np.asarray(rho, dtype=float),
        "rho_sum": float(rho_sum),
        "rho_imag_max": float(rho_imag_max),
        "rho_entropy": float(rho_entropy),
        "rho_floor_rate": float(rho_floor_rate),
        "rho_renorm_applied": bool(rho_renorm_applied),
        "rho_renorm_delta": float(rho_renorm_delta),
        "c_norm": np.asarray(c_norm, dtype=float),
        "c_sum": float(c_sum),
        "c_edge_sum": float(c_edge_sum),
        "c_entropy": float(c_entropy),
        "c_norm_entropy": float(c_norm_entropy),
        "trace_heat": float(trace_heat),
        "lambda_min": float(lambda_min),
        "lambda_max": float(lambda_max),
    }
    _ROW_STATIC_CACHE[key] = cached
    return cached


def _solve_scf_full_functional_v1_edge_coherence(
    *,
    laplacian_base: np.ndarray,
    v0: np.ndarray,
    rho_base: np.ndarray,
    c_norm: np.ndarray,
    kappa: float,
    phi_eps: float,
    sc_iters: int,
    sc_eta: float,
    sc_damping: float,
    sc_clip: float,
    sc_max_backtracks: int,
    coef_alpha: float,
    coef_beta: float,
    coef_gamma: float,
    mvec: np.ndarray,
    gauge_fix_phi_mean: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, bool, float, float, list[dict[str, object]]]:
    lap = np.asarray(laplacian_base, dtype=float)
    if lap.ndim != 2 or lap.shape[0] != lap.shape[1]:
        raise AccuracyA35Error("laplacian_base must be square")
    n = int(lap.shape[0])

    v0_vec = np.asarray(v0, dtype=float).reshape(-1)
    if v0_vec.size != n:
        raise AccuracyA35Error("v0 must have shape (n,)")

    iters = int(sc_iters)
    if iters < 0 or iters > 5:
        raise AccuracyA35Error("sc_iters must be in [0,5]")

    eta0 = float(sc_eta)
    damping = float(sc_damping)
    clip = float(sc_clip)
    max_bt = int(sc_max_backtracks)
    if not math.isfinite(eta0):
        raise AccuracyA35Error("sc_eta must be finite")
    if damping < 0.0 or damping > 1.0 or not math.isfinite(damping):
        raise AccuracyA35Error("sc_damping must be in [0,1] and finite")
    if clip <= 0.0 or not math.isfinite(clip):
        raise AccuracyA35Error("sc_clip must be > 0 and finite")
    if max_bt < 0 or max_bt > 3:
        raise AccuracyA35Error("sc_max_backtracks must be in [0,3]")

    m0 = np.asarray(mvec, dtype=float).reshape(-1)
    if m0.size != n:
        raise AccuracyA35Error("mvec must have shape (n,)")

    alpha = float(coef_alpha)
    beta = float(coef_beta)
    gamma = float(coef_gamma)
    if not (math.isfinite(alpha) and math.isfinite(beta) and math.isfinite(gamma)):
        raise AccuracyA35Error("coef_alpha/beta/gamma must be finite")

    rho0 = np.asarray(rho_base, dtype=float).reshape(-1)
    c0 = np.asarray(c_norm, dtype=float).reshape(-1)
    if rho0.size != n or c0.size != n:
        raise AccuracyA35Error("rho_base/c_norm must have shape (n,)")

    def _state() -> tuple[np.ndarray, np.ndarray, np.ndarray, float, bool, float, float]:
        rho_eff, rho_eff_renorm_applied, rho_eff_sum = _mix_rho_and_c(rho=rho0, c_norm=c0, kappa=float(kappa))
        rho_eff_entropy = _entropy(rho_eff)
        phi = _phi_from_rho(rho=rho_eff, phi_eps=float(phi_eps), gauge_fix_phi_mean=bool(gauge_fix_phi_mean))
        curvature = -np.asarray(lap @ phi, dtype=float)

        grad_energy = float(phi.T @ (lap @ phi))
        curv_l1 = float(np.sum(np.abs(curvature)))
        mass_phi = float(np.sum(m0 * phi))
        E = float(alpha * grad_energy + beta * curv_l1 + gamma * mass_phi)
        if not math.isfinite(E):
            raise AccuracyA35Error("non-finite E in SCF")
        return (
            np.asarray(rho_eff, dtype=float),
            np.asarray(phi, dtype=float),
            np.asarray(curvature, dtype=float),
            float(E),
            bool(rho_eff_renorm_applied),
            float(rho_eff_sum),
            float(rho_eff_entropy),
        )

    v = v0_vec.copy()
    rho_eff, phi, curvature, E, rho_eff_renorm_applied, rho_eff_sum, rho_eff_entropy = _state()

    trace: list[dict[str, object]] = []
    for t in range(iters):
        eta_k = float(eta0)
        accepted = False
        bt_used = 0
        E_prev = float(E)
        residual_inf = 0.0

        for bt in range(max_bt + 1):
            bt_used = int(bt)
            upd = float(eta_k) * np.clip(curvature, -clip, clip)
            v_star = v0_vec + upd
            v_cand = (1.0 - damping) * v + damping * v_star

            rho_eff_c, phi_c, curvature_c, E_c, ren_c, sum_c, rho_eff_ent_c = _state()
            dv = np.asarray(v_cand - v, dtype=float)
            residual_inf = float(np.max(np.abs(dv))) if dv.size else 0.0

            if E_c <= E_prev:
                accepted = True
                v = np.asarray(v_cand, dtype=float)
                rho_eff = np.asarray(rho_eff_c, dtype=float)
                phi = np.asarray(phi_c, dtype=float)
                curvature = np.asarray(curvature_c, dtype=float)
                E = float(E_c)
                rho_eff_renorm_applied = bool(ren_c)
                rho_eff_sum = float(sum_c)
                rho_eff_entropy = float(rho_eff_ent_c)
                break
            eta_k = float(eta_k) * 0.5

        trace.append(
            {
                "iter": int(t),
                "E_prev": float(E_prev),
                "E": float(E),
                "accepted": bool(accepted),
                "accepted_backtracks": int(bt_used),
                "eta_final": float(eta_k),
                "residual_inf": float(residual_inf),
            }
        )

    return (
        np.asarray(v, dtype=float),
        np.asarray(rho_eff, dtype=float),
        np.asarray(phi, dtype=float),
        np.asarray(curvature, dtype=float),
        float(E),
        bool(rho_eff_renorm_applied),
        float(rho_eff_sum),
        float(rho_eff_entropy),
        list(trace),
    )


def _predict_one(
    row: _Row,
    *,
    atoms_db: AtomsDbV1,
    cfg: _A35Config,
    kappa: float,
) -> tuple[dict[str, object], dict[str, object]]:
    st = _get_row_static(row, atoms_db=atoms_db, cfg=cfg)
    lap_base = np.asarray(st["lap_base"], dtype=float)
    v0 = np.asarray(st["v0"], dtype=float)
    mvec = np.asarray(st["mvec"], dtype=float)
    rho_base = np.asarray(st["rho"], dtype=float)
    c_norm = np.asarray(st["c_norm"], dtype=float)

    (
        _v_final,
        rho_eff,
        _phi,
        _curvature,
        E,
        rho_eff_renorm_applied,
        rho_eff_sum,
        rho_eff_entropy,
        trace,
    ) = _solve_scf_full_functional_v1_edge_coherence(
        laplacian_base=lap_base,
        v0=v0,
        rho_base=rho_base,
        c_norm=c_norm,
        kappa=float(kappa),
        phi_eps=float(cfg.phi_eps),
        sc_iters=int(cfg.sc_iters),
        sc_eta=float(cfg.sc_eta),
        sc_damping=float(cfg.sc_damping),
        sc_clip=float(cfg.sc_clip),
        sc_max_backtracks=int(cfg.sc_max_backtracks),
        coef_alpha=float(cfg.coef_alpha),
        coef_beta=float(cfg.coef_beta),
        coef_gamma=float(cfg.coef_gamma),
        mvec=mvec,
        gauge_fix_phi_mean=bool(cfg.gauge_fix_phi_mean),
    )

    pred_record: dict[str, object] = {
        "id": str(row.mid),
        "group_id": str(row.gid),
        "smiles": str(row.smiles),
        "truth_rel_kcalmol": float(row.truth_rel),
        "pred_raw": float(E),
        "phi_fixed": float(PHI_FIXED),
        "kappa_selected": float(kappa),
        "rho_mode": "rho_eff",
    }

    diag_record: dict[str, object] = {
        "id": str(row.mid),
        "group_id": str(row.gid),
        "smiles": str(row.smiles),
        "n_heavy_atoms": int(row.n_heavy_atoms),
        "n_rings": int(row.n_rings),
        "n_ring_edges": int(row.n_ring_edges),
        "n_shared_ring_edges": int(row.n_shared_ring_edges),
        "phi_fixed": float(PHI_FIXED),
        "kappa_selected": float(kappa),
        "rho_mode": "rho_eff",
        "rho_sum": float(st["rho_sum"]),
        "rho_imag_max": float(st["rho_imag_max"]),
        "rho_entropy": float(st["rho_entropy"]),
        "rho_floor_rate": float(st["rho_floor_rate"]),
        "rho_renorm_applied": bool(st["rho_renorm_applied"]),
        "rho_renorm_delta": float(st["rho_renorm_delta"]),
        "c_sum": float(st["c_sum"]),
        "c_edge_sum": float(st["c_edge_sum"]),
        "c_entropy": float(st["c_entropy"]),
        "c_norm_entropy": float(st["c_norm_entropy"]),
        "rho_eff_entropy": float(rho_eff_entropy),
        "rho_eff_renorm_applied": bool(rho_eff_renorm_applied),
        "rho_eff_sum": float(rho_eff_sum),
        "trace_heat": float(st["trace_heat"]),
        "lambda_min": float(st["lambda_min"]),
        "lambda_max": float(st["lambda_max"]),
        "pred_raw": float(E),
        "trace": json.dumps(list(trace), ensure_ascii=False),
    }
    _ = rho_eff
    return pred_record, diag_record


def _select_kappa_nested_train_only(
    *,
    atoms_db: AtomsDbV1,
    cfg: _A35Config,
    train_rows: list[_Row],
    fold_id: int,
    test_gid: str,
) -> tuple[float, list[dict[str, object]]]:
    inner_group_ids = sorted({r.gid for r in train_rows})
    if len(inner_group_ids) < 2:
        return float(min(KAPPA_CANDIDATES)), []

    results: list[dict[str, object]] = []
    scored: list[tuple[int, float, float, float]] = []

    for kappa in KAPPA_CANDIDATES:
        inner_pred_records: list[dict[str, object]] = []
        for inner_test_gid in inner_group_ids:
            inner_test_rows = [r for r in train_rows if str(r.gid) == str(inner_test_gid)]
            for r in inner_test_rows:
                pred_rec, _diag = _predict_one(r, atoms_db=atoms_db, cfg=cfg, kappa=float(kappa))
                inner_pred_records.append(pred_rec)

        _attach_pred_rel(inner_pred_records)
        gm_all = _compute_group_metrics(inner_pred_records, pred_rel_key="pred_rel")
        agg = _aggregate_loocv_metrics(gm_all)

        num_neg = int(agg.get("num_groups_spearman_negative", 0) or 0)
        median_s = float(agg.get("median_spearman_by_group", float("nan")))
        mean_s = float(agg.get("mean_spearman_by_group", float("nan")))
        pairwise = float(agg.get("pairwise_order_accuracy_overall", float("nan")))
        top1 = float(agg.get("top1_accuracy_mean", float("nan")))
        groups_total = int(agg.get("groups_total", 0) or 0)

        median_key = float(median_s) if math.isfinite(float(median_s)) else float("-inf")
        scored.append((int(num_neg), -float(median_key), -float(kappa), float(kappa)))

        results.append(
            {
                "fold_id": int(fold_id),
                "test_group_id": str(test_gid),
                "kappa_candidate": float(kappa),
                "num_negative_train_inner": int(num_neg),
                "median_spearman_train_inner": float(median_s),
                "mean_spearman_train_inner": float(mean_s),
                "pairwise_train_inner": float(pairwise),
                "top1_train_inner": float(top1),
                "train_groups_total": int(groups_total),
            }
        )

    scored.sort(key=lambda t: (int(t[0]), float(t[1]), float(t[2])))
    chosen_kappa = float(scored[0][3])
    for r in results:
        r["kappa_selected"] = float(chosen_kappa)
        r["selected"] = bool(abs(float(r.get("kappa_candidate") or 0.0) - float(chosen_kappa)) < 1e-12)
    return float(chosen_kappa), results


def _kappa_sweep_test_metrics(
    *,
    atoms_db: AtomsDbV1,
    cfg: _A35Config,
    rows_sorted: list[_Row],
    fold_order: list[str],
) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    for kappa in KAPPA_CANDIDATES:
        pred_records: list[dict[str, object]] = []
        for fold_id, test_gid in enumerate(fold_order, start=1):
            test_rows = [r for r in rows_sorted if str(r.gid) == str(test_gid)]
            for r in test_rows:
                pred_rec, _diag = _predict_one(r, atoms_db=atoms_db, cfg=cfg, kappa=float(kappa))
                pred_rec["fold_id"] = int(fold_id)
                pred_records.append(pred_rec)

        _attach_pred_rel(pred_records)
        gm_all = _compute_group_metrics(pred_records, pred_rel_key="pred_rel")
        agg = _aggregate_loocv_metrics(gm_all)

        out.append(
            {
                "kappa": float(kappa),
                "groups_total": int(agg.get("groups_total") or 0),
                "num_groups_spearman_negative_test": int(agg.get("num_groups_spearman_negative") or 0),
                "negative_spearman_groups_test": json.dumps(
                    list(agg.get("negative_spearman_groups") or []), ensure_ascii=False
                ),
                "median_spearman_by_group_test": float(agg.get("median_spearman_by_group") or float("nan")),
                "pairwise_order_accuracy_overall_test": float(agg.get("pairwise_order_accuracy_overall") or float("nan")),
                "top1_accuracy_mean_test": float(agg.get("top1_accuracy_mean") or float("nan")),
            }
        )
    return out


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


def run_a3_5(
    *, input_csv: Path, out_dir: Path, seed: int, experiment_id: str, force_kappa: float | None = None
) -> None:
    cfg = _A35Config()
    atoms_db = load_atoms_db_v1()

    rows = _load_rows(input_csv)
    rows_sorted = sorted(rows, key=lambda r: (str(r.gid), str(r.mid)))
    group_ids = sorted({r.gid for r in rows_sorted})
    if len(group_ids) < 2:
        raise AccuracyA35Error("need at least 2 groups for LOOCV")

    rng = random.Random(int(seed))
    fold_order = list(group_ids)
    rng.shuffle(fold_order)

    all_pred_records: list[dict[str, object]] = []
    all_diag_records: list[dict[str, object]] = []
    fold_rows: list[dict[str, object]] = []
    selected_kappa_by_fold: dict[str, float] = {}
    search_results: list[dict[str, object]] = []

    forced_kappa_val: float | None
    if force_kappa is None:
        forced_kappa_val = None
    else:
        forced_kappa_val = float(force_kappa)
        if forced_kappa_val < 0.0 or forced_kappa_val > 1.0 or not math.isfinite(forced_kappa_val):
            raise AccuracyA35Error("--force_kappa must be in [0,1] and finite")
        if forced_kappa_val not in KAPPA_CANDIDATES:
            raise AccuracyA35Error(f"--force_kappa must be one of {KAPPA_CANDIDATES}")

    for fold_id, test_gid in enumerate(fold_order, start=1):
        test_rows = [r for r in rows_sorted if str(r.gid) == str(test_gid)]
        train_rows = [r for r in rows_sorted if str(r.gid) != str(test_gid)]
        if not test_rows:
            raise AccuracyA35Error("empty test group in LOOCV")

        if forced_kappa_val is None:
            kappa_sel, fold_search = _select_kappa_nested_train_only(
                atoms_db=atoms_db, cfg=cfg, train_rows=train_rows, fold_id=int(fold_id), test_gid=str(test_gid)
            )
            search_results.extend(list(fold_search))
        else:
            kappa_sel = float(forced_kappa_val)
            fold_search = []

        selected_kappa_by_fold[str(test_gid)] = float(kappa_sel)

        test_recs: list[dict[str, object]] = []
        test_diags: list[dict[str, object]] = []
        for r in test_rows:
            pred_rec, diag_rec = _predict_one(r, atoms_db=atoms_db, cfg=cfg, kappa=float(kappa_sel))
            pred_rec["fold_id"] = int(fold_id)
            diag_rec["fold_id"] = int(fold_id)
            test_recs.append(pred_rec)
            test_diags.append(diag_rec)

        _attach_pred_rel(test_recs)
        all_pred_records.extend(test_recs)
        all_diag_records.extend(test_diags)

        gm_test = _compute_group_metrics(test_recs, pred_rel_key="pred_rel").get(str(test_gid))
        if gm_test is None:
            raise AccuracyA35Error("missing group metrics for test group")
        fold_rows.append(
            {
                "fold_id": int(fold_id),
                "test_group_id": str(test_gid),
                "group_id": str(test_gid),
                "phi_fixed": float(PHI_FIXED),
                "kappa_selected": float(kappa_sel),
                **{k: gm_test.get(k, "") for k in gm_test.keys()},
            }
        )

    group_metrics_all = _compute_group_metrics(all_pred_records, pred_rel_key="pred_rel")
    metrics_test = _aggregate_loocv_metrics(group_metrics_all)
    worst_groups = _worst_groups_by_spearman(group_metrics_all, n=3)

    kappa_sweep_rows = _kappa_sweep_test_metrics(atoms_db=atoms_db, cfg=cfg, rows_sorted=rows_sorted, fold_order=fold_order)

    rho_imag_max_max = float(
        max(float(r.get("rho_imag_max") or 0.0) for r in all_diag_records) if all_diag_records else 0.0
    )
    c_sum_max = float(max(float(r.get("c_sum") or 0.0) for r in all_diag_records) if all_diag_records else 0.0)

    _write_outputs_a3_5(
        input_csv=input_csv,
        out_dir=out_dir,
        experiment_id=experiment_id,
        seed=seed,
        cfg=cfg,
        all_pred_records=all_pred_records,
        all_diag_records=all_diag_records,
        fold_rows=fold_rows,
        group_metrics_all=group_metrics_all,
        metrics_test=metrics_test,
        worst_groups=worst_groups,
        selected_kappa_by_fold=selected_kappa_by_fold,
        search_results=search_results,
        kappa_sweep_rows=kappa_sweep_rows,
        force_kappa=forced_kappa_val,
        rho_imag_max_max=rho_imag_max_max,
        c_sum_max=c_sum_max,
    )


def _write_outputs_a3_5(
    *,
    input_csv: Path,
    out_dir: Path,
    experiment_id: str,
    seed: int,
    cfg: _A35Config,
    all_pred_records: list[dict[str, object]],
    all_diag_records: list[dict[str, object]],
    fold_rows: list[dict[str, object]],
    group_metrics_all: dict[str, dict[str, object]],
    metrics_test: dict[str, object],
    worst_groups: list[dict[str, object]],
    selected_kappa_by_fold: dict[str, float],
    search_results: list[dict[str, object]],
    kappa_sweep_rows: list[dict[str, object]],
    force_kappa: float | None,
    rho_imag_max_max: float,
    c_sum_max: float,
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
            "pred_raw",
            "pred_rel",
            "rho_mode",
            "phi_fixed",
            "kappa_selected",
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
            "phi_fixed",
            "kappa_selected",
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

    rho_compare_path = out_dir / "rho_compare.csv"
    with rho_compare_path.open("w", encoding="utf-8", newline="") as f:
        fieldnames = [
            "fold_id",
            "id",
            "group_id",
            "smiles",
            "rho_mode",
            "phi_fixed",
            "kappa_selected",
            "rho_sum",
            "rho_imag_max",
            "rho_entropy",
            "rho_floor_rate",
            "rho_renorm_applied",
            "rho_renorm_delta",
            "c_sum",
            "c_edge_sum",
            "c_entropy",
            "c_norm_entropy",
            "rho_eff_entropy",
            "rho_eff_renorm_applied",
            "rho_eff_sum",
            "n_heavy_atoms",
            "n_rings",
            "n_ring_edges",
            "n_shared_ring_edges",
        ]
        w = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
        w.writeheader()
        for rec in sorted(
            all_diag_records, key=lambda rr: (int(rr.get("fold_id") or 0), str(rr.get("group_id")), str(rr.get("id")))
        ):
            w.writerow({k: rec.get(k, "") for k in fieldnames})

    coherence_summary_path = out_dir / "coherence_summary.csv"
    by_gid: dict[str, list[dict[str, object]]] = {}
    for rec in all_diag_records:
        by_gid.setdefault(str(rec.get("group_id") or ""), []).append(rec)

    with coherence_summary_path.open("w", encoding="utf-8", newline="") as f:
        fieldnames = [
            "group_id",
            "n_rows",
            "mean_c_sum",
            "median_c_sum",
            "max_c_sum",
            "mean_c_entropy",
            "median_c_entropy",
            "mean_c_norm_entropy",
            "median_c_norm_entropy",
        ]
        w = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
        w.writeheader()
        for gid, rows in sorted(by_gid.items(), key=lambda x: str(x[0])):
            c_sums = [float(r.get("c_sum") or 0.0) for r in rows]
            c_entropies = [float(r.get("c_entropy") or 0.0) for r in rows]
            c_norm_entropies = [float(r.get("c_norm_entropy") or 0.0) for r in rows]

            w.writerow(
                {
                    "group_id": str(gid),
                    "n_rows": int(len(rows)),
                    "mean_c_sum": float(statistics.fmean(c_sums)) if c_sums else 0.0,
                    "median_c_sum": float(statistics.median(c_sums)) if c_sums else 0.0,
                    "max_c_sum": float(max(c_sums)) if c_sums else 0.0,
                    "mean_c_entropy": float(statistics.fmean(c_entropies)) if c_entropies else 0.0,
                    "median_c_entropy": float(statistics.median(c_entropies)) if c_entropies else 0.0,
                    "mean_c_norm_entropy": float(statistics.fmean(c_norm_entropies)) if c_norm_entropies else 0.0,
                    "median_c_norm_entropy": float(statistics.median(c_norm_entropies)) if c_norm_entropies else 0.0,
                }
            )

    search_results_path = out_dir / "search_results.csv"
    with search_results_path.open("w", encoding="utf-8", newline="") as f:
        fieldnames = [
            "fold_id",
            "test_group_id",
            "kappa_candidate",
            "num_negative_train_inner",
            "median_spearman_train_inner",
            "mean_spearman_train_inner",
            "pairwise_train_inner",
            "top1_train_inner",
            "train_groups_total",
            "kappa_selected",
            "selected",
        ]
        w = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
        w.writeheader()
        for rec in sorted(search_results, key=lambda rr: (int(rr.get("fold_id") or 0), float(rr.get("kappa_candidate") or 0.0))):
            w.writerow({k: rec.get(k, "") for k in fieldnames})

    kappa_sweep_path = out_dir / KAPPA_SWEEP_FILE
    with kappa_sweep_path.open("w", encoding="utf-8", newline="") as f:
        fieldnames = [
            "kappa",
            "groups_total",
            "num_groups_spearman_negative_test",
            "negative_spearman_groups_test",
            "median_spearman_by_group_test",
            "pairwise_order_accuracy_overall_test",
            "top1_accuracy_mean_test",
        ]
        w = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
        w.writeheader()
        for rec in sorted(kappa_sweep_rows, key=lambda rr: float(rr.get("kappa") or 0.0)):
            w.writerow({k: rec.get(k, "") for k in fieldnames})

    best_cfg: dict[str, object] = {
        "nested_selection": force_kappa is None,
        "phi_fixed": float(PHI_FIXED),
        "kappa_candidates": list(KAPPA_CANDIDATES),
        "selected_kappa_by_outer_fold": dict(selected_kappa_by_fold),
        "search_space_size": int(len(KAPPA_CANDIDATES)),
        "seed": int(seed),
        "selection_metric_primary": "num_groups_spearman_negative_train_inner",
        "selection_metric_secondary": "median_spearman_by_group_train_inner",
        "selection_tie_breaker": "prefer_larger_kappa",
        "operator": {
            "edge_weight_mode": str(cfg.edge_weight_mode),
            "edge_aromatic_mult": float(cfg.edge_aromatic_mult),
            "edge_delta_chi_alpha": float(cfg.edge_delta_chi_alpha),
            "gamma": float(cfg.gamma),
            "potential_variant": str(cfg.potential_variant),
        },
        "scf": {
            "heat_tau": float(cfg.heat_tau),
            "phi_eps": float(cfg.phi_eps),
            "rho_floor": float(cfg.rho_floor),
            "sc_iters": int(cfg.sc_iters),
            "sc_eta": float(cfg.sc_eta),
            "sc_damping": float(cfg.sc_damping),
            "sc_clip": float(cfg.sc_clip),
            "sc_max_backtracks": int(cfg.sc_max_backtracks),
            "gauge_fix_phi_mean": bool(cfg.gauge_fix_phi_mean),
            "coef_alpha": float(cfg.coef_alpha),
            "coef_beta": float(cfg.coef_beta),
            "coef_gamma": float(cfg.coef_gamma),
            "mass_mode": str(cfg.mass_mode),
        },
    }
    if force_kappa is not None:
        best_cfg["force_kappa"] = float(force_kappa)
    (out_dir / "best_config.json").write_text(
        json.dumps(best_cfg, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )

    kpi_verdict = "PASS" if int(metrics_test["num_groups_spearman_negative"]) == 0 else "FAIL"
    kappa_distribution: dict[str, int] = {}
    for v in selected_kappa_by_fold.values():
        k = str(float(v))
        kappa_distribution[k] = int(kappa_distribution.get(k, 0)) + 1

    metrics_payload: dict[str, object] = {
        "schema_version": "accuracy_a1_isomers_a3_5.v1",
        "experiment_id": str(experiment_id),
        "dataset": {"rows_total": int(len(all_pred_records)), "groups_total": int(len(group_metrics_all))},
        "input_csv": input_csv.as_posix(),
        "input_sha256_normalized": _sha256_text_normalized(input_csv),
        "best_config": best_cfg,
        "kpi": {
            "verdict": str(kpi_verdict),
            "must_num_groups_spearman_negative_test": 0,
            "num_groups_spearman_negative_test": int(metrics_test["num_groups_spearman_negative"]),
            "negative_spearman_groups_test": list(metrics_test["negative_spearman_groups"]),
        },
        "metrics_loocv_test_functional_only": metrics_test,
        "worst_groups_test": worst_groups,
        "rho_imag_max_max": float(rho_imag_max_max),
        "c_sum_max": float(c_sum_max),
        "kappa_distribution": kappa_distribution,
        "files": {
            "summary_csv": "summary.csv",
            "predictions_csv": "predictions.csv",
            "fold_metrics_csv": "fold_metrics.csv",
            "group_metrics_csv": "group_metrics.csv",
            "rho_compare_csv": "rho_compare.csv",
            "coherence_summary_csv": "coherence_summary.csv",
            "search_results_csv": "search_results.csv",
            "kappa_sweep_test_csv": KAPPA_SWEEP_FILE,
            "metrics_json": "metrics.json",
            "best_config_json": "best_config.json",
            "provenance_json": "provenance.json",
            "manifest_json": "manifest.json",
            "checksums_sha256": "checksums.sha256",
            "index_md": "index.md",
        },
    }
    (out_dir / "metrics.json").write_text(
        json.dumps(metrics_payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )

    index_lines = [
        f"# {experiment_id} (Isomers) A3.5 rho+edge coherence",
        "",
        "LOOCV (by group_id) metrics (functional_only):",
        f"- mean_spearman_by_group: {metrics_test.get('mean_spearman_by_group')}",
        f"- median_spearman_by_group: {metrics_test.get('median_spearman_by_group')}",
        f"- pairwise_order_accuracy_overall: {metrics_test.get('pairwise_order_accuracy_overall')} ({metrics_test.get('pairwise_correct')}/{metrics_test.get('pairwise_total')})",
        f"- top1_accuracy_mean: {metrics_test.get('top1_accuracy_mean')}",
        f"- num_groups_spearman_negative_test: {metrics_test.get('num_groups_spearman_negative')}",
        "",
        "Nested kappa selection:",
        "```json",
        json.dumps(best_cfg.get("selected_kappa_by_outer_fold", {}), ensure_ascii=False, sort_keys=True, indent=2),
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
        (
            Path("docs/specs/accuracy_a3_5_edge_coherence_condensate.md"),
            repo_root / "docs/specs/accuracy_a3_5_edge_coherence_condensate.md",
        ),
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
        "best_config": best_cfg,
        "rho_imag_max_max": float(rho_imag_max_max),
        "phi_fixed": float(PHI_FIXED),
    }
    _write_provenance(out_dir, payload=provenance)

    config_for_manifest = {
        "experiment_id": str(experiment_id),
        "input_csv": input_csv.as_posix(),
        "input_sha256_normalized": _sha256_text_normalized(input_csv),
        "best_config": best_cfg,
    }
    file_infos = _compute_file_infos(out_dir, skip_names={"manifest.json", "checksums.sha256", "evidence_pack.zip"})
    manifest_files = list(file_infos)
    manifest_files.append({"path": "./manifest.json", "size_bytes": None, "sha256": None})
    _write_manifest(out_dir, config=config_for_manifest, files=manifest_files)
    file_infos_final = _compute_file_infos(out_dir, skip_names={"checksums.sha256", "evidence_pack.zip"})
    _write_checksums(out_dir, file_infos_final)
    _write_zip_pack(out_dir, zip_name="evidence_pack.zip")


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="ACCURACY-A3.5 rho+edge coherence runner (opt-in).")
    p.add_argument("--experiment_id", type=str, default="ACCURACY-A3.5", help="Experiment identifier for outputs.")
    p.add_argument("--input_csv", type=Path, default=DEFAULT_INPUT_CSV, help="Canonical isomer truth CSV.")
    p.add_argument("--out_dir", type=Path, default=DEFAULT_OUT_DIR, help="Output directory.")
    p.add_argument("--seed", type=int, default=0, help="Seed for fold order shuffling.")
    p.add_argument(
        "--force_kappa",
        type=float,
        default=None,
        help="Force a fixed kappa for all outer folds (disables nested selection). Must be one of {0,0.25,0.5,1.0}.",
    )
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_arg_parser().parse_args(list(argv) if argv is not None else None)
    run_a3_5(
        input_csv=Path(args.input_csv),
        out_dir=Path(args.out_dir),
        seed=int(args.seed),
        experiment_id=str(args.experiment_id),
        force_kappa=None if args.force_kappa is None else float(args.force_kappa),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
