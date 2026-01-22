from __future__ import annotations

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
from hetero2.physics_operator import load_atoms_db_v1


class AccuracyA1PairwiseRankError(ValueError):
    pass


DEFAULT_INPUT_CSV = Path("data/accuracy/isomer_truth.v1.csv")
DEFAULT_OUT_DIR = Path("out_accuracy_a1_isomers_a1_5")


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
        raise AccuracyA1PairwiseRankError("missing CSV header")
    missing = [c for c in required if c not in fieldnames]
    if missing:
        raise AccuracyA1PairwiseRankError(f"missing required columns: {missing}")


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


def _write_manifest(out_dir: Path, *, config: dict[str, object], files: list[dict[str, object]]) -> None:
    payload: dict[str, object] = {
        "git_sha": _detect_git_sha(),
        "python_version": platform.python_version(),
        "timestamp_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "command": " ".join([Path(sys.argv[0]).name] + sys.argv[1:]),
        "config": dict(config),
        "files": sorted(list(files), key=lambda x: str(x.get("path", ""))),
    }
    (out_dir / "manifest.json").write_text(json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def _write_provenance(out_dir: Path, *, payload: dict[str, object]) -> None:
    (out_dir / "provenance.json").write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )


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


def _laplacian_from_adjacency(adj: np.ndarray) -> np.ndarray:
    deg = np.diag(adj.sum(axis=1))
    return deg - adj


def _spectral_entropy_beta(eigvals: np.ndarray, *, beta: float, eps: float = 1e-12) -> float:
    vals = np.asarray(eigvals, dtype=float).reshape(-1)
    if vals.size == 0:
        return float("nan")
    b = float(beta)
    if b <= 0.0:
        raise AccuracyA1PairwiseRankError("beta must be > 0")
    x = -b * vals
    x = x - float(np.max(x))
    w = np.exp(x)
    total = float(np.sum(w))
    if total <= 0.0 or not math.isfinite(total):
        return float("nan")
    p = w / total
    return float(-np.sum(p * np.log(p + float(eps))))


def _logdet_shifted_eps(eigvals: np.ndarray, *, eps: float, shift: float) -> float:
    vals = np.asarray(eigvals, dtype=float).reshape(-1)
    if vals.size == 0:
        return float("nan")
    e = float(eps)
    s = float(shift)
    if e <= 0.0:
        raise AccuracyA1PairwiseRankError("eps must be > 0")
    if s < 0.0:
        raise AccuracyA1PairwiseRankError("shift must be >= 0")
    shifted = vals - float(np.min(vals))
    return float(np.sum(np.log(shifted + s + e)))


def _heat_trace_beta(eigvals: np.ndarray, *, beta: float) -> float:
    vals = np.asarray(eigvals, dtype=float).reshape(-1)
    if vals.size == 0:
        return float("nan")
    b = float(beta)
    if b <= 0.0:
        raise AccuracyA1PairwiseRankError("beta must be > 0")
    return float(np.sum(np.exp(-b * vals)))


def _free_energy_beta(eigvals: np.ndarray, *, beta: float) -> float:
    vals = np.asarray(eigvals, dtype=float).reshape(-1)
    if vals.size == 0:
        return float("nan")
    b = float(beta)
    if b <= 0.0:
        raise AccuracyA1PairwiseRankError("beta must be > 0")
    x = -b * vals
    m = float(np.max(x))
    x = x - m
    z = float(np.sum(np.exp(x)))
    if z <= 0.0 or not math.isfinite(z):
        return float("nan")
    return float(-(1.0 / b) * (math.log(z) + m))


def _sigmoid(z: np.ndarray) -> np.ndarray:
    z0 = np.asarray(z, dtype=float)
    out = np.empty_like(z0, dtype=float)
    pos = z0 >= 0.0
    out[pos] = 1.0 / (1.0 + np.exp(-z0[pos]))
    ez = np.exp(z0[~pos])
    out[~pos] = ez / (1.0 + ez)
    return out


def _fit_pairwise_logistic(
    X_pairs: np.ndarray,
    y_pairs: np.ndarray,
    *,
    l2_lambda: float,
    lr: float,
    max_iter: int,
    tol: float,
) -> np.ndarray:
    X = np.asarray(X_pairs, dtype=float)
    y = np.asarray(y_pairs, dtype=float).reshape(-1)
    if X.ndim != 2 or y.ndim != 1 or X.shape[0] != y.shape[0]:
        raise AccuracyA1PairwiseRankError("invalid shapes for logistic fit")
    if X.shape[0] < 1:
        raise AccuracyA1PairwiseRankError("no training pairs")

    lam = float(l2_lambda)
    if lam < 0.0:
        raise AccuracyA1PairwiseRankError("l2_lambda must be >= 0")

    w = np.zeros(X.shape[1], dtype=float)
    n = float(X.shape[0])
    for _ in range(int(max_iter)):
        z = X @ w
        p = _sigmoid(z)
        grad = (X.T @ (p - y)) / n + lam * w
        gnorm = float(np.linalg.norm(grad))
        w = w - float(lr) * grad
        if gnorm <= float(tol):
            break
    return w


def _fit_pairwise_rank_ridge(X_pairs: np.ndarray, y_deltas: np.ndarray, *, ridge_lambda: float) -> np.ndarray:
    X = np.asarray(X_pairs, dtype=float)
    y = np.asarray(y_deltas, dtype=float).reshape(-1)
    if X.ndim != 2 or y.ndim != 1 or X.shape[0] != y.shape[0]:
        raise AccuracyA1PairwiseRankError("invalid shapes for rank ridge fit")
    if X.shape[0] < 1:
        raise AccuracyA1PairwiseRankError("no training pairs")

    lam = float(ridge_lambda)
    if lam < 0.0:
        raise AccuracyA1PairwiseRankError("ridge_lambda must be >= 0")

    reg = np.eye(X.shape[1], dtype=float) * lam
    w = np.linalg.solve(X.T @ X + reg, X.T @ y)
    return np.asarray(w, dtype=float)


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


def _node_features(mol, heavy: list[int]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    z: list[float] = []
    degree: list[float] = []
    valence: list[float] = []
    aromatic: list[float] = []
    formal_charge: list[float] = []
    in_ring: list[float] = []
    for idx in heavy:
        atom = mol.GetAtomWithIdx(int(idx))
        z.append(float(atom.GetAtomicNum()))
        degree.append(float(atom.GetDegree()))
        try:
            valence.append(float(atom.GetTotalValence()))
        except Exception:
            valence.append(float(atom.GetDegree()))
        aromatic.append(1.0 if atom.GetIsAromatic() else 0.0)
        formal_charge.append(float(atom.GetFormalCharge()))
        in_ring.append(1.0 if atom.IsInRing() else 0.0)
    return (
        np.asarray(z, dtype=float),
        np.asarray(degree, dtype=float),
        np.asarray(valence, dtype=float),
        np.asarray(aromatic, dtype=float),
        np.asarray(formal_charge, dtype=float),
        np.asarray(in_ring, dtype=float),
    )


def _weighted_adjacency_from_bonds(n: int, bonds: Sequence[tuple[int, int, float, int]], *, aromatic_mult: float) -> np.ndarray:
    w_adj = np.zeros((int(n), int(n)), dtype=float)
    am = float(aromatic_mult)
    for i, j, bond_order, aromatic in bonds:
        a, b = int(i), int(j)
        if a == b:
            continue
        w = float(bond_order)
        if int(aromatic):
            w = w * (1.0 + am)
        w_adj[a, b] = w
        w_adj[b, a] = w
    return w_adj


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
    n_rings: int
    n_aromatic_rings: int
    n_aromatic_atoms: int
    n_hetero_atoms: int


def _load_rows(input_csv: Path) -> list[_Row]:
    reader = csv.DictReader(input_csv.read_text(encoding="utf-8").splitlines())
    _require_columns(reader.fieldnames, required=["id", "group_id", "smiles", "energy_rel_kcalmol"])
    raw_rows = [dict(r) for r in reader]
    if not raw_rows:
        raise AccuracyA1PairwiseRankError("input_csv has no data rows")

    rows: list[_Row] = []
    for r in raw_rows:
        mid = str(r.get("id") or "").strip()
        gid = str(r.get("group_id") or "").strip()
        smiles = str(r.get("smiles") or "").strip()
        if not mid or not gid or not smiles:
            raise AccuracyA1PairwiseRankError(f"invalid row (id/group_id/smiles required): {r}")
        truth_rel = float(str(r.get("energy_rel_kcalmol") or "").strip())

        g = ChemGraph(smiles=smiles)
        mol = g.mol
        heavy, mapping = _heavy_atom_mapping(mol)
        types = tuple(int(mol.GetAtomWithIdx(int(idx)).GetAtomicNum()) for idx in heavy)
        bonds = _heavy_bonds_with_attrs(mol, mapping)
        _, degree, valence, aromatic, formal_charge, in_ring = _node_features(mol, heavy)

        ring_info = g.ring_info()
        n_rings = int(ring_info.get("n_rings", 0))
        n_aromatic_rings = int(ring_info.get("n_aromatic_rings", 0))
        n_aromatic_atoms = int(sum(1 for idx in heavy if bool(mol.GetAtomWithIdx(int(idx)).GetIsAromatic())))
        n_hetero_atoms = int(sum(1 for idx in heavy if int(mol.GetAtomWithIdx(int(idx)).GetAtomicNum()) not in {1, 6}))

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
                n_rings=n_rings,
                n_aromatic_rings=n_aromatic_rings,
                n_aromatic_atoms=n_aromatic_atoms,
                n_hetero_atoms=n_hetero_atoms,
            )
        )
    return rows


def _build_operator_eigvals(
    row: _Row,
    *,
    gamma: float,
    potential_variant: str,
    v_deg_coeff: float,
    v_valence_coeff: float,
    v_arom_coeff: float,
    v_ring_coeff: float,
    v_charge_coeff: float,
    v_chi_coeff: float,
    edge_aromatic_mult: float,
) -> np.ndarray:
    atoms_db = load_atoms_db_v1()
    eps_by_z = atoms_db.potential_by_atomic_num
    chi_by_z = atoms_db.chi_by_atomic_num
    missing_eps = sorted({int(z) for z in row.types if int(z) not in eps_by_z})
    if missing_eps:
        raise AccuracyA1PairwiseRankError(f"missing atoms_db epsilon for Z={missing_eps}")

    epsilon = np.asarray([float(eps_by_z[int(z)]) for z in row.types], dtype=float)
    chi = np.asarray([float(chi_by_z.get(int(z), 0.0)) for z in row.types], dtype=float)

    variant = str(potential_variant)
    if variant == "epsilon_z":
        v = epsilon
    elif variant == "epsilon_z_plus_features_v2":
        v = (
            epsilon
            + float(v_deg_coeff) * row.node_degree
            + float(v_valence_coeff) * row.node_valence
            + float(v_arom_coeff) * row.node_aromatic
            + float(v_ring_coeff) * row.node_in_ring
            + float(v_charge_coeff) * row.node_formal_charge
            + float(v_chi_coeff) * chi
        )
    else:
        raise AccuracyA1PairwiseRankError(f"invalid potential_variant: {variant}")

    w_adj = _weighted_adjacency_from_bonds(row.n_heavy_atoms, row.bonds, aromatic_mult=float(edge_aromatic_mult))
    lap = _laplacian_from_adjacency(w_adj)
    H = np.asarray(lap + float(gamma) * np.diag(v), dtype=float)
    vals = np.linalg.eigvalsh(H)
    return np.sort(np.asarray(vals, dtype=float))


def _feature_vector(
    eigvals: np.ndarray,
    row: _Row,
    *,
    logdet_eps_values: Sequence[float],
    logdet_shift: float,
    heat_betas: Sequence[float],
    free_energy_betas: Sequence[float],
    entropy_beta: float,
) -> dict[str, float]:
    vals = np.asarray(eigvals, dtype=float).reshape(-1)
    features: dict[str, float] = {
        "trace_H": float(np.sum(vals)) if vals.size else float("nan"),
        "entropy_beta": float(_spectral_entropy_beta(vals, beta=float(entropy_beta))),
    }
    for eps in logdet_eps_values:
        features[f"logdet_shifted_eps_{float(eps)}"] = float(_logdet_shifted_eps(vals, eps=float(eps), shift=float(logdet_shift)))
    for b in heat_betas:
        features[f"heat_trace_beta_{float(b)}"] = float(_heat_trace_beta(vals, beta=float(b)))
    for b in free_energy_betas:
        features[f"free_energy_beta_{float(b)}"] = float(_free_energy_beta(vals, beta=float(b)))

    features["n_rings"] = float(row.n_rings)
    features["n_aromatic_rings"] = float(row.n_aromatic_rings)
    features["n_aromatic_atoms"] = float(row.n_aromatic_atoms)
    features["n_hetero_atoms"] = float(row.n_hetero_atoms)

    deg = np.asarray(row.node_degree, dtype=float).reshape(-1)
    if deg.size:
        features["deg_mean"] = float(np.mean(deg))
        features["deg_max"] = float(np.max(deg))
    else:
        features["deg_mean"] = float("nan")
        features["deg_max"] = float("nan")

    return features


def _is_finite_dict(d: dict[str, float]) -> bool:
    return all(math.isfinite(float(v)) for v in d.values())


def _group_truth_min_center(rows: Sequence[_Row]) -> dict[str, float]:
    by_group: dict[str, list[_Row]] = {}
    for r in rows:
        by_group.setdefault(r.gid, []).append(r)

    out: dict[str, float] = {}
    for gid, group in by_group.items():
        min_truth = min(float(r.truth_rel) for r in group)
        for r in group:
            out[r.mid] = float(float(r.truth_rel) - float(min_truth))
    return out


def _compute_group_metrics(records: list[dict[str, object]]) -> dict[str, dict[str, object]]:
    by_group: dict[str, list[dict[str, object]]] = {}
    for rec in records:
        gid = str(rec.get("group_id") or "")
        by_group.setdefault(gid, []).append(rec)

    out: dict[str, dict[str, object]] = {}
    for gid, group in sorted(by_group.items(), key=lambda x: x[0]):
        group_sorted = sorted(group, key=lambda r: str(r.get("id", "")))
        truth = [float(r["truth_rel_kcalmol"]) for r in group_sorted]
        pred = [float(r["pred_rel"]) for r in group_sorted]
        spearman = float(_spearman_corr(truth, pred))
        correct, total, acc = _pairwise_order_accuracy(truth, pred)
        min_truth = min(truth)
        truth_best_ids = [str(r["id"]) for r in group_sorted if float(r["truth_rel_kcalmol"]) == float(min_truth)]
        pred_best = min(group_sorted, key=lambda r: float(r["pred_rel"]))
        pred_best_id = str(pred_best["id"])
        top1 = 1.0 if pred_best_id in set(truth_best_ids) else 0.0
        truth_spread = float(max(truth) - min(truth)) if truth else float("nan")
        out[gid] = {
            "group_id": gid,
            "n": int(len(group_sorted)),
            "truth_spread_kcalmol": float(truth_spread),
            "spearman_pred_vs_truth": float(spearman),
            "pairwise_order_accuracy": float(acc),
            "pairwise_correct": int(correct),
            "pairwise_total": int(total),
            "top1_accuracy": float(top1),
            "pred_best_id": pred_best_id,
            "truth_best_ids": list(truth_best_ids),
        }
    return out


def run_accuracy_a1_isomers_pairwise_rank(
    *,
    experiment_id: str,
    input_csv: Path,
    out_dir: Path,
    seed: int,
    gamma: float,
    potential_variant: str,
    v_deg_coeff: float,
    v_valence_coeff: float,
    v_arom_coeff: float,
    v_ring_coeff: float,
    v_charge_coeff: float,
    v_chi_coeff: float,
    edge_aromatic_mult: float,
    logdet_eps_values: Sequence[float],
    logdet_shift: float,
    heat_betas: Sequence[float],
    free_energy_betas: Sequence[float],
    entropy_beta: float,
    model_type: str,
    model_ridge_lambda: float,
    model_l2_lambda: float,
    model_lr: float,
    model_max_iter: int,
    model_tol: float,
    kpi_mean_spearman_by_group_test_min: float,
    kpi_median_spearman_by_group_test_min: float,
    kpi_pairwise_order_accuracy_overall_test_min: float,
    kpi_top1_accuracy_mean_test_min: float,
) -> None:
    rows = _load_rows(input_csv)
    rows_sorted = sorted(rows, key=lambda r: (str(r.gid), str(r.mid)))
    group_ids = sorted({r.gid for r in rows_sorted})
    if len(group_ids) < 2:
        raise AccuracyA1PairwiseRankError("need at least 2 groups for LOOCV")

    feature_names = sorted(
        _feature_vector(
            np.asarray([0.0, 1.0], dtype=float),
            rows_sorted[0],
            logdet_eps_values=list(logdet_eps_values),
            logdet_shift=float(logdet_shift),
            heat_betas=list(heat_betas),
            free_energy_betas=list(free_energy_betas),
            entropy_beta=float(entropy_beta),
        ).keys()
    )

    features_by_id: dict[str, dict[str, float]] = {}
    for r in rows_sorted:
        eigvals = _build_operator_eigvals(
            r,
            gamma=float(gamma),
            potential_variant=str(potential_variant),
            v_deg_coeff=float(v_deg_coeff),
            v_valence_coeff=float(v_valence_coeff),
            v_arom_coeff=float(v_arom_coeff),
            v_ring_coeff=float(v_ring_coeff),
            v_charge_coeff=float(v_charge_coeff),
            v_chi_coeff=float(v_chi_coeff),
            edge_aromatic_mult=float(edge_aromatic_mult),
        )
        feats = _feature_vector(
            eigvals,
            r,
            logdet_eps_values=list(logdet_eps_values),
            logdet_shift=float(logdet_shift),
            heat_betas=list(heat_betas),
            free_energy_betas=list(free_energy_betas),
            entropy_beta=float(entropy_beta),
        )
        if set(feats.keys()) != set(feature_names):
            raise AccuracyA1PairwiseRankError("feature name mismatch across rows")
        if not _is_finite_dict(feats):
            raise AccuracyA1PairwiseRankError(f"non-finite features for id={r.mid} group={r.gid}")
        features_by_id[r.mid] = feats

    truth_rel_by_id = _group_truth_min_center(rows_sorted)

    rng = random.Random(int(seed))
    fold_order = list(group_ids)
    rng.shuffle(fold_order)

    all_records: list[dict[str, object]] = []
    fold_rows: list[dict[str, object]] = []
    fold_weights: list[dict[str, object]] = []

    for fold_id, test_gid in enumerate(fold_order):
        train_rows = [r for r in rows_sorted if r.gid != test_gid]
        test_rows = [r for r in rows_sorted if r.gid == test_gid]
        if not test_rows:
            raise AccuracyA1PairwiseRankError("empty test group")

        X_train = np.asarray([[float(features_by_id[r.mid][k]) for k in feature_names] for r in train_rows], dtype=float)
        mean = np.mean(X_train, axis=0)
        std = np.std(X_train, axis=0)
        std = np.where(std == 0.0, 1.0, std)

        X_std_by_id: dict[str, np.ndarray] = {}
        for r in rows_sorted:
            x = np.asarray([float(features_by_id[r.mid][k]) for k in feature_names], dtype=float)
            X_std_by_id[r.mid] = (x - mean) / std

        X_pairs_list: list[np.ndarray] = []
        y_labels_list: list[float] = []
        y_deltas_list: list[float] = []

        train_by_group: dict[str, list[_Row]] = {}
        for r in train_rows:
            train_by_group.setdefault(r.gid, []).append(r)

        for gid, group in train_by_group.items():
            group_sorted = sorted(group, key=lambda rr: str(rr.mid))
            for i in range(len(group_sorted)):
                for j in range(i + 1, len(group_sorted)):
                    a = group_sorted[i]
                    b = group_sorted[j]
                    ta = float(truth_rel_by_id[a.mid])
                    tb = float(truth_rel_by_id[b.mid])
                    if ta == tb:
                        continue
                    xa = X_std_by_id[a.mid]
                    xb = X_std_by_id[b.mid]
                    diff = xa - xb
                    delta = float(ta - tb)
                    y = 1.0 if delta > 0.0 else 0.0
                    X_pairs_list.append(diff)
                    y_labels_list.append(float(y))
                    y_deltas_list.append(float(delta))
                    X_pairs_list.append(-diff)
                    y_labels_list.append(float(1.0 - y))
                    y_deltas_list.append(float(-delta))

        X_pairs = np.asarray(X_pairs_list, dtype=float)
        y_labels = np.asarray(y_labels_list, dtype=float)
        y_deltas = np.asarray(y_deltas_list, dtype=float)
        mtype = str(model_type)
        if mtype == "pairwise_rank_ridge":
            w = _fit_pairwise_rank_ridge(X_pairs, y_deltas, ridge_lambda=float(model_ridge_lambda))
        elif mtype == "pairwise_logistic_l2":
            w = _fit_pairwise_logistic(
                X_pairs,
                y_labels,
                l2_lambda=float(model_l2_lambda),
                lr=float(model_lr),
                max_iter=int(model_max_iter),
                tol=float(model_tol),
            )
        else:
            raise AccuracyA1PairwiseRankError(f"invalid model_type: {mtype}")

        fold_weights.append(
            {
                "fold_id": int(fold_id),
                "test_group_id": str(test_gid),
                "n_train_groups": int(len(train_by_group)),
                "train_groups": sorted(train_by_group.keys()),
                "n_train_pairs": int(X_pairs.shape[0]),
                "standardization_mean": [float(x) for x in mean.tolist()],
                "standardization_std": [float(x) for x in std.tolist()],
                "weights": {feature_names[i]: float(w[i]) for i in range(len(feature_names))},
            }
        )

        pred_raw_by_id: dict[str, float] = {}
        for r in test_rows:
            pred_raw_by_id[r.mid] = float(np.dot(w, X_std_by_id[r.mid]))

        min_pred = min(float(pred_raw_by_id[r.mid]) for r in test_rows)
        for r in test_rows:
            all_records.append(
                {
                    "fold_id": int(fold_id),
                    "id": str(r.mid),
                    "group_id": str(r.gid),
                    "smiles": str(r.smiles),
                    "n_heavy_atoms": int(r.n_heavy_atoms),
                    "truth_rel_kcalmol": float(truth_rel_by_id[r.mid]),
                    "pred_raw": float(pred_raw_by_id[r.mid]),
                    "pred_rel": float(float(pred_raw_by_id[r.mid]) - float(min_pred)),
                }
            )

        group_metrics = _compute_group_metrics([rr for rr in all_records if str(rr.get("group_id")) == str(test_gid)])
        gm = group_metrics.get(str(test_gid))
        if gm is None:
            raise AccuracyA1PairwiseRankError("missing group metrics for fold")
        fold_rows.append(
            {
                "fold_id": int(fold_id),
                "test_group_id": str(test_gid),
                **{k: gm.get(k) for k in gm.keys()},
            }
        )

    group_metrics_all = _compute_group_metrics(all_records)
    spearmans = [float(v["spearman_pred_vs_truth"]) for v in group_metrics_all.values() if math.isfinite(float(v["spearman_pred_vs_truth"]))]
    mean_spearman = float(statistics.fmean(spearmans)) if spearmans else float("nan")
    median_spearman = float(statistics.median(spearmans)) if spearmans else float("nan")
    top1s = [float(v["top1_accuracy"]) for v in group_metrics_all.values() if math.isfinite(float(v["top1_accuracy"]))]
    top1_mean = float(statistics.fmean(top1s)) if top1s else float("nan")
    pair_correct = sum(int(v.get("pairwise_correct") or 0) for v in group_metrics_all.values())
    pair_total = sum(int(v.get("pairwise_total") or 0) for v in group_metrics_all.values())
    pair_acc = float(pair_correct) / float(pair_total) if pair_total else float("nan")

    metrics_test = {
        "mean_spearman_by_group": float(mean_spearman),
        "median_spearman_by_group": float(median_spearman),
        "pairwise_order_accuracy_overall": float(pair_acc),
        "pairwise_correct": int(pair_correct),
        "pairwise_total": int(pair_total),
        "top1_accuracy_mean": float(top1_mean),
    }

    ok = (
        math.isfinite(float(metrics_test["mean_spearman_by_group"]))
        and float(metrics_test["mean_spearman_by_group"]) >= float(kpi_mean_spearman_by_group_test_min)
        and math.isfinite(float(metrics_test["median_spearman_by_group"]))
        and float(metrics_test["median_spearman_by_group"]) >= float(kpi_median_spearman_by_group_test_min)
        and math.isfinite(float(metrics_test["pairwise_order_accuracy_overall"]))
        and float(metrics_test["pairwise_order_accuracy_overall"]) >= float(kpi_pairwise_order_accuracy_overall_test_min)
        and math.isfinite(float(metrics_test["top1_accuracy_mean"]))
        and float(metrics_test["top1_accuracy_mean"]) >= float(kpi_top1_accuracy_mean_test_min)
    )

    kpi_payload: dict[str, object] = {
        "mean_spearman_by_group_test": float(metrics_test["mean_spearman_by_group"]),
        "median_spearman_by_group_test": float(metrics_test["median_spearman_by_group"]),
        "pairwise_order_accuracy_overall_test": float(metrics_test["pairwise_order_accuracy_overall"]),
        "top1_accuracy_mean_test": float(metrics_test["top1_accuracy_mean"]),
        "verdict": "PASS" if ok else "FAIL",
        "reason": (
            f"mean_spearman_by_group_test={metrics_test['mean_spearman_by_group']}, "
            f"median_spearman_by_group_test={metrics_test['median_spearman_by_group']}, "
            f"pairwise_order_accuracy_overall_test={metrics_test['pairwise_order_accuracy_overall']}, "
            f"top1_accuracy_mean_test={metrics_test['top1_accuracy_mean']}"
        ),
    }

    out_dir.mkdir(parents=True, exist_ok=True)

    predictions_path = out_dir / "predictions.csv"
    with predictions_path.open("w", encoding="utf-8", newline="") as f:
        fieldnames = [
            "fold_id",
            "id",
            "group_id",
            "smiles",
            "n_heavy_atoms",
            "truth_rel_kcalmol",
            "pred_raw",
            "pred_rel",
        ]
        w = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
        w.writeheader()
        for rec in sorted(all_records, key=lambda rr: (str(rr["group_id"]), str(rr["id"]))):
            w.writerow({k: rec.get(k, "") for k in fieldnames})

    shutil.copyfile(predictions_path, out_dir / "summary.csv")

    fold_metrics_path = out_dir / "fold_metrics.csv"
    with fold_metrics_path.open("w", encoding="utf-8", newline="") as f:
        fieldnames = [
            "fold_id",
            "test_group_id",
            "n",
            "truth_spread_kcalmol",
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
        for rec in sorted(fold_rows, key=lambda rr: int(rr["fold_id"])):
            row = {k: rec.get(k, "") for k in fieldnames}
            row["truth_best_ids"] = json.dumps(list(rec.get("truth_best_ids") or []), ensure_ascii=False)
            w.writerow(row)

    best_cfg: dict[str, object] = {
        "operator": {
            "edge_weight_mode": "bond_order",
            "edge_aromatic_mult": float(edge_aromatic_mult),
            "gamma": float(gamma),
            "potential_variant": str(potential_variant),
            "v_deg_coeff": float(v_deg_coeff),
            "v_valence_coeff": float(v_valence_coeff),
            "v_arom_coeff": float(v_arom_coeff),
            "v_ring_coeff": float(v_ring_coeff),
            "v_charge_coeff": float(v_charge_coeff),
            "v_chi_coeff": float(v_chi_coeff),
        },
        "features": {
            "names": list(feature_names),
            "logdet_eps_values": [float(x) for x in logdet_eps_values],
            "logdet_shift": float(logdet_shift),
            "heat_betas": [float(x) for x in heat_betas],
            "free_energy_betas": [float(x) for x in free_energy_betas],
            "entropy_beta": float(entropy_beta),
        },
        "model": {
            "type": str(model_type),
            "ridge_lambda": float(model_ridge_lambda),
            "l2_lambda": float(model_l2_lambda),
            "lr": float(model_lr),
            "max_iter": int(model_max_iter),
            "tol": float(model_tol),
            "fold_weights": fold_weights,
        },
        "cv": {
            "method": "LOOCV_GROUP_ID",
            "seed": int(seed),
            "fold_order": list(fold_order),
        },
    }
    (out_dir / "best_config.json").write_text(json.dumps(best_cfg, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")

    metrics_payload: dict[str, object] = {
        "schema_version": "accuracy_a1_isomers_a1_5.v1",
        "experiment_id": str(experiment_id),
        "dataset": {
            "rows_total": int(len(all_records)),
            "groups_total": int(len(group_ids)),
        },
        "cv": best_cfg.get("cv", {}),
        "best_config": best_cfg,
        "metrics": {
            "test": metrics_test,
        },
        "kpi": kpi_payload,
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics_payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")

    index_lines = [
        f"# {experiment_id} (Isomers) A1.5 pairwise ranking",
        "",
        "KPI (LOOCV test folds):",
        f"- verdict: {kpi_payload.get('verdict')} ({kpi_payload.get('reason')})",
        "",
        "LOOCV test metrics:",
        f"- mean_spearman_by_group_test: {metrics_test.get('mean_spearman_by_group')}",
        f"- median_spearman_by_group_test: {metrics_test.get('median_spearman_by_group')}",
        f"- pairwise_order_accuracy_overall_test: {metrics_test.get('pairwise_order_accuracy_overall')}",
        f"- top1_accuracy_mean_test: {metrics_test.get('top1_accuracy_mean')}",
        "",
        "Evidence files:",
        "- predictions.csv (out-of-sample per row; one fold per group)",
        "- fold_metrics.csv (per test group/fold)",
        "- metrics.json (summary + KPI verdict)",
        "- best_config.json (operator + features + per-fold weights)",
        "- provenance.json / manifest.json / checksums.sha256",
        "",
    ]
    (out_dir / "index.md").write_text("\n".join(index_lines), encoding="utf-8")

    raw_truth = Path("data/accuracy/raw/dft_golden_isomers_v2_spice2_0_1.csv")
    raw_truth_sha = raw_truth.with_suffix(raw_truth.suffix + ".sha256")
    contract = Path("docs/contracts/isomer_truth.v1.md")
    atoms_db = Path("data/atoms_db_v1.json")
    for p in [input_csv, raw_truth, raw_truth_sha, contract, atoms_db]:
        if not p.exists():
            raise AccuracyA1PairwiseRankError(f"missing required input for pack: {p.as_posix()}")

    (out_dir / "data/accuracy/raw").mkdir(parents=True, exist_ok=True)
    (out_dir / "data/accuracy").mkdir(parents=True, exist_ok=True)
    (out_dir / "docs/contracts").mkdir(parents=True, exist_ok=True)
    (out_dir / "data").mkdir(parents=True, exist_ok=True)

    shutil.copyfile(input_csv, out_dir / "data/accuracy/isomer_truth.v1.csv")
    shutil.copyfile(raw_truth, out_dir / "data/accuracy/raw/dft_golden_isomers_v2_spice2_0_1.csv")
    shutil.copyfile(raw_truth_sha, out_dir / "data/accuracy/raw/dft_golden_isomers_v2_spice2_0_1.csv.sha256")
    shutil.copyfile(contract, out_dir / "docs/contracts/isomer_truth.v1.md")
    shutil.copyfile(atoms_db, out_dir / "data/atoms_db_v1.json")

    provenance_payload: dict[str, object] = {
        "git_sha": _detect_git_sha(),
        "python_version": platform.python_version(),
        "timestamp_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "command": " ".join([Path(sys.argv[0]).name] + sys.argv[1:]),
        "experiment_id": str(experiment_id),
        "truth": {
            "raw_csv_path": raw_truth.as_posix(),
            "raw_csv_sha256": _sha256_file(raw_truth),
            "raw_csv_sha256_file_sha256_text_normalized": _sha256_text_normalized(raw_truth_sha),
            "canonical_csv_path": input_csv.as_posix(),
            "canonical_csv_sha256": _sha256_file(input_csv),
            "contract_path": contract.as_posix(),
            "contract_sha256_text_normalized": _sha256_text_normalized(contract),
        },
        "config": best_cfg,
    }
    _write_provenance(out_dir, payload=provenance_payload)

    config_for_manifest: dict[str, object] = {
        "experiment_id": str(experiment_id),
        **best_cfg,
    }
    manifest_files = _compute_file_infos(out_dir, skip_names={"manifest.json", "checksums.sha256", "evidence_pack.zip"})
    _write_manifest(out_dir, config=config_for_manifest, files=manifest_files)
    file_infos_final = _compute_file_infos(out_dir, skip_names={"checksums.sha256", "evidence_pack.zip"})
    _write_checksums(out_dir, file_infos_final)
    _write_zip_pack(out_dir, zip_name="evidence_pack.zip")


def main() -> int:
    args = _parse_args()
    run_accuracy_a1_isomers_pairwise_rank(
        experiment_id=str(args.experiment_id),
        input_csv=args.input_csv,
        out_dir=args.out_dir,
        seed=int(args.seed),
        gamma=float(args.gamma),
        potential_variant=str(args.potential_variant),
        v_deg_coeff=float(args.v_deg_coeff),
        v_valence_coeff=float(args.v_valence_coeff),
        v_arom_coeff=float(args.v_arom_coeff),
        v_ring_coeff=float(args.v_ring_coeff),
        v_charge_coeff=float(args.v_charge_coeff),
        v_chi_coeff=float(args.v_chi_coeff),
        edge_aromatic_mult=float(args.edge_aromatic_mult),
        logdet_eps_values=list(args.logdet_eps_values),
        logdet_shift=float(args.logdet_shift),
        heat_betas=list(args.heat_betas),
        free_energy_betas=list(args.free_energy_betas),
        entropy_beta=float(args.entropy_beta),
        model_type=str(args.model_type),
        model_ridge_lambda=float(args.model_ridge_lambda),
        model_l2_lambda=float(args.model_l2_lambda),
        model_lr=float(args.model_lr),
        model_max_iter=int(args.model_max_iter),
        model_tol=float(args.model_tol),
        kpi_mean_spearman_by_group_test_min=float(args.kpi_mean_spearman_by_group_test_min),
        kpi_median_spearman_by_group_test_min=float(args.kpi_median_spearman_by_group_test_min),
        kpi_pairwise_order_accuracy_overall_test_min=float(args.kpi_pairwise_order_accuracy_overall_test_min),
        kpi_top1_accuracy_mean_test_min=float(args.kpi_top1_accuracy_mean_test_min),
    )
    print(f"wrote: {args.out_dir.as_posix()}")
    print(f"zip: {(args.out_dir / 'evidence_pack.zip').as_posix()}")
    return 0


def _parse_list_str(value: str) -> list[str]:
    return [v.strip() for v in str(value).split(",") if v.strip()]


def _parse_list_float(values: Sequence[str]) -> list[float]:
    out: list[float] = []
    for v in values:
        if not str(v).strip():
            continue
        out.append(float(v))
    return out


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run ACCURACY-A1.5 isomers pairwise ranking (LOOCV by group_id) and build an evidence pack.")
    p.add_argument("--experiment_id", type=str, default="ACCURACY-A1.5", help="Experiment/Roadmap label to embed in the pack.")
    p.add_argument("--input_csv", type=Path, default=DEFAULT_INPUT_CSV, help="Canonical isomer truth CSV.")
    p.add_argument("--out_dir", type=Path, default=DEFAULT_OUT_DIR, help="Output directory.")
    p.add_argument("--seed", type=int, default=0, help="Deterministic fold order seed (group-level).")
    p.add_argument("--gamma", type=float, default=0.28, help="Potential scale gamma for H = L_w + gamma*diag(V).")
    p.add_argument(
        "--potential_variant",
        type=str,
        default="epsilon_z_plus_features_v2",
        help="Potential variant: epsilon_z or epsilon_z_plus_features_v2",
    )
    p.add_argument("--v_deg_coeff", type=float, default=0.10, help="Potential add-on coefficient for node degree.")
    p.add_argument("--v_valence_coeff", type=float, default=0.05, help="Potential add-on coefficient for node valence.")
    p.add_argument("--v_arom_coeff", type=float, default=0.20, help="Potential add-on coefficient for aromatic flag.")
    p.add_argument("--v_ring_coeff", type=float, default=0.20, help="Potential add-on coefficient for ring membership.")
    p.add_argument("--v_charge_coeff", type=float, default=0.50, help="Potential add-on coefficient for formal charge.")
    p.add_argument("--v_chi_coeff", type=float, default=0.00, help="Potential add-on coefficient for chi(Z) proxy.")
    p.add_argument("--edge_aromatic_mult", type=float, default=0.0, help="Multiplier for aromatic bonds in weighted adjacency.")
    p.add_argument("--logdet_eps_values", type=str, default="1e-6,1e-4", help="Comma-separated eps list for logdet features.")
    p.add_argument("--logdet_shift", type=float, default=0.0, help="Shift for logdet_shifted_eps features.")
    p.add_argument("--heat_betas", type=str, default="0.5,1.0,2.0", help="Comma-separated list of betas for heat_trace features.")
    p.add_argument("--free_energy_betas", type=str, default="0.5,1.0", help="Comma-separated list of betas for free_energy features.")
    p.add_argument("--entropy_beta", type=float, default=1.0, help="Beta for spectral entropy feature.")
    p.add_argument(
        "--model_type",
        type=str,
        default="pairwise_logistic_l2",
        help="Model type: pairwise_rank_ridge or pairwise_logistic_l2",
    )
    p.add_argument("--model_ridge_lambda", type=float, default=1e-3, help="Ridge lambda for pairwise_rank_ridge.")
    p.add_argument("--model_l2_lambda", type=float, default=1e-3, help="L2 regularization for pairwise logistic.")
    p.add_argument("--model_lr", type=float, default=0.1, help="Learning rate for pairwise logistic.")
    p.add_argument("--model_max_iter", type=int, default=2000, help="Max iterations for pairwise logistic.")
    p.add_argument("--model_tol", type=float, default=1e-6, help="Gradient norm tolerance for pairwise logistic.")
    p.add_argument("--kpi_mean_spearman_by_group_test_min", type=float, default=0.55, help="KPI gate (LOOCV test): mean spearman by group >= value.")
    p.add_argument("--kpi_median_spearman_by_group_test_min", type=float, default=0.55, help="KPI gate (LOOCV test): median spearman by group >= value.")
    p.add_argument(
        "--kpi_pairwise_order_accuracy_overall_test_min",
        type=float,
        default=0.70,
        help="KPI gate (LOOCV test): overall pairwise order accuracy >= value.",
    )
    p.add_argument("--kpi_top1_accuracy_mean_test_min", type=float, default=0.40, help="KPI gate (LOOCV test): top1 mean accuracy >= value.")
    args = p.parse_args(argv)
    args.logdet_eps_values = _parse_list_float(_parse_list_str(args.logdet_eps_values))
    args.heat_betas = _parse_list_float(_parse_list_str(args.heat_betas))
    args.free_energy_betas = _parse_list_float(_parse_list_str(args.free_energy_betas))
    return args


if __name__ == "__main__":
    raise SystemExit(main())
