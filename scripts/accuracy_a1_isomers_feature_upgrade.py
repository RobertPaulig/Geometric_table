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


class AccuracyA1FeatureUpgradeError(ValueError):
    pass


DEFAULT_INPUT_CSV = Path("data/accuracy/isomer_truth.v1.csv")
DEFAULT_OUT_DIR = Path("out_accuracy_a1_isomers_a1_4")


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
        raise AccuracyA1FeatureUpgradeError("missing CSV header")
    missing = [c for c in required if c not in fieldnames]
    if missing:
        raise AccuracyA1FeatureUpgradeError(f"missing required columns: {missing}")


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
        raise AccuracyA1FeatureUpgradeError("beta must be > 0")
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
        raise AccuracyA1FeatureUpgradeError("eps must be > 0")
    if s < 0.0:
        raise AccuracyA1FeatureUpgradeError("shift must be >= 0")
    shifted = vals - float(np.min(vals))
    return float(np.sum(np.log(shifted + s + e)))


def _heat_trace_beta(eigvals: np.ndarray, *, beta: float) -> float:
    vals = np.asarray(eigvals, dtype=float).reshape(-1)
    if vals.size == 0:
        return float("nan")
    b = float(beta)
    if b <= 0.0:
        raise AccuracyA1FeatureUpgradeError("beta must be > 0")
    return float(np.sum(np.exp(-b * vals)))


def _heavy_atom_mapping(mol) -> tuple[list[int], dict[int, int]]:
    heavy: list[int] = []
    mapping: dict[int, int] = {}
    for idx, atom in enumerate(mol.GetAtoms()):
        if int(atom.GetAtomicNum()) > 1:
            mapping[int(idx)] = len(heavy)
            heavy.append(int(idx))
    return heavy, mapping


def _heavy_bonds_with_order(mol, mapping: dict[int, int]) -> tuple[tuple[int, int, float], ...]:
    edges: set[tuple[int, int, float]] = set()
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
            edges.add((i, j, bo))
    return tuple(sorted(edges))


def _node_features(mol, heavy: list[int]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    z: list[float] = []
    degree: list[float] = []
    aromatic: list[float] = []
    formal_charge: list[float] = []
    for idx in heavy:
        atom = mol.GetAtomWithIdx(int(idx))
        z.append(float(atom.GetAtomicNum()))
        aromatic.append(1.0 if atom.GetIsAromatic() else 0.0)
        formal_charge.append(float(atom.GetFormalCharge()))
        degree.append(float(atom.GetDegree()))
    return (
        np.asarray(z, dtype=float),
        np.asarray(degree, dtype=float),
        np.asarray(aromatic, dtype=float),
        np.asarray(formal_charge, dtype=float),
    )


def _weighted_adjacency_from_bonds(n: int, bonds: Sequence[tuple[int, int, float]]) -> np.ndarray:
    w_adj = np.zeros((int(n), int(n)), dtype=float)
    for i, j, bond_order in bonds:
        a, b = int(i), int(j)
        if a == b:
            continue
        w = float(bond_order)
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
    bonds: tuple[tuple[int, int, float], ...]
    node_degree: np.ndarray
    node_aromatic: np.ndarray
    node_formal_charge: np.ndarray


def _load_rows(input_csv: Path) -> list[_Row]:
    reader = csv.DictReader(input_csv.read_text(encoding="utf-8").splitlines())
    _require_columns(reader.fieldnames, required=["id", "group_id", "smiles", "energy_rel_kcalmol"])
    raw_rows = [dict(r) for r in reader]
    if not raw_rows:
        raise AccuracyA1FeatureUpgradeError("input_csv has no data rows")

    rows: list[_Row] = []
    for r in raw_rows:
        mid = str(r.get("id") or "").strip()
        gid = str(r.get("group_id") or "").strip()
        smiles = str(r.get("smiles") or "").strip()
        if not mid or not gid or not smiles:
            raise AccuracyA1FeatureUpgradeError(f"invalid row (id/group_id/smiles required): {r}")
        truth_rel = float(str(r.get("energy_rel_kcalmol") or "").strip())

        g = ChemGraph(smiles=smiles)
        mol = g.mol
        heavy, mapping = _heavy_atom_mapping(mol)
        types = tuple(int(mol.GetAtomWithIdx(int(idx)).GetAtomicNum()) for idx in heavy)
        bonds = _heavy_bonds_with_order(mol, mapping)
        _, degree, aromatic, formal_charge = _node_features(mol, heavy)
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
                node_aromatic=np.asarray(aromatic, dtype=float),
                node_formal_charge=np.asarray(formal_charge, dtype=float),
            )
        )
    return rows


def _split_groups(*, group_ids: Sequence[str], seed: int, n_train_groups: int) -> tuple[list[str], list[str]]:
    gids = sorted({str(g) for g in group_ids})
    if n_train_groups <= 0 or n_train_groups >= len(gids):
        raise AccuracyA1FeatureUpgradeError("invalid n_train_groups")
    rng = random.Random(int(seed))
    rng.shuffle(gids)
    train = list(gids[: int(n_train_groups)])
    test = list(gids[int(n_train_groups) :])
    return train, test


def _build_operator_eigvals(
    row: _Row,
    *,
    gamma: float,
    potential_variant: str,
    v_deg_coeff: float,
    v_arom_coeff: float,
    v_charge_coeff: float,
) -> np.ndarray:
    atoms_db = load_atoms_db_v1()
    eps_by_z = atoms_db.potential_by_atomic_num
    missing = sorted({int(z) for z in row.types if int(z) not in eps_by_z})
    if missing:
        raise AccuracyA1FeatureUpgradeError(f"missing atoms_db epsilon for Z={missing}")
    epsilon = np.asarray([float(eps_by_z[int(z)]) for z in row.types], dtype=float)

    variant = str(potential_variant)
    if variant == "epsilon_z":
        v = epsilon
    elif variant == "epsilon_z_plus_features":
        v = (
            epsilon
            + float(v_deg_coeff) * row.node_degree
            + float(v_arom_coeff) * row.node_aromatic
            + float(v_charge_coeff) * row.node_formal_charge
        )
    else:
        raise AccuracyA1FeatureUpgradeError(f"invalid potential_variant: {variant}")

    w_adj = _weighted_adjacency_from_bonds(row.n_heavy_atoms, row.bonds)
    lap = _laplacian_from_adjacency(w_adj)
    H = np.asarray(lap + float(gamma) * np.diag(v), dtype=float)
    vals = np.linalg.eigvalsh(H)
    return np.sort(np.asarray(vals, dtype=float))


def _feature_vector(
    eigvals: np.ndarray,
    *,
    logdet_eps: float,
    logdet_shift: float,
    heat_betas: Sequence[float],
    entropy_beta: float,
) -> dict[str, float]:
    vals = np.asarray(eigvals, dtype=float).reshape(-1)
    features: dict[str, float] = {
        "trace_H": float(np.sum(vals)) if vals.size else float("nan"),
        "logdet_shifted_eps": float(_logdet_shifted_eps(vals, eps=float(logdet_eps), shift=float(logdet_shift))),
        "entropy_beta": float(_spectral_entropy_beta(vals, beta=float(entropy_beta))),
    }
    for b in heat_betas:
        features[f"heat_trace_beta_{float(b)}"] = float(_heat_trace_beta(vals, beta=float(b)))
    return features


def _group_min_center(features_by_id: dict[str, dict[str, float]], rows: Sequence[_Row]) -> dict[str, dict[str, float]]:
    by_group: dict[str, list[str]] = {}
    for r in rows:
        by_group.setdefault(r.gid, []).append(r.mid)

    keys: set[str] = set()
    for f in features_by_id.values():
        keys.update(f.keys())

    out: dict[str, dict[str, float]] = {}
    for gid, mids in by_group.items():
        mins: dict[str, float] = {}
        for k in keys:
            vals = [float(features_by_id[mid][k]) for mid in mids]
            mins[k] = float(min(vals))
        for mid in mids:
            centered = {k: float(features_by_id[mid][k]) - float(mins[k]) for k in keys}
            out[mid] = centered
    return out


def _fit_ridge(
    X: np.ndarray,
    y: np.ndarray,
    *,
    ridge_lambda: float,
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    X0 = np.asarray(X, dtype=float)
    y0 = np.asarray(y, dtype=float).reshape(-1)
    if X0.ndim != 2 or y0.ndim != 1 or X0.shape[0] != y0.shape[0]:
        raise AccuracyA1FeatureUpgradeError("invalid shapes for regression")

    mean = np.mean(X0, axis=0)
    std = np.std(X0, axis=0)
    std = np.where(std == 0.0, 1.0, std)
    Xs = (X0 - mean) / std

    ones = np.ones((Xs.shape[0], 1), dtype=float)
    Xa = np.concatenate([ones, Xs], axis=1)

    lam = float(ridge_lambda)
    reg = np.eye(Xa.shape[1], dtype=float) * lam
    reg[0, 0] = 0.0
    w = np.linalg.solve(Xa.T @ Xa + reg, Xa.T @ y0)
    intercept = float(w[0])
    coeffs = np.asarray(w[1:], dtype=float)
    return intercept, coeffs, mean, std


def _predict(
    features: dict[str, float],
    *,
    feature_names: list[str],
    intercept: float,
    coeffs: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
) -> float:
    x = np.array([float(features[name]) for name in feature_names], dtype=float)
    xs = (x - mean) / std
    return float(intercept + float(np.dot(coeffs, xs)))


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


def _group_pred_min_center(pred_by_id: dict[str, float], rows: Sequence[_Row]) -> dict[str, float]:
    by_group: dict[str, list[str]] = {}
    for r in rows:
        by_group.setdefault(r.gid, []).append(r.mid)

    out: dict[str, float] = {}
    for gid, mids in by_group.items():
        min_pred = min(float(pred_by_id[mid]) for mid in mids)
        for mid in mids:
            out[mid] = float(float(pred_by_id[mid]) - float(min_pred))
    return out


def _is_finite_dict(d: dict[str, float]) -> bool:
    return all(math.isfinite(float(v)) for v in d.values())


def _compute_group_metrics(records: list[dict[str, object]]) -> list[dict[str, object]]:
    by_group: dict[str, list[dict[str, object]]] = {}
    for rec in records:
        gid = str(rec.get("group_id") or "")
        by_group.setdefault(gid, []).append(rec)

    per_group: list[dict[str, object]] = []
    for gid, group in sorted(by_group.items(), key=lambda x: x[0]):
        group_sorted = sorted(group, key=lambda r: str(r.get("id", "")))
        truth = [float(r["truth_rel_kcalmol"]) for r in group_sorted]
        pred = [float(r["pred_rel"]) for r in group_sorted]
        sp = _spearman_corr(pred, truth)
        c, t, acc = _pairwise_order_accuracy(truth, pred)

        truth_min = min(truth)
        truth_best_ids = {str(r["id"]) for r, tr in zip(group_sorted, truth) if tr == truth_min}
        pred_best_id = str(min(group_sorted, key=lambda r: float(r["pred_rel"]))["id"])
        top1_ok = int(pred_best_id in truth_best_ids)

        per_group.append(
            {
                "group_id": gid,
                "split": str(group_sorted[0].get("split") or ""),
                "n": int(len(group_sorted)),
                "truth_spread_kcalmol": float(max(truth) - min(truth)) if truth else float("nan"),
                "spearman_pred_vs_truth": float(sp),
                "pairwise_order_accuracy": float(acc),
                "pairwise_correct": int(c),
                "pairwise_total": int(t),
                "truth_best_ids": sorted(truth_best_ids),
                "pred_best_id": pred_best_id,
                "top1_accuracy": float(top1_ok),
            }
        )
    return per_group


def _aggregate_metrics(per_group: list[dict[str, object]], *, records_total: int) -> dict[str, object]:
    spearmans: list[float] = []
    pair_accs: list[float] = []
    top1_hits = 0
    pairwise_correct = 0
    pairwise_total = 0

    for g in per_group:
        sp = float(g.get("spearman_pred_vs_truth", float("nan")))
        if math.isfinite(sp):
            spearmans.append(float(sp))
        acc = float(g.get("pairwise_order_accuracy", float("nan")))
        if math.isfinite(acc):
            pair_accs.append(float(acc))
        top1_hits += int(float(g.get("top1_accuracy", 0.0)))
        pairwise_correct += int(g.get("pairwise_correct", 0))
        pairwise_total += int(g.get("pairwise_total", 0))

    groups_total = int(len(per_group))
    out: dict[str, object] = {
        "rows_total": int(records_total),
        "groups_total": groups_total,
        "mean_spearman_by_group": float(statistics.fmean(spearmans)) if spearmans else float("nan"),
        "median_spearman_by_group": float(statistics.median(spearmans)) if spearmans else float("nan"),
        "pairwise_order_accuracy_overall": (float(pairwise_correct) / float(pairwise_total)) if pairwise_total else float("nan"),
        "pairwise_order_accuracy_by_group_mean": float(statistics.fmean(pair_accs)) if pair_accs else float("nan"),
        "top1_accuracy_mean": float(top1_hits) / float(groups_total) if groups_total else float("nan"),
        "pairwise_correct": int(pairwise_correct),
        "pairwise_total": int(pairwise_total),
    }
    return out


def run_accuracy_a1_isomers_feature_upgrade(
    *,
    experiment_id: str,
    input_csv: Path,
    out_dir: Path,
    seed: int,
    n_train_groups: int,
    gamma: float,
    potential_variant: str,
    v_deg_coeff: float,
    v_arom_coeff: float,
    v_charge_coeff: float,
    logdet_eps: float,
    logdet_shift: float,
    heat_betas: Sequence[float],
    entropy_beta: float,
    ridge_lambda: float,
    kpi_mean_spearman_by_group_test_min: float,
    kpi_median_spearman_by_group_test_min: float,
    kpi_pairwise_order_accuracy_overall_test_min: float,
    kpi_top1_accuracy_mean_test_min: float,
) -> None:
    if not input_csv.exists():
        raise AccuracyA1FeatureUpgradeError(f"missing input_csv: {input_csv}")
    if float(gamma) < 0.0:
        raise AccuracyA1FeatureUpgradeError("gamma must be >= 0")

    rows = _load_rows(input_csv)
    if not rows:
        raise AccuracyA1FeatureUpgradeError("no rows loaded")

    by_group: dict[str, list[_Row]] = {}
    for r in rows:
        by_group.setdefault(r.gid, []).append(r)
    bad_groups = sorted([gid for gid, group in by_group.items() if len(group) < 2])
    if bad_groups:
        raise AccuracyA1FeatureUpgradeError(f"group_id has <2 rows: {bad_groups}")

    train_groups, test_groups = _split_groups(group_ids=[r.gid for r in rows], seed=int(seed), n_train_groups=int(n_train_groups))
    split_by_gid = {gid: "train" for gid in train_groups} | {gid: "test" for gid in test_groups}

    eigvals_by_id: dict[str, np.ndarray] = {}
    features_by_id: dict[str, dict[str, float]] = {}
    for r in rows:
        vals = _build_operator_eigvals(
            r,
            gamma=float(gamma),
            potential_variant=str(potential_variant),
            v_deg_coeff=float(v_deg_coeff),
            v_arom_coeff=float(v_arom_coeff),
            v_charge_coeff=float(v_charge_coeff),
        )
        eigvals_by_id[r.mid] = vals
        feats = _feature_vector(
            vals,
            logdet_eps=float(logdet_eps),
            logdet_shift=float(logdet_shift),
            heat_betas=list(heat_betas),
            entropy_beta=float(entropy_beta),
        )
        if not _is_finite_dict(feats):
            raise AccuracyA1FeatureUpgradeError(f"non-finite feature(s) for id={r.mid}")
        features_by_id[r.mid] = feats

    centered_features_by_id = _group_min_center(features_by_id, rows)
    truth_rel_by_id = _group_truth_min_center(rows)

    feature_names = sorted(next(iter(centered_features_by_id.values())).keys())
    X = np.asarray([[float(centered_features_by_id[r.mid][k]) for k in feature_names] for r in rows], dtype=float)
    y = np.asarray([float(truth_rel_by_id[r.mid]) for r in rows], dtype=float)

    train_mask = np.asarray([r.gid in set(train_groups) for r in rows], dtype=bool)
    X_train = X[train_mask]
    y_train = y[train_mask]
    intercept, coeffs, mean, std = _fit_ridge(X_train, y_train, ridge_lambda=float(ridge_lambda))

    pred_raw_by_id: dict[str, float] = {}
    for r in rows:
        pred_raw_by_id[r.mid] = _predict(
            centered_features_by_id[r.mid],
            feature_names=feature_names,
            intercept=float(intercept),
            coeffs=coeffs,
            mean=mean,
            std=std,
        )
        if not math.isfinite(float(pred_raw_by_id[r.mid])):
            raise AccuracyA1FeatureUpgradeError(f"non-finite pred_raw for id={r.mid}")

    pred_rel_by_id = _group_pred_min_center(pred_raw_by_id, rows)

    records: list[dict[str, object]] = []
    for r in rows:
        records.append(
            {
                "id": r.mid,
                "group_id": r.gid,
                "split": split_by_gid.get(r.gid, ""),
                "smiles": r.smiles,
                "n_heavy_atoms": int(r.n_heavy_atoms),
                "truth_rel_kcalmol": float(truth_rel_by_id[r.mid]),
                "pred_raw": float(pred_raw_by_id[r.mid]),
                "pred_rel": float(pred_rel_by_id[r.mid]),
            }
        )

    per_group = _compute_group_metrics(records)
    per_group_train = [g for g in per_group if str(g.get("split")) == "train"]
    per_group_test = [g for g in per_group if str(g.get("split")) == "test"]

    metrics_overall = _aggregate_metrics(per_group, records_total=len(records))
    metrics_train = _aggregate_metrics(per_group_train, records_total=int(sum(g.get("n", 0) for g in per_group_train)))
    metrics_test = _aggregate_metrics(per_group_test, records_total=int(sum(g.get("n", 0) for g in per_group_test)))

    kpi_payload = {
        "mean_spearman_by_group_test_min": float(kpi_mean_spearman_by_group_test_min),
        "median_spearman_by_group_test_min": float(kpi_median_spearman_by_group_test_min),
        "pairwise_order_accuracy_overall_test_min": float(kpi_pairwise_order_accuracy_overall_test_min),
        "top1_accuracy_mean_test_min": float(kpi_top1_accuracy_mean_test_min),
        "mean_spearman_by_group_test": float(metrics_test.get("mean_spearman_by_group", float("nan"))),
        "median_spearman_by_group_test": float(metrics_test.get("median_spearman_by_group", float("nan"))),
        "pairwise_order_accuracy_overall_test": float(metrics_test.get("pairwise_order_accuracy_overall", float("nan"))),
        "top1_accuracy_mean_test": float(metrics_test.get("top1_accuracy_mean", float("nan"))),
    }
    ok = (
        math.isfinite(float(kpi_payload["mean_spearman_by_group_test"]))
        and float(kpi_payload["mean_spearman_by_group_test"]) >= float(kpi_mean_spearman_by_group_test_min)
        and math.isfinite(float(kpi_payload["median_spearman_by_group_test"]))
        and float(kpi_payload["median_spearman_by_group_test"]) >= float(kpi_median_spearman_by_group_test_min)
        and math.isfinite(float(kpi_payload["pairwise_order_accuracy_overall_test"]))
        and float(kpi_payload["pairwise_order_accuracy_overall_test"]) >= float(kpi_pairwise_order_accuracy_overall_test_min)
        and math.isfinite(float(kpi_payload["top1_accuracy_mean_test"]))
        and float(kpi_payload["top1_accuracy_mean_test"]) >= float(kpi_top1_accuracy_mean_test_min)
    )
    kpi_payload["verdict"] = "PASS" if ok else "FAIL"
    kpi_payload["reason"] = (
        f"mean_spearman_by_group_test={kpi_payload['mean_spearman_by_group_test']}, "
        f"median_spearman_by_group_test={kpi_payload['median_spearman_by_group_test']}, "
        f"pairwise_order_accuracy_overall_test={kpi_payload['pairwise_order_accuracy_overall_test']}, "
        f"top1_accuracy_mean_test={kpi_payload['top1_accuracy_mean_test']}"
    )

    out_dir.mkdir(parents=True, exist_ok=True)

    predictions_path = out_dir / "predictions.csv"
    with predictions_path.open("w", encoding="utf-8", newline="") as f:
        fieldnames = ["id", "group_id", "split", "smiles", "n_heavy_atoms", "truth_rel_kcalmol", "pred_raw", "pred_rel"]
        w = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
        w.writeheader()
        for rec in sorted(records, key=lambda rr: (str(rr["group_id"]), str(rr["id"]))):
            w.writerow({k: rec.get(k, "") for k in fieldnames})

    summary_path = out_dir / "summary.csv"
    shutil.copyfile(predictions_path, summary_path)

    group_metrics_path = out_dir / "group_metrics.csv"
    with group_metrics_path.open("w", encoding="utf-8", newline="") as f:
        fieldnames = [
            "group_id",
            "split",
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
        for g in per_group:
            row = {k: g.get(k, "") for k in fieldnames}
            row["truth_best_ids"] = json.dumps(list(g.get("truth_best_ids") or []), ensure_ascii=False)
            w.writerow(row)

    best_cfg: dict[str, object] = {
        "operator": {
            "edge_weight_mode": "bond_order",
            "gamma": float(gamma),
            "potential_variant": str(potential_variant),
            "v_deg_coeff": float(v_deg_coeff),
            "v_arom_coeff": float(v_arom_coeff),
            "v_charge_coeff": float(v_charge_coeff),
        },
        "features": {
            "names": list(feature_names),
            "logdet_eps": float(logdet_eps),
            "logdet_shift": float(logdet_shift),
            "heat_betas": [float(x) for x in heat_betas],
            "entropy_beta": float(entropy_beta),
        },
        "model": {
            "ridge_lambda": float(ridge_lambda),
            "intercept": float(intercept),
            "coeffs": {name: float(coeffs[i]) for i, name in enumerate(feature_names)},
            "feature_standardization_mean": [float(x) for x in mean.tolist()],
            "feature_standardization_std": [float(x) for x in std.tolist()],
        },
        "split": {
            "seed": int(seed),
            "n_train_groups": int(n_train_groups),
            "train_groups": list(train_groups),
            "test_groups": list(test_groups),
        },
    }
    (out_dir / "best_config.json").write_text(json.dumps(best_cfg, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")

    metrics_payload: dict[str, object] = {
        "schema_version": "accuracy_a1_isomers_a1_4.v1",
        "experiment_id": str(experiment_id),
        "dataset": {
            "rows_total": int(len(records)),
            "groups_total": int(len(by_group)),
            "n_train_groups": int(n_train_groups),
            "n_test_groups": int(len(test_groups)),
        },
        "split": {
            "seed": int(seed),
            "train_groups": list(train_groups),
            "test_groups": list(test_groups),
        },
        "best_config": best_cfg,
        "metrics": {
            "overall": metrics_overall,
            "train": metrics_train,
            "test": metrics_test,
        },
        "kpi": kpi_payload,
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics_payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")

    index_lines = [
        f"# {experiment_id} (Isomers) A1.4 feature upgrade",
        "",
        "KPI (test split):",
        f"- verdict: {kpi_payload.get('verdict')} ({kpi_payload.get('reason')})",
        "",
        "Test metrics:",
        f"- mean_spearman_by_group: {metrics_test.get('mean_spearman_by_group')}",
        f"- median_spearman_by_group: {metrics_test.get('median_spearman_by_group')}",
        f"- pairwise_order_accuracy_overall: {metrics_test.get('pairwise_order_accuracy_overall')}",
        f"- top1_accuracy_mean: {metrics_test.get('top1_accuracy_mean')}",
        "",
        "Best config:",
        "```json",
        json.dumps(best_cfg, ensure_ascii=False, sort_keys=True, indent=2),
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
    ]
    for rel_dst, src in extra_files:
        try:
            if not src.exists():
                continue
            dst = out_dir / rel_dst
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src, dst)
        except Exception as exc:
            raise AccuracyA1FeatureUpgradeError(f"failed to copy provenance file {src} -> {rel_dst}") from exc

    provenance: dict[str, object] = {
        "experiment_id": str(experiment_id),
        "git_sha": _detect_git_sha(),
        "python_version": platform.python_version(),
        "timestamp_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "command": " ".join([Path(sys.argv[0]).name] + sys.argv[1:]),
        "input_csv": input_csv.as_posix(),
        "input_sha256_normalized": _sha256_text_normalized(input_csv),
        "best_config": best_cfg,
        "kpi": kpi_payload,
    }
    _write_provenance(out_dir, payload=provenance)

    config_for_manifest = {
        "experiment_id": str(experiment_id),
        "input_csv": input_csv.as_posix(),
        "input_sha256_normalized": _sha256_text_normalized(input_csv),
        "best_config": best_cfg,
        "kpi": kpi_payload,
    }
    file_infos = _compute_file_infos(out_dir, skip_names={"manifest.json", "checksums.sha256", "evidence_pack.zip"})
    manifest_files = list(file_infos)
    manifest_files.append({"path": "./manifest.json", "size_bytes": None, "sha256": None})
    _write_manifest(out_dir, config=config_for_manifest, files=manifest_files)
    file_infos_final = _compute_file_infos(out_dir, skip_names={"checksums.sha256", "evidence_pack.zip"})
    _write_checksums(out_dir, file_infos_final)
    _write_zip_pack(out_dir, zip_name="evidence_pack.zip")


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
    p = argparse.ArgumentParser(description="Run ACCURACY-A1.4 isomers feature upgrade and build an evidence pack.")
    p.add_argument("--experiment_id", type=str, default="ACCURACY-A1.4", help="Experiment/Roadmap label to embed in the pack.")
    p.add_argument("--input_csv", type=Path, default=DEFAULT_INPUT_CSV, help="Canonical isomer truth CSV.")
    p.add_argument("--out_dir", type=Path, default=DEFAULT_OUT_DIR, help="Output directory.")
    p.add_argument("--seed", type=int, default=0, help="Deterministic split seed (group-level).")
    p.add_argument("--n_train_groups", type=int, default=7, help="Number of train groups (out of total groups).")
    p.add_argument("--gamma", type=float, default=0.28, help="Potential scale gamma for H = L_w + gamma*diag(V).")
    p.add_argument(
        "--potential_variant",
        type=str,
        default="epsilon_z_plus_features",
        help="Potential variant: epsilon_z or epsilon_z_plus_features",
    )
    p.add_argument("--v_deg_coeff", type=float, default=0.10, help="Potential add-on coefficient for node degree.")
    p.add_argument("--v_arom_coeff", type=float, default=0.20, help="Potential add-on coefficient for aromatic flag.")
    p.add_argument("--v_charge_coeff", type=float, default=0.50, help="Potential add-on coefficient for formal charge.")
    p.add_argument("--logdet_eps", type=float, default=1e-6, help="Epsilon for logdet_shifted_eps feature.")
    p.add_argument("--logdet_shift", type=float, default=0.0, help="Shift for logdet_shifted_eps feature.")
    p.add_argument("--heat_betas", type=str, default="0.5,1.0,2.0", help="Comma-separated list of betas for heat_trace features.")
    p.add_argument("--entropy_beta", type=float, default=1.0, help="Beta for spectral entropy feature.")
    p.add_argument("--ridge_lambda", type=float, default=1e-3, help="Ridge regularization strength.")
    p.add_argument("--kpi_mean_spearman_by_group_test_min", type=float, default=0.55, help="KPI gate (test): mean spearman by group >= value.")
    p.add_argument("--kpi_median_spearman_by_group_test_min", type=float, default=0.55, help="KPI gate (test): median spearman by group >= value.")
    p.add_argument(
        "--kpi_pairwise_order_accuracy_overall_test_min",
        type=float,
        default=0.65,
        help="KPI gate (test): overall pairwise order accuracy >= value.",
    )
    p.add_argument("--kpi_top1_accuracy_mean_test_min", type=float, default=0.40, help="KPI gate (test): top1 mean accuracy >= value.")
    args = p.parse_args(argv)
    args.heat_betas = _parse_list_float(_parse_list_str(args.heat_betas))
    return args


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    run_accuracy_a1_isomers_feature_upgrade(
        experiment_id=str(args.experiment_id),
        input_csv=args.input_csv,
        out_dir=args.out_dir,
        seed=int(args.seed),
        n_train_groups=int(args.n_train_groups),
        gamma=float(args.gamma),
        potential_variant=str(args.potential_variant),
        v_deg_coeff=float(args.v_deg_coeff),
        v_arom_coeff=float(args.v_arom_coeff),
        v_charge_coeff=float(args.v_charge_coeff),
        logdet_eps=float(args.logdet_eps),
        logdet_shift=float(args.logdet_shift),
        heat_betas=list(args.heat_betas),
        entropy_beta=float(args.entropy_beta),
        ridge_lambda=float(args.ridge_lambda),
        kpi_mean_spearman_by_group_test_min=float(args.kpi_mean_spearman_by_group_test_min),
        kpi_median_spearman_by_group_test_min=float(args.kpi_median_spearman_by_group_test_min),
        kpi_pairwise_order_accuracy_overall_test_min=float(args.kpi_pairwise_order_accuracy_overall_test_min),
        kpi_top1_accuracy_mean_test_min=float(args.kpi_top1_accuracy_mean_test_min),
    )
    print(f"wrote: {args.out_dir.as_posix()}")
    print(f"zip: {(args.out_dir / 'evidence_pack.zip').as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
