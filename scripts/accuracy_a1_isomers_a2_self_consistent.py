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
from hetero2.physics_operator import AtomsDbV1, load_atoms_db_v1


class AccuracyA2SelfConsistentError(ValueError):
    pass


DEFAULT_INPUT_CSV = Path("data/accuracy/isomer_truth.v1.csv")
DEFAULT_OUT_DIR = Path("out_accuracy_a1_isomers_a2")


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
        raise AccuracyA2SelfConsistentError("missing CSV header")
    missing = [c for c in required if c not in fieldnames]
    if missing:
        raise AccuracyA2SelfConsistentError(f"missing required columns: {missing}")


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
        raise AccuracyA2SelfConsistentError("invalid shapes for logistic fit")
    if X.shape[0] < 1:
        raise AccuracyA2SelfConsistentError("no training pairs")

    lam = float(l2_lambda)
    if lam < 0.0:
        raise AccuracyA2SelfConsistentError("l2_lambda must be >= 0")

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
        raise AccuracyA2SelfConsistentError("invalid shapes for rank ridge fit")
    if X.shape[0] < 1:
        raise AccuracyA2SelfConsistentError("no training pairs")

    lam = float(ridge_lambda)
    if lam < 0.0:
        raise AccuracyA2SelfConsistentError("ridge_lambda must be >= 0")

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


def _weighted_adjacency_from_bonds(
    *,
    n: int,
    bonds: Sequence[tuple[int, int, float, int]],
    types: Sequence[int],
    atoms_db: AtomsDbV1,
    edge_weight_mode: str,
    aromatic_mult: float,
    delta_chi_alpha: float,
) -> np.ndarray:
    mode = str(edge_weight_mode)
    if mode not in {"bond_order", "bond_order_delta_chi"}:
        raise AccuracyA2SelfConsistentError(f"invalid edge_weight_mode: {mode}")

    am = float(aromatic_mult)
    alpha = float(delta_chi_alpha)
    chi: np.ndarray | None = None
    if mode == "bond_order_delta_chi":
        missing = sorted({int(z) for z in types if int(z) not in atoms_db.chi_by_atomic_num})
        if missing:
            raise AccuracyA2SelfConsistentError(f"missing atoms_db chi for Z={missing}")
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
    deg = np.diag(adj.sum(axis=1))
    return deg - adj


def _softmax(x: np.ndarray) -> np.ndarray:
    vals = np.asarray(x, dtype=float)
    if vals.size == 0:
        return np.array([], dtype=float)
    z = vals - float(np.max(vals))
    w = np.exp(z)
    s = float(np.sum(w))
    if s <= 0.0 or not math.isfinite(s):
        return np.zeros_like(vals)
    return w / s


def _build_base_potential(
    row: "_Row",
    *,
    atoms_db: AtomsDbV1,
    gamma: float,
    potential_variant: str,
    v_deg_coeff: float,
    v_valence_coeff: float,
    v_arom_coeff: float,
    v_ring_coeff: float,
    v_charge_coeff: float,
    v_chi_coeff: float,
) -> np.ndarray:
    eps_by_z = atoms_db.potential_by_atomic_num
    missing_eps = sorted({int(z) for z in row.types if int(z) not in eps_by_z})
    if missing_eps:
        raise AccuracyA2SelfConsistentError(f"missing atoms_db epsilon for Z={missing_eps}")
    epsilon = np.asarray([float(eps_by_z[int(z)]) for z in row.types], dtype=float)

    chi_raw: np.ndarray | None = None
    if float(v_chi_coeff) != 0.0:
        chi_by_z = atoms_db.chi_by_atomic_num
        missing_chi = sorted({int(z) for z in row.types if int(z) not in chi_by_z})
        if missing_chi:
            raise AccuracyA2SelfConsistentError(f"missing atoms_db chi for Z={missing_chi}")
        chi_raw = np.asarray([float(chi_by_z[int(z)]) for z in row.types], dtype=float)

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
        )
        if chi_raw is not None:
            v = v + float(v_chi_coeff) * chi_raw
    else:
        raise AccuracyA2SelfConsistentError(f"invalid potential_variant: {variant}")

    return np.asarray(float(gamma) * v, dtype=float)


def _solve_self_consistent_functional(
    *,
    laplacian: np.ndarray,
    v0: np.ndarray,
    occ_k: int,
    tau: float,
    max_iter: int,
    tol: float,
    damping: float,
    phi_eps: float,
    eta_a: float,
    eta_phi: float,
    update_clip: float | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[dict[str, object]], bool, int, float]:
    lap = np.asarray(laplacian, dtype=float)
    if lap.ndim != 2 or lap.shape[0] != lap.shape[1]:
        raise AccuracyA2SelfConsistentError("laplacian must be square")

    v0_vec = np.asarray(v0, dtype=float).reshape(-1)
    n = int(lap.shape[0])
    if v0_vec.size != n:
        raise AccuracyA2SelfConsistentError("v0 must have shape (n,)")

    k = int(occ_k)
    if k < 1:
        raise AccuracyA2SelfConsistentError("occ_k must be >= 1")
    tau_val = float(tau)
    if tau_val <= 0.0 or not math.isfinite(tau_val):
        raise AccuracyA2SelfConsistentError("tau must be > 0 and finite")
    mi = int(max_iter)
    if mi < 1:
        raise AccuracyA2SelfConsistentError("max_iter must be >= 1")
    tol_val = float(tol)
    if tol_val <= 0.0 or not math.isfinite(tol_val):
        raise AccuracyA2SelfConsistentError("tol must be > 0 and finite")
    damp = float(damping)
    if not (0.0 < damp <= 1.0) or not math.isfinite(damp):
        raise AccuracyA2SelfConsistentError("damping must be in (0,1] and finite")
    eps = float(phi_eps)
    if eps <= 0.0 or not math.isfinite(eps):
        raise AccuracyA2SelfConsistentError("phi_eps must be > 0 and finite")

    v = v0_vec.copy()
    rho_prev = np.zeros((n,), dtype=float)
    trace: list[dict[str, object]] = []
    converged = False
    residual_final = float("nan")
    iters = 0

    rho = np.zeros((n,), dtype=float)
    phi = np.zeros((n,), dtype=float)
    curvature = np.zeros((n,), dtype=float)

    clip = None if update_clip is None else float(update_clip)
    if clip is not None and (clip <= 0.0 or not math.isfinite(clip)):
        raise AccuracyA2SelfConsistentError("update_clip must be > 0 and finite (or omitted)")

    for t in range(mi):
        iters = t + 1
        H = lap + np.diag(v)
        eigvals, eigvecs = np.linalg.eigh(H)
        idx = np.argsort(eigvals)
        eigvals = np.asarray(eigvals[idx], dtype=float)
        eigvecs = np.asarray(eigvecs[:, idx], dtype=float)
        kk = int(min(int(k), int(eigvals.size)))
        vals_k = np.asarray(eigvals[:kk], dtype=float)
        vecs_k = np.asarray(eigvecs[:, :kk], dtype=float)

        weights = _softmax(-vals_k / tau_val)
        rho = (vecs_k**2) @ weights
        delta_rho_l1 = float(np.sum(np.abs(rho - rho_prev)))
        rho_prev = rho.copy()

        phi = -np.log(rho + eps)
        phi_tilde = phi - float(np.mean(phi))
        curvature = -np.asarray(lap @ phi, dtype=float)

        upd = float(eta_a) * curvature + float(eta_phi) * phi_tilde
        if clip is not None:
            upd = np.clip(upd, -clip, clip)
        v_proposed = v0_vec + upd
        v_next = (1.0 - damp) * v + damp * v_proposed
        dv = np.asarray(v_next - v, dtype=float)
        residual_inf = float(np.max(np.abs(dv)))
        residual_mean = float(np.mean(np.abs(dv)))
        residual_final = residual_inf

        stop_reason = "iterating"
        if residual_inf < tol_val:
            stop_reason = "converged"
        elif (t + 1) >= mi:
            stop_reason = "max_iters"

        trace.append(
            {
                "iter": int(t + 1),
                "residual_inf": residual_inf,
                "residual_mean": residual_mean,
                "delta_rho_l1": delta_rho_l1,
                "eta_a": float(eta_a),
                "eta_phi": float(eta_phi),
                "damping": float(damp),
                "min_V": float(np.min(v_next)),
                "max_V": float(np.max(v_next)),
                "mean_V": float(np.mean(v_next)),
                "min_rho": float(np.min(rho)),
                "max_rho": float(np.max(rho)),
                "mean_rho": float(np.mean(rho)),
                "converged": bool(residual_inf < tol_val),
                "status": "CONVERGED" if residual_inf < tol_val else "ITERATING",
                "stop_reason": stop_reason,
            }
        )

        v = v_next
        if residual_inf < tol_val:
            converged = True
            break

    return v, rho, phi, curvature, trace, converged, int(iters), float(residual_final)


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


def _load_rows(input_csv: Path) -> list[_Row]:
    reader = csv.DictReader(input_csv.read_text(encoding="utf-8").splitlines())
    _require_columns(reader.fieldnames, required=["id", "group_id", "smiles", "energy_rel_kcalmol"])
    raw_rows = [dict(r) for r in reader]
    if not raw_rows:
        raise AccuracyA2SelfConsistentError("input_csv has no data rows")

    rows: list[_Row] = []
    for r in raw_rows:
        mid = str(r.get("id") or "").strip()
        gid = str(r.get("group_id") or "").strip()
        smiles = str(r.get("smiles") or "").strip()
        if not mid or not gid or not smiles:
            raise AccuracyA2SelfConsistentError(f"invalid row (id/group_id/smiles required): {r}")
        truth_rel = float(str(r.get("energy_rel_kcalmol") or "").strip())

        g = ChemGraph(smiles=smiles)
        mol = g.mol
        heavy, mapping = _heavy_atom_mapping(mol)
        types = tuple(int(mol.GetAtomWithIdx(int(idx)).GetAtomicNum()) for idx in heavy)
        bonds = _heavy_bonds_with_attrs(mol, mapping)
        _, degree, valence, aromatic, formal_charge, in_ring = _node_features(mol, heavy)

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
            )
        )
    return rows


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
    for r in records:
        gid = str(r.get("group_id") or "")
        by_group.setdefault(gid, []).append(r)

    out: dict[str, dict[str, object]] = {}
    for gid, group in by_group.items():
        group_sorted = sorted(group, key=lambda rr: str(rr.get("id")))
        truth = [float(rr["truth_rel_kcalmol"]) for rr in group_sorted]
        pred = [float(rr["pred_rel"]) for rr in group_sorted]
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


def _is_finite_dict(d: dict[str, float]) -> bool:
    return all(math.isfinite(float(v)) for v in d.values())


def _functional_features_for_row(
    row: _Row,
    *,
    atoms_db: AtomsDbV1,
    gamma: float,
    potential_variant: str,
    v_deg_coeff: float,
    v_valence_coeff: float,
    v_arom_coeff: float,
    v_ring_coeff: float,
    v_charge_coeff: float,
    v_chi_coeff: float,
    edge_weight_mode: str,
    edge_aromatic_mult: float,
    edge_delta_chi_alpha: float,
    occ_k: int,
    tau: float,
    sc_max_iter: int,
    sc_tol: float,
    sc_damping: float,
    phi_eps: float,
    eta_a: float,
    eta_phi: float,
    update_clip: float | None,
) -> dict[str, float]:
    w_adj = _weighted_adjacency_from_bonds(
        n=int(row.n_heavy_atoms),
        bonds=row.bonds,
        types=row.types,
        atoms_db=atoms_db,
        edge_weight_mode=str(edge_weight_mode),
        aromatic_mult=float(edge_aromatic_mult),
        delta_chi_alpha=float(edge_delta_chi_alpha),
    )
    lap = _laplacian_from_adjacency(w_adj)
    v0 = _build_base_potential(
        row,
        atoms_db=atoms_db,
        gamma=float(gamma),
        potential_variant=str(potential_variant),
        v_deg_coeff=float(v_deg_coeff),
        v_valence_coeff=float(v_valence_coeff),
        v_arom_coeff=float(v_arom_coeff),
        v_ring_coeff=float(v_ring_coeff),
        v_charge_coeff=float(v_charge_coeff),
        v_chi_coeff=float(v_chi_coeff),
    )

    missing = sorted({int(z) for z in row.types if int(z) not in atoms_db.chi_by_atomic_num})
    if missing:
        raise AccuracyA2SelfConsistentError(f"missing atoms_db chi for Z={missing}")
    c_vec = np.asarray([float(atoms_db.chi_by_atomic_num[int(z)]) for z in row.types], dtype=float)

    _, rho, phi, curvature, _, _, _, _ = _solve_self_consistent_functional(
        laplacian=lap,
        v0=v0,
        occ_k=int(occ_k),
        tau=float(tau),
        max_iter=int(sc_max_iter),
        tol=float(sc_tol),
        damping=float(sc_damping),
        phi_eps=float(phi_eps),
        eta_a=float(eta_a),
        eta_phi=float(eta_phi),
        update_clip=update_clip,
    )

    if not math.isfinite(float(np.min(rho))) or float(np.min(rho)) < 0.0:
        raise AccuracyA2SelfConsistentError(f"invalid rho for id={row.mid}")

    grad_term = float(phi.T @ (lap @ phi))
    curvature_l1 = float(np.sum(np.abs(curvature)))
    mass_phi = float(np.sum(c_vec * phi))

    feats = {
        "phi_grad_energy": float(grad_term),
        "curvature_l1": float(curvature_l1),
        "mass_phi_c": float(mass_phi),
    }

    if not _is_finite_dict(feats):
        raise AccuracyA2SelfConsistentError(f"non-finite features for id={row.mid}: {feats}")

    return feats


def run_accuracy_a1_isomers_a2_self_consistent(
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
    edge_weight_mode: str,
    edge_aromatic_mult: float,
    edge_delta_chi_alpha: float,
    sc_occ_k: int,
    sc_tau: float,
    sc_max_iter: int,
    sc_tol: float,
    sc_damping: float,
    phi_eps: float,
    eta_a: float,
    eta_phi: float,
    update_clip: float | None,
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
    if not input_csv.exists():
        raise AccuracyA2SelfConsistentError(f"missing input_csv: {input_csv}")

    rows = _load_rows(input_csv)
    rows_sorted = sorted(rows, key=lambda rr: (str(rr.gid), str(rr.mid)))
    group_ids = sorted({r.gid for r in rows_sorted})
    if len(group_ids) < 2:
        raise AccuracyA2SelfConsistentError("need at least 2 groups for LOOCV")

    atoms_db = load_atoms_db_v1()
    if not atoms_db.chi_by_atomic_num:
        raise AccuracyA2SelfConsistentError("atoms_db_v1 must provide chi(Z) for A2")

    feature_names = ["phi_grad_energy", "curvature_l1", "mass_phi_c"]
    features_by_id: dict[str, dict[str, float]] = {}
    for r in rows_sorted:
        feats = _functional_features_for_row(
            r,
            atoms_db=atoms_db,
            gamma=float(gamma),
            potential_variant=str(potential_variant),
            v_deg_coeff=float(v_deg_coeff),
            v_valence_coeff=float(v_valence_coeff),
            v_arom_coeff=float(v_arom_coeff),
            v_ring_coeff=float(v_ring_coeff),
            v_charge_coeff=float(v_charge_coeff),
            v_chi_coeff=float(v_chi_coeff),
            edge_weight_mode=str(edge_weight_mode),
            edge_aromatic_mult=float(edge_aromatic_mult),
            edge_delta_chi_alpha=float(edge_delta_chi_alpha),
            occ_k=int(sc_occ_k),
            tau=float(sc_tau),
            sc_max_iter=int(sc_max_iter),
            sc_tol=float(sc_tol),
            sc_damping=float(sc_damping),
            phi_eps=float(phi_eps),
            eta_a=float(eta_a),
            eta_phi=float(eta_phi),
            update_clip=update_clip,
        )
        if set(feats.keys()) != set(feature_names):
            raise AccuracyA2SelfConsistentError("feature name mismatch across rows")
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
            raise AccuracyA2SelfConsistentError("empty test group")

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
            raise AccuracyA2SelfConsistentError(f"invalid model_type: {mtype}")

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
                    "truth_rel_kcalmol": float(truth_rel_by_id[r.mid]),
                    "pred_raw": float(pred_raw_by_id[r.mid]),
                    "pred_rel": float(float(pred_raw_by_id[r.mid]) - float(min_pred)),
                }
            )

        group_metrics = _compute_group_metrics([rr for rr in all_records if str(rr.get("group_id")) == str(test_gid)])
        gm = group_metrics.get(str(test_gid))
        if gm is None:
            raise AccuracyA2SelfConsistentError("missing group metrics for fold")
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

    def _finite_or(value: float, fallback: float) -> float:
        return float(value) if math.isfinite(float(value)) else float(fallback)

    worst_groups = sorted(
        [
            {
                "group_id": str(gid),
                "spearman_pred_vs_truth": float(gm.get("spearman_pred_vs_truth", float("nan"))),
                "pairwise_order_accuracy": float(gm.get("pairwise_order_accuracy", float("nan"))),
                "top1_accuracy": float(gm.get("top1_accuracy", float("nan"))),
            }
            for gid, gm in group_metrics_all.items()
        ],
        key=lambda x: (
            _finite_or(float(x["spearman_pred_vs_truth"]), 0.0),
            _finite_or(float(x["top1_accuracy"]), 0.0),
            _finite_or(float(x["pairwise_order_accuracy"]), 0.0),
        ),
    )[:3]

    kpi_payload: dict[str, object] = {
        "mean_spearman_by_group_test_min": float(kpi_mean_spearman_by_group_test_min),
        "median_spearman_by_group_test_min": float(kpi_median_spearman_by_group_test_min),
        "pairwise_order_accuracy_overall_test_min": float(kpi_pairwise_order_accuracy_overall_test_min),
        "top1_accuracy_mean_test_min": float(kpi_top1_accuracy_mean_test_min),
        "mean_spearman_by_group_test": float(metrics_test["mean_spearman_by_group"]),
        "median_spearman_by_group_test": float(metrics_test["median_spearman_by_group"]),
        "pairwise_order_accuracy_overall_test": float(metrics_test["pairwise_order_accuracy_overall"]),
        "top1_accuracy_mean_test": float(metrics_test["top1_accuracy_mean"]),
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
        fieldnames = ["fold_id", "id", "group_id", "smiles", "truth_rel_kcalmol", "pred_raw", "pred_rel"]
        w = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
        w.writeheader()
        for rec in sorted(all_records, key=lambda rr: (int(rr["fold_id"]), str(rr["group_id"]), str(rr["id"]))):
            w.writerow({k: rec.get(k, "") for k in fieldnames})

    summary_path = out_dir / "summary.csv"
    shutil.copyfile(predictions_path, summary_path)

    fold_metrics_path = out_dir / "fold_metrics.csv"
    with fold_metrics_path.open("w", encoding="utf-8", newline="") as f:
        fieldnames = [
            "fold_id",
            "test_group_id",
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

    best_cfg: dict[str, object] = {
        "operator": {
            "edge_weight_mode": str(edge_weight_mode),
            "edge_aromatic_mult": float(edge_aromatic_mult),
            "edge_delta_chi_alpha": float(edge_delta_chi_alpha),
            "gamma": float(gamma),
            "potential_variant": str(potential_variant),
            "v_deg_coeff": float(v_deg_coeff),
            "v_valence_coeff": float(v_valence_coeff),
            "v_arom_coeff": float(v_arom_coeff),
            "v_ring_coeff": float(v_ring_coeff),
            "v_charge_coeff": float(v_charge_coeff),
            "v_chi_coeff": float(v_chi_coeff),
        },
        "self_consistent": {
            "occ_k": int(sc_occ_k),
            "tau": float(sc_tau),
            "max_iter": int(sc_max_iter),
            "tol": float(sc_tol),
            "damping": float(sc_damping),
            "phi_eps": float(phi_eps),
            "eta_a": float(eta_a),
            "eta_phi": float(eta_phi),
            "update_clip": None if update_clip is None else float(update_clip),
        },
        "features": {"names": list(feature_names)},
        "model": {
            "model_type": str(model_type),
            "ridge_lambda": float(model_ridge_lambda),
            "l2_lambda": float(model_l2_lambda),
            "lr": float(model_lr),
            "max_iter": int(model_max_iter),
            "tol": float(model_tol),
        },
        "cv": {
            "seed": int(seed),
            "fold_order": list(fold_order),
            "fold_weights": list(fold_weights),
        },
    }
    (out_dir / "best_config.json").write_text(json.dumps(best_cfg, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")

    metrics_payload: dict[str, object] = {
        "schema_version": "accuracy_a1_isomers_a2.v1",
        "experiment_id": str(experiment_id),
        "dataset": {
            "rows_total": int(len(rows_sorted)),
            "groups_total": int(len(group_ids)),
        },
        "input_csv": input_csv.as_posix(),
        "input_sha256_normalized": _sha256_text_normalized(input_csv),
        "best_config": best_cfg,
        "model_type": str(model_type),
        "metrics_loocv_test": metrics_test,
        "kpi": kpi_payload,
        "worst_groups": worst_groups,
        "files": {
            "summary_csv": "summary.csv",
            "predictions_csv": "predictions.csv",
            "fold_metrics_csv": "fold_metrics.csv",
            "group_metrics_csv": "group_metrics.csv",
            "metrics_json": "metrics.json",
            "best_config_json": "best_config.json",
            "provenance_json": "provenance.json",
            "manifest_json": "manifest.json",
            "checksums_sha256": "checksums.sha256",
            "index_md": "index.md",
        },
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics_payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")

    index_lines = [
        f"# {experiment_id} (Isomers) A2 self-consistent functional",
        "",
        "LOOCV (by group_id) metrics:",
        f"- mean_spearman_by_group: {metrics_test.get('mean_spearman_by_group')}",
        f"- median_spearman_by_group: {metrics_test.get('median_spearman_by_group')}",
        f"- pairwise_order_accuracy_overall: {metrics_test.get('pairwise_order_accuracy_overall')} ({metrics_test.get('pairwise_correct')}/{metrics_test.get('pairwise_total')})",
        f"- top1_accuracy_mean: {metrics_test.get('top1_accuracy_mean')}",
        "",
        "KPI:",
        f"- verdict: {kpi_payload.get('verdict')}",
        f"- reason: {kpi_payload.get('reason')}",
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
        (Path("data/accuracy/raw/dft_golden_isomers_v2_spice2_0_1.csv"), repo_root / "data/accuracy/raw/dft_golden_isomers_v2_spice2_0_1.csv"),
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
            raise AccuracyA2SelfConsistentError(f"failed to copy provenance file {src} -> {rel_dst}") from exc

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


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run ACCURACY-A2 isomers self-consistent functional (LOOCV by group_id) and build an evidence pack.")
    p.add_argument("--experiment_id", type=str, default="ACCURACY-A2", help="Experiment/Roadmap label to embed in the pack.")
    p.add_argument("--input_csv", type=Path, default=DEFAULT_INPUT_CSV, help="Canonical isomer truth CSV.")
    p.add_argument("--out_dir", type=Path, default=DEFAULT_OUT_DIR, help="Output directory.")
    p.add_argument("--seed", type=int, default=0, help="Deterministic fold order seed (group-level).")
    p.add_argument("--gamma", type=float, default=0.28, help="Potential scale gamma for H = L_w + gamma*diag(V).")
    p.add_argument("--potential_variant", type=str, default="epsilon_z", help="Potential variant: epsilon_z or epsilon_z_plus_features_v2")
    p.add_argument("--v_deg_coeff", type=float, default=0.10, help="Potential add-on coefficient for node degree.")
    p.add_argument("--v_valence_coeff", type=float, default=0.05, help="Potential add-on coefficient for node valence.")
    p.add_argument("--v_arom_coeff", type=float, default=0.20, help="Potential add-on coefficient for aromatic flag.")
    p.add_argument("--v_ring_coeff", type=float, default=0.20, help="Potential add-on coefficient for ring membership.")
    p.add_argument("--v_charge_coeff", type=float, default=0.50, help="Potential add-on coefficient for formal charge.")
    p.add_argument("--v_chi_coeff", type=float, default=0.00, help="Potential add-on coefficient for chi(Z) proxy.")

    p.add_argument("--edge_weight_mode", type=str, default="bond_order_delta_chi", help="Edge weights: bond_order or bond_order_delta_chi")
    p.add_argument("--edge_aromatic_mult", type=float, default=0.0, help="Multiplier for aromatic bonds in weighted adjacency.")
    p.add_argument("--edge_delta_chi_alpha", type=float, default=1.0, help="Alpha coefficient for delta-chi edge weights.")

    p.add_argument("--sc_occ_k", type=int, default=5, help="Occupied eigenvectors (k) for rho estimate.")
    p.add_argument("--sc_tau", type=float, default=1.0, help="Softmax temperature tau for rho estimate.")
    p.add_argument("--sc_max_iter", type=int, default=5, help="Self-consistent iterations (2-5 recommended).")
    p.add_argument("--sc_tol", type=float, default=1e-6, help="Self-consistent convergence tolerance (inf-norm on dV).")
    p.add_argument("--sc_damping", type=float, default=0.5, help="Self-consistent damping in (0,1].")
    p.add_argument("--phi_eps", type=float, default=1e-6, help="Epsilon for phi=-ln(rho+eps).")
    p.add_argument("--eta_a", type=float, default=0.1, help="Self-consistent update coeff for curvature a=-Lphi.")
    p.add_argument("--eta_phi", type=float, default=0.0, help="Self-consistent update coeff for centered phi.")
    p.add_argument("--update_clip", type=float, default=0.5, help="Clip for update vector (abs).")

    p.add_argument("--model_type", type=str, default="pairwise_rank_ridge", help="Model type: pairwise_rank_ridge or pairwise_logistic_l2")
    p.add_argument("--model_ridge_lambda", type=float, default=1e-3, help="Ridge lambda for pairwise_rank_ridge.")
    p.add_argument("--model_l2_lambda", type=float, default=1e-3, help="L2 regularization for pairwise logistic.")
    p.add_argument("--model_lr", type=float, default=0.1, help="Learning rate for pairwise logistic.")
    p.add_argument("--model_max_iter", type=int, default=2000, help="Max iterations for pairwise logistic.")
    p.add_argument("--model_tol", type=float, default=1e-6, help="Gradient norm tolerance for pairwise logistic.")

    p.add_argument("--kpi_mean_spearman_by_group_test_min", type=float, default=0.55, help="KPI gate (LOOCV test): mean spearman by group >= value.")
    p.add_argument("--kpi_median_spearman_by_group_test_min", type=float, default=0.55, help="KPI gate (LOOCV test): median spearman by group >= value.")
    p.add_argument("--kpi_pairwise_order_accuracy_overall_test_min", type=float, default=0.70, help="KPI gate (LOOCV test): overall pairwise order accuracy >= value.")
    p.add_argument("--kpi_top1_accuracy_mean_test_min", type=float, default=0.40, help="KPI gate (LOOCV test): top1 mean accuracy >= value.")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    run_accuracy_a1_isomers_a2_self_consistent(
        experiment_id=str(args.experiment_id),
        input_csv=Path(args.input_csv),
        out_dir=Path(args.out_dir),
        seed=int(args.seed),
        gamma=float(args.gamma),
        potential_variant=str(args.potential_variant),
        v_deg_coeff=float(args.v_deg_coeff),
        v_valence_coeff=float(args.v_valence_coeff),
        v_arom_coeff=float(args.v_arom_coeff),
        v_ring_coeff=float(args.v_ring_coeff),
        v_charge_coeff=float(args.v_charge_coeff),
        v_chi_coeff=float(args.v_chi_coeff),
        edge_weight_mode=str(args.edge_weight_mode),
        edge_aromatic_mult=float(args.edge_aromatic_mult),
        edge_delta_chi_alpha=float(args.edge_delta_chi_alpha),
        sc_occ_k=int(args.sc_occ_k),
        sc_tau=float(args.sc_tau),
        sc_max_iter=int(args.sc_max_iter),
        sc_tol=float(args.sc_tol),
        sc_damping=float(args.sc_damping),
        phi_eps=float(args.phi_eps),
        eta_a=float(args.eta_a),
        eta_phi=float(args.eta_phi),
        update_clip=float(args.update_clip) if args.update_clip is not None else None,
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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
