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
class _A2FullFunctionalConfig:
    config_id: str
    heat_tau: float
    phi_eps: float
    sc_iters: int
    sc_eta: float
    sc_clip: float
    coef_alpha: float
    coef_beta: float
    coef_gamma: float
    mass_mode: str  # "Z" or "sqrtZ"
    rho_floor: float = 0.0
    sc_damping: float = 1.0
    sc_max_backtracks: int = 3
    gauge_fix_phi_mean: bool = False
    # ACCURACY-A2.4:
    # - "heat_diag": trace-normalized shifted heat-kernel diagonal (A2.3 rho-law, baseline)
    # - "soft_occupancy_ldos": soft-occupancy LDOS from eigenpairs (A2.4 rho-law, new)
    rho_mode: str = "heat_diag"
    rho_ldos_k: int = 0
    rho_ldos_beta: float = 0.0
    rho_ldos_deg_tol: float = 1e-8


def _heat_kernel_diagonal(H: np.ndarray, *, tau: float) -> np.ndarray:
    tau_val = float(tau)
    if tau_val <= 0.0 or not math.isfinite(tau_val):
        raise AccuracyA2SelfConsistentError("heat_tau must be > 0 and finite")
    H0 = np.asarray(H, dtype=float)
    if H0.ndim != 2 or H0.shape[0] != H0.shape[1]:
        raise AccuracyA2SelfConsistentError("H must be square")
    eigvals, eigvecs = np.linalg.eigh(H0)
    eigvals = np.asarray(eigvals, dtype=float).reshape(-1)
    eigvecs = np.asarray(eigvecs, dtype=float)
    weights = np.exp(-tau_val * eigvals)
    rho = (eigvecs**2) @ weights
    return np.asarray(rho, dtype=float).reshape(-1)


def _heat_kernel_diagonal_shifted_trace_normalized(
    H: np.ndarray, *, tau: float
) -> tuple[np.ndarray, float, float, float, float]:
    """
    ACCURACY-A2.3 rho-law:
      rho = diag(exp(-tau * (H - lambda_min I))) / trace(exp(-tau * (H - lambda_min I))).

    Returns:
      (rho, trace_heat, lambda_min, lambda_max, rho_trace_norm)
    """
    tau_val = float(tau)
    if tau_val <= 0.0 or not math.isfinite(tau_val):
        raise AccuracyA2SelfConsistentError("heat_tau must be > 0 and finite")
    H0 = np.asarray(H, dtype=float)
    if H0.ndim != 2 or H0.shape[0] != H0.shape[1]:
        raise AccuracyA2SelfConsistentError("H must be square")

    eigvals, eigvecs = np.linalg.eigh(H0)
    eigvals = np.asarray(eigvals, dtype=float).reshape(-1)
    eigvecs = np.asarray(eigvecs, dtype=float)
    if eigvals.size == 0:
        raise AccuracyA2SelfConsistentError("H must be non-empty")

    lambda_min = float(eigvals[0])
    lambda_max = float(eigvals[-1])
    weights = np.exp(-tau_val * (eigvals - lambda_min))
    diag_heat = (eigvecs**2) @ weights

    trace_heat = float(np.sum(weights))
    if not math.isfinite(trace_heat) or trace_heat <= 0.0:
        raise AccuracyA2SelfConsistentError("invalid trace_heat in rho-law")

    rho = np.asarray(diag_heat, dtype=float).reshape(-1) / float(trace_heat)
    rho_trace_norm = float(np.sum(rho))
    return rho, float(trace_heat), float(lambda_min), float(lambda_max), float(rho_trace_norm)


def _soft_occupancy_ldos_rho(
    H: np.ndarray, *, beta: float, k: int, deg_tol: float
) -> tuple[np.ndarray, float, float, float, float, float, int, bool, float, bool, float, float, float]:
    """
    ACCURACY-A2.4 rho-law (mode B: soft-occupancy LDOS):

      eig(H) = (lambda_j, u_j)
      w_j = softmax(-beta * (lambda_j - lambda_min_window)) over the lowest-k eigenpairs
      rho_i = sum_j w_j * |u_{ij}|^2

    Returns:
      (rho, trace_weights, lambda_min, lambda_max, rho_trace_norm, rho_entropy,
       k_eff, degeneracy_guard_applied, rho_sum, rho_renorm_applied, rho_renorm_delta,
       lambda_min_window, lambda_gap)
    """
    beta_val = float(beta)
    if beta_val <= 0.0 or not math.isfinite(beta_val):
        raise AccuracyA2SelfConsistentError("rho_ldos_beta must be > 0 and finite")
    k_val = int(k)
    if k_val <= 0:
        raise AccuracyA2SelfConsistentError("rho_ldos_k must be >= 1")
    deg_tol_val = float(deg_tol)
    if deg_tol_val < 0.0 or not math.isfinite(deg_tol_val):
        raise AccuracyA2SelfConsistentError("rho_ldos_deg_tol must be >= 0 and finite")

    H0 = np.asarray(H, dtype=float)
    if H0.ndim != 2 or H0.shape[0] != H0.shape[1]:
        raise AccuracyA2SelfConsistentError("H must be square")

    eigvals, eigvecs = np.linalg.eigh(H0)
    eigvals = np.asarray(eigvals, dtype=float).reshape(-1)
    eigvecs = np.asarray(eigvecs, dtype=float)
    n = int(eigvals.size)
    if n == 0:
        raise AccuracyA2SelfConsistentError("H must be non-empty")
    if k_val > n:
        raise AccuracyA2SelfConsistentError("rho_ldos_k must be <= n")

    lambda_min = float(eigvals[0])
    lambda_max = float(eigvals[-1])

    cutoff_lambda = float(eigvals[k_val - 1])
    k_eff = int(k_val)
    while k_eff < n and abs(float(eigvals[k_eff]) - cutoff_lambda) <= deg_tol_val:
        k_eff += 1
    degeneracy_guard_applied = bool(k_eff != k_val)

    eigvals_k = eigvals[:k_eff]
    eigvecs_k = eigvecs[:, :k_eff]
    lambda_min_window = float(eigvals_k[0])
    lambda_gap = float(float(eigvals_k[-1]) - lambda_min_window)

    logits = -beta_val * (eigvals_k - float(lambda_min_window))
    logits = logits - float(np.max(logits))
    exp_logits = np.exp(logits)
    Z = float(np.sum(exp_logits))
    if not math.isfinite(Z) or Z <= 0.0:
        raise AccuracyA2SelfConsistentError("invalid trace_weights in rho-law")
    w = exp_logits / float(Z)

    rho = (np.abs(eigvecs_k) ** 2) @ w
    rho = np.asarray(rho, dtype=float).reshape(-1)
    rho_sum = float(np.sum(rho))
    rho_renorm_delta = float(rho_sum - 1.0)
    rho_sum_tol = 1e-8
    if abs(rho_renorm_delta) > rho_sum_tol and math.isfinite(rho_sum) and rho_sum > 0.0:
        rho = np.asarray(rho / float(rho_sum), dtype=float)
        rho_renorm_applied = True
    else:
        rho_renorm_applied = False
    rho_trace_norm = float(np.sum(rho))
    rho_entropy = float(-np.sum(rho * np.log(rho + 1e-300)))
    return (
        rho,
        float(Z),
        float(lambda_min),
        float(lambda_max),
        float(rho_trace_norm),
        float(rho_entropy),
        int(k_eff),
        bool(degeneracy_guard_applied),
        float(rho_sum),
        bool(rho_renorm_applied),
        float(rho_renorm_delta),
        float(lambda_min_window),
        float(lambda_gap),
    )


def _solve_scf_full_functional_v1(
    *,
    laplacian: np.ndarray,
    v0: np.ndarray,
    heat_tau: float,
    phi_eps: float,
    sc_iters: int,
    sc_eta: float,
    sc_clip: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[dict[str, object]], bool, int, float]:
    lap = np.asarray(laplacian, dtype=float)
    if lap.ndim != 2 or lap.shape[0] != lap.shape[1]:
        raise AccuracyA2SelfConsistentError("laplacian must be square")

    v0_vec = np.asarray(v0, dtype=float).reshape(-1)
    n = int(lap.shape[0])
    if v0_vec.size != n:
        raise AccuracyA2SelfConsistentError("v0 must have shape (n,)")

    eps = float(phi_eps)
    if eps <= 0.0 or not math.isfinite(eps):
        raise AccuracyA2SelfConsistentError("phi_eps must be > 0 and finite")

    iters = int(sc_iters)
    if iters < 0 or iters > 5:
        raise AccuracyA2SelfConsistentError("sc_iters must be in [0,5]")

    eta = float(sc_eta)
    if not math.isfinite(eta):
        raise AccuracyA2SelfConsistentError("sc_eta must be finite")

    clip = float(sc_clip)
    if clip <= 0.0 or not math.isfinite(clip):
        raise AccuracyA2SelfConsistentError("sc_clip must be > 0 and finite")

    v = v0_vec.copy()
    trace: list[dict[str, object]] = []
    residual_final = 0.0

    for t in range(iters):
        H = lap + np.diag(v)
        rho = _heat_kernel_diagonal(H, tau=float(heat_tau))
        phi = -np.log(rho + eps)
        curvature = -np.asarray(lap @ phi, dtype=float)

        upd = float(eta) * np.clip(curvature, -clip, clip)
        v_next = v0_vec + upd
        dv = np.asarray(v_next - v, dtype=float)
        residual_inf = float(np.max(np.abs(dv))) if dv.size else 0.0
        residual_final = float(residual_inf)

        trace.append(
            {
                "iter": int(t + 1),
                "residual_inf": float(residual_inf),
                "min_V": float(np.min(v_next)),
                "max_V": float(np.max(v_next)),
                "mean_V": float(np.mean(v_next)),
                "min_rho": float(np.min(rho)),
                "max_rho": float(np.max(rho)),
                "mean_rho": float(np.mean(rho)),
            }
        )
        v = v_next

    H_final = lap + np.diag(v)
    rho_final = _heat_kernel_diagonal(H_final, tau=float(heat_tau))
    phi_final = -np.log(rho_final + eps)
    curvature_final = -np.asarray(lap @ phi_final, dtype=float)
    converged = bool(residual_final <= 1e-6) if iters else True

    return (
        np.asarray(v, dtype=float),
        np.asarray(rho_final, dtype=float),
        np.asarray(phi_final, dtype=float),
        np.asarray(curvature_final, dtype=float),
        list(trace),
        bool(converged),
        int(iters),
        float(residual_final),
    )


def _solve_scf_full_functional_v1_variationally_stable(
    *,
    laplacian: np.ndarray,
    v0: np.ndarray,
    heat_tau: float,
    phi_eps: float,
    rho_floor: float,
    sc_iters: int,
    sc_eta: float,
    sc_damping: float,
    sc_clip: float,
    sc_max_backtracks: int,
    coef_alpha: float,
    coef_beta: float,
    coef_gamma: float,
    mvec: np.ndarray,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    list[dict[str, object]],
    bool,
    int,
    float,
    list[float],
    list[int],
    list[float],
]:
    """
    ACCURACY-A2.2: variationally-stable SCF (monotone relaxation on the final E).

    Law:
      - rho = diag(exp(-tau * H))  (heat-kernel diagonal)
      - phi = -log(max(rho, rho_floor) + eps), then gauge-fix: phi -= mean(phi)
      - a = -L phi
      - V* = V0 + eta * clip(a, c)
      - V_next = (1-d) * V + d * V*
      - backtracking: if E_next > E_prev then eta *= 0.5 (max 3)
    """
    lap = np.asarray(laplacian, dtype=float)
    if lap.ndim != 2 or lap.shape[0] != lap.shape[1]:
        raise AccuracyA2SelfConsistentError("laplacian must be square")

    v0_vec = np.asarray(v0, dtype=float).reshape(-1)
    n = int(lap.shape[0])
    if v0_vec.size != n:
        raise AccuracyA2SelfConsistentError("v0 must have shape (n,)")

    eps = float(phi_eps)
    if eps <= 0.0 or not math.isfinite(eps):
        raise AccuracyA2SelfConsistentError("phi_eps must be > 0 and finite")

    iters = int(sc_iters)
    if iters < 0 or iters > 5:
        raise AccuracyA2SelfConsistentError("sc_iters must be in [0,5]")

    eta0 = float(sc_eta)
    if not math.isfinite(eta0):
        raise AccuracyA2SelfConsistentError("sc_eta must be finite")

    damping = float(sc_damping)
    if damping < 0.0 or damping > 1.0 or not math.isfinite(damping):
        raise AccuracyA2SelfConsistentError("sc_damping must be in [0,1] and finite")

    clip = float(sc_clip)
    if clip <= 0.0 or not math.isfinite(clip):
        raise AccuracyA2SelfConsistentError("sc_clip must be > 0 and finite")

    rho_floor_val = float(rho_floor)
    if rho_floor_val < 0.0 or not math.isfinite(rho_floor_val):
        raise AccuracyA2SelfConsistentError("rho_floor must be >= 0 and finite")

    max_bt = int(sc_max_backtracks)
    if max_bt < 0 or max_bt > 3:
        raise AccuracyA2SelfConsistentError("sc_max_backtracks must be in [0,3]")

    m0 = np.asarray(mvec, dtype=float).reshape(-1)
    if m0.size != n:
        raise AccuracyA2SelfConsistentError("mvec must have shape (n,)")

    alpha = float(coef_alpha)
    beta = float(coef_beta)
    gamma = float(coef_gamma)
    if not (math.isfinite(alpha) and math.isfinite(beta) and math.isfinite(gamma)):
        raise AccuracyA2SelfConsistentError("coef_alpha/beta/gamma must be finite")

    def _state_for_v(v_vec: np.ndarray) -> tuple[np.ndarray, float, np.ndarray, np.ndarray, float]:
        H = lap + np.diag(v_vec)
        rho_raw = _heat_kernel_diagonal(H, tau=float(heat_tau))
        if rho_raw.size != n:
            raise AccuracyA2SelfConsistentError("invalid rho shape")

        if rho_floor_val > 0.0:
            floor_mask = rho_raw < rho_floor_val
            floor_rate = float(np.mean(floor_mask)) if floor_mask.size else 0.0
            rho = np.maximum(rho_raw, rho_floor_val)
        else:
            floor_rate = 0.0
            rho = rho_raw

        phi = -np.log(rho + eps)
        phi = np.asarray(phi - float(np.mean(phi)), dtype=float)  # gauge-fix
        curvature = -np.asarray(lap @ phi, dtype=float)

        grad_energy = float(phi.T @ (lap @ phi))
        curv_l1 = float(np.sum(np.abs(curvature)))
        mass_phi = float(np.sum(m0 * phi))
        E = float(alpha * grad_energy + beta * curv_l1 + gamma * mass_phi)
        if not math.isfinite(E):
            raise AccuracyA2SelfConsistentError("non-finite E in SCF")

        return rho_raw, float(floor_rate), phi, curvature, float(E)

    v = v0_vec.copy()
    rho_raw, floor_rate, phi, curvature, E = _state_for_v(v)

    residual_final = 0.0
    E_trace: list[float] = [float(E)]
    accepted_backtracks: list[int] = []
    rho_floor_rate_trace: list[float] = [float(floor_rate)]
    trace: list[dict[str, object]] = []
    monotonic_ok = True
    iters_done = 0

    for t in range(iters):
        iters_done = int(t + 1)
        eta_k = float(eta0)
        accepted = False
        bt_used = 0
        E_prev = float(E)
        v_prev = v.copy()
        residual_inf = 0.0

        for bt in range(max_bt + 1):
            bt_used = int(bt)
            upd = float(eta_k) * np.clip(curvature, -clip, clip)
            v_star = v0_vec + upd
            v_cand = (1.0 - damping) * v + damping * v_star

            rho_raw_c, floor_rate_c, phi_c, curvature_c, E_c = _state_for_v(v_cand)
            dv = np.asarray(v_cand - v, dtype=float)
            residual_inf = float(np.max(np.abs(dv))) if dv.size else 0.0

            if E_c <= E_prev:
                accepted = True
                v = np.asarray(v_cand, dtype=float)
                rho_raw = np.asarray(rho_raw_c, dtype=float)
                floor_rate = float(floor_rate_c)
                phi = np.asarray(phi_c, dtype=float)
                curvature = np.asarray(curvature_c, dtype=float)
                E = float(E_c)
                residual_final = float(residual_inf)
                break

            eta_k *= 0.5

        trace.append(
            {
                "iter": int(t + 1),
                "accepted": bool(accepted),
                "accepted_backtracks": int(bt_used),
                "eta_used": float(eta_k),
                "E_prev": float(E_prev),
                "E_next": float(E),
                "residual_inf": float(residual_inf),
                "rho_floor_rate": float(floor_rate),
                "min_V": float(np.min(v)),
                "max_V": float(np.max(v)),
                "mean_V": float(np.mean(v)),
                "min_rho": float(np.min(rho_raw)),
                "max_rho": float(np.max(rho_raw)),
                "mean_rho": float(np.mean(rho_raw)),
            }
        )

        accepted_backtracks.append(int(bt_used))
        E_trace.append(float(E))
        rho_floor_rate_trace.append(float(floor_rate))

        if not accepted:
            monotonic_ok = False
            v = v_prev
            break

    return (
        np.asarray(v, dtype=float),
        np.asarray(rho_raw, dtype=float),
        np.asarray(phi, dtype=float),
        np.asarray(curvature, dtype=float),
        list(trace),
        bool(monotonic_ok),
        int(iters_done),
        float(residual_final),
        list(E_trace),
        list(accepted_backtracks),
        list(rho_floor_rate_trace),
    )


def _solve_scf_full_functional_v1_variationally_stable_rho_trace_normalized(
    *,
    laplacian: np.ndarray,
    v0: np.ndarray,
    heat_tau: float,
    phi_eps: float,
    rho_floor: float,
    sc_iters: int,
    sc_eta: float,
    sc_damping: float,
    sc_clip: float,
    sc_max_backtracks: int,
    coef_alpha: float,
    coef_beta: float,
    coef_gamma: float,
    mvec: np.ndarray,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    float,
    float,
    float,
    float,
    list[dict[str, object]],
    bool,
    int,
    float,
    list[float],
    list[int],
    list[float],
]:
    """
    ACCURACY-A2.3: keep A2.2 SCF discipline fixed, change ONLY the rho-law.

    rho-law:
      rho = diag(exp(-tau * (H - lambda_min I))) / trace(exp(-tau * (H - lambda_min I)))
    """
    lap = np.asarray(laplacian, dtype=float)
    if lap.ndim != 2 or lap.shape[0] != lap.shape[1]:
        raise AccuracyA2SelfConsistentError("laplacian must be square")

    v0_vec = np.asarray(v0, dtype=float).reshape(-1)
    n = int(lap.shape[0])
    if v0_vec.size != n:
        raise AccuracyA2SelfConsistentError("v0 must have shape (n,)")

    eps = float(phi_eps)
    if eps <= 0.0 or not math.isfinite(eps):
        raise AccuracyA2SelfConsistentError("phi_eps must be > 0 and finite")

    iters = int(sc_iters)
    if iters < 0 or iters > 5:
        raise AccuracyA2SelfConsistentError("sc_iters must be in [0,5]")

    eta0 = float(sc_eta)
    if not math.isfinite(eta0):
        raise AccuracyA2SelfConsistentError("sc_eta must be finite")

    damping = float(sc_damping)
    if damping < 0.0 or damping > 1.0 or not math.isfinite(damping):
        raise AccuracyA2SelfConsistentError("sc_damping must be in [0,1] and finite")

    clip = float(sc_clip)
    if clip <= 0.0 or not math.isfinite(clip):
        raise AccuracyA2SelfConsistentError("sc_clip must be > 0 and finite")

    rho_floor_val = float(rho_floor)
    if rho_floor_val < 0.0 or not math.isfinite(rho_floor_val):
        raise AccuracyA2SelfConsistentError("rho_floor must be >= 0 and finite")

    max_bt = int(sc_max_backtracks)
    if max_bt < 0 or max_bt > 3:
        raise AccuracyA2SelfConsistentError("sc_max_backtracks must be in [0,3]")

    m0 = np.asarray(mvec, dtype=float).reshape(-1)
    if m0.size != n:
        raise AccuracyA2SelfConsistentError("mvec must have shape (n,)")

    alpha = float(coef_alpha)
    beta = float(coef_beta)
    gamma = float(coef_gamma)
    if not (math.isfinite(alpha) and math.isfinite(beta) and math.isfinite(gamma)):
        raise AccuracyA2SelfConsistentError("coef_alpha/beta/gamma must be finite")

    def _state_for_v(v_vec: np.ndarray) -> tuple[np.ndarray, float, float, float, float, float, np.ndarray, np.ndarray, float]:
        H = lap + np.diag(v_vec)
        rho_norm, trace_heat, lambda_min, lambda_max, rho_trace_norm = _heat_kernel_diagonal_shifted_trace_normalized(
            H, tau=float(heat_tau)
        )
        if rho_norm.size != n:
            raise AccuracyA2SelfConsistentError("invalid rho shape")

        if rho_floor_val > 0.0:
            floor_mask = rho_norm < rho_floor_val
            floor_rate = float(np.mean(floor_mask)) if floor_mask.size else 0.0
            rho_used = np.maximum(rho_norm, rho_floor_val)
        else:
            floor_rate = 0.0
            rho_used = rho_norm

        phi = -np.log(rho_used + eps)
        phi = np.asarray(phi - float(np.mean(phi)), dtype=float)  # gauge-fix (fixed)
        curvature = -np.asarray(lap @ phi, dtype=float)

        grad_energy = float(phi.T @ (lap @ phi))
        curv_l1 = float(np.sum(np.abs(curvature)))
        mass_phi = float(np.sum(m0 * phi))
        E = float(alpha * grad_energy + beta * curv_l1 + gamma * mass_phi)
        if not math.isfinite(E):
            raise AccuracyA2SelfConsistentError("non-finite E in SCF")

        return (
            np.asarray(rho_norm, dtype=float),
            float(floor_rate),
            float(trace_heat),
            float(lambda_min),
            float(lambda_max),
            float(rho_trace_norm),
            np.asarray(phi, dtype=float),
            np.asarray(curvature, dtype=float),
            float(E),
        )

    v = v0_vec.copy()
    rho, floor_rate, trace_heat, lambda_min, lambda_max, rho_trace_norm, phi, curvature, E = _state_for_v(v)

    residual_final = 0.0
    E_trace: list[float] = [float(E)]
    accepted_backtracks: list[int] = []
    rho_floor_rate_trace: list[float] = [float(floor_rate)]
    trace: list[dict[str, object]] = []
    monotonic_ok = True
    iters_done = 0

    for t in range(iters):
        iters_done = int(t + 1)
        eta_k = float(eta0)
        accepted = False
        bt_used = 0
        E_prev = float(E)
        v_prev = v.copy()
        residual_inf = 0.0

        for bt in range(max_bt + 1):
            bt_used = int(bt)
            upd = float(eta_k) * np.clip(curvature, -clip, clip)
            v_star = v0_vec + upd
            v_cand = (1.0 - damping) * v + damping * v_star

            rho_c, floor_rate_c, trace_heat_c, lambda_min_c, lambda_max_c, rho_trace_norm_c, phi_c, curvature_c, E_c = _state_for_v(v_cand)
            dv = np.asarray(v_cand - v, dtype=float)
            residual_inf = float(np.max(np.abs(dv))) if dv.size else 0.0

            if E_c <= E_prev:
                accepted = True
                v = np.asarray(v_cand, dtype=float)
                rho = np.asarray(rho_c, dtype=float)
                floor_rate = float(floor_rate_c)
                trace_heat = float(trace_heat_c)
                lambda_min = float(lambda_min_c)
                lambda_max = float(lambda_max_c)
                rho_trace_norm = float(rho_trace_norm_c)
                phi = np.asarray(phi_c, dtype=float)
                curvature = np.asarray(curvature_c, dtype=float)
                E = float(E_c)
                residual_final = float(residual_inf)
                break

            eta_k *= 0.5

        trace.append(
            {
                "iter": int(t + 1),
                "accepted": bool(accepted),
                "accepted_backtracks": int(bt_used),
                "eta_used": float(eta_k),
                "E_prev": float(E_prev),
                "E_next": float(E),
                "residual_inf": float(residual_inf),
                "rho_floor_rate": float(floor_rate),
                "rho_trace_norm": float(rho_trace_norm),
                "trace_heat": float(trace_heat),
                "lambda_min": float(lambda_min),
                "lambda_max": float(lambda_max),
                "min_V": float(np.min(v)),
                "max_V": float(np.max(v)),
                "mean_V": float(np.mean(v)),
                "rho_min": float(np.min(rho)),
                "rho_max": float(np.max(rho)),
            }
        )

        accepted_backtracks.append(int(bt_used))
        E_trace.append(float(E))
        rho_floor_rate_trace.append(float(floor_rate))

        if not accepted:
            monotonic_ok = False
            v = v_prev
            break

    return (
        np.asarray(v, dtype=float),
        np.asarray(rho, dtype=float),
        np.asarray(phi, dtype=float),
        np.asarray(curvature, dtype=float),
        float(trace_heat),
        float(lambda_min),
        float(lambda_max),
        float(rho_trace_norm),
        list(trace),
        bool(monotonic_ok),
        int(iters_done),
        float(residual_final),
        list(E_trace),
        list(accepted_backtracks),
        list(rho_floor_rate_trace),
    )


def _solve_scf_full_functional_v1_variationally_stable_rho_a2_4(
    *,
    laplacian: np.ndarray,
    v0: np.ndarray,
    heat_tau: float,
    phi_eps: float,
    rho_floor: float,
    sc_iters: int,
    sc_eta: float,
    sc_damping: float,
    sc_clip: float,
    sc_max_backtracks: int,
    coef_alpha: float,
    coef_beta: float,
    coef_gamma: float,
    mvec: np.ndarray,
    rho_mode: str,
    rho_ldos_beta: float,
    rho_ldos_k: int,
    rho_ldos_deg_tol: float,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    float,
    float,
    float,
    float,
    float,
    float,
    bool,
    float,
    int,
    bool,
    float,
    float,
    list[dict[str, object]],
    bool,
    int,
    float,
    list[float],
    list[int],
    list[float],
]:
    """
    ACCURACY-A2.4: keep A2.2 SCF discipline fixed, change ONLY the rho-law via rho_mode:

      - rho_mode="heat_diag": A2.3 rho-law baseline (trace-normalized shifted heat diag)
      - rho_mode="soft_occupancy_ldos": A2.4 rho-law (soft-occupancy LDOS)
    """
    lap = np.asarray(laplacian, dtype=float)
    if lap.ndim != 2 or lap.shape[0] != lap.shape[1]:
        raise AccuracyA2SelfConsistentError("laplacian must be square")

    v0_vec = np.asarray(v0, dtype=float).reshape(-1)
    n = int(lap.shape[0])
    if v0_vec.size != n:
        raise AccuracyA2SelfConsistentError("v0 must have shape (n,)")

    eps = float(phi_eps)
    if eps <= 0.0 or not math.isfinite(eps):
        raise AccuracyA2SelfConsistentError("phi_eps must be > 0 and finite")

    iters = int(sc_iters)
    if iters < 0 or iters > 5:
        raise AccuracyA2SelfConsistentError("sc_iters must be in [0,5]")

    eta0 = float(sc_eta)
    if not math.isfinite(eta0):
        raise AccuracyA2SelfConsistentError("sc_eta must be finite")

    damping = float(sc_damping)
    if damping < 0.0 or damping > 1.0 or not math.isfinite(damping):
        raise AccuracyA2SelfConsistentError("sc_damping must be in [0,1] and finite")

    clip = float(sc_clip)
    if clip <= 0.0 or not math.isfinite(clip):
        raise AccuracyA2SelfConsistentError("sc_clip must be > 0 and finite")

    rho_floor_val = float(rho_floor)
    if rho_floor_val < 0.0 or not math.isfinite(rho_floor_val):
        raise AccuracyA2SelfConsistentError("rho_floor must be >= 0 and finite")

    max_bt = int(sc_max_backtracks)
    if max_bt < 0 or max_bt > 3:
        raise AccuracyA2SelfConsistentError("sc_max_backtracks must be in [0,3]")

    m0 = np.asarray(mvec, dtype=float).reshape(-1)
    if m0.size != n:
        raise AccuracyA2SelfConsistentError("mvec must have shape (n,)")

    alpha = float(coef_alpha)
    beta = float(coef_beta)
    gamma = float(coef_gamma)
    if not (math.isfinite(alpha) and math.isfinite(beta) and math.isfinite(gamma)):
        raise AccuracyA2SelfConsistentError("coef_alpha/beta/gamma must be finite")

    rho_mode_val = str(rho_mode)
    if rho_mode_val not in {"heat_diag", "soft_occupancy_ldos"}:
        raise AccuracyA2SelfConsistentError(f"invalid rho_mode: {rho_mode_val}")

    def _state_for_v(
        v_vec: np.ndarray,
    ) -> tuple[
        np.ndarray,
        float,
        float,
        float,
        float,
        float,
        float,
        np.ndarray,
        np.ndarray,
        float,
        float,
        bool,
        float,
        int,
        bool,
        float,
        float,
    ]:
        H = lap + np.diag(v_vec)
        if rho_mode_val == "heat_diag":
            rho_norm, trace_heat, lambda_min, lambda_max, rho_trace_norm = _heat_kernel_diagonal_shifted_trace_normalized(
                H, tau=float(heat_tau)
            )
            trace_weights = float(trace_heat)
            rho_entropy = float(-np.sum(rho_norm * np.log(rho_norm + 1e-300)))
            rho_sum = float(rho_trace_norm)
            rho_renorm_applied = False
            rho_renorm_delta = float(rho_sum - 1.0)
            k_eff = 0
            degeneracy_guard_applied = False
            lambda_min_window = float(lambda_min)
            lambda_gap = float(float(lambda_max) - float(lambda_min))
        else:
            (
                rho_norm,
                trace_weights,
                lambda_min,
                lambda_max,
                rho_trace_norm,
                rho_entropy,
                k_eff,
                degeneracy_guard_applied,
                rho_sum,
                rho_renorm_applied,
                rho_renorm_delta,
                lambda_min_window,
                lambda_gap,
            ) = _soft_occupancy_ldos_rho(
                H,
                beta=float(rho_ldos_beta),
                k=int(rho_ldos_k),
                deg_tol=float(rho_ldos_deg_tol),
            )

        if rho_norm.size != n:
            raise AccuracyA2SelfConsistentError("invalid rho shape")

        if rho_floor_val > 0.0:
            floor_mask = rho_norm < rho_floor_val
            floor_rate = float(np.mean(floor_mask)) if floor_mask.size else 0.0
            rho_used = np.maximum(rho_norm, rho_floor_val)
        else:
            floor_rate = 0.0
            rho_used = rho_norm

        phi = -np.log(rho_used + eps)
        phi = np.asarray(phi - float(np.mean(phi)), dtype=float)  # gauge-fix (fixed)
        curvature = -np.asarray(lap @ phi, dtype=float)

        grad_energy = float(phi.T @ (lap @ phi))
        curv_l1 = float(np.sum(np.abs(curvature)))
        mass_phi = float(np.sum(m0 * phi))
        E = float(alpha * grad_energy + beta * curv_l1 + gamma * mass_phi)
        if not math.isfinite(E):
            raise AccuracyA2SelfConsistentError("non-finite E in SCF")

        return (
            np.asarray(rho_norm, dtype=float),
            float(floor_rate),
            float(trace_weights),
            float(lambda_min),
            float(lambda_max),
            float(rho_trace_norm),
            float(rho_entropy),
            np.asarray(phi, dtype=float),
            np.asarray(curvature, dtype=float),
            float(E),
            float(rho_sum),
            bool(rho_renorm_applied),
            float(rho_renorm_delta),
            int(k_eff),
            bool(degeneracy_guard_applied),
            float(lambda_min_window),
            float(lambda_gap),
        )

    v = v0_vec.copy()
    (
        rho,
        floor_rate,
        trace_weights,
        lambda_min,
        lambda_max,
        rho_trace_norm,
        rho_entropy,
        phi,
        curvature,
        E,
        rho_sum,
        rho_renorm_applied,
        rho_renorm_delta,
        rho_ldos_k_eff,
        rho_ldos_degeneracy_guard_applied,
        lambda_min_window,
        lambda_gap,
    ) = _state_for_v(v)

    residual_final = 0.0
    E_trace: list[float] = [float(E)]
    accepted_backtracks: list[int] = []
    rho_floor_rate_trace: list[float] = [float(floor_rate)]
    trace: list[dict[str, object]] = []
    monotonic_ok = True
    iters_done = 0

    for t in range(iters):
        iters_done = int(t + 1)
        eta_k = float(eta0)
        accepted = False
        bt_used = 0
        E_prev = float(E)
        v_prev = v.copy()
        residual_inf = 0.0

        for bt in range(max_bt + 1):
            bt_used = int(bt)
            upd = float(eta_k) * np.clip(curvature, -clip, clip)
            v_star = v0_vec + upd
            v_cand = (1.0 - damping) * v + damping * v_star

            (
                rho_c,
                floor_rate_c,
                trace_weights_c,
                lambda_min_c,
                lambda_max_c,
                rho_trace_norm_c,
                rho_entropy_c,
                phi_c,
                curvature_c,
                E_c,
                rho_sum_c,
                rho_renorm_applied_c,
                rho_renorm_delta_c,
                rho_ldos_k_eff_c,
                rho_ldos_degeneracy_guard_applied_c,
                lambda_min_window_c,
                lambda_gap_c,
            ) = _state_for_v(v_cand)
            dv = np.asarray(v_cand - v, dtype=float)
            residual_inf = float(np.max(np.abs(dv))) if dv.size else 0.0

            if E_c <= E_prev:
                accepted = True
                v = np.asarray(v_cand, dtype=float)
                rho = np.asarray(rho_c, dtype=float)
                floor_rate = float(floor_rate_c)
                trace_weights = float(trace_weights_c)
                lambda_min = float(lambda_min_c)
                lambda_max = float(lambda_max_c)
                rho_trace_norm = float(rho_trace_norm_c)
                rho_entropy = float(rho_entropy_c)
                rho_sum = float(rho_sum_c)
                rho_renorm_applied = bool(rho_renorm_applied_c)
                rho_renorm_delta = float(rho_renorm_delta_c)
                rho_ldos_k_eff = int(rho_ldos_k_eff_c)
                rho_ldos_degeneracy_guard_applied = bool(rho_ldos_degeneracy_guard_applied_c)
                lambda_min_window = float(lambda_min_window_c)
                lambda_gap = float(lambda_gap_c)
                phi = np.asarray(phi_c, dtype=float)
                curvature = np.asarray(curvature_c, dtype=float)
                E = float(E_c)
                residual_final = float(residual_inf)
                break

            eta_k *= 0.5

        trace.append(
            {
                "iter": int(t + 1),
                "accepted": bool(accepted),
                "accepted_backtracks": int(bt_used),
                "eta_used": float(eta_k),
                "E_prev": float(E_prev),
                "E_next": float(E),
                "residual_inf": float(residual_inf),
                "rho_floor_rate": float(floor_rate),
                "rho_trace_norm": float(rho_trace_norm),
                "rho_sum": float(rho_sum),
                "rho_renorm_applied": bool(rho_renorm_applied),
                "rho_renorm_delta": float(rho_renorm_delta),
                "rho_entropy": float(rho_entropy),
                "trace_weights": float(trace_weights),
                "rho_ldos_k_eff": int(rho_ldos_k_eff),
                "rho_ldos_deg_tol": float(rho_ldos_deg_tol),
                "rho_ldos_degeneracy_guard_applied": bool(rho_ldos_degeneracy_guard_applied),
                "lambda_min": float(lambda_min),
                "lambda_max": float(lambda_max),
                "lambda_min_window": float(lambda_min_window),
                "lambda_gap": float(lambda_gap),
                "min_V": float(np.min(v)),
                "max_V": float(np.max(v)),
                "mean_V": float(np.mean(v)),
                "rho_min": float(np.min(rho)),
                "rho_max": float(np.max(rho)),
            }
        )

        accepted_backtracks.append(int(bt_used))
        E_trace.append(float(E))
        rho_floor_rate_trace.append(float(floor_rate))

        if not accepted:
            monotonic_ok = False
            v = v_prev
            break

    return (
        np.asarray(v, dtype=float),
        np.asarray(rho, dtype=float),
        np.asarray(phi, dtype=float),
        np.asarray(curvature, dtype=float),
        float(trace_weights),
        float(lambda_min),
        float(lambda_max),
        float(rho_trace_norm),
        float(rho_entropy),
        float(rho_sum),
        bool(rho_renorm_applied),
        float(rho_renorm_delta),
        int(rho_ldos_k_eff),
        bool(rho_ldos_degeneracy_guard_applied),
        float(lambda_min_window),
        float(lambda_gap),
        list(trace),
        bool(monotonic_ok),
        int(iters_done),
        float(residual_final),
        list(E_trace),
        list(accepted_backtracks),
        list(rho_floor_rate_trace),
    )


def _mass_vector(types: Sequence[int], *, mode: str) -> np.ndarray:
    mz = str(mode)
    z = np.asarray([float(int(t)) for t in types], dtype=float)
    if mz == "Z":
        return z
    if mz == "sqrtZ":
        return np.sqrt(z)
    raise AccuracyA2SelfConsistentError(f"invalid mass_mode: {mz}")


def _a2_full_functional_default_configs() -> list[_A2FullFunctionalConfig]:
    configs: list[_A2FullFunctionalConfig] = []
    phi_eps_default = 1e-6
    sc_clip_default = 0.5

    base_tau = [0.5, 1.0, 2.0]
    mass_modes = ["Z", "sqrtZ"]
    coef_sets = [(1.0, 1.0, 1.0), (1.0, 0.5, 1.0)]
    for tau in base_tau:
        for mz in mass_modes:
            for a, b, g in coef_sets:
                cfg_id = f"ffv1_{len(configs) + 1:03d}"
                configs.append(
                    _A2FullFunctionalConfig(
                        config_id=cfg_id,
                        heat_tau=float(tau),
                        phi_eps=float(phi_eps_default),
                        sc_iters=0,
                        sc_eta=0.0,
                        sc_clip=float(sc_clip_default),
                        coef_alpha=float(a),
                        coef_beta=float(b),
                        coef_gamma=float(g),
                        mass_mode=str(mz),
                    )
                )

    sc_tau = [0.5, 1.0, 2.0, 4.0]
    for tau in sc_tau:
        for mz in mass_modes:
            cfg_id = f"ffv1_{len(configs) + 1:03d}"
            configs.append(
                _A2FullFunctionalConfig(
                    config_id=cfg_id,
                    heat_tau=float(tau),
                    phi_eps=float(phi_eps_default),
                    sc_iters=3,
                    sc_eta=0.1,
                    sc_clip=float(sc_clip_default),
                    coef_alpha=1.0,
                    coef_beta=1.0,
                    coef_gamma=1.0,
                    mass_mode=str(mz),
                )
            )

    if len(configs) > 20:
        raise AccuracyA2SelfConsistentError("A2.1 search budget exceeded: configs > 20")
    return configs


def _a2_full_functional_a2_2_default_configs() -> list[_A2FullFunctionalConfig]:
    """
    ACCURACY-A2.2: keep the functional fixed; search only SCF stability knobs (budget <= 20).
    """
    configs: list[_A2FullFunctionalConfig] = []
    phi_eps_default = 1e-6
    sc_clip_default = 0.5

    base_tau = [0.5, 1.0, 2.0]
    base_eta = [0.1, 0.05]
    base_floor = [0.0, 1e-12]

    # Base: K=3, damping=0.5 (12 configs)
    for tau in base_tau:
        for eta in base_eta:
            for rf in base_floor:
                cfg_id = f"a2_2_{len(configs) + 1:03d}"
                configs.append(
                    _A2FullFunctionalConfig(
                        config_id=cfg_id,
                        heat_tau=float(tau),
                        phi_eps=float(phi_eps_default),
                        rho_floor=float(rf),
                        sc_iters=3,
                        sc_eta=float(eta),
                        sc_damping=0.5,
                        sc_clip=float(sc_clip_default),
                        sc_max_backtracks=3,
                        gauge_fix_phi_mean=True,
                        coef_alpha=1.0,
                        coef_beta=1.0,
                        coef_gamma=1.0,
                        mass_mode="Z",
                    )
                )

    # Extra: smaller damping at tau=1.0 (4 configs)
    for eta in base_eta:
        for rf in base_floor:
            cfg_id = f"a2_2_{len(configs) + 1:03d}"
            configs.append(
                _A2FullFunctionalConfig(
                    config_id=cfg_id,
                    heat_tau=1.0,
                    phi_eps=float(phi_eps_default),
                    rho_floor=float(rf),
                    sc_iters=3,
                    sc_eta=float(eta),
                    sc_damping=0.2,
                    sc_clip=float(sc_clip_default),
                    sc_max_backtracks=3,
                    gauge_fix_phi_mean=True,
                    coef_alpha=1.0,
                    coef_beta=1.0,
                    coef_gamma=1.0,
                    mass_mode="Z",
                )
            )

    # Extra: more iterations at tau=1.0 (4 configs) -> total 20
    for eta in base_eta:
        for rf in base_floor:
            cfg_id = f"a2_2_{len(configs) + 1:03d}"
            configs.append(
                _A2FullFunctionalConfig(
                    config_id=cfg_id,
                    heat_tau=1.0,
                    phi_eps=float(phi_eps_default),
                    rho_floor=float(rf),
                    sc_iters=5,
                    sc_eta=float(eta),
                    sc_damping=0.5,
                    sc_clip=float(sc_clip_default),
                    sc_max_backtracks=3,
                    gauge_fix_phi_mean=True,
                    coef_alpha=1.0,
                    coef_beta=1.0,
                    coef_gamma=1.0,
                    mass_mode="Z",
                )
            )

    if len(configs) > 20:
        raise AccuracyA2SelfConsistentError("A2.2 search budget exceeded: configs > 20")
    return configs


def _a2_full_functional_a2_3_default_configs() -> list[_A2FullFunctionalConfig]:
    """
    ACCURACY-A2.3: same knobs/budget as A2.2; rho-law is the only intended change.
    """
    configs: list[_A2FullFunctionalConfig] = []
    phi_eps_default = 1e-6
    sc_clip_default = 0.5

    base_tau = [0.5, 1.0, 2.0]
    base_eta = [0.1, 0.05]
    base_floor = [0.0, 1e-12]

    # Base: K=3, damping=0.5 (12 configs)
    for tau in base_tau:
        for eta in base_eta:
            for rf in base_floor:
                cfg_id = f"a2_3_{len(configs) + 1:03d}"
                configs.append(
                    _A2FullFunctionalConfig(
                        config_id=cfg_id,
                        heat_tau=float(tau),
                        phi_eps=float(phi_eps_default),
                        rho_floor=float(rf),
                        sc_iters=3,
                        sc_eta=float(eta),
                        sc_damping=0.5,
                        sc_clip=float(sc_clip_default),
                        sc_max_backtracks=3,
                        gauge_fix_phi_mean=True,
                        coef_alpha=1.0,
                        coef_beta=1.0,
                        coef_gamma=1.0,
                        mass_mode="Z",
                    )
                )

    # Extra: smaller damping at tau=1.0 (4 configs)
    for eta in base_eta:
        for rf in base_floor:
            cfg_id = f"a2_3_{len(configs) + 1:03d}"
            configs.append(
                _A2FullFunctionalConfig(
                    config_id=cfg_id,
                    heat_tau=1.0,
                    phi_eps=float(phi_eps_default),
                    rho_floor=float(rf),
                    sc_iters=3,
                    sc_eta=float(eta),
                    sc_damping=0.2,
                    sc_clip=float(sc_clip_default),
                    sc_max_backtracks=3,
                    gauge_fix_phi_mean=True,
                    coef_alpha=1.0,
                    coef_beta=1.0,
                    coef_gamma=1.0,
                    mass_mode="Z",
                )
            )

    # Extra: more iterations at tau=1.0 (4 configs) -> total 20
    for eta in base_eta:
        for rf in base_floor:
            cfg_id = f"a2_3_{len(configs) + 1:03d}"
            configs.append(
                _A2FullFunctionalConfig(
                    config_id=cfg_id,
                    heat_tau=1.0,
                    phi_eps=float(phi_eps_default),
                    rho_floor=float(rf),
                    sc_iters=5,
                    sc_eta=float(eta),
                    sc_damping=0.5,
                    sc_clip=float(sc_clip_default),
                    sc_max_backtracks=3,
                    gauge_fix_phi_mean=True,
                    coef_alpha=1.0,
                    coef_beta=1.0,
                    coef_gamma=1.0,
                    mass_mode="Z",
                )
            )

    if len(configs) > 20:
        raise AccuracyA2SelfConsistentError("A2.3 search budget exceeded: configs > 20")
    return configs


def _a2_full_functional_a2_4_default_configs() -> list[_A2FullFunctionalConfig]:
    """
    ACCURACY-A2.4: keep A2.2 SCF discipline fixed; extend search space with rho_mode=B (soft-occupancy LDOS).
    Budget <= 20 total (rho_mode=A baseline + rho_mode=B candidates).
    """
    configs: list[_A2FullFunctionalConfig] = []
    phi_eps_default = 1e-6
    sc_clip_default = 0.5

    # Baseline (rho_mode=A): same rho-law as A2.3 (trace-normalized heat diag)
    base_tau = [0.5, 1.0, 2.0]
    base_eta = [0.1, 0.05]
    for tau in base_tau:
        for eta in base_eta:
            cfg_id = f"a2_4_{len(configs) + 1:03d}"
            configs.append(
                _A2FullFunctionalConfig(
                    config_id=cfg_id,
                    heat_tau=float(tau),
                    phi_eps=float(phi_eps_default),
                    rho_floor=0.0,
                    sc_iters=3,
                    sc_eta=float(eta),
                    sc_damping=0.5,
                    sc_clip=float(sc_clip_default),
                    sc_max_backtracks=3,
                    gauge_fix_phi_mean=True,
                    coef_alpha=1.0,
                    coef_beta=1.0,
                    coef_gamma=1.0,
                    mass_mode="Z",
                    rho_mode="heat_diag",
                )
            )

    # New (rho_mode=B): soft-occupancy LDOS.
    # NOTE: k must be <= min(heavy_atoms) across isomer_truth.v1 (currently 14).
    ldos_beta = [0.5, 1.0, 2.0, 4.0]
    ldos_k = [6, 10, 14]
    for beta in ldos_beta:
        for k in ldos_k:
            cfg_id = f"a2_4_{len(configs) + 1:03d}"
            configs.append(
                _A2FullFunctionalConfig(
                    config_id=cfg_id,
                    heat_tau=1.0,  # unused by rho_mode=B; fixed for comparability/logging
                    phi_eps=float(phi_eps_default),
                    rho_floor=0.0,
                    sc_iters=3,
                    sc_eta=0.1,
                    sc_damping=0.5,
                    sc_clip=float(sc_clip_default),
                    sc_max_backtracks=3,
                    gauge_fix_phi_mean=True,
                    coef_alpha=1.0,
                    coef_beta=1.0,
                    coef_gamma=1.0,
                    mass_mode="Z",
                    rho_mode="soft_occupancy_ldos",
                    rho_ldos_beta=float(beta),
                    rho_ldos_k=int(k),
                )
            )

    # Two extra B candidates with smaller eta (total <= 20).
    for k in [10, 14]:
        cfg_id = f"a2_4_{len(configs) + 1:03d}"
        configs.append(
            _A2FullFunctionalConfig(
                config_id=cfg_id,
                heat_tau=1.0,
                phi_eps=float(phi_eps_default),
                rho_floor=0.0,
                sc_iters=3,
                sc_eta=0.05,
                sc_damping=0.5,
                sc_clip=float(sc_clip_default),
                sc_max_backtracks=3,
                gauge_fix_phi_mean=True,
                coef_alpha=1.0,
                coef_beta=1.0,
                coef_gamma=1.0,
                mass_mode="Z",
                rho_mode="soft_occupancy_ldos",
                rho_ldos_beta=2.0,
                rho_ldos_k=int(k),
            )
        )

    if len(configs) > 20:
        raise AccuracyA2SelfConsistentError("A2.4 search budget exceeded: configs > 20")
    return configs


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


def _compute_group_metrics_for_pred_rel(records: list[dict[str, object]], *, pred_rel_key: str) -> dict[str, dict[str, object]]:
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


def _aggregate_metrics_from_group_metrics(group_metrics: dict[str, dict[str, object]]) -> dict[str, float | int]:
    spearmans = [
        float(v.get("spearman_pred_vs_truth", float("nan")))
        for v in group_metrics.values()
        if math.isfinite(float(v.get("spearman_pred_vs_truth", float("nan"))))
    ]
    mean_spearman = float(statistics.fmean(spearmans)) if spearmans else float("nan")
    median_spearman = float(statistics.median(spearmans)) if spearmans else float("nan")

    top1s = [
        float(v.get("top1_accuracy", float("nan")))
        for v in group_metrics.values()
        if math.isfinite(float(v.get("top1_accuracy", float("nan"))))
    ]
    top1_mean = float(statistics.fmean(top1s)) if top1s else float("nan")

    pair_correct = sum(int(v.get("pairwise_correct") or 0) for v in group_metrics.values())
    pair_total = sum(int(v.get("pairwise_total") or 0) for v in group_metrics.values())
    pair_acc = float(pair_correct) / float(pair_total) if pair_total else float("nan")

    num_negative = sum(
        1
        for v in group_metrics.values()
        if not math.isfinite(float(v.get("spearman_pred_vs_truth", float("nan"))))
        or float(v.get("spearman_pred_vs_truth", float("nan"))) < 0.0
    )

    return {
        "mean_spearman_by_group": float(mean_spearman),
        "median_spearman_by_group": float(median_spearman),
        "pairwise_order_accuracy_overall": float(pair_acc),
        "pairwise_correct": int(pair_correct),
        "pairwise_total": int(pair_total),
        "top1_accuracy_mean": float(top1_mean),
        "num_groups_spearman_negative": int(num_negative),
    }


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


def _full_functional_terms_for_row(
    row: _Row,
    *,
    atoms_db: AtomsDbV1,
    potential_gamma: float,
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
    cfg: _A2FullFunctionalConfig,
    scf_mode: str,
) -> dict[str, float | int | bool]:
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
        gamma=float(potential_gamma),
        potential_variant=str(potential_variant),
        v_deg_coeff=float(v_deg_coeff),
        v_valence_coeff=float(v_valence_coeff),
        v_arom_coeff=float(v_arom_coeff),
        v_ring_coeff=float(v_ring_coeff),
        v_charge_coeff=float(v_charge_coeff),
        v_chi_coeff=float(v_chi_coeff),
    )

    mvec = _mass_vector(row.types, mode=str(cfg.mass_mode))
    mode = str(scf_mode)
    if mode == "a2_2":
        (
            v,
            rho,
            phi,
            curvature,
            _,
            sc_converged,
            sc_iters,
            residual_final,
            E_trace,
            accepted_backtracks,
            rho_floor_rate_trace,
        ) = _solve_scf_full_functional_v1_variationally_stable(
            laplacian=lap,
            v0=v0,
            heat_tau=float(cfg.heat_tau),
            phi_eps=float(cfg.phi_eps),
            rho_floor=float(cfg.rho_floor),
            sc_iters=int(cfg.sc_iters),
            sc_eta=float(cfg.sc_eta),
            sc_damping=float(cfg.sc_damping),
            sc_clip=float(cfg.sc_clip),
            sc_max_backtracks=int(cfg.sc_max_backtracks),
            coef_alpha=float(cfg.coef_alpha),
            coef_beta=float(cfg.coef_beta),
            coef_gamma=float(cfg.coef_gamma),
            mvec=mvec,
        )
        trace_heat = None
        lambda_min = None
        lambda_max = None
        rho_trace_norm = None
        trace_weights = None
        rho_entropy = None
        rho_sum = None
        rho_renorm_applied = None
        rho_renorm_delta = None
        rho_ldos_k_eff = None
        rho_ldos_degeneracy_guard_applied = None
        lambda_min_window = None
        lambda_gap = None
    elif mode == "a2_3":
        (
            v,
            rho,
            phi,
            curvature,
            trace_heat,
            lambda_min,
            lambda_max,
            rho_trace_norm,
            _,
            sc_converged,
            sc_iters,
            residual_final,
            E_trace,
            accepted_backtracks,
            rho_floor_rate_trace,
        ) = _solve_scf_full_functional_v1_variationally_stable_rho_trace_normalized(
            laplacian=lap,
            v0=v0,
            heat_tau=float(cfg.heat_tau),
            phi_eps=float(cfg.phi_eps),
            rho_floor=float(cfg.rho_floor),
            sc_iters=int(cfg.sc_iters),
            sc_eta=float(cfg.sc_eta),
            sc_damping=float(cfg.sc_damping),
            sc_clip=float(cfg.sc_clip),
            sc_max_backtracks=int(cfg.sc_max_backtracks),
            coef_alpha=float(cfg.coef_alpha),
            coef_beta=float(cfg.coef_beta),
            coef_gamma=float(cfg.coef_gamma),
            mvec=mvec,
        )
        trace_weights = None
        rho_entropy = None
        rho_sum = None
        rho_renorm_applied = None
        rho_renorm_delta = None
        rho_ldos_k_eff = None
        rho_ldos_degeneracy_guard_applied = None
        lambda_min_window = None
        lambda_gap = None
    elif mode == "a2_4":
        (
            v,
            rho,
            phi,
            curvature,
            trace_weights,
            lambda_min,
            lambda_max,
            rho_trace_norm,
            rho_entropy,
            rho_sum,
            rho_renorm_applied,
            rho_renorm_delta,
            rho_ldos_k_eff,
            rho_ldos_degeneracy_guard_applied,
            lambda_min_window,
            lambda_gap,
            _,
            sc_converged,
            sc_iters,
            residual_final,
            E_trace,
            accepted_backtracks,
            rho_floor_rate_trace,
        ) = _solve_scf_full_functional_v1_variationally_stable_rho_a2_4(
            laplacian=lap,
            v0=v0,
            heat_tau=float(cfg.heat_tau),
            phi_eps=float(cfg.phi_eps),
            rho_floor=float(cfg.rho_floor),
            sc_iters=int(cfg.sc_iters),
            sc_eta=float(cfg.sc_eta),
            sc_damping=float(cfg.sc_damping),
            sc_clip=float(cfg.sc_clip),
            sc_max_backtracks=int(cfg.sc_max_backtracks),
            coef_alpha=float(cfg.coef_alpha),
            coef_beta=float(cfg.coef_beta),
            coef_gamma=float(cfg.coef_gamma),
            mvec=mvec,
            rho_mode=str(cfg.rho_mode),
            rho_ldos_beta=float(cfg.rho_ldos_beta),
            rho_ldos_k=int(cfg.rho_ldos_k),
            rho_ldos_deg_tol=float(cfg.rho_ldos_deg_tol),
        )
        trace_heat = None
    else:
        v, rho, phi, curvature, _, sc_converged, sc_iters, residual_final = _solve_scf_full_functional_v1(
            laplacian=lap,
            v0=v0,
            heat_tau=float(cfg.heat_tau),
            phi_eps=float(cfg.phi_eps),
            sc_iters=int(cfg.sc_iters),
            sc_eta=float(cfg.sc_eta),
            sc_clip=float(cfg.sc_clip),
        )
        E_trace = []
        accepted_backtracks = []
        rho_floor_rate_trace = []
        trace_heat = None
        lambda_min = None
        lambda_max = None
        rho_trace_norm = None
        trace_weights = None
        rho_entropy = None
        rho_sum = None
        rho_renorm_applied = None
        rho_renorm_delta = None
        rho_ldos_k_eff = None
        rho_ldos_degeneracy_guard_applied = None
        lambda_min_window = None
        lambda_gap = None

    if rho.size != int(row.n_heavy_atoms) or phi.size != int(row.n_heavy_atoms) or curvature.size != int(row.n_heavy_atoms):
        raise AccuracyA2SelfConsistentError(f"invalid SCF output shapes for id={row.mid}")
    if not math.isfinite(float(np.min(rho))) or float(np.min(rho)) < 0.0:
        raise AccuracyA2SelfConsistentError(f"invalid rho for id={row.mid}")

    grad_energy = float(phi.T @ (lap @ phi))
    curv_l1 = float(np.sum(np.abs(curvature)))
    curv_l2 = float(np.linalg.norm(curvature))
    curv_maxabs = float(np.max(np.abs(curvature))) if curvature.size else 0.0
    mass_phi = float(np.sum(mvec * phi))

    term_grad = float(cfg.coef_alpha) * float(grad_energy)
    term_curv = float(cfg.coef_beta) * float(curv_l1)
    term_mass = float(cfg.coef_gamma) * float(mass_phi)
    E = float(term_grad + term_curv + term_mass)

    phi_std = float(np.std(phi)) if phi.size else 0.0
    payload: dict[str, float | int | bool] = {
        "E": float(E),
        "grad_energy": float(grad_energy),
        "curv_l1": float(curv_l1),
        "curv_l2": float(curv_l2),
        "curv_maxabs": float(curv_maxabs),
        "mass_phi": float(mass_phi),
        "term_grad": float(term_grad),
        "term_curv": float(term_curv),
        "term_mass": float(term_mass),
        "rho_min": float(np.min(rho)),
        "rho_max": float(np.max(rho)),
        "rho_mean": float(np.mean(rho)),
        "phi_min": float(np.min(phi)),
        "phi_max": float(np.max(phi)),
        "phi_std": float(phi_std),
        "sc_iters": int(sc_iters),
        "sc_residual_final": float(residual_final),
        "sc_converged": bool(sc_converged),
        "sc_rho_floor_rate_max": float(max(rho_floor_rate_trace) if rho_floor_rate_trace else 0.0),
        "sc_rho_floor_rate_final": float(rho_floor_rate_trace[-1] if rho_floor_rate_trace else 0.0),
        "sc_accepted_backtracks_mean": float(
            float(sum(int(x) for x in accepted_backtracks)) / float(len(accepted_backtracks)) if accepted_backtracks else 0.0
        ),
        "sc_accepted_backtracks_max": int(max((int(x) for x in accepted_backtracks), default=0)),
        "sc_E_trace": "|".join(f"{float(x):.12g}" for x in E_trace) if E_trace else "",
        "min_V": float(np.min(v)),
        "max_V": float(np.max(v)),
        "mean_V": float(np.mean(v)),
    }
    if mode == "a2_3":
        if trace_heat is None or lambda_min is None or lambda_max is None or rho_trace_norm is None:
            raise AccuracyA2SelfConsistentError("missing A2.3 rho-law diagnostics")
        payload.update(
            {
                "rho_trace_norm": float(rho_trace_norm),
                "trace_heat": float(trace_heat),
                "lambda_min": float(lambda_min),
                "lambda_max": float(lambda_max),
            }
        )
    elif mode == "a2_4":
        if (
            trace_weights is None
            or lambda_min is None
            or lambda_max is None
            or rho_trace_norm is None
            or rho_entropy is None
            or rho_sum is None
            or rho_renorm_applied is None
            or rho_renorm_delta is None
            or rho_ldos_k_eff is None
            or rho_ldos_degeneracy_guard_applied is None
            or lambda_min_window is None
            or lambda_gap is None
        ):
            raise AccuracyA2SelfConsistentError("missing A2.4 rho-law diagnostics")
        payload.update(
            {
                "rho_trace_norm": float(rho_trace_norm),
                "rho_sum": float(rho_sum),
                "rho_renorm_applied": bool(rho_renorm_applied),
                "rho_renorm_delta": float(rho_renorm_delta),
                "trace_weights": float(trace_weights),
                "lambda_min": float(lambda_min),
                "lambda_max": float(lambda_max),
                "lambda_min_window": float(lambda_min_window),
                "lambda_gap": float(lambda_gap),
                "rho_entropy": float(rho_entropy),
                "rho_ldos_beta": float(cfg.rho_ldos_beta),
                "rho_ldos_k": int(cfg.rho_ldos_k),
                "rho_ldos_k_eff": int(rho_ldos_k_eff),
                "rho_ldos_deg_tol": float(cfg.rho_ldos_deg_tol),
                "rho_ldos_degeneracy_guard_applied": bool(rho_ldos_degeneracy_guard_applied),
            }
        )
    non_numeric_keys = {"sc_iters", "sc_converged", "sc_E_trace"}
    if not all(isinstance(payload[k], bool) or math.isfinite(float(payload[k])) for k in payload if k not in non_numeric_keys):
        raise AccuracyA2SelfConsistentError(f"non-finite functional terms for id={row.mid}: {payload}")
    return payload


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


def run_accuracy_a1_isomers_a2_full_functional_v1(
    *,
    experiment_id: str,
    input_csv: Path,
    out_dir: Path,
    seed: int,
    potential_gamma: float,
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
    calibrator_ridge_lambda: float,
    kpi_mean_spearman_by_group_test_min: float,
    kpi_pairwise_order_accuracy_overall_test_min: float,
    kpi_median_spearman_by_group_test_min: float,
    kpi_top1_accuracy_mean_test_min: float,
    full_functional_mode: str = "a2_1",
) -> None:
    if not input_csv.exists():
        raise AccuracyA2SelfConsistentError(f"missing input_csv: {input_csv}")

    rows = _load_rows(input_csv)
    rows_sorted = sorted(rows, key=lambda rr: (str(rr.gid), str(rr.mid)))
    group_ids = sorted({r.gid for r in rows_sorted})
    if len(group_ids) < 2:
        raise AccuracyA2SelfConsistentError("need at least 2 groups for LOOCV")

    atoms_db = load_atoms_db_v1()
    if not atoms_db.potential_by_atomic_num:
        raise AccuracyA2SelfConsistentError("atoms_db_v1 must provide epsilon(Z)")

    truth_rel_by_id = _group_truth_min_center(rows_sorted)

    mode = str(full_functional_mode)
    if mode == "a2_4":
        configs = _a2_full_functional_a2_4_default_configs()
    elif mode == "a2_3":
        configs = _a2_full_functional_a2_3_default_configs()
    elif mode == "a2_2":
        configs = _a2_full_functional_a2_2_default_configs()
    else:
        configs = _a2_full_functional_default_configs()
    cfg_by_id = {c.config_id: c for c in configs}
    if len(cfg_by_id) != len(configs):
        raise AccuracyA2SelfConsistentError("duplicate config_id in full functional search space")

    eval_by_config: dict[str, dict[str, dict[str, float | int | bool]]] = {}
    group_metrics_by_config: dict[str, dict[str, dict[str, object]]] = {}
    search_rows: list[dict[str, object]] = []

    for cfg in configs:
        terms_by_id: dict[str, dict[str, float | int | bool]] = {}
        for r in rows_sorted:
            terms_by_id[r.mid] = _full_functional_terms_for_row(
                r,
                atoms_db=atoms_db,
                potential_gamma=float(potential_gamma),
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
                cfg=cfg,
                scf_mode=str(mode),
            )

        eval_by_config[str(cfg.config_id)] = terms_by_id

        min_E_by_gid: dict[str, float] = {}
        for gid in group_ids:
            e_vals = [float(terms_by_id[r.mid]["E"]) for r in rows_sorted if r.gid == gid]
            if not e_vals:
                raise AccuracyA2SelfConsistentError("empty group while building config metrics")
            min_E_by_gid[str(gid)] = float(min(e_vals))

        records_cfg: list[dict[str, object]] = []
        for r in rows_sorted:
            e = float(terms_by_id[r.mid]["E"])
            records_cfg.append(
                {
                    "id": str(r.mid),
                    "group_id": str(r.gid),
                    "smiles": str(r.smiles),
                    "truth_rel_kcalmol": float(truth_rel_by_id[r.mid]),
                    "pred_raw": float(e),
                    "pred_rel": float(float(e) - float(min_E_by_gid[str(r.gid)])),
                }
            )
        gm = _compute_group_metrics(records_cfg)
        group_metrics_by_config[str(cfg.config_id)] = gm
        agg = _aggregate_metrics_from_group_metrics(gm)
        sr: dict[str, object] = {
            "config_id": str(cfg.config_id),
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
        }
        if mode == "a2_4":
            sr.update(
                {
                    "rho_mode": str(cfg.rho_mode),
                    "rho_ldos_beta": float(cfg.rho_ldos_beta),
                    "rho_ldos_k": int(cfg.rho_ldos_k),
                    "rho_ldos_deg_tol": float(cfg.rho_ldos_deg_tol),
                }
            )
        sr.update({k: agg[k] for k in sorted(agg.keys())})
        search_rows.append(sr)

    rng = random.Random(int(seed))
    fold_order = list(group_ids)
    rng.shuffle(fold_order)

    def _finite_or(value: float, fallback: float) -> float:
        return float(value) if math.isfinite(float(value)) else float(fallback)

    fold_selection: list[dict[str, object]] = []
    selected_cfg_by_test_gid: dict[str, str] = {}
    for fold_id, test_gid in enumerate(fold_order):
        train_gids = [g for g in group_ids if g != test_gid]
        best_key: tuple[object, ...] | None = None
        best_cfg_id: str | None = None
        best_train_stats: dict[str, object] = {}
        for cfg in configs:
            gm = group_metrics_by_config[str(cfg.config_id)]
            spearmans = [float(gm[str(g)].get("spearman_pred_vs_truth", float("nan"))) for g in train_gids]
            num_neg = sum(1 for s in spearmans if not math.isfinite(float(s)) or float(s) < 0.0)
            spearmans_for_median = [_finite_or(float(s), -1.0) for s in spearmans]
            median = float(statistics.median(spearmans_for_median)) if spearmans_for_median else float("nan")
            mean = float(statistics.fmean(spearmans_for_median)) if spearmans_for_median else float("nan")
            key = (int(num_neg), -float(median), -float(mean), str(cfg.config_id))
            if best_key is None or key < best_key:
                best_key = key
                best_cfg_id = str(cfg.config_id)
                best_train_stats = {
                    "train_num_groups_spearman_negative": int(num_neg),
                    "train_median_spearman_by_group": float(median),
                    "train_mean_spearman_by_group": float(mean),
                }

        if best_cfg_id is None:
            raise AccuracyA2SelfConsistentError("failed to select config for fold")
        selected_cfg_by_test_gid[str(test_gid)] = str(best_cfg_id)
        fold_selection.append(
            {
                "fold_id": int(fold_id),
                "test_group_id": str(test_gid),
                "selected_config_id": str(best_cfg_id),
                **best_train_stats,
            }
        )

    fold_selection_by_id: dict[int, dict[str, object]] = {int(r.get("fold_id", -1)): r for r in fold_selection}
    if len(fold_selection_by_id) != len(fold_selection):
        raise AccuracyA2SelfConsistentError("duplicate fold_id in fold_selection")

    metrics_test_func_by_rho_mode: dict[str, dict[str, object]] | None = None
    negative_spearman_groups_test_by_rho_mode: dict[str, list[str]] | None = None
    if mode == "a2_4":
        configs_a = [c for c in configs if str(c.rho_mode) == "heat_diag"]
        configs_b = [c for c in configs if str(c.rho_mode) == "soft_occupancy_ldos"]

        def _select_configs(configs_subset: list[_A2FullFunctionalConfig]) -> dict[str, str]:
            if not configs_subset:
                raise AccuracyA2SelfConsistentError("empty configs_subset while selecting rho_mode baseline")
            sel: dict[str, str] = {}
            for test_gid in fold_order:
                train_gids = [g for g in group_ids if g != test_gid]
                best_key: tuple[object, ...] | None = None
                best_cfg_id: str | None = None
                for cfg in configs_subset:
                    gm = group_metrics_by_config[str(cfg.config_id)]
                    spearmans = [float(gm[str(g)].get("spearman_pred_vs_truth", float("nan"))) for g in train_gids]
                    num_neg = sum(1 for s in spearmans if not math.isfinite(float(s)) or float(s) < 0.0)
                    spearmans_for_median = [_finite_or(float(s), -1.0) for s in spearmans]
                    median = float(statistics.median(spearmans_for_median)) if spearmans_for_median else float("nan")
                    mean = float(statistics.fmean(spearmans_for_median)) if spearmans_for_median else float("nan")
                    key = (int(num_neg), -float(median), -float(mean), str(cfg.config_id))
                    if best_key is None or key < best_key:
                        best_key = key
                        best_cfg_id = str(cfg.config_id)
                if best_cfg_id is None:
                    raise AccuracyA2SelfConsistentError("failed to select rho_mode baseline config for fold")
                sel[str(test_gid)] = str(best_cfg_id)
            return sel

        def _metrics_for_selected_cfgs(selected: dict[str, str]) -> tuple[dict[str, object], list[str]]:
            records: list[dict[str, object]] = []
            for fold_id, test_gid in enumerate(fold_order):
                cfg_id = str(selected[str(test_gid)])
                terms_by_id = eval_by_config[cfg_id]
                test_rows = [r for r in rows_sorted if r.gid == test_gid]
                if not test_rows:
                    raise AccuracyA2SelfConsistentError("empty test group while building rho_mode metrics")
                pred_raw: dict[str, float] = {r.mid: float(terms_by_id[r.mid]["E"]) for r in test_rows}
                min_pred = min(float(pred_raw[r.mid]) for r in test_rows)
                for r in test_rows:
                    e = float(pred_raw[r.mid])
                    records.append(
                        {
                            "fold_id": int(fold_id),
                            "selected_config_id": str(cfg_id),
                            "rho_mode": str(cfg_by_id[cfg_id].rho_mode),
                            "id": str(r.mid),
                            "group_id": str(r.gid),
                            "smiles": str(r.smiles),
                            "truth_rel_kcalmol": float(truth_rel_by_id[r.mid]),
                            "pred_raw": float(e),
                            "pred_rel": float(e - float(min_pred)),
                        }
                    )
            gm = _compute_group_metrics(records)
            agg = _aggregate_metrics_from_group_metrics(gm)
            neg_groups = [
                str(gid)
                for gid, rec in gm.items()
                if math.isfinite(float(rec.get("spearman_pred_vs_truth", float("nan"))))
                and float(rec.get("spearman_pred_vs_truth", float("nan"))) < 0.0
            ]
            return {k: agg[k] for k in sorted(agg.keys())}, sorted(neg_groups)

        selected_a = _select_configs(configs_a)
        selected_b = _select_configs(configs_b)
        metrics_a, neg_a = _metrics_for_selected_cfgs(selected_a)
        metrics_b, neg_b = _metrics_for_selected_cfgs(selected_b)
        metrics_test_func_by_rho_mode = {"heat_diag": metrics_a, "soft_occupancy_ldos": metrics_b}
        negative_spearman_groups_test_by_rho_mode = {"heat_diag": neg_a, "soft_occupancy_ldos": neg_b}

    feature_names_cal = ["grad_energy", "curv_l1", "mass_phi"]
    calibrator_lambda = float(calibrator_ridge_lambda)
    if calibrator_lambda < 0.0 or not math.isfinite(calibrator_lambda):
        raise AccuracyA2SelfConsistentError("calibrator_ridge_lambda must be >= 0 and finite")

    all_records: list[dict[str, object]] = []
    fold_rows: list[dict[str, object]] = []
    fold_weights: list[dict[str, object]] = []
    diagnostics_rows: list[dict[str, object]] = []

    for fold_id, test_gid in enumerate(fold_order):
        cfg_id = str(selected_cfg_by_test_gid[str(test_gid)])
        cfg = cfg_by_id[cfg_id]
        terms_by_id = eval_by_config[cfg_id]

        train_rows = [r for r in rows_sorted if r.gid != test_gid]
        test_rows = [r for r in rows_sorted if r.gid == test_gid]
        if not test_rows:
            raise AccuracyA2SelfConsistentError("empty test group")

        pred_raw_func: dict[str, float] = {r.mid: float(terms_by_id[r.mid]["E"]) for r in test_rows}
        min_func = min(float(pred_raw_func[r.mid]) for r in test_rows)

        X_train = np.asarray([[float(terms_by_id[r.mid][k]) for k in feature_names_cal] for r in train_rows], dtype=float)
        mean = np.mean(X_train, axis=0)
        std = np.std(X_train, axis=0)
        std = np.where(std == 0.0, 1.0, std)

        X_std_by_id: dict[str, np.ndarray] = {}
        for r in rows_sorted:
            x = np.asarray([float(terms_by_id[r.mid][k]) for k in feature_names_cal], dtype=float)
            X_std_by_id[r.mid] = (x - mean) / std

        X_pairs_list: list[np.ndarray] = []
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
                    X_pairs_list.append(diff)
                    y_deltas_list.append(float(delta))
                    X_pairs_list.append(-diff)
                    y_deltas_list.append(float(-delta))

        X_pairs = np.asarray(X_pairs_list, dtype=float)
        y_deltas = np.asarray(y_deltas_list, dtype=float)
        w = _fit_pairwise_rank_ridge(X_pairs, y_deltas, ridge_lambda=float(calibrator_lambda))

        fold_weights.append(
            {
                "fold_id": int(fold_id),
                "test_group_id": str(test_gid),
                "selected_config_id": str(cfg_id),
                "n_train_groups": int(len(train_by_group)),
                "train_groups": sorted(train_by_group.keys()),
                "n_train_pairs": int(X_pairs.shape[0]),
                "standardization_mean": [float(x) for x in mean.tolist()],
                "standardization_std": [float(x) for x in std.tolist()],
                "weights": {feature_names_cal[i]: float(w[i]) for i in range(len(feature_names_cal))},
            }
        )

        pred_raw_cal: dict[str, float] = {}
        for r in test_rows:
            pred_raw_cal[r.mid] = float(np.dot(w, X_std_by_id[r.mid]))
        min_cal = min(float(pred_raw_cal[r.mid]) for r in test_rows)

        for r in test_rows:
            rec: dict[str, object] = {
                "fold_id": int(fold_id),
                "selected_config_id": str(cfg_id),
                "id": str(r.mid),
                "group_id": str(r.gid),
                "smiles": str(r.smiles),
                "truth_rel_kcalmol": float(truth_rel_by_id[r.mid]),
                "pred_raw": float(pred_raw_func[r.mid]),
                "pred_rel": float(float(pred_raw_func[r.mid]) - float(min_func)),
                "pred_raw_calibrated_linear": float(pred_raw_cal[r.mid]),
                "pred_rel_calibrated_linear": float(float(pred_raw_cal[r.mid]) - float(min_cal)),
            }
            if mode == "a2_4":
                rec["rho_mode"] = str(cfg.rho_mode)
            all_records.append(rec)

            t = terms_by_id[r.mid]
            diag_rec: dict[str, object] = {
                "fold_id": int(fold_id),
                "selected_config_id": str(cfg_id),
                "id": str(r.mid),
                "group_id": str(r.gid),
                "smiles": str(r.smiles),
                "n_heavy_atoms": int(r.n_heavy_atoms),
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
                "rho_min": float(t["rho_min"]),
                "rho_mean": float(t["rho_mean"]),
                "phi_min": float(t["phi_min"]),
                "phi_max": float(t["phi_max"]),
                "phi_std": float(t["phi_std"]),
                "curv_l1": float(t["curv_l1"]),
                "curv_l2": float(t["curv_l2"]),
                "curv_maxabs": float(t["curv_maxabs"]),
                "term_grad": float(t["term_grad"]),
                "term_curv": float(t["term_curv"]),
                "term_mass": float(t["term_mass"]),
                "E": float(t["E"]),
                "sc_residual_final": float(t["sc_residual_final"]),
                "sc_converged": bool(t["sc_converged"]),
                "sc_rho_floor_rate_max": float(t.get("sc_rho_floor_rate_max", 0.0)),
                "sc_rho_floor_rate_final": float(t.get("sc_rho_floor_rate_final", 0.0)),
                "sc_accepted_backtracks_mean": float(t.get("sc_accepted_backtracks_mean", 0.0)),
                "sc_accepted_backtracks_max": int(t.get("sc_accepted_backtracks_max", 0) or 0),
                "sc_E_trace": str(t.get("sc_E_trace", "")),
            }
            if mode == "a2_3":
                diag_rec.update(
                    {
                        "rho_trace_norm": float(t.get("rho_trace_norm", float("nan"))),
                        "rho_max": float(t.get("rho_max", float("nan"))),
                        "trace_heat": float(t.get("trace_heat", float("nan"))),
                        "lambda_min": float(t.get("lambda_min", float("nan"))),
                        "lambda_max": float(t.get("lambda_max", float("nan"))),
                    }
                )
            elif mode == "a2_4":
                diag_rec.update(
                    {
                        "rho_mode": str(cfg.rho_mode),
                        "rho_ldos_beta": float(cfg.rho_ldos_beta),
                        "rho_ldos_k": int(cfg.rho_ldos_k),
                        "rho_ldos_deg_tol": float(cfg.rho_ldos_deg_tol),
                        "rho_max": float(t.get("rho_max", float("nan"))),
                        "rho_trace_norm": float(t.get("rho_trace_norm", float("nan"))),
                        "rho_sum": float(t.get("rho_sum", float("nan"))),
                        "rho_renorm_applied": bool(t.get("rho_renorm_applied", False)),
                        "rho_renorm_delta": float(t.get("rho_renorm_delta", float("nan"))),
                        "rho_entropy": float(t.get("rho_entropy", float("nan"))),
                        "trace_weights": float(t.get("trace_weights", float("nan"))),
                        "lambda_min": float(t.get("lambda_min", float("nan"))),
                        "lambda_max": float(t.get("lambda_max", float("nan"))),
                        "lambda_min_window": float(t.get("lambda_min_window", float("nan"))),
                        "lambda_gap": float(t.get("lambda_gap", float("nan"))),
                        "rho_ldos_k_eff": int(t.get("rho_ldos_k_eff", 0) or 0),
                        "rho_ldos_degeneracy_guard_applied": bool(t.get("rho_ldos_degeneracy_guard_applied", False)),
                    }
                )
            diagnostics_rows.append(diag_rec)

        test_records = [rr for rr in all_records if str(rr.get("group_id")) == str(test_gid)]
        group_metrics_fold_func = _compute_group_metrics(test_records)
        gm_func = group_metrics_fold_func.get(str(test_gid))
        if gm_func is None:
            raise AccuracyA2SelfConsistentError("missing fold group metrics (functional_only)")

        group_metrics_fold_cal = _compute_group_metrics_for_pred_rel(test_records, pred_rel_key="pred_rel_calibrated_linear")
        gm_cal = group_metrics_fold_cal.get(str(test_gid))
        if gm_cal is None:
            raise AccuracyA2SelfConsistentError("missing fold group metrics (calibrated_linear)")

        sel = fold_selection_by_id.get(int(fold_id))
        if sel is None:
            raise AccuracyA2SelfConsistentError("missing fold selection record")

        fold_rows.append(
            {
                "fold_id": int(fold_id),
                "test_group_id": str(test_gid),
                "selected_config_id": str(cfg_id),
                "train_num_groups_spearman_negative": int(sel.get("train_num_groups_spearman_negative") or 0),
                "train_median_spearman_by_group": float(sel.get("train_median_spearman_by_group") or float("nan")),
                "train_mean_spearman_by_group": float(sel.get("train_mean_spearman_by_group") or float("nan")),
                "n": int(gm_func.get("n") or 0),
                "spearman_pred_vs_truth": float(gm_func.get("spearman_pred_vs_truth") or float("nan")),
                "pairwise_order_accuracy": float(gm_func.get("pairwise_order_accuracy") or float("nan")),
                "pairwise_correct": int(gm_func.get("pairwise_correct") or 0),
                "pairwise_total": int(gm_func.get("pairwise_total") or 0),
                "top1_accuracy": float(gm_func.get("top1_accuracy") or float("nan")),
                "pred_best_id": str(gm_func.get("pred_best_id") or ""),
                "truth_best_ids": gm_func.get("truth_best_ids") or [],
                "spearman_pred_vs_truth_calibrated_linear": float(gm_cal.get("spearman_pred_vs_truth") or float("nan")),
                "pairwise_order_accuracy_calibrated_linear": float(gm_cal.get("pairwise_order_accuracy") or float("nan")),
                "top1_accuracy_calibrated_linear": float(gm_cal.get("top1_accuracy") or float("nan")),
            }
        )

    group_metrics_func = _compute_group_metrics(all_records)
    group_metrics_cal = _compute_group_metrics_for_pred_rel(all_records, pred_rel_key="pred_rel_calibrated_linear")
    metrics_test_func = _aggregate_metrics_from_group_metrics(group_metrics_func)
    metrics_test_cal = _aggregate_metrics_from_group_metrics(group_metrics_cal)

    negative_spearman_groups_test = sorted(
        [
            str(gid)
            for gid, gm in group_metrics_func.items()
            if not math.isfinite(float(gm.get("spearman_pred_vs_truth", float("nan"))))
            or float(gm.get("spearman_pred_vs_truth", float("nan"))) < 0.0
        ]
    )

    worst_groups = sorted(
        [
            {
                "group_id": str(gid),
                "spearman_pred_vs_truth": float(gm.get("spearman_pred_vs_truth", float("nan"))),
                "pairwise_order_accuracy": float(gm.get("pairwise_order_accuracy", float("nan"))),
                "top1_accuracy": float(gm.get("top1_accuracy", float("nan"))),
            }
            for gid, gm in group_metrics_func.items()
        ],
        key=lambda x: (
            _finite_or(float(x["spearman_pred_vs_truth"]), 0.0),
            _finite_or(float(x["top1_accuracy"]), 0.0),
            _finite_or(float(x["pairwise_order_accuracy"]), 0.0),
        ),
    )[:3]

    worst_groups_cal = sorted(
        [
            {
                "group_id": str(gid),
                "spearman_pred_vs_truth": float(gm.get("spearman_pred_vs_truth", float("nan"))),
                "pairwise_order_accuracy": float(gm.get("pairwise_order_accuracy", float("nan"))),
                "top1_accuracy": float(gm.get("top1_accuracy", float("nan"))),
            }
            for gid, gm in group_metrics_cal.items()
        ],
        key=lambda x: (
            _finite_or(float(x["spearman_pred_vs_truth"]), 0.0),
            _finite_or(float(x["top1_accuracy"]), 0.0),
            _finite_or(float(x["pairwise_order_accuracy"]), 0.0),
        ),
    )[:3]

    kpi_payload: dict[str, object] = {
        "mean_spearman_by_group_test_min": float(kpi_mean_spearman_by_group_test_min),
        "pairwise_order_accuracy_overall_test_min": float(kpi_pairwise_order_accuracy_overall_test_min),
        "median_spearman_by_group_test_min": float(kpi_median_spearman_by_group_test_min),
        "top1_accuracy_mean_test_min": float(kpi_top1_accuracy_mean_test_min),
        "mean_spearman_by_group_test": float(metrics_test_func["mean_spearman_by_group"]),
        "pairwise_order_accuracy_overall_test": float(metrics_test_func["pairwise_order_accuracy_overall"]),
        "median_spearman_by_group_test": float(metrics_test_func["median_spearman_by_group"]),
        "top1_accuracy_mean_test": float(metrics_test_func["top1_accuracy_mean"]),
        "num_groups_spearman_negative_test": int(metrics_test_func["num_groups_spearman_negative"]),
        "num_groups_spearman_negative_test_max": 0,
    }

    ok = (
        math.isfinite(float(kpi_payload["mean_spearman_by_group_test"]))
        and float(kpi_payload["mean_spearman_by_group_test"]) >= float(kpi_payload["mean_spearman_by_group_test_min"])
        and math.isfinite(float(kpi_payload["median_spearman_by_group_test"]))
        and float(kpi_payload["median_spearman_by_group_test"]) >= float(kpi_payload["median_spearman_by_group_test_min"])
        and math.isfinite(float(kpi_payload["pairwise_order_accuracy_overall_test"]))
        and float(kpi_payload["pairwise_order_accuracy_overall_test"]) >= float(kpi_payload["pairwise_order_accuracy_overall_test_min"])
        and math.isfinite(float(kpi_payload["top1_accuracy_mean_test"]))
        and float(kpi_payload["top1_accuracy_mean_test"]) >= float(kpi_payload["top1_accuracy_mean_test_min"])
        and int(kpi_payload["num_groups_spearman_negative_test"]) <= int(kpi_payload["num_groups_spearman_negative_test_max"])
    )
    kpi_payload["verdict"] = "PASS" if ok else "FAIL"
    kpi_payload["reason"] = (
        f"mean_spearman_by_group_test={kpi_payload['mean_spearman_by_group_test']}, "
        f"median_spearman_by_group_test={kpi_payload['median_spearman_by_group_test']}, "
        f"pairwise_order_accuracy_overall_test={kpi_payload['pairwise_order_accuracy_overall_test']}, "
        f"top1_accuracy_mean_test={kpi_payload['top1_accuracy_mean_test']}, "
        f"num_groups_spearman_negative_test={kpi_payload['num_groups_spearman_negative_test']}"
    )

    out_dir.mkdir(parents=True, exist_ok=True)

    search_results_path = out_dir / "search_results.csv"
    with search_results_path.open("w", encoding="utf-8", newline="") as f:
        fieldnames = list(search_rows[0].keys()) if search_rows else []
        w = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
        w.writeheader()
        for rec in search_rows:
            w.writerow({k: rec.get(k, "") for k in fieldnames})

    predictions_path = out_dir / "predictions.csv"
    with predictions_path.open("w", encoding="utf-8", newline="") as f:
        fieldnames = [
            "fold_id",
            "selected_config_id",
            "id",
            "group_id",
            "smiles",
            "truth_rel_kcalmol",
            "pred_raw",
            "pred_rel",
            "pred_raw_calibrated_linear",
            "pred_rel_calibrated_linear",
        ]
        if mode == "a2_4":
            fieldnames.insert(2, "rho_mode")
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
            "selected_config_id",
            "train_num_groups_spearman_negative",
            "train_median_spearman_by_group",
            "train_mean_spearman_by_group",
            "n",
            "spearman_pred_vs_truth",
            "pairwise_order_accuracy",
            "pairwise_correct",
            "pairwise_total",
            "top1_accuracy",
            "pred_best_id",
            "truth_best_ids",
            "spearman_pred_vs_truth_calibrated_linear",
            "pairwise_order_accuracy_calibrated_linear",
            "top1_accuracy_calibrated_linear",
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
            "spearman_pred_vs_truth_calibrated_linear",
            "pairwise_order_accuracy_calibrated_linear",
            "top1_accuracy_calibrated_linear",
        ]
        w = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
        w.writeheader()
        for gid, gm in sorted(group_metrics_func.items(), key=lambda x: str(x[0])):
            row = {k: gm.get(k, "") for k in fieldnames}
            row["truth_best_ids"] = json.dumps(list(gm.get("truth_best_ids") or []), ensure_ascii=False)
            gm_cal = group_metrics_cal.get(str(gid), {})
            row["spearman_pred_vs_truth_calibrated_linear"] = gm_cal.get("spearman_pred_vs_truth", "")
            row["pairwise_order_accuracy_calibrated_linear"] = gm_cal.get("pairwise_order_accuracy", "")
            row["top1_accuracy_calibrated_linear"] = gm_cal.get("top1_accuracy", "")
            w.writerow(row)

    diagnostics_path = out_dir / "diagnostics.csv"
    with diagnostics_path.open("w", encoding="utf-8", newline="") as f:
        fieldnames = list(diagnostics_rows[0].keys()) if diagnostics_rows else []
        w = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
        w.writeheader()
        for rec in sorted(diagnostics_rows, key=lambda rr: (int(rr.get("fold_id") or 0), str(rr.get("group_id") or ""), str(rr.get("id") or ""))):
            w.writerow({k: rec.get(k, "") for k in fieldnames})

    if mode == "a2_4":
        rho_compare_path = out_dir / "rho_compare.csv"
        with rho_compare_path.open("w", encoding="utf-8", newline="") as f:
            fieldnames = [
                "fold_id",
                "selected_config_id",
                "rho_mode",
                "rho_ldos_beta",
                "rho_ldos_k",
                "rho_ldos_deg_tol",
                "rho_ldos_k_eff",
                "rho_ldos_degeneracy_guard_applied",
                "id",
                "group_id",
                "smiles",
                "n_heavy_atoms",
                "rho_min",
                "rho_mean",
                "rho_max",
                "rho_entropy",
                "rho_trace_norm",
                "rho_sum",
                "rho_renorm_applied",
                "rho_renorm_delta",
                "sc_rho_floor_rate_final",
                "sc_rho_floor_rate_max",
                "trace_weights",
                "lambda_min",
                "lambda_max",
                "lambda_min_window",
                "lambda_gap",
                "heat_tau",
                "phi_eps",
                "rho_floor",
                "sc_iters",
                "sc_eta",
                "sc_damping",
                "sc_clip",
                "sc_max_backtracks",
            ]
            w = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
            w.writeheader()
            for rec in sorted(
                diagnostics_rows,
                key=lambda rr: (
                    int(rr.get("fold_id") or 0),
                    str(rr.get("group_id") or ""),
                    str(rr.get("id") or ""),
                ),
            ):
                w.writerow({k: rec.get(k, "") for k in fieldnames})

    if mode == "a2_3":
        schema_version = "accuracy_a1_isomers_a2_3.v1"
        a2_variant_label = "full_functional_v1_heat_kernel_a2_3"
    elif mode == "a2_2":
        schema_version = "accuracy_a1_isomers_a2_2.v1"
        a2_variant_label = "full_functional_v1_heat_kernel_a2_2"
    elif mode == "a2_4":
        schema_version = "accuracy_a1_isomers_a2_4.v1"
        a2_variant_label = "full_functional_v1_heat_kernel_a2_4"
    else:
        schema_version = "accuracy_a1_isomers_a2_1.v1"
        a2_variant_label = "full_functional_v1_heat_kernel"
    best_cfg: dict[str, object] = {
        "a2_variant": str(a2_variant_label),
        "operator_fixed": {
            "potential_gamma": float(potential_gamma),
            "potential_variant": str(potential_variant),
            "v_deg_coeff": float(v_deg_coeff),
            "v_valence_coeff": float(v_valence_coeff),
            "v_arom_coeff": float(v_arom_coeff),
            "v_ring_coeff": float(v_ring_coeff),
            "v_charge_coeff": float(v_charge_coeff),
            "v_chi_coeff": float(v_chi_coeff),
            "edge_weight_mode": str(edge_weight_mode),
            "edge_aromatic_mult": float(edge_aromatic_mult),
            "edge_delta_chi_alpha": float(edge_delta_chi_alpha),
        },
        "search": {
            "space_size": int(len(configs)),
            "budget": int(len(configs)),
            "chosen_by_train_only": True,
            "train_selection_metric_primary": "num_negative_spearman",
            "train_selection_metric_secondary": "median_spearman",
            "results_csv": "search_results.csv",
        },
        "candidates": [
            {
                "config_id": str(c.config_id),
                "heat_tau": float(c.heat_tau),
                "phi_eps": float(c.phi_eps),
                "rho_floor": float(c.rho_floor),
                "sc_iters": int(c.sc_iters),
                "sc_eta": float(c.sc_eta),
                "sc_damping": float(c.sc_damping),
                "sc_clip": float(c.sc_clip),
                "sc_max_backtracks": int(c.sc_max_backtracks),
                "gauge_fix_phi_mean": bool(c.gauge_fix_phi_mean),
                "coef_alpha": float(c.coef_alpha),
                "coef_beta": float(c.coef_beta),
                "coef_gamma": float(c.coef_gamma),
                "mass_mode": str(c.mass_mode),
                "rho_mode": str(c.rho_mode),
                "rho_ldos_beta": float(c.rho_ldos_beta),
                "rho_ldos_k": int(c.rho_ldos_k),
                "rho_ldos_deg_tol": float(c.rho_ldos_deg_tol),
            }
            for c in configs
        ],
        "fold_selection": list(fold_selection),
        "calibrator": {
            "mode": "calibrated_linear_pairwise_rank_ridge",
            "feature_names": list(feature_names_cal),
            "ridge_lambda": float(calibrator_lambda),
            "fold_weights": list(fold_weights),
        },
        "cv": {
            "seed": int(seed),
            "fold_order": list(fold_order),
        },
    }
    (out_dir / "best_config.json").write_text(json.dumps(best_cfg, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")

    files_payload: dict[str, str] = {
        "summary_csv": "summary.csv",
        "predictions_csv": "predictions.csv",
        "fold_metrics_csv": "fold_metrics.csv",
        "group_metrics_csv": "group_metrics.csv",
        "diagnostics_csv": "diagnostics.csv",
        "metrics_json": "metrics.json",
        "best_config_json": "best_config.json",
        "provenance_json": "provenance.json",
        "manifest_json": "manifest.json",
        "checksums_sha256": "checksums.sha256",
        "index_md": "index.md",
        "search_results_csv": "search_results.csv",
    }
    if mode == "a2_4":
        files_payload["rho_compare_csv"] = "rho_compare.csv"

    rho_mode_selected: dict[str, object] | None = None
    if mode == "a2_4":
        by_fold: list[dict[str, object]] = []
        dist: dict[str, int] = {}
        for rec in fold_selection:
            cfg_id = str(rec.get("selected_config_id") or "")
            cfg_sel = cfg_by_id.get(cfg_id)
            rho_mode_val = str(cfg_sel.rho_mode) if cfg_sel is not None else ""
            by_fold.append(
                {
                    "fold_id": int(rec.get("fold_id") or 0),
                    "test_group_id": str(rec.get("test_group_id") or ""),
                    "selected_config_id": str(cfg_id),
                    "rho_mode": str(rho_mode_val),
                }
            )
            if rho_mode_val:
                dist[rho_mode_val] = int(dist.get(rho_mode_val, 0)) + 1
        rho_mode_selected = {"by_fold": by_fold, "distribution": dist}

    metrics_payload: dict[str, object] = {
        "schema_version": str(schema_version),
        "experiment_id": str(experiment_id),
        "dataset": {
            "rows_total": int(len(rows_sorted)),
            "groups_total": int(len(group_ids)),
        },
        "input_csv": input_csv.as_posix(),
        "input_sha256_normalized": _sha256_text_normalized(input_csv),
        "best_config": best_cfg,
        "metrics_loocv_test_functional_only": dict(metrics_test_func),
        "metrics_loocv_test_calibrated_linear": dict(metrics_test_cal),
        "metrics_loocv_test_functional_only_by_rho_mode": dict(metrics_test_func_by_rho_mode)
        if metrics_test_func_by_rho_mode is not None
        else None,
        "kpi": kpi_payload,
        "negative_spearman_groups_test": list(negative_spearman_groups_test),
        "negative_spearman_groups_test_by_rho_mode": dict(negative_spearman_groups_test_by_rho_mode)
        if negative_spearman_groups_test_by_rho_mode is not None
        else None,
        "worst_groups": worst_groups,
        "worst_groups_calibrated_linear": worst_groups_cal,
        "rho_mode_selected": dict(rho_mode_selected) if rho_mode_selected is not None else None,
        "files": dict(files_payload),
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics_payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")

    if mode == "a2_4":
        title = "A2.4 full functional (soft-occupancy LDOS rho-law + variational SCF)"
    elif mode == "a2_3":
        title = "A2.3 full functional (trace-normalized heat diag + variational SCF)"
    elif mode == "a2_2":
        title = "A2.2 variationally-stable SCF"
    else:
        title = "A2.1 full functional (heat-kernel diag + SCF)"
    index_lines = [
        f"# {experiment_id} (Isomers) {title}",
        "",
        "LOOCV (by group_id) metrics:",
        "",
        "Functional-only:",
        f"- mean_spearman_by_group: {metrics_test_func.get('mean_spearman_by_group')}",
        f"- median_spearman_by_group: {metrics_test_func.get('median_spearman_by_group')}",
        f"- pairwise_order_accuracy_overall: {metrics_test_func.get('pairwise_order_accuracy_overall')} ({metrics_test_func.get('pairwise_correct')}/{metrics_test_func.get('pairwise_total')})",
        f"- top1_accuracy_mean: {metrics_test_func.get('top1_accuracy_mean')}",
        f"- num_groups_spearman_negative: {metrics_test_func.get('num_groups_spearman_negative')}",
        "",
        "Calibrated-linear (pairwise rank ridge):",
        f"- mean_spearman_by_group: {metrics_test_cal.get('mean_spearman_by_group')}",
        f"- median_spearman_by_group: {metrics_test_cal.get('median_spearman_by_group')}",
        f"- pairwise_order_accuracy_overall: {metrics_test_cal.get('pairwise_order_accuracy_overall')} ({metrics_test_cal.get('pairwise_correct')}/{metrics_test_cal.get('pairwise_total')})",
        f"- top1_accuracy_mean: {metrics_test_cal.get('top1_accuracy_mean')}",
        f"- num_groups_spearman_negative: {metrics_test_cal.get('num_groups_spearman_negative')}",
        "",
        "KPI (functional-only):",
        f"- verdict: {kpi_payload.get('verdict')}",
        f"- reason: {kpi_payload.get('reason')}",
        "",
        "Worst groups (functional-only):",
        "```json",
        json.dumps(worst_groups, ensure_ascii=False, sort_keys=True, indent=2),
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
        "search_space_size": int(len(configs)),
        "budget": int(len(configs)),
        "chosen_by_train_only": True,
        "train_selection_metric_primary": "num_negative_spearman",
        "train_selection_metric_secondary": "median_spearman",
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
    p.add_argument(
        "--a2_variant",
        type=str,
        default="self_consistent_v1",
        choices=[
            "self_consistent_v1",
            "full_functional_v1",
            "full_functional_v1_a2_2",
            "full_functional_v1_a2_3",
            "full_functional_v1_a2_4",
        ],
        help=(
            "A2 runner variant: legacy self-consistent functional (v1), "
            "A2.1 full functional (heat-kernel diag + SCF), or "
            "A2.2 full functional with variationally-stable SCF (monotone backtracking), or "
            "A2.3 same SCF discipline with rho-law trace-normalized heat diag, or "
            "A2.4 same SCF discipline with rho-law soft-occupancy LDOS (baseline+new rho_mode)."
        ),
    )
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

    p.add_argument("--calibrator_ridge_lambda", type=float, default=1e-3, help="Ridge lambda for calibrated_linear mode (A2.1).")

    p.add_argument("--kpi_mean_spearman_by_group_test_min", type=float, default=0.55, help="KPI gate (LOOCV test): mean spearman by group >= value.")
    p.add_argument("--kpi_median_spearman_by_group_test_min", type=float, default=0.55, help="KPI gate (LOOCV test): median spearman by group >= value.")
    p.add_argument("--kpi_pairwise_order_accuracy_overall_test_min", type=float, default=0.70, help="KPI gate (LOOCV test): overall pairwise order accuracy >= value.")
    p.add_argument("--kpi_top1_accuracy_mean_test_min", type=float, default=0.40, help="KPI gate (LOOCV test): top1 mean accuracy >= value.")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    variant = str(args.a2_variant)
    if variant == "full_functional_v1":
        run_accuracy_a1_isomers_a2_full_functional_v1(
            experiment_id=str(args.experiment_id),
            input_csv=Path(args.input_csv),
            out_dir=Path(args.out_dir),
            seed=int(args.seed),
            potential_gamma=float(args.gamma),
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
            calibrator_ridge_lambda=float(args.calibrator_ridge_lambda),
            kpi_mean_spearman_by_group_test_min=float(args.kpi_mean_spearman_by_group_test_min),
            kpi_pairwise_order_accuracy_overall_test_min=float(args.kpi_pairwise_order_accuracy_overall_test_min),
            kpi_median_spearman_by_group_test_min=float(args.kpi_median_spearman_by_group_test_min),
            kpi_top1_accuracy_mean_test_min=float(args.kpi_top1_accuracy_mean_test_min),
        )
    elif variant == "full_functional_v1_a2_2":
        run_accuracy_a1_isomers_a2_full_functional_v1(
            experiment_id=str(args.experiment_id),
            input_csv=Path(args.input_csv),
            out_dir=Path(args.out_dir),
            seed=int(args.seed),
            potential_gamma=float(args.gamma),
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
            calibrator_ridge_lambda=float(args.calibrator_ridge_lambda),
            kpi_mean_spearman_by_group_test_min=float(args.kpi_mean_spearman_by_group_test_min),
            kpi_pairwise_order_accuracy_overall_test_min=float(args.kpi_pairwise_order_accuracy_overall_test_min),
            kpi_median_spearman_by_group_test_min=float(args.kpi_median_spearman_by_group_test_min),
            kpi_top1_accuracy_mean_test_min=float(args.kpi_top1_accuracy_mean_test_min),
            full_functional_mode="a2_2",
        )
    elif variant == "full_functional_v1_a2_3":
        run_accuracy_a1_isomers_a2_full_functional_v1(
            experiment_id=str(args.experiment_id),
            input_csv=Path(args.input_csv),
            out_dir=Path(args.out_dir),
            seed=int(args.seed),
            potential_gamma=float(args.gamma),
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
            calibrator_ridge_lambda=float(args.calibrator_ridge_lambda),
            kpi_mean_spearman_by_group_test_min=float(args.kpi_mean_spearman_by_group_test_min),
            kpi_pairwise_order_accuracy_overall_test_min=float(args.kpi_pairwise_order_accuracy_overall_test_min),
            kpi_median_spearman_by_group_test_min=float(args.kpi_median_spearman_by_group_test_min),
            kpi_top1_accuracy_mean_test_min=float(args.kpi_top1_accuracy_mean_test_min),
            full_functional_mode="a2_3",
        )
    elif variant == "full_functional_v1_a2_4":
        run_accuracy_a1_isomers_a2_full_functional_v1(
            experiment_id=str(args.experiment_id),
            input_csv=Path(args.input_csv),
            out_dir=Path(args.out_dir),
            seed=int(args.seed),
            potential_gamma=float(args.gamma),
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
            calibrator_ridge_lambda=float(args.calibrator_ridge_lambda),
            kpi_mean_spearman_by_group_test_min=float(args.kpi_mean_spearman_by_group_test_min),
            kpi_pairwise_order_accuracy_overall_test_min=float(args.kpi_pairwise_order_accuracy_overall_test_min),
            kpi_median_spearman_by_group_test_min=float(args.kpi_median_spearman_by_group_test_min),
            kpi_top1_accuracy_mean_test_min=float(args.kpi_top1_accuracy_mean_test_min),
            full_functional_mode="a2_4",
        )
    elif variant == "self_consistent_v1":
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
    else:
        raise AccuracyA2SelfConsistentError(f"invalid a2_variant: {variant}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
