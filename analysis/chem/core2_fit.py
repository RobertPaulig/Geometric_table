from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np


def compute_p_pred(
    g: Mapping[str, int],
    e_ref: Mapping[str, float],
    *,
    T: float,
    lam: float,
) -> Dict[str, float]:
    """
    Compute degeneracy-aware predicted topology probabilities:
        P_pred(topo) ∝ g(topo) * exp(-lam * E_ref(topo) / T)

    Notes:
      - E_ref must be the *raw* reference energy/score (no coupling pre-applied).
      - If T<=0, this returns a delta mass on the minimum E_ref topology (ties split by g).
    """
    keys = [k for k in g.keys() if k in e_ref]
    if not keys:
        return {}

    if T <= 0.0:
        e_min = min(float(e_ref[k]) for k in keys)
        tied = [k for k in keys if float(e_ref[k]) == e_min]
        z = sum(float(g[k]) for k in tied) or 1.0
        return {k: (float(g[k]) / z if k in tied else 0.0) for k in keys}

    beta = float(lam) / float(T)
    # Stabilize exponent by subtracting min(beta*E).
    vals = np.asarray([beta * float(e_ref[k]) for k in keys], dtype=float)
    shift = float(np.min(vals))
    weights: Dict[str, float] = {}
    for k in keys:
        w = float(g[k]) * math.exp(-(beta * float(e_ref[k]) - shift))
        weights[k] = w
    z = sum(weights.values()) or 1.0
    return {k: (weights[k] / z) for k in keys}


def kl_divergence(
    p_obs: Mapping[str, float],
    p_pred: Mapping[str, float],
    *,
    eps: float = 1e-12,
) -> float:
    """
    KL(P_obs || P_pred) with small epsilon smoothing for numerical stability.
    """
    keys = set(p_obs.keys()) | set(p_pred.keys())
    kl = 0.0
    for k in keys:
        p = max(float(p_obs.get(k, 0.0)), 0.0)
        q = max(float(p_pred.get(k, 0.0)), 0.0)
        p = max(p, eps)
        q = max(q, eps)
        kl += p * math.log(p / q)
    return float(kl)


@dataclass(frozen=True)
class LambdaFitRow:
    lam: float
    kl: float
    log_ratio_pred: Dict[str, float]


@dataclass(frozen=True)
class LambdaFitResult:
    lam_star: float
    kl_min: float
    table: Tuple[LambdaFitRow, ...]


def _default_lam_grid() -> np.ndarray:
    return np.linspace(0.0, 10.0, 401, dtype=float)


def fit_lambda(
    p_obs: Mapping[str, float],
    g: Mapping[str, int],
    e_ref: Mapping[str, float],
    *,
    T: float,
    lam_grid: Sequence[float] | None = None,
) -> LambdaFitResult:
    """
    Grid-search fit for lambda that minimizes KL(P_obs || P_pred(lambda)).
    Returns lambda*, KL_min, and a table with per-lambda diagnostics.
    """
    if lam_grid is None:
        grid = _default_lam_grid()
    else:
        grid = np.asarray(list(lam_grid), dtype=float)

    rows: List[LambdaFitRow] = []
    best_lam = float(grid[0]) if grid.size else 0.0
    best_kl = float("inf")

    keys = [k for k in g.keys() if k in e_ref]
    if len(keys) < 2:
        return LambdaFitResult(lam_star=0.0, kl_min=0.0, table=tuple())

    ref_key = keys[0]
    for lam in grid:
        p_pred = compute_p_pred(g, e_ref, T=float(T), lam=float(lam))
        kl = kl_divergence(p_obs, p_pred)
        log_ratios: Dict[str, float] = {}
        for k in keys:
            if k == ref_key:
                continue
            pn = max(float(p_pred.get(ref_key, 0.0)), 1e-300)
            pk = max(float(p_pred.get(k, 0.0)), 1e-300)
            log_ratios[f"log(P({k})/P({ref_key}))"] = float(math.log(pk / pn))
        rows.append(LambdaFitRow(lam=float(lam), kl=float(kl), log_ratio_pred=log_ratios))
        if kl < best_kl:
            best_kl = float(kl)
            best_lam = float(lam)

    # Local refinement around best (optional, deterministic).
    if grid.size >= 3:
        step = float(np.min(np.diff(np.unique(grid))))
        if step > 0:
            fine = np.linspace(best_lam - step, best_lam + step, 201, dtype=float)
            for lam in fine:
                if lam < 0:
                    continue
                p_pred = compute_p_pred(g, e_ref, T=float(T), lam=float(lam))
                kl = kl_divergence(p_obs, p_pred)
                if kl < best_kl:
                    best_kl = float(kl)
                    best_lam = float(lam)

    return LambdaFitResult(lam_star=best_lam, kl_min=best_kl, table=tuple(rows))

