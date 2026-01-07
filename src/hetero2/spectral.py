from __future__ import annotations

from typing import Dict, Iterable, List

import numpy as np


def laplacian_eigvals(laplacian: np.ndarray) -> np.ndarray:
    eigvals = np.linalg.eigvalsh(laplacian)
    return np.sort(eigvals)


def compute_stability_metrics(
    eigvals: Iterable[float],
    *,
    eps: float = 1e-9,
    round_decimals: int = 8,
) -> Dict[str, float]:
    vals = np.array(list(eigvals), dtype=float)
    if vals.size == 0:
        return {"spectral_gap": float("nan"), "spectral_entropy": float("nan"), "spectral_entropy_norm": float("nan")}
    pos = vals[vals > eps]
    if pos.size == 0:
        return {"spectral_gap": float("nan"), "spectral_entropy": float("nan"), "spectral_entropy_norm": float("nan")}
    spectral_gap = float(pos.min())
    total = float(pos.sum())
    if total <= 0.0:
        entropy = float("nan")
        entropy_norm = float("nan")
    else:
        p = pos / total
        entropy = float(-np.sum(p * np.log(p)))
        if p.size > 1:
            entropy_norm = float(entropy / np.log(float(p.size)))
        else:
            entropy_norm = 0.0
    return {
        "spectral_gap": float(np.round(spectral_gap, round_decimals)),
        "spectral_entropy": float(np.round(entropy, round_decimals)),
        "spectral_entropy_norm": float(np.round(entropy_norm, round_decimals)),
    }


def spectral_fp_from_laplacian(
    laplacian: np.ndarray,
    *,
    k: int | None = None,
    round_decimals: int = 12,
) -> List[float]:
    eigvals = laplacian_eigvals(laplacian)
    if k is not None:
        eigvals = eigvals[: int(k)]
    return [float(x) for x in np.round(eigvals, round_decimals)]


def ldos_fp(
    eigvals: Iterable[float],
    *,
    grid_n: int = 64,
    gamma: float = 0.05,
    round_decimals: int = 12,
) -> List[float]:
    vals = np.array(list(eigvals), dtype=float)
    if vals.size == 0:
        return []
    vmin = float(vals.min())
    vmax = float(vals.max())
    if vmin == vmax:
        vmin -= 1.0
        vmax += 1.0
    grid = np.linspace(vmin, vmax, int(grid_n), dtype=float)
    gamma_sq = float(gamma) ** 2
    denom = np.pi * gamma
    ldos = []
    for x in grid:
        diff = (x - vals) ** 2 + gamma_sq
        ldos.append(float(np.sum(gamma_sq / (diff * denom))))
    return [float(x) for x in np.round(ldos, round_decimals)]
