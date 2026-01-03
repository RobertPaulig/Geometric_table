from __future__ import annotations

from typing import Iterable, List

import numpy as np


def spectral_fp_from_laplacian(
    laplacian: np.ndarray,
    *,
    k: int | None = None,
    round_decimals: int = 12,
) -> List[float]:
    eigvals = np.linalg.eigvalsh(laplacian)
    eigvals = np.sort(eigvals)
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
