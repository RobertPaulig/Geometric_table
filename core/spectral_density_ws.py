from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import math
from typing import Callable

import numpy as np

from core.nuclear_spectrum_ws import build_radial_hamiltonian_ws


@dataclass(frozen=True)
class WSRadialParams:
    R_max: float = 12.0
    R_well: float = 5.0
    V0: float = 40.0
    N_grid: int = 220
    ell: int = 0
    state_index: int = 0


def _radial_grid(R_max: float, N_grid: int) -> tuple[np.ndarray, float]:
    dr = float(R_max) / float(N_grid)
    # Важно: r начинается с dr, чтобы избегать r=0
    r = np.linspace(dr, float(R_max), int(N_grid), dtype=float)
    return r, dr


def _normalize_u(u: np.ndarray, dr: float) -> np.ndarray:
    # нормировка ∫|u|^2 dr = 1
    norm = math.sqrt(float(np.sum((u * u) * dr)))
    if norm <= 0.0:
        return u
    return u / norm


def _rho3d_from_u(u: np.ndarray, r: np.ndarray) -> np.ndarray:
    # u(r) = r * R(r). Тогда ρ_3d(r) = |u|^2 / (4π r^2)
    denom = 4.0 * math.pi * np.maximum(r * r, 1e-12)
    rho = (u * u) / denom
    return rho


@lru_cache(maxsize=256)
def make_ws_rho3d_interpolator(Z: int, params: WSRadialParams) -> Callable[[np.ndarray], np.ndarray]:
    """
    Построить интерполятор ρ_3d(r) из WS-спектра для данного Z и параметров.

    Z здесь включён только для кэш-ключа; текущая реализация не использует его
    явно, но в будущем параметры потенциала могут зависеть от Z.
    """
    # Радиальная сетка и шаг
    r, dr = _radial_grid(params.R_max, params.N_grid)

    # Построение гамильтониана WS (радиальная задача)
    evals, evecs = build_radial_hamiltonian_ws(
        R_max=params.R_max,
        R0=params.R_well,
        a=0.7,
        V0=params.V0,
        N_grid=params.N_grid,
        ell=params.ell,
    )

    # Берём bound-состояния по энергии < 0 (если есть), иначе просто state_index
    idx = int(params.state_index)
    bound_idx = np.where(evals < 0.0)[0]
    if bound_idx.size > 0:
        idx = int(bound_idx[min(idx, bound_idx.size - 1)])

    u = np.array(evecs[:, idx], dtype=float)
    u = _normalize_u(u, dr)
    rho3d = _rho3d_from_u(u, r)

    # возвращаем callable, принимающий radii (np.ndarray)
    def rho_fn(radii: np.ndarray) -> np.ndarray:
        x = np.asarray(radii, dtype=float)
        x = np.clip(x, 0.0, params.R_max)
        # r начинается с dr, поэтому для r≈0 используем rho(r=dr)
        x_safe = np.maximum(x, r[0])
        return np.interp(x_safe, r, rho3d, left=rho3d[0], right=0.0)

    return rho_fn

