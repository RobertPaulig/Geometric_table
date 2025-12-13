from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import math
from typing import Callable, Tuple

import numpy as np

from core.nuclear_spectrum_ws import build_radial_hamiltonian_ws
from core.thermo_config import get_current_thermo_config
from core.ws_param_scaling import apply_ws_Z_scaling, WSRadialParams


@dataclass(frozen=True)
class WSRadialDiagnostics:
    r: np.ndarray
    u: np.ndarray
    cdf: np.ndarray
    r_mean: float
    r_rms: float
    r_99: float


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


def _compute_diagnostics(r: np.ndarray, u: np.ndarray, dr: float) -> WSRadialDiagnostics:
    pdf = (u * u)
    cdf = np.cumsum(pdf * dr)
    total = float(cdf[-1]) if cdf.size > 0 else 1.0
    if total <= 0.0:
        cdf_norm = cdf
    else:
        cdf_norm = cdf / total

    r_mean = float(np.sum(r * pdf * dr))
    r_rms = math.sqrt(float(np.sum((r * r) * pdf * dr)))

    r_99 = float(r[-1])
    if cdf_norm.size > 0:
        idx = np.searchsorted(cdf_norm, 0.99)
        idx = min(idx, len(r) - 1)
        r_99 = float(r[idx])

    return WSRadialDiagnostics(r=r, u=u, cdf=cdf_norm, r_mean=r_mean, r_rms=r_rms, r_99=r_99)


@lru_cache(maxsize=256)
def make_ws_rho3d_with_diagnostics(Z: int, params: WSRadialParams) -> Tuple[Callable[[np.ndarray], np.ndarray], WSRadialDiagnostics]:
    """
    Построить интерполятор ρ_3d(r) из WS-спектра для данного Z и параметров и вернуть
    диагностическую информацию по радиальной волновой функции.
    """
    # Радиальная сетка и шаг (с учётом Z-скейлинга, если включён)
    thermo = get_current_thermo_config()
    params_eff = apply_ws_Z_scaling(
        params,
        Z=int(Z),
        coupling_ws_Z=getattr(thermo, "coupling_ws_Z", 0.0),
        Z_ref=getattr(thermo, "ws_Z_ref", 10.0),
        alpha=getattr(thermo, "ws_Z_alpha", 1.0 / 3.0),
    )

    r, dr = _radial_grid(params_eff.R_max, params_eff.N_grid)

    # Построение гамильтониана WS (радиальная задача)
    evals, evecs = build_radial_hamiltonian_ws(
        R_max=params_eff.R_max,
        R0=params_eff.R_well,
        a=0.7,
        V0=params_eff.V0,
        N_grid=params_eff.N_grid,
        ell=params_eff.ell,
    )

    # Берём bound-состояния по энергии < 0 (если есть), иначе просто state_index
    idx = int(params_eff.state_index)
    bound_idx = np.where(evals < 0.0)[0]
    if bound_idx.size > 0:
        idx = int(bound_idx[min(idx, bound_idx.size - 1)])

    u = np.array(evecs[:, idx], dtype=float)
    u = _normalize_u(u, dr)
    rho3d = _rho3d_from_u(u, r)

    diag = _compute_diagnostics(r, u, dr)

    # возвращаем callable, принимающий radii (np.ndarray)
    def rho_fn(radii: np.ndarray) -> np.ndarray:
        x = np.asarray(radii, dtype=float)
        x = np.clip(x, 0.0, params_eff.R_max)
        # r начинается с dr, поэтому для r≈0 используем rho(r=dr)
        x_safe = np.maximum(x, r[0])
        return np.interp(x_safe, r, rho3d, left=rho3d[0], right=0.0)

    return rho_fn, diag


@lru_cache(maxsize=256)
def make_ws_rho3d_interpolator(Z: int, params: WSRadialParams) -> Callable[[np.ndarray], np.ndarray]:
    """
    Обёртка над make_ws_rho3d_with_diagnostics, сохраняющая старый интерфейс.
    """
    rho_fn, _ = make_ws_rho3d_with_diagnostics(Z, params)
    return rho_fn
