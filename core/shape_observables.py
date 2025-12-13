from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Tuple

import math
import numpy as np

from core.thermo_config import ThermoConfig, get_current_thermo_config
from core.spectral_density_ws import WSRadialParams, make_ws_rho3d_with_diagnostics
from core.density_models import beta_effective
# Excess kurtosis for radial variable of 3D Gaussian (Maxwell distribution).
KURTOSIS_EXCESS_GAUSS_RADIAL = (
    4.0 * (-96.0 - 3.0 * math.pi ** 2 + 40.0 * math.pi)
    / (-48.0 * math.pi + 64.0 + 9.0 * math.pi ** 2)
)


def radial_kurtosis_excess_from_rho3d(r: np.ndarray, rho3d: np.ndarray) -> float:
    """
    rho3d(r): 3D density (expected normalized: ∫ rho dV = 1).
    Computes excess kurtosis of radial random variable r under p(r)=4π r^2 rho3d(r).
    """
    r = np.asarray(r, dtype=float)
    rho3d = np.asarray(rho3d, dtype=float)

    p = 4.0 * math.pi * (r ** 2) * rho3d
    norm = np.trapz(p, r)
    if not np.isfinite(norm) or norm <= 0:
        return float("nan")
    p = p / norm

    mu = np.trapz(r * p, r)
    var = np.trapz((r - mu) ** 2 * p, r)
    if not np.isfinite(var) or var <= 0:
        return float("nan")
    mu4 = np.trapz((r - mu) ** 4 * p, r)
    return float(mu4 / (var ** 2) - 3.0)


@dataclass(frozen=True)
class ShapeObs:
    r_rms_ws: float
    kurt_ws: float
    r_rms_g: float
    kurt_g: float
    delta_r: float
    delta_k: float


def _ws_params_from_thermo(cfg: ThermoConfig) -> WSRadialParams:
    return WSRadialParams(
        R_max=float(getattr(cfg, "ws_R_max", 12.0)),
        R_well=float(getattr(cfg, "ws_R_well", 5.0)),
        V0=float(getattr(cfg, "ws_V0", 40.0)),
        N_grid=int(getattr(cfg, "ws_N_grid", 220)),
        ell=int(getattr(cfg, "ws_ell", 0)),
        state_index=int(getattr(cfg, "ws_state_index", 0)),
    )


def _gaussian_r_rms(beta: float) -> float:
    beta = max(float(beta), 1e-15)
    return math.sqrt(3.0 / (2.0 * beta))


def thermo_fingerprint_for_shape(cfg: ThermoConfig) -> Tuple:
    """
    Строим компактный кортеж из параметров WS/WSZ, влияющих на форму.
    Используется как ключ для lru_cache, чтобы избежать утечек памяти.
    """
    return (
        float(cfg.ws_R_max),
        float(cfg.ws_R_well),
        float(cfg.ws_V0),
        int(cfg.ws_N_grid),
        int(cfg.ws_ell),
        int(cfg.ws_state_index),
        float(cfg.ws_Z_ref),
        float(cfg.ws_Z_alpha),
        float(cfg.coupling_ws_Z),
    )


@lru_cache(maxsize=256)
def get_shape_observables(Z: int, thermo_fp: Tuple) -> ShapeObs:
    """
    Возвращает shape-подпись атома по Z и термо-профилю (через fingerprint).
    Функция кэшируется по (Z, thermo_fp).
    """
    cfg = get_current_thermo_config()
    params = _ws_params_from_thermo(cfg)
    rho_fn, diag = make_ws_rho3d_with_diagnostics(int(Z), params)

    r_rms_ws = float(diag.r_rms)

    r_grid = np.linspace(0.0, float(params.R_max), 4096)
    rho_grid = rho_fn(r_grid)
    kurt_ws = float(radial_kurtosis_excess_from_rho3d(r_grid, rho_grid))

    beta = beta_effective(
        int(Z),
        getattr(cfg, "coupling_density", 0.0),
        model=getattr(cfg, "density_model", "tf_radius"),
        blend=getattr(cfg, "density_blend", "linear"),
        Z_ref=getattr(cfg, "density_Z_ref", 10.0),
    )
    r_rms_g = _gaussian_r_rms(beta)
    kurt_g = float(KURTOSIS_EXCESS_GAUSS_RADIAL)

    delta_r = r_rms_ws / max(r_rms_g, 1e-15)
    delta_k = kurt_ws - kurt_g

    return ShapeObs(
        r_rms_ws=r_rms_ws,
        kurt_ws=kurt_ws,
        r_rms_g=r_rms_g,
        kurt_g=kurt_g,
        delta_r=delta_r,
        delta_k=delta_k,
    )
