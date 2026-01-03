from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Tuple

import math
import numpy as np

_trapz = getattr(np, "trapezoid", np.trapz)

from core.thermo_config import ThermoConfig, get_current_thermo_config
from core.spectral_density_ws import WSRadialParams, make_ws_rho3d_with_diagnostics
from core.density_models import beta_effective
from core.fdm import FDMIntegrator, make_tensor_grid_ifs
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
    norm = _trapz(p, r)
    if not np.isfinite(norm) or norm <= 0:
        return float("nan")
    p = p / norm

    mu = _trapz(r * p, r)
    var = _trapz((r - mu) ** 2 * p, r)
    if not np.isfinite(var) or var <= 0:
        return float("nan")
    mu4 = _trapz((r - mu) ** 4 * p, r)
    return float(mu4 / (var ** 2) - 3.0)


@dataclass(frozen=True)
class ShapeObs:
    r_rms_ws: float
    kurt_ws: float
    r_rms_g: float
    kurt_g: float
    delta_r: float
    delta_k: float
    effective_volume_ws: float
    softness_integral_ws: float
    density_overlap_ws: float


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
        cfg.ws_integrator,
        int(cfg.ws_fdm_depth),
        int(cfg.ws_fdm_base),
        float(cfg.coupling_density),
        cfg.density_model,
        cfg.density_blend,
        float(cfg.density_Z_ref),
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

    beta = beta_effective(
        int(Z),
        getattr(cfg, "coupling_density", 0.0),
        model=getattr(cfg, "density_model", "tf_radius"),
        blend=getattr(cfg, "density_blend", "linear"),
        Z_ref=getattr(cfg, "density_Z_ref", 10.0),
    )

    def _gaussian_rho3d(r: np.ndarray) -> np.ndarray:
        r = np.asarray(r, dtype=float)
        beta_pos = max(float(beta), 1e-15)
        norm = (beta_pos / math.pi) ** 1.5
        return norm * np.exp(-beta_pos * (r ** 2))

    # Compute kurtosis via selected integrator
    R_max = float(params.R_max)
    method = getattr(cfg, "ws_integrator", "trapz")

    if method == "fdm":
        base = int(getattr(cfg, "ws_fdm_base", 2))
        depth = int(getattr(cfg, "ws_fdm_depth", 9))
        ifs = make_tensor_grid_ifs(dim=1, base=base)
        fdm = FDMIntegrator(ifs)

        def moments_func(u: np.ndarray) -> np.ndarray:
            # u shape: (N,1) in [0,1]; map to r in [0, R_max]
            r = (u[:, 0]) * R_max
            rho = rho_fn(r)
            p = 4.0 * math.pi * (r ** 2) * rho
            rho_g = _gaussian_rho3d(r)
            # raw moments m0..m4 for the integrand
            m0 = p
            m1 = r * p
            m2 = (r ** 2) * p
            m3 = (r ** 3) * p
            m4 = (r ** 4) * p
            # additional channels for FAST-SPECTRUM-2
            i_rho2 = 4.0 * math.pi * (r ** 2) * (rho ** 2)
            i_soft = np.exp(-beta * (r ** 2)) * p
            i_overlap = 4.0 * math.pi * (r ** 2) * rho * rho_g
            return np.stack(
                [m0, m1, m2, m3, m4, i_rho2, i_soft, i_overlap],
                axis=1,
            )

        # Векторизованный расчёт моментов по всем FDM-точкам
        samples = fdm.sample(depth=depth, dim=1)  # (N, 1)
        vals = moments_func(samples)              # (N, 8)
        m_vec = R_max * vals.mean(axis=0)
        (
            m0,
            m1,
            m2,
            m3,
            m4,
            I_rho2,
            I_soft,
            I_overlap,
        ) = [float(x) for x in m_vec]
        # effective volume: (m0^2) / I_rho2
        if not np.isfinite(I_rho2) or I_rho2 <= 0.0:
            effective_volume_ws = float("nan")
        else:
            effective_volume_ws = (m0 * m0) / I_rho2
        # softness integral: (1/m0) * ∫ exp(-β r^2) p(r) dr
        if not np.isfinite(m0) or m0 <= 0.0:
            softness_integral_ws = float("nan")
        else:
            softness_integral_ws = I_soft / m0
        density_overlap_ws = I_overlap
        if not np.isfinite(m0) or m0 <= 0.0:
            kurt_ws = float("nan")
        else:
            mu = m1 / m0
            # central moments via raw moments
            mu2 = m2 / m0 - mu ** 2
            mu3 = m3 / m0 - 3 * mu * (m2 / m0) + 2 * mu ** 3
            mu4 = m4 / m0 - 4 * mu * (m3 / m0) + 6 * (mu ** 2) * (m2 / m0) - 3 * mu ** 4
            if not np.isfinite(mu2) or mu2 <= 0.0:
                kurt_ws = float("nan")
            else:
                kurt_ws = float(mu4 / (mu2 ** 2) - 3.0)
    else:
        # Legacy trapz-based computation on fixed grid
        r_grid = np.linspace(0.0, R_max, 4096)
        rho_grid = rho_fn(r_grid)
        kurt_ws = float(radial_kurtosis_excess_from_rho3d(r_grid, rho_grid))
        p_grid = 4.0 * math.pi * (r_grid ** 2) * rho_grid
        # m0 and I_rho2 for effective volume
        m0 = float(_trapz(p_grid, r_grid))
        rho2_grid = 4.0 * math.pi * (r_grid ** 2) * (rho_grid ** 2)
        I_rho2 = float(_trapz(rho2_grid, r_grid))
        if not np.isfinite(I_rho2) or I_rho2 <= 0.0:
            effective_volume_ws = float("nan")
        else:
            effective_volume_ws = (m0 * m0) / I_rho2
        # softness integral: (1/m0) * ∫ exp(-β r^2) p(r) dr
        soft_grid = np.exp(-beta * (r_grid ** 2)) * p_grid
        if not np.isfinite(m0) or m0 <= 0.0:
            softness_integral_ws = float("nan")
        else:
            softness_integral_ws = float(_trapz(soft_grid, r_grid) / m0)
        # overlap of WS and Gaussian densities in 3D
        rho_g_grid = _gaussian_rho3d(r_grid)
        overlap_grid = 4.0 * math.pi * (r_grid ** 2) * rho_grid * rho_g_grid
        density_overlap_ws = float(_trapz(overlap_grid, r_grid))
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
        effective_volume_ws=effective_volume_ws,
        softness_integral_ws=softness_integral_ws,
        density_overlap_ws=density_overlap_ws,
    )
