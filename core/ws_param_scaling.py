from __future__ import annotations

import math

from dataclasses import dataclass


@dataclass(frozen=True)
class WSRadialParams:
    R_max: float = 12.0
    R_well: float = 5.0
    V0: float = 40.0
    N_grid: int = 220
    ell: int = 0
    state_index: int = 0


def clamp01(x: float) -> float:
    return max(0.0, min(float(x), 1.0))


def ws_scale_factor(Z: int, Z_ref: float, alpha: float) -> float:
    """
    Масштабный множитель для радиусов WS-потенциала:
    s(Z) = (Z_ref / max(Z,1))^alpha.

    При alpha>0 радиусы уменьшаются с ростом Z (TF-radius-подобное поведение).
    """
    z = max(float(Z), 1.0)
    z_ref = max(float(Z_ref), 1.0)
    return (z_ref / z) ** float(alpha)


def apply_ws_Z_scaling(
    params: WSRadialParams,
    Z: int,
    coupling_ws_Z: float,
    Z_ref: float,
    alpha: float,
) -> WSRadialParams:
    """
    Применяет Z-зависимое масштабирование к R_max и R_well с плавным blend по coupling_ws_Z.
    При coupling_ws_Z=0 возвращает исходные параметры.
    """
    c = clamp01(coupling_ws_Z)
    if c <= 0.0:
        return params

    s = ws_scale_factor(Z, Z_ref, alpha)

    R_max_base = float(params.R_max)
    R_well_base = float(params.R_well)

    R_max_scaled = R_max_base * s
    R_well_scaled = R_well_base * s

    R_max_eff = (1.0 - c) * R_max_base + c * R_max_scaled
    R_well_eff = (1.0 - c) * R_well_base + c * R_well_scaled

    return WSRadialParams(
        R_max=R_max_eff,
        R_well=R_well_eff,
        V0=params.V0,
        N_grid=params.N_grid,
        ell=params.ell,
        state_index=params.state_index,
    )
