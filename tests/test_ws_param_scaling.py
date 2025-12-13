from __future__ import annotations

from core.spectral_density_ws import WSRadialParams
from core.ws_param_scaling import apply_ws_Z_scaling


def test_ws_param_scaling_coupling_zero_identity() -> None:
    params = WSRadialParams(R_max=12.0, R_well=5.0, V0=40.0, N_grid=220, ell=0, state_index=0)
    scaled = apply_ws_Z_scaling(params, Z=10, coupling_ws_Z=0.0, Z_ref=10.0, alpha=1.0 / 3.0)
    assert scaled == params


def test_ws_param_scaling_radius_decreases_with_Z() -> None:
    params = WSRadialParams(R_max=12.0, R_well=5.0, V0=40.0, N_grid=220, ell=0, state_index=0)
    scaled_lowZ = apply_ws_Z_scaling(params, Z=5, coupling_ws_Z=1.0, Z_ref=10.0, alpha=1.0 / 3.0)
    scaled_highZ = apply_ws_Z_scaling(params, Z=20, coupling_ws_Z=1.0, Z_ref=10.0, alpha=1.0 / 3.0)
    assert scaled_lowZ.R_well > scaled_highZ.R_well

