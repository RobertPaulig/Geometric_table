from __future__ import annotations

from core.spectral_density_ws import WSRadialParams
from core.port_geometry_spectral import ws_sp_gap
from core.thermo_config import ThermoConfig, override_thermo_config


def test_ws_sp_gap_constant_when_coupling_ws_Z_zero() -> None:
    params = WSRadialParams()
    cfg = ThermoConfig(coupling_ws_Z=0.0)
    with override_thermo_config(cfg):
        gap_5 = ws_sp_gap(5, params)
        gap_20 = ws_sp_gap(20, params)
    assert abs(gap_5 - gap_20) < 1e-9


def test_ws_sp_gap_varies_with_Z_when_coupling_ws_Z_one() -> None:
    params = WSRadialParams()
    cfg = ThermoConfig(coupling_ws_Z=1.0, ws_Z_ref=10.0, ws_Z_alpha=1.0 / 3.0)
    with override_thermo_config(cfg):
        gap_5 = ws_sp_gap(5, params)
        gap_20 = ws_sp_gap(20, params)
    assert abs(gap_5 - gap_20) > 1e-4

