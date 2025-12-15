from __future__ import annotations

from core.shape_observables import get_shape_observables, thermo_fingerprint_for_shape
from core.thermo_config import ThermoConfig, override_thermo_config


def _kurt_for_Z(Z: int, ws_integrator: str) -> float:
    cfg = ThermoConfig(ws_integrator=ws_integrator, ws_fdm_depth=9, ws_fdm_base=2)
    with override_thermo_config(cfg):
        fp = thermo_fingerprint_for_shape(cfg)
        obs = get_shape_observables(Z, fp)
    return float(obs.kurt_ws)


def test_ws_integrator_fdm_close_to_trapz_for_selected_Z():
    # A few representative Z values
    for Z in [1, 8, 26]:
        k_trapz = _kurt_for_Z(Z, ws_integrator="trapz")
        k_fdm = _kurt_for_Z(Z, ws_integrator="fdm")
        # FDM approximation should be reasonably close to trapz baseline
        assert abs(k_fdm - k_trapz) < 0.1

