from __future__ import annotations

from core.shape_observables import get_shape_observables, thermo_fingerprint_for_shape
from core.thermo_config import ThermoConfig, override_thermo_config


def _kurt_for_Z(Z: int, ws_integrator: str, base: int, depth: int) -> float:
    cfg = ThermoConfig(
        ws_integrator=ws_integrator,
        ws_fdm_depth=depth,
        ws_fdm_base=base,
    )
    with override_thermo_config(cfg):
        fp = thermo_fingerprint_for_shape(cfg)
        obs = get_shape_observables(Z, fp)
    return float(obs.kurt_ws)


def test_ws_integrator_fdm_meets_accuracy_DoD_for_selected_Z():
    # Selected (base*, depth*) should satisfy max_abs_err <= 0.05 on key Z values
    base_star = 2
    depth_star = 9
    max_err = 0.0
    for Z in [1, 6, 8, 14, 26]:
        k_trapz = _kurt_for_Z(Z, ws_integrator="trapz", base=base_star, depth=depth_star)
        k_fdm = _kurt_for_Z(Z, ws_integrator="fdm", base=base_star, depth=depth_star)
        max_err = max(max_err, abs(k_fdm - k_trapz))
    assert max_err <= 0.05
