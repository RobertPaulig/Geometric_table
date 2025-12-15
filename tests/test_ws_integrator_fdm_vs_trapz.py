from __future__ import annotations

import math

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


def _obs_for_Z(Z: int, ws_integrator: str, base: int, depth: int):
    cfg = ThermoConfig(
        ws_integrator=ws_integrator,
        ws_fdm_depth=depth,
        ws_fdm_base=base,
    )
    with override_thermo_config(cfg):
        fp = thermo_fingerprint_for_shape(cfg)
        obs = get_shape_observables(Z, fp)
    return obs


def test_ws_integrator_trapz_vs_fdm_new_observables_agree_on_key_Z():
    base_star = 2
    depth_star = 5
    max_rel_err_volume = 0.0
    max_rel_err_softness = 0.0
    max_rel_err_overlap = 0.0

    for Z in [1, 6, 8, 14]:
        obs_trapz = _obs_for_Z(Z, ws_integrator="trapz", base=base_star, depth=depth_star)
        obs_fdm = _obs_for_Z(Z, ws_integrator="fdm", base=base_star, depth=depth_star)

        # effective volume
        v_t = float(obs_trapz.effective_volume_ws)
        v_f = float(obs_fdm.effective_volume_ws)
        if math.isfinite(v_t) and v_t > 0.0 and math.isfinite(v_f) and v_f > 0.0:
            max_rel_err_volume = max(max_rel_err_volume, abs(v_f - v_t) / v_t)

        # softness integral
        s_t = float(obs_trapz.softness_integral_ws)
        s_f = float(obs_fdm.softness_integral_ws)
        if math.isfinite(s_t) and s_t != 0.0 and math.isfinite(s_f):
            max_rel_err_softness = max(max_rel_err_softness, abs(s_f - s_t) / abs(s_t))

        # density overlap: only sanity (>0, finite)
        o_t = float(obs_trapz.density_overlap_ws)
        o_f = float(obs_fdm.density_overlap_ws)
        assert o_t > 0.0 and math.isfinite(o_t)
        assert o_f > 0.0 and math.isfinite(o_f)
        max_rel_err_overlap = max(max_rel_err_overlap, abs(o_f - o_t) / o_t if o_t != 0.0 else 0.0)

    assert max_rel_err_volume <= 0.10
    assert max_rel_err_softness <= 0.05
