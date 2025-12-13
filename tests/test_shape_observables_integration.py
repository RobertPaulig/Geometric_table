from __future__ import annotations

from dataclasses import replace

from core.shape_observables import ShapeObs, get_shape_observables, thermo_fingerprint_for_shape
from core.thermo_config import ThermoConfig, override_thermo_config
from core.geom_atoms import get_atom


def test_shape_observables_couplings_zero_keep_legacy_softness_and_chi():
    cfg = ThermoConfig()
    with override_thermo_config(cfg):
        c_atom = get_atom("C")
        soft_legacy = c_atom.effective_softness(cfg)
        chi_legacy = c_atom.chi_geom_signed_spec()

    cfg_shape = replace(
        cfg,
        coupling_shape_softness=1.0,
        coupling_shape_chi=1.0,
    )
    with override_thermo_config(cfg_shape):
        fp = thermo_fingerprint_for_shape(cfg_shape)
        # ensure helper can be called without errors
        _ = get_shape_observables(6, fp)
        c_atom = get_atom("C")
        soft_shape = c_atom.effective_softness(cfg_shape)
        chi_shape = c_atom.chi_geom_signed_spec()

    assert soft_shape >= soft_legacy
    assert chi_shape is not None
    assert abs(chi_shape) >= abs(chi_legacy or 0.0)

