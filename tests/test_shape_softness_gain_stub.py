from __future__ import annotations

from dataclasses import replace

from core.geom_atoms import get_atom
from core.thermo_config import ThermoConfig, override_thermo_config


def test_shape_softness_gain_zero_disables_shape_softness():
    cfg_base = ThermoConfig()

    atom = get_atom("C")

    # Legacy softness without shape coupling
    with override_thermo_config(cfg_base):
        soft_legacy = atom.effective_softness(cfg_base)

    # Shape coupling on, but gain set to 0 -> поведение должно совпасть с legacy
    cfg_shape = replace(
        cfg_base,
        coupling_shape_softness=1.0,
        shape_softness_gain=0.0,
    )
    with override_thermo_config(cfg_shape):
        soft_shape = atom.effective_softness(cfg_shape)

    assert soft_shape == soft_legacy

