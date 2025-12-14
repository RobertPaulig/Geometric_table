from __future__ import annotations

from dataclasses import replace

from core.geom_atoms import get_atom
from core.thermo_config import ThermoConfig, override_thermo_config


def _bond_class(delta_chi: float) -> str:
    d = abs(float(delta_chi))
    if d > 1.5:
        return "ionic_strong"
    if d > 0.8:
        return "ionic_polar"
    return "covalent"


def test_shape_chi_gain_guardrails():
    cfg = replace(
        ThermoConfig(),
        coupling_ws_Z=1.0,
        coupling_shape_chi=1.0,
        coupling_shape_softness=1.0,
        shape_chi_gain=0.30,
    )

    pairs = [
        ("Na", "Cl", "ionic_strong"),
        ("H", "F", "ionic_strong"),
        ("Si", "O", "ionic_polar"),
        ("C", "O", "covalent"),
    ]

    with override_thermo_config(cfg):
        for a, b, expected in pairs:
            chi1 = get_atom(a).chi_geom_signed_spec() or 0.0
            chi2 = get_atom(b).chi_geom_signed_spec() or 0.0
            delta = chi2 - chi1
            got = _bond_class(delta)
            assert got == expected, f"{a}-{b}: Δχ={delta:.6f} expected={expected} got={got}"

