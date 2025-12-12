from __future__ import annotations

from core.geom_atoms import AtomGraph, W_COMPLEXITY, compute_W_complexity_eff
from core.thermo_config import ThermoConfig, override_thermo_config


def test_W_complexity_eff_blend() -> None:
    base = W_COMPLEXITY
    T = 2.0

    assert compute_W_complexity_eff(base, coupling=0.0, temperature=T) == base
    assert compute_W_complexity_eff(base, coupling=1.0, temperature=T) == base * T
    mid = compute_W_complexity_eff(base, coupling=0.5, temperature=T)
    assert base < mid < base * T


def test_F_geom_respects_complexity_coupling() -> None:
    atom = AtomGraph(
        name="X",
        Z=6,
        nodes=4,
        edges=4,
        ports=4,
        symmetry_score=1.0,
        port_geometry="tetra",
        role="hub",
    )

    cfg_cold = ThermoConfig(temperature=0.5, coupling_complexity=1.0)
    cfg_hot = ThermoConfig(temperature=2.0, coupling_complexity=1.0)

    with override_thermo_config(cfg_cold):
        F_cold = atom.F_geom()

    with override_thermo_config(cfg_hot):
        F_hot = atom.F_geom()

    assert F_hot > F_cold

