from __future__ import annotations

from core.geom_atoms import AtomGraph
from core.thermo_config import ThermoConfig


def test_effective_softness_legacy_vs_coupled() -> None:
    atom = AtomGraph(
        name="Si",
        Z=14,
        nodes=4,
        edges=4,
        ports=4,
        symmetry_score=1.0,
        port_geometry="tetra",
        role="hub",
        softness=0.3,
    )

    thermo_legacy = ThermoConfig(coupling_softness=0.0)
    thermo_coupled = ThermoConfig(temperature=2.0, coupling_softness=1.0)

    s_legacy = atom.effective_softness(thermo_legacy)
    s_coupled = atom.effective_softness(thermo_coupled)

    assert abs(s_legacy - 0.3) < 1e-9
    assert 0.0 <= s_coupled <= 0.95
    assert s_coupled != s_legacy

