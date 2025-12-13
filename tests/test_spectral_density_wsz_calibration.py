from __future__ import annotations

from core.geom_atoms import estimate_atom_energy_fdm
from core.thermo_config import ThermoConfig, override_thermo_config


def test_mass_ratio_box_ws_z_on_Z1_reasonable_scale() -> None:
    """
    При включённом ws Z-coupling и ws_radial источнике плотности
    энергия для Z=1 остаётся конечной и позитивной (масса в коробке
    не "теряется").
    """
    cfg = ThermoConfig(
        coupling_density=1.0,
        coupling_density_shape=1.0,
        density_source="ws_radial",
        coupling_ws_Z=1.0,
        ws_Z_ref=10.0,
        ws_Z_alpha=1.0 / 3.0,
    )
    with override_thermo_config(cfg):
        val = estimate_atom_energy_fdm(1, 1.0)
    assert val > 0.0

