from __future__ import annotations

import numpy as np

from core.geom_atoms import estimate_atom_energy_fdm
from core.spectral_density_ws import WSRadialParams, make_ws_rho3d_interpolator
from core.thermo_config import ThermoConfig
from core.thermo_config import override_thermo_config


def test_ws_rho_is_finite_and_nonnegative() -> None:
    params = WSRadialParams()
    rho_fn = make_ws_rho3d_interpolator(6, params)
    radii = np.linspace(0.0, params.R_max, 128)
    rho = rho_fn(radii)
    assert rho.shape == radii.shape
    assert np.all(np.isfinite(rho))
    assert np.all(rho >= 0.0)


def test_estimate_atom_energy_fdm_legacy_unchanged_when_density_source_gaussian() -> None:
    atom_z = 8
    e_port = 1.0

    # Baseline with some nonzero coupling_density (beta-physics) but no WS shape
    cfg_baseline = ThermoConfig(coupling_density=1.0, density_source="gaussian")
    with override_thermo_config(cfg_baseline):
        baseline = estimate_atom_energy_fdm(atom_z, e_port)

    # The same, но с изменёнными WS-параметрами и ws Z-coupling — не должно влиять
    # на gaussian-ветку.
    cfg_shape = ThermoConfig(
        coupling_density=1.0,
        coupling_density_shape=1.0,
        density_source="gaussian",
        coupling_ws_Z=1.0,
        ws_Z_ref=5.0,
        ws_Z_alpha=0.5,
    )
    with override_thermo_config(cfg_shape):
        val = estimate_atom_energy_fdm(atom_z, e_port)

    # Должно совпадать с baseline в пределах небольшого численного шума
    assert val == baseline or abs(val - baseline) / baseline < 1e-6


def test_estimate_atom_energy_fdm_ws_radial_is_reasonable_scale() -> None:
    atom_z = 6
    e_port = 1.0

    # Legacy Gaussian-only estimate
    legacy = estimate_atom_energy_fdm(atom_z, e_port)
    assert legacy > 0.0
    assert np.isfinite(legacy)

    cfg = ThermoConfig(
        coupling_density=1.0,
        coupling_density_shape=1.0,
        density_source="ws_radial",
    )
    with override_thermo_config(cfg):
        ws_val = estimate_atom_energy_fdm(atom_z, e_port)

    assert ws_val > 0.0
    assert np.isfinite(ws_val)
    # Масштабы не должны взрываться/обнуляться относительно legacy
    assert 0.1 * legacy < ws_val < 10.0 * legacy
