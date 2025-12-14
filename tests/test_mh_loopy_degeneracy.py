from __future__ import annotations

import numpy as np

from core.grower import grow_molecule_loopy
from core.thermo_config import ThermoConfig, override_thermo_config


def _normalize_bonds(bonds):
    norm = set()
    for i, j in bonds:
        if i == j:
            continue
        a, b = (i, j) if i < j else (j, i)
        norm.add((a, b))
    return norm


def _grow_loopy_legacy(seed: int = 42):
    rng = np.random.default_rng(seed)
    thermo = ThermoConfig(grower_use_mh=False)
    with override_thermo_config(thermo):
        mol = grow_molecule_loopy("C", rng=rng)
    return mol


def _grow_loopy_mh(coupling_delta_G: float, temperature_T: float, seed: int = 42):
    rng = np.random.default_rng(seed)
    thermo = ThermoConfig(
        grower_use_mh=True,
        coupling_delta_G=coupling_delta_G,
        temperature_T=temperature_T,
    )
    with override_thermo_config(thermo):
        mol = grow_molecule_loopy("C", rng=rng)
    return mol


def test_mh_loopy_degeneracy_coupling_zero_matches_legacy():
    mol_legacy = _grow_loopy_legacy(seed=42)
    mol_mh = _grow_loopy_mh(coupling_delta_G=0.0, temperature_T=1.0, seed=42)

    assert len(mol_legacy.atoms) == len(mol_mh.atoms)
    assert _normalize_bonds(mol_legacy.bonds) == _normalize_bonds(mol_mh.bonds)


def test_mh_loopy_degeneracy_T_infinite_matches_legacy():
    mol_legacy = _grow_loopy_legacy(seed=43)
    mol_mh = _grow_loopy_mh(coupling_delta_G=1.0, temperature_T=1e9, seed=43)

    assert len(mol_legacy.atoms) == len(mol_mh.atoms)
    assert _normalize_bonds(mol_legacy.bonds) == _normalize_bonds(mol_mh.bonds)

