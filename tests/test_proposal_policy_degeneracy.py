from __future__ import annotations

import numpy as np

from core.grower import GrowthParams, grow_molecule_christmas_tree, grow_molecule_loopy
from core.thermo_config import ThermoConfig, override_thermo_config


def _normalize_bonds(bonds):
    norm = set()
    for i, j in bonds:
        if i == j:
            continue
        a, b = (i, j) if i < j else (j, i)
        norm.add((a, b))
    return norm


def _grow_tree_uniform(seed: int = 42):
    rng = np.random.default_rng(seed)
    params = GrowthParams(max_depth=3, max_atoms=15)
    thermo = ThermoConfig(
        grower_use_mh=False,
        grower_proposal_policy="uniform",
        proposal_beta=0.0,
        proposal_ports_gamma=0.0,
    )
    with override_thermo_config(thermo):
        mol = grow_molecule_christmas_tree("C", params=params, rng=rng)
    return mol


def _grow_tree_ctt_degenerate(seed: int = 42):
    rng = np.random.default_rng(seed)
    params = GrowthParams(max_depth=3, max_atoms=15)
    thermo = ThermoConfig(
        grower_use_mh=False,
        grower_proposal_policy="ctt_biased",
        proposal_beta=0.0,
        proposal_ports_gamma=0.0,
    )
    with override_thermo_config(thermo):
        mol = grow_molecule_christmas_tree("C", params=params, rng=rng)
    return mol


def test_proposal_policy_degeneracy_tree_uniform_vs_ctt_beta0_gamma0():
    mol_u = _grow_tree_uniform(seed=123)
    mol_c = _grow_tree_ctt_degenerate(seed=123)

    assert len(mol_u.atoms) == len(mol_c.atoms)
    assert _normalize_bonds(mol_u.bonds) == _normalize_bonds(mol_c.bonds)


def _grow_loopy_uniform(seed: int = 42):
    rng = np.random.default_rng(seed)
    thermo = ThermoConfig(
        grower_use_mh=False,
        grower_proposal_policy="uniform",
        proposal_beta=0.0,
        proposal_ports_gamma=0.0,
    )
    with override_thermo_config(thermo):
        mol = grow_molecule_loopy("C", rng=rng)
    return mol


def _grow_loopy_ctt_degenerate(seed: int = 42):
    rng = np.random.default_rng(seed)
    thermo = ThermoConfig(
        grower_use_mh=False,
        grower_proposal_policy="ctt_biased",
        proposal_beta=0.0,
        proposal_ports_gamma=0.0,
    )
    with override_thermo_config(thermo):
        mol = grow_molecule_loopy("C", rng=rng)
    return mol


def test_proposal_policy_degeneracy_loopy_uniform_vs_ctt_beta0_gamma0():
    mol_u = _grow_loopy_uniform(seed=321)
    mol_c = _grow_loopy_ctt_degenerate(seed=321)

    assert len(mol_u.atoms) == len(mol_c.atoms)
    assert _normalize_bonds(mol_u.bonds) == _normalize_bonds(mol_c.bonds)

