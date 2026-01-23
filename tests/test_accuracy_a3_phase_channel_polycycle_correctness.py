import os

import numpy as np
import pytest


RUN_A3 = os.environ.get("RUN_A3_TESTS") == "1"
pytestmark = pytest.mark.skipif(not RUN_A3, reason="A3.2 polycycle tests are opt-in; set RUN_A3_TESTS=1")


if RUN_A3:
    from rdkit import Chem

    from hetero2.phase_channel import (
        PhaseChannelConfig,
        magnetic_laplacian,
        phase_matrix_from_mol_sssr,
        sssr_cycles_from_mol,
    )


def _bond_weight_matrix(mol: "Chem.Mol") -> np.ndarray:
    n = mol.GetNumAtoms()
    w = np.zeros((n, n), dtype=float)
    for b in mol.GetBonds():
        i = int(b.GetBeginAtomIdx())
        j = int(b.GetEndAtomIdx())
        w[i, j] = 1.0
        w[j, i] = 1.0
    return w


def _sorted_eigvals(H: np.ndarray) -> np.ndarray:
    return np.sort(np.linalg.eigvalsh(H))


def test_a3_2_sssr_cycles_naphthalene_are_deterministic() -> None:
    mol = Chem.MolFromSmiles("c1cccc2c1cccc2")
    assert mol is not None

    cycles = sssr_cycles_from_mol(mol)
    assert cycles == [[0, 1, 2, 3, 4, 5], [4, 5, 6, 7, 8, 9]]


def test_a3_2_shared_edge_flux_is_sum_over_sssr_cycles() -> None:
    mol = Chem.MolFromSmiles("c1cccc2c1cccc2")  # naphthalene: two fused 6-cycles
    assert mol is not None

    cycles = sssr_cycles_from_mol(mol)
    assert cycles == [[0, 1, 2, 3, 4, 5], [4, 5, 6, 7, 8, 9]]

    phi = 0.9
    A = phase_matrix_from_mol_sssr(mol=mol, config=PhaseChannelConfig(flux_phi=phi))

    # With deterministic cycle orientation (min atom + lexicographically-min traversal),
    # shared edge (4,5) is traversed as 4->5 in BOTH rings, so contributions sum:
    # A[4,5] = +phi/6 + +phi/6 = phi/3.
    assert float(A[4, 5]) == pytest.approx(phi / 3.0, abs=1e-12)
    assert float(A[5, 4]) == pytest.approx(-phi / 3.0, abs=1e-12)


def test_a3_2_zero_field_limit_polycycle() -> None:
    mol = Chem.MolFromSmiles("c1cccc2c1cccc2")
    assert mol is not None

    w = _bond_weight_matrix(mol)
    A0 = phase_matrix_from_mol_sssr(mol=mol, config=PhaseChannelConfig(flux_phi=0.0))
    L0 = magnetic_laplacian(weights=w, A=A0)

    D = np.diag(np.sum(w, axis=1))
    L_base = D - w
    assert np.allclose(L0, L_base, atol=1e-12)


def test_a3_2_two_pi_periodicity_spectrum_polycycle() -> None:
    mol = Chem.MolFromSmiles("c1cccc2c1cccc2")
    assert mol is not None

    w = _bond_weight_matrix(mol)
    phi = 0.7
    ev1 = _sorted_eigvals(magnetic_laplacian(weights=w, A=phase_matrix_from_mol_sssr(mol=mol, config=PhaseChannelConfig(flux_phi=phi))))
    ev2 = _sorted_eigvals(magnetic_laplacian(weights=w, A=phase_matrix_from_mol_sssr(mol=mol, config=PhaseChannelConfig(flux_phi=phi + 2.0 * np.pi))))
    assert float(np.max(np.abs(ev1 - ev2))) <= 1e-8

