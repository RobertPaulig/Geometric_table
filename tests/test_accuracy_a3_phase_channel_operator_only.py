import os

import numpy as np
import pytest


RUN_A3 = os.environ.get("RUN_A3_TESTS") == "1"
pytestmark = pytest.mark.skipif(not RUN_A3, reason="A3.1 operator-only tests are opt-in; set RUN_A3_TESTS=1")


if RUN_A3:
    from rdkit import Chem

    from hetero2.phase_channel import (
        PhaseChannelConfig,
        gauge_transform_A,
        magnetic_laplacian,
        normalize_flux_phi,
        phase_matrix_from_mol_sssr,
        ring_edges_from_cycles,
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


def test_a3_1_hermiticity() -> None:
    mol = Chem.MolFromSmiles("C1CCCCC1")  # cyclohexane
    assert mol is not None

    w = _bond_weight_matrix(mol)
    A = phase_matrix_from_mol_sssr(mol=mol, config=PhaseChannelConfig(flux_phi=0.7))
    L = magnetic_laplacian(weights=w, A=A)
    assert np.allclose(L, L.conj().T, atol=1e-10)


def test_a3_1_gauge_invariance_spectrum() -> None:
    mol = Chem.MolFromSmiles("c1ccccc1")  # benzene
    assert mol is not None

    w = _bond_weight_matrix(mol)
    A = phase_matrix_from_mol_sssr(mol=mol, config=PhaseChannelConfig(flux_phi=0.9))

    rng = np.random.default_rng(0)
    theta = rng.normal(size=mol.GetNumAtoms())
    A2 = gauge_transform_A(A=A, theta=theta)

    ev1 = _sorted_eigvals(magnetic_laplacian(weights=w, A=A))
    ev2 = _sorted_eigvals(magnetic_laplacian(weights=w, A=A2))
    assert float(np.max(np.abs(ev1 - ev2))) <= 1e-8


def test_a3_1_zero_field_limit() -> None:
    mol = Chem.MolFromSmiles("CC1CCCCC1")  # methylcyclohexane: ring + one non-ring edge
    assert mol is not None

    w = _bond_weight_matrix(mol)
    A0 = phase_matrix_from_mol_sssr(mol=mol, config=PhaseChannelConfig(flux_phi=0.0))
    L0 = magnetic_laplacian(weights=w, A=A0)

    # Baseline real Laplacian.
    D = np.diag(np.sum(w, axis=1))
    L_base = D - w
    assert np.allclose(L0, L_base, atol=1e-12)


def test_a3_1_time_reversal_invariance_spectrum() -> None:
    mol = Chem.MolFromSmiles("C1CCCCC1")
    assert mol is not None

    w = _bond_weight_matrix(mol)
    phi = 0.8
    ev_plus = _sorted_eigvals(magnetic_laplacian(weights=w, A=phase_matrix_from_mol_sssr(mol=mol, config=PhaseChannelConfig(flux_phi=phi))))
    ev_minus = _sorted_eigvals(magnetic_laplacian(weights=w, A=phase_matrix_from_mol_sssr(mol=mol, config=PhaseChannelConfig(flux_phi=-phi))))
    assert float(np.max(np.abs(ev_plus - ev_minus))) <= 1e-8


def test_a3_1_two_pi_periodicity_spectrum() -> None:
    mol = Chem.MolFromSmiles("c1ccccc1")
    assert mol is not None

    w = _bond_weight_matrix(mol)
    phi = 0.8
    ev1 = _sorted_eigvals(magnetic_laplacian(weights=w, A=phase_matrix_from_mol_sssr(mol=mol, config=PhaseChannelConfig(flux_phi=phi))))
    ev2 = _sorted_eigvals(magnetic_laplacian(weights=w, A=phase_matrix_from_mol_sssr(mol=mol, config=PhaseChannelConfig(flux_phi=phi + 2.0 * np.pi))))
    assert float(np.max(np.abs(ev1 - ev2))) <= 1e-8


def test_a3_1_dof_guard_only_ring_edges_nonzero() -> None:
    mol = Chem.MolFromSmiles("CC1CCCCC1")
    assert mol is not None

    cycles = sssr_cycles_from_mol(mol)
    ring_edges = ring_edges_from_cycles(cycles)
    assert len(ring_edges) > 0

    A = phase_matrix_from_mol_sssr(mol=mol, config=PhaseChannelConfig(flux_phi=0.7))
    n = mol.GetNumAtoms()
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if abs(float(A[i, j])) <= 1e-12:
                continue
            assert frozenset((i, j)) in ring_edges


def test_a3_1_flux_normalization_interval() -> None:
    x = normalize_flux_phi(123.0)
    assert -np.pi <= float(x) <= np.pi

