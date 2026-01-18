import numpy as np
import pytest

from hetero2.physics_operator import (
    AtomsDbV1,
    MissingPhysicsParams,
    compute_physics_features,
    eigvals_symmetric,
    laplacian_from_adjacency,
    load_atoms_db_v1,
    weighted_adjacency_from_bonds,
)


def test_weighted_edges_make_laplacian_sensitive_to_types() -> None:
    # Path graph on 4 nodes: 0-1-2-3 (same topology, different atom types).
    adj = np.array(
        [
            [0.0, 1.0, 0.0, 0.0],
            [1.0, 0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0, 1.0],
            [0.0, 0.0, 1.0, 0.0],
        ],
        dtype=float,
    )
    bonds = ((0, 1, 1.0), (1, 2, 1.0), (2, 3, 1.0))
    atoms_db = load_atoms_db_v1()

    # Unweighted Laplacian is blind to atom types.
    spec_L_a = eigvals_symmetric(laplacian_from_adjacency(adj))
    spec_L_b = eigvals_symmetric(laplacian_from_adjacency(adj))

    # Weighted Laplacian depends on Δchi (SoT: atoms_db_v1.json).
    w_adj_a = weighted_adjacency_from_bonds(
        n=4,
        bonds=bonds,
        types=(6, 6, 6, 6),
        edge_weight_mode="bond_order_delta_chi",
        atoms_db=atoms_db,
    )
    w_adj_b = weighted_adjacency_from_bonds(
        n=4,
        bonds=bonds,
        types=(6, 6, 8, 6),
        edge_weight_mode="bond_order_delta_chi",
        atoms_db=atoms_db,
    )
    spec_W_a = eigvals_symmetric(laplacian_from_adjacency(w_adj_a))
    spec_W_b = eigvals_symmetric(laplacian_from_adjacency(w_adj_b))

    assert np.allclose(spec_L_a, spec_L_b)
    assert not np.allclose(spec_W_a, spec_W_b)


def test_weighted_edges_require_chi_and_hamiltonian_requires_V() -> None:
    adj = np.array(
        [
            [0.0, 1.0, 0.0, 0.0],
            [1.0, 0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0, 1.0],
            [0.0, 0.0, 1.0, 0.0],
        ],
        dtype=float,
    )
    bonds = ((0, 1, 1.0), (1, 2, 1.0), (2, 3, 1.0))

    # Missing chi -> explicit error (no silent defaults).
    atoms_db_missing_chi = AtomsDbV1(
        source_path="test",
        potential_by_atomic_num={6: -1.0, 8: -1.0},
        chi_by_atomic_num={6: 2.55},
        symbol_by_atomic_num={6: "C", 8: "O"},
    )
    with pytest.raises(MissingPhysicsParams) as excinfo:
        weighted_adjacency_from_bonds(
            n=4,
            bonds=bonds,
            types=(6, 6, 8, 6),
            edge_weight_mode="bond_order_delta_chi",
            atoms_db=atoms_db_missing_chi,
        )
    assert excinfo.value.missing_key == "chi"

    # Missing V (epsilon) -> explicit error when physics_mode needs H.
    atoms_db_missing_eps = AtomsDbV1(
        source_path="test",
        potential_by_atomic_num={6: -1.0},
        chi_by_atomic_num={6: 2.55, 8: 3.44},
        symbol_by_atomic_num={6: "C", 8: "O"},
    )
    with pytest.raises(MissingPhysicsParams) as excinfo:
        compute_physics_features(adjacency=adj, types=(6, 6, 8, 6), physics_mode="hamiltonian", atoms_db=atoms_db_missing_eps)
    assert excinfo.value.missing_key == "epsilon"
