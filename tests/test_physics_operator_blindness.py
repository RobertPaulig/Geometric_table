import numpy as np

from hetero2.physics_operator import compute_spectra, load_atoms_db_v1


def test_physics_operator_blindness_topology_vs_types() -> None:
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
    atoms_db = load_atoms_db_v1()

    spec_a = compute_spectra(adjacency=adj, types=(6, 6, 6, 6), physics_mode="both", atoms_db=atoms_db)
    spec_b = compute_spectra(adjacency=adj, types=(6, 6, 8, 6), physics_mode="both", atoms_db=atoms_db)

    assert np.allclose(spec_a["L"], spec_b["L"])
    assert not np.allclose(spec_a["H"], spec_b["H"])

