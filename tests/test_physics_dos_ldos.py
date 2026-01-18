import numpy as np

from hetero2.physics_operator import DOS_ETA_DEFAULT, compute_dos_curve, compute_dos_ldos_payload


def test_dos_curve_is_deterministic_for_fixed_inputs() -> None:
    eigvals = [0.0, 1.0, 2.0, 3.0]
    energy = np.linspace(-1.0, 4.0, 64, dtype=float)
    dos_a = compute_dos_curve(eigenvalues=eigvals, energy_grid=energy, eta=float(DOS_ETA_DEFAULT))
    dos_b = compute_dos_curve(eigenvalues=eigvals, energy_grid=energy, eta=float(DOS_ETA_DEFAULT))
    assert np.allclose(dos_a, dos_b)


def test_dos_blindness_preserved_for_L_and_vision_enabled_for_H() -> None:
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
    # Bonds are needed for payload schema, even if edge weights are unweighted.
    bonds = ((0, 1, 1.0), (1, 2, 1.0), (2, 3, 1.0))

    payload_a = compute_dos_ldos_payload(
        adjacency=adj,
        bonds=bonds,
        types=(6, 6, 6, 6),
        physics_mode="both",
        edge_weight_mode="unweighted",
    )
    payload_b = compute_dos_ldos_payload(
        adjacency=adj,
        bonds=bonds,
        types=(6, 6, 8, 6),
        physics_mode="both",
        edge_weight_mode="unweighted",
    )

    eig_L_a = payload_a["eigvals_L"]
    eig_L_b = payload_b["eigvals_L"]
    eig_H_a = payload_a["eigvals_H"]
    eig_H_b = payload_b["eigvals_H"]

    assert np.allclose(eig_L_a, eig_L_b)
    assert not np.allclose(eig_H_a, eig_H_b)

    # Same grid for both (run-level determinism).
    all_H = np.array(list(eig_H_a) + list(eig_H_b), dtype=float)
    e_min = float(np.min(all_H)) - 3.0 * float(DOS_ETA_DEFAULT)
    e_max = float(np.max(all_H)) + 3.0 * float(DOS_ETA_DEFAULT)
    energy = np.linspace(e_min, e_max, 64, dtype=float)

    dos_L_a = compute_dos_curve(eigenvalues=eig_L_a, energy_grid=energy, eta=float(DOS_ETA_DEFAULT))
    dos_L_b = compute_dos_curve(eigenvalues=eig_L_b, energy_grid=energy, eta=float(DOS_ETA_DEFAULT))
    dos_H_a = compute_dos_curve(eigenvalues=eig_H_a, energy_grid=energy, eta=float(DOS_ETA_DEFAULT))
    dos_H_b = compute_dos_curve(eigenvalues=eig_H_b, energy_grid=energy, eta=float(DOS_ETA_DEFAULT))

    assert np.allclose(dos_L_a, dos_L_b)
    assert not np.allclose(dos_H_a, dos_H_b)

