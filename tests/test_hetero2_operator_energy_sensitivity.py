from hetero2.chem_operator import h_operator_energy_from_edges, laplacian_energy_from_edges


def test_h_operator_energy_changes_with_atom_types_for_same_topology() -> None:
    edges = ((0, 1),)
    e_lap_1 = laplacian_energy_from_edges(2, edges)
    e_lap_2 = laplacian_energy_from_edges(2, edges)
    assert e_lap_1 == e_lap_2

    e_h_c = h_operator_energy_from_edges(2, edges, (6, 6))
    e_h_o = h_operator_energy_from_edges(2, edges, (8, 8))
    assert abs(e_h_c - e_h_o) > 1e-6

