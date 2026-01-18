from hetero2.physics_operator import load_atoms_db_v1


def test_atoms_db_v1_has_potential_for_common_elements() -> None:
    atoms_db = load_atoms_db_v1()
    for z in [1, 6, 7, 8, 9, 15, 16, 17, 35, 53]:
        assert z in atoms_db.potential_by_atomic_num
        assert z in atoms_db.symbol_by_atomic_num
        assert atoms_db.symbol_by_atomic_num[z]

