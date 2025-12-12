from __future__ import annotations

from core.geom_atoms import _make_base_atoms


def test_atoms_db_not_empty() -> None:
    atoms = _make_base_atoms()
    assert len(atoms) >= 10

