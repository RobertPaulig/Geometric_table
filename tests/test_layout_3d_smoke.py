from __future__ import annotations

import numpy as np

from core.geom_atoms import Molecule, PERIODIC_TABLE
from core.layout_3d import layout_molecule_3d


def test_layout_molecule_3d_smoke() -> None:
    C = PERIODIC_TABLE["C"]
    mol = Molecule(name="chain_C3", atoms=[C, C, C], bonds=[(0, 1), (1, 2)])
    pos = layout_molecule_3d(mol, n_steps=50, seed=123)
    assert pos.shape == (3, 3)
    assert np.all(np.isfinite(pos))

