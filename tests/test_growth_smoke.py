from __future__ import annotations

from core.growth_config import load_growth_config
from core.grower import grow_molecule_christmas_tree


def test_cy1a_growth_smoke() -> None:
    cfg = load_growth_config("configs/growth_cy1a.yaml")
    params = cfg.to_growth_params()
    mol = grow_molecule_christmas_tree("C", params)
    assert len(mol.nodes) >= 1


