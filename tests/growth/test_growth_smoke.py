from __future__ import annotations

from core.growth_config import load_growth_config
from core.grower import grow_molecule_christmas_tree


def test_cy1a_growth_smoke() -> None:
    cfg = load_growth_config("configs/growth_cy1a.yaml")
    params = cfg.to_growth_params()
    mol = grow_molecule_christmas_tree("C", params)
    assert len(mol.nodes) >= 1


def test_cy1a_avg_depth_and_size_reasonable() -> None:
    cfg = load_growth_config("configs/growth_cy1a.yaml")
    params = cfg.to_growth_params()

    depths = []
    sizes = []
    for _ in range(20):
        mol = grow_molecule_christmas_tree("C", params)
        depths.append(mol.depth)
        sizes.append(len(mol.nodes))

    avg_depth = sum(depths) / len(depths)
    avg_size = sum(sizes) / len(sizes)

    assert 2.0 <= avg_depth <= 6.0
    assert 3.0 <= avg_size <= 40.0

