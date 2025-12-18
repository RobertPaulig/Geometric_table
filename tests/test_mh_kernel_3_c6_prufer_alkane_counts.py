from __future__ import annotations

from collections import Counter

import numpy as np

from analysis.chem.chem_validation_1b_hexane import HEXANE_DEGENERACY, classify_hexane_topology
from analysis.chem.exact_trees import enumerate_labeled_trees


def test_c6_prufer_alkane_filter_count_and_topology_counts() -> None:
    trees = enumerate_labeled_trees(6)
    assert len(trees) == 6 ** (6 - 2)  # Cayley

    alkane = []
    for adj in trees:
        deg = np.sum(adj > 0, axis=1).astype(int)
        if int(np.max(deg)) <= 4:
            alkane.append(adj)
    assert len(alkane) == 1290  # excludes only the star K1,5

    counts: Counter[str] = Counter()
    for adj in alkane:
        counts[str(classify_hexane_topology(adj))] += 1

    for topo, g in HEXANE_DEGENERACY.items():
        assert counts[topo] == g

