from __future__ import annotations

import numpy as np

from analysis.chem.chem_validation_1b_hexane import classify_hexane_topology


def _adj(edges: list[tuple[int, int]]) -> np.ndarray:
    n = 6
    adj = np.zeros((n, n), dtype=float)
    for a, b in edges:
        adj[a, b] = adj[b, a] = 1.0
    return adj


def test_classify_hexane_topology_references() -> None:
    # n-hexane: 0-1-2-3-4-5
    assert classify_hexane_topology(_adj([(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)])) == "n_hexane"

    # 2-methylpentane: chain 0-1-2-3-4, methyl at 1 (node 5)
    assert (
        classify_hexane_topology(_adj([(0, 1), (1, 2), (2, 3), (3, 4), (1, 5)]))
        == "2_methylpentane"
    )

    # 3-methylpentane: chain 0-1-2-3-4, methyl at 2 (node 5)
    assert (
        classify_hexane_topology(_adj([(0, 1), (1, 2), (2, 3), (3, 4), (2, 5)]))
        == "3_methylpentane"
    )

    # 2,2-dimethylbutane: chain 0-1-2-3, two methyl at 1 (4,5)
    assert (
        classify_hexane_topology(_adj([(0, 1), (1, 2), (2, 3), (1, 4), (1, 5)]))
        == "2,2_dimethylbutane"
    )

    # 2,3-dimethylbutane: chain 0-1-2-3, methyl at 1 (4) and 2 (5)
    assert (
        classify_hexane_topology(_adj([(0, 1), (1, 2), (2, 3), (1, 4), (2, 5)]))
        == "2,3_dimethylbutane"
    )

