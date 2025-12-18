from __future__ import annotations

import numpy as np


def _adj(n: int, edges: list[tuple[int, int]]) -> np.ndarray:
    a = np.zeros((n, n), dtype=float)
    for i, j in edges:
        a[i, j] = 1.0
        a[j, i] = 1.0
    return a


def test_tree_automorphism_size_known_cases() -> None:
    from analysis.chem.alkane_exact_1 import tree_automorphism_size

    # C5 n-pentane (path): |Aut|=2
    adj_path = _adj(5, [(0, 1), (1, 2), (2, 3), (3, 4)])
    assert tree_automorphism_size(adj_path) == 2

    # C5 neopentane (star): |Aut|=4! = 24
    adj_star = _adj(5, [(0, 1), (0, 2), (0, 3), (0, 4)])
    assert tree_automorphism_size(adj_star) == 24

    # C6 2,2-dimethylbutane (degree-4 center with 3 identical leaves): |Aut|=6
    adj_22 = _adj(6, [(0, 1), (0, 2), (0, 3), (0, 5), (1, 4)])
    assert tree_automorphism_size(adj_22) == 6

