from __future__ import annotations

import numpy as np

from core.entanglement_3d import segment_segment_distance, entanglement_score


def test_segment_segment_distance_basic() -> None:
    p1 = np.array([0.0, 0.0, 0.0])
    p2 = np.array([1.0, 0.0, 0.0])
    q1 = np.array([0.0, 1.0, 0.0])
    q2 = np.array([1.0, 1.0, 0.0])
    d = segment_segment_distance(p1, p2, q1, q2)
    assert np.isfinite(d)
    assert d > 0.0


def test_entanglement_score_tree_vs_dense() -> None:
    # простое дерево
    pos_tree = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
        ],
        dtype=float,
    )
    edges_tree = [(0, 1), (1, 2)]
    score_tree = entanglement_score(pos_tree, edges_tree, sigma=0.5)
    assert np.isfinite(score_tree)
    assert score_tree >= 0.0

    # более плотный граф на тех же позициях
    edges_dense = [(0, 1), (1, 2), (0, 2)]
    score_dense = entanglement_score(pos_tree, edges_dense, sigma=0.5)
    assert np.isfinite(score_dense)
    assert score_dense >= 0.0
    assert score_dense >= score_tree

