from __future__ import annotations

import numpy as np

from core.fixedn_tree_mcmc import apply_move, leaf_rewire_moves


def _adj_from_edges(n: int, edges: list[tuple[int, int]]) -> np.ndarray:
    adj = np.zeros((n, n), dtype=float)
    for a, b in edges:
        adj[a, b] = 1.0
        adj[b, a] = 1.0
    return adj


def test_leaf_rewire_move_preserves_tree_and_degree() -> None:
    # Start from a path on 6 nodes.
    n = 6
    adj0 = _adj_from_edges(n, [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)])
    moves = leaf_rewire_moves(adj0, max_degree=4)
    assert moves
    move = moves[0]
    adj1 = apply_move(adj0, move)

    assert adj1.shape == (n, n)
    assert np.allclose(adj1, adj1.T)

    # Still has n-1 edges.
    m = int(np.sum(adj1) // 2)
    assert m == n - 1

    deg = np.sum(adj1 > 0, axis=1).astype(int)
    assert int(np.max(deg)) <= 4

