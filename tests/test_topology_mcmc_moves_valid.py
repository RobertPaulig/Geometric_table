from __future__ import annotations

from analysis.chem.topology_mcmc import (
    apply_leaf_rewire_move,
    enumerate_leaf_rewire_moves,
    is_tree,
)


def test_leaf_rewire_keeps_tree_n4() -> None:
    n = 4
    edges = [(0, 1), (1, 2), (2, 3)]
    assert is_tree(n, edges)
    moves = enumerate_leaf_rewire_moves(n, edges, max_valence=4)
    assert moves
    for mv in moves[:10]:
        e2 = apply_leaf_rewire_move(edges, mv)
        assert is_tree(n, e2)


def test_leaf_rewire_keeps_tree_n5() -> None:
    n = 5
    edges = [(0, 1), (1, 2), (2, 3), (3, 4)]
    assert is_tree(n, edges)
    moves = enumerate_leaf_rewire_moves(n, edges, max_valence=4)
    assert moves
    for mv in moves[:20]:
        e2 = apply_leaf_rewire_move(edges, mv)
        assert is_tree(n, e2)

