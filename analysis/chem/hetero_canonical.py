from __future__ import annotations

from typing import List, Sequence, Tuple

from core.tree_canonical_colored import canonical_colored_tree_permutation

Edge = Tuple[int, int]


def _edges_to_adj_list(n: int, edges: Sequence[Edge]) -> List[List[int]]:
    adj: List[List[int]] = [[] for _ in range(n)]
    for u, v in edges:
        if not (0 <= int(u) < n and 0 <= int(v) < n):
            raise ValueError("edge endpoints must be within [0, n)")
        if int(u) == int(v):
            raise ValueError("self-loops are not allowed")
        adj[int(u)].append(int(v))
        adj[int(v)].append(int(u))
    for i in range(n):
        adj[i].sort()
    return adj


def canonicalize_hetero_state(
    n: int,
    edges: Sequence[Edge],
    types: Sequence[int],
) -> Tuple[List[Edge], List[int], str]:
    if len(types) != n:
        raise ValueError("types length must match n")
    if len(edges) != n - 1:
        raise ValueError("tree must have n-1 edges")

    adj_list = _edges_to_adj_list(n, edges)
    perm = canonical_colored_tree_permutation(adj_list, types)
    inv = {int(old): int(new) for new, old in enumerate(perm)}

    types_can = [int(types[int(old)]) for old in perm]
    edges_can = sorted(
        (min(inv[int(u)], inv[int(v)]), max(inv[int(u)], inv[int(v)])) for u, v in edges
    )
    edges_str = ",".join(f"{u}-{v}" for u, v in edges_can)
    types_str = ",".join(str(t) for t in types_can)
    state_id = f"het:edges={edges_str};types={types_str}"
    return edges_can, types_can, state_id
