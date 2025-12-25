from __future__ import annotations

from collections import deque
from typing import Dict, List, Sequence, Tuple


def _tree_centers(adj_list: List[List[int]]) -> List[int]:
    n = len(adj_list)
    if n == 0:
        return []
    deg = [len(adj_list[i]) for i in range(n)]
    leaves = [i for i, d in enumerate(deg) if d <= 1]
    removed = len(leaves)
    while removed < n:
        new_leaves: List[int] = []
        for u in leaves:
            deg[u] = 0
            for v in adj_list[u]:
                if deg[v] > 0:
                    deg[v] -= 1
                    if deg[v] == 1:
                        new_leaves.append(v)
        removed += len(new_leaves)
        leaves = new_leaves
    return leaves if leaves else [0]


def _rooted_colored_ahu_code(
    root: int,
    parent: int,
    adj_list: Sequence[Sequence[int]],
    labels: Sequence[int],
    cache: Dict[int, Tuple],
) -> Tuple:
    children = [int(v) for v in adj_list[int(root)] if int(v) != int(parent)]
    encs = [_rooted_colored_ahu_code(v, root, adj_list, labels, cache) for v in children]
    encs.sort()
    code = (int(labels[int(root)]), tuple(encs))
    cache[int(root)] = code
    return code


def canonical_colored_tree_permutation(
    adj_list: Sequence[Sequence[int]],
    labels: Sequence[int],
) -> List[int]:
    """
    Canonical relabeling permutation for colored trees using AHU encoding.

    Returns `perm` where `perm[new_index] = old_index`.
    """
    n = int(len(adj_list))
    if n != len(labels):
        raise ValueError("labels must match the number of nodes")
    if n <= 2:
        return list(range(n))

    adj_local = [list(nei) for nei in adj_list]
    centers = _tree_centers(adj_local)

    best_root = int(centers[0])
    best_code = None
    best_cache: Dict[int, Tuple] = {}
    for c in centers:
        cache: Dict[int, Tuple] = {}
        code = _rooted_colored_ahu_code(int(c), -1, adj_local, labels, cache)
        if best_code is None or code < best_code:
            best_code = code
            best_root = int(c)
            best_cache = cache

    perm: List[int] = []
    queue: deque[Tuple[int, int]] = deque([(best_root, -1)])
    while queue:
        u, parent = queue.popleft()
        perm.append(int(u))
        children = [int(v) for v in adj_local[int(u)] if int(v) != int(parent)]
        children.sort(key=lambda x: best_cache.get(int(x), ()))
        for v in children:
            queue.append((int(v), int(u)))
    return perm


def relabel_adj_list(adj_list: Sequence[Sequence[int]], perm: Sequence[int]) -> List[List[int]]:
    """
    Apply permutation `perm[new]=old` to adjacency list.
    """
    n = len(adj_list)
    if len(perm) != n:
        raise ValueError("perm length must match adjacency size")

    inv = [0] * n
    for new_i, old_i in enumerate(perm):
        inv[int(old_i)] = int(new_i)

    out: List[List[int]] = [[] for _ in range(n)]
    for new_u, old_u in enumerate(perm):
        out[new_u] = sorted(inv[int(v)] for v in adj_list[int(old_u)])
    return out
