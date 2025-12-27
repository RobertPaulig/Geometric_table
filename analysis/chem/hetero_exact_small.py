from __future__ import annotations

import itertools
import math
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np

from analysis.chem.exact_trees import enumerate_labeled_trees
from analysis.chem.hetero_canonical import canonicalize_hetero_state
from analysis.chem.hetero_operator import hetero_energy_from_state

Edge = Tuple[int, int]

DEFAULT_VALENCE = {0: 4, 1: 3, 2: 2}
DEFAULT_RHO = {0: 0.0, 1: 0.2, 2: 0.5}
DEFAULT_ALPHA_H = 0.5


def _adj_to_edges(adj: np.ndarray) -> List[Edge]:
    n = int(adj.shape[0])
    edges: List[Edge] = []
    for i in range(n):
        for j in range(i + 1, n):
            if adj[i, j] > 0:
                edges.append((i, j))
    return edges


def _degrees(n: int, edges: Sequence[Edge]) -> List[int]:
    deg = [0] * n
    for u, v in edges:
        deg[int(u)] += 1
        deg[int(v)] += 1
    return deg


def _unique_type_assignments(type_counts: Mapping[int, int]) -> Iterable[Tuple[int, ...]]:
    base: List[int] = []
    for t, cnt in type_counts.items():
        base.extend([int(t)] * int(cnt))
    if not base:
        return []
    seen: set[Tuple[int, ...]] = set()
    for perm in itertools.permutations(base):
        if perm in seen:
            continue
        seen.add(perm)
        yield tuple(int(x) for x in perm)


def exact_distribution_for_type_counts(
    type_counts: Mapping[int, int],
    *,
    beta: float,
    rho_by_type: Mapping[int, float] | None = None,
    alpha_H: float = DEFAULT_ALPHA_H,
    valence_by_type: Mapping[int, int] | None = None,
) -> Dict[str, float]:
    n = int(sum(int(v) for v in type_counts.values()))
    if n <= 0:
        raise ValueError("type_counts must define at least one vertex")
    rho = dict(DEFAULT_RHO if rho_by_type is None else rho_by_type)
    valence = dict(DEFAULT_VALENCE if valence_by_type is None else valence_by_type)
    weights: Dict[str, float] = {}
    assignments = list(_unique_type_assignments(type_counts))
    trees = enumerate_labeled_trees(n)
    for adj in trees:
        if isinstance(adj, np.ndarray):
            edges = _adj_to_edges(adj)
        else:
            edges = list(adj)  # type: ignore[assignment]
        deg = _degrees(n, edges)
        for types in assignments:
            if any(deg[i] > valence.get(int(types[i]), 0) for i in range(n)):
                continue
            edges_can, types_can, state_id = canonicalize_hetero_state(n, edges, types)
            energy = hetero_energy_from_state(
                n,
                edges_can,
                types_can,
                rho_by_type=rho,
                alpha_H=float(alpha_H),
                valence_by_type=valence,
            )
            w = math.exp(-float(beta) * float(energy))
            weights[state_id] = weights.get(state_id, 0.0) + float(w)
    total = sum(weights.values())
    if total <= 0:
        raise RuntimeError("No valid states generated for type counts")
    return {k: float(v) / float(total) for k, v in weights.items()}


def exact_distribution_for_formula_C3H8O(
    *,
    beta: float,
    rho_by_type: Dict[int, float] | None = None,
    alpha_H: float = DEFAULT_ALPHA_H,
    valence_by_type: Dict[int, int] | None = None,
) -> Dict[str, float]:
    return exact_distribution_for_type_counts(
        {0: 3, 2: 1},
        beta=beta,
        rho_by_type=rho_by_type,
        alpha_H=alpha_H,
        valence_by_type=valence_by_type,
    )


def exact_distribution_for_formula_C2H6O(
    *,
    beta: float,
    rho_by_type: Dict[int, float] | None = None,
    alpha_H: float = DEFAULT_ALPHA_H,
    valence_by_type: Dict[int, int] | None = None,
) -> Dict[str, float]:
    return exact_distribution_for_type_counts(
        {0: 2, 2: 1},
        beta=beta,
        rho_by_type=rho_by_type,
        alpha_H=alpha_H,
        valence_by_type=valence_by_type,
    )


def exact_distribution_for_formula_C2H7N(
    *,
    beta: float,
    rho_by_type: Dict[int, float] | None = None,
    alpha_H: float = DEFAULT_ALPHA_H,
    valence_by_type: Dict[int, int] | None = None,
) -> Dict[str, float]:
    return exact_distribution_for_type_counts(
        {0: 2, 1: 1},
        beta=beta,
        rho_by_type=rho_by_type,
        alpha_H=alpha_H,
        valence_by_type=valence_by_type,
    )
