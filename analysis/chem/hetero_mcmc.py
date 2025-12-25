from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Dict, List, Sequence, Tuple

import numpy as np

from analysis.chem.hetero_canonical import canonicalize_hetero_state

Edge = Tuple[int, int]


def _canonical_edge(i: int, j: int) -> Edge:
    a, b = int(i), int(j)
    return (a, b) if a < b else (b, a)


def _build_adj_list(n: int, edges: Sequence[Edge]) -> List[List[int]]:
    adj: List[List[int]] = [[] for _ in range(n)]
    for u, v in edges:
        a, b = _canonical_edge(u, v)
        adj[a].append(b)
        adj[b].append(a)
    for i in range(n):
        adj[i].sort()
    return adj


def _degrees_from_adj(adj: Sequence[Sequence[int]]) -> List[int]:
    return [len(nei) for nei in adj]


def _validate_valence(deg: Sequence[int], types: Sequence[int], valence_by_type: Dict[int, int]) -> bool:
    for idx, d in enumerate(deg):
        val = int(valence_by_type.get(int(types[idx]), 0))
        if d > val:
            return False
    return True


def _leaf_rewire_candidates(
    adj: Sequence[Sequence[int]],
    types: Sequence[int],
    valence_by_type: Dict[int, int],
) -> List[Tuple[int, int, int]]:
    n = len(adj)
    deg = _degrees_from_adj(adj)
    candidates: List[Tuple[int, int, int]] = []
    valence_default = {int(k): int(v) for k, v in valence_by_type.items()}
    for leaf in range(n):
        if deg[leaf] != 1:
            continue
        old_parent = int(adj[leaf][0])
        for new_parent in range(n):
            if new_parent == leaf or new_parent == old_parent:
                continue
            limit = valence_default.get(int(types[new_parent]), 0)
            if deg[new_parent] >= limit:
                continue
            candidates.append((leaf, old_parent, new_parent))
    return candidates


def _swap_candidates(
    adj: Sequence[Sequence[int]],
    types: Sequence[int],
    valence_by_type: Dict[int, int],
) -> List[Tuple[int, int]]:
    deg = _degrees_from_adj(adj)
    n = len(adj)
    candidates: List[Tuple[int, int]] = []
    for i in range(n):
        for j in range(i + 1, n):
            if int(types[i]) == int(types[j]):
                continue
            val_i = int(valence_by_type.get(int(types[j]), 0))
            val_j = int(valence_by_type.get(int(types[i]), 0))
            if deg[i] > val_i or deg[j] > val_j:
                continue
            candidates.append((i, j))
    return candidates


def _apply_rewire(
    n: int,
    edges: Sequence[Edge],
    leaf: int,
    old_parent: int,
    new_parent: int,
) -> Tuple[Tuple[Edge, ...], List[List[int]]]:
    edge_set = {_canonical_edge(u, v) for (u, v) in edges}
    edge_set.discard(_canonical_edge(leaf, old_parent))
    edge_set.add(_canonical_edge(leaf, new_parent))
    new_edges = tuple(sorted(edge_set))
    adj = _build_adj_list(n, new_edges)
    return new_edges, adj


def _apply_swap(types: Sequence[int], i: int, j: int) -> Tuple[int, ...]:
    types_new = list(types)
    types_new[i], types_new[j] = types_new[j], types_new[i]
    return tuple(types_new)


@dataclass(frozen=True)
class HeteroState:
    n: int
    edges: Tuple[Edge, ...]
    types: Tuple[int, ...]


@dataclass
class HeteroMCMCSummary:
    steps: int
    burnin: int
    thin: int
    accepted: int
    proposals: int
    p_accept: float


def step_hetero_mcmc(
    state: HeteroState,
    *,
    rng: np.random.Generator,
    p_rewire: float,
    valence_by_type: Dict[int, int],
) -> Tuple[HeteroState, float, bool]:
    adj = _build_adj_list(state.n, state.edges)
    move_choice = float(rng.random())

    if move_choice < float(p_rewire):
        candidates = _leaf_rewire_candidates(adj, state.types, valence_by_type)
        total_moves = len(candidates)
        if total_moves == 0:
            return state, 0.0, False
        leaf, old_parent, new_parent = candidates[int(rng.integers(0, total_moves))]
        new_edges, adj_new = _apply_rewire(state.n, state.edges, leaf, old_parent, new_parent)
        candidates_y = _leaf_rewire_candidates(adj_new, state.types, valence_by_type)
        total_moves_y = len(candidates_y)
        if total_moves_y == 0:
            return state, 0.0, False
        log_qratio = math.log(float(total_moves) / float(total_moves_y))
        return HeteroState(n=state.n, edges=new_edges, types=state.types), float(log_qratio), True

    candidates_swap = _swap_candidates(adj, state.types, valence_by_type)
    total_swaps = len(candidates_swap)
    if total_swaps == 0:
        return state, 0.0, False
    i, j = candidates_swap[int(rng.integers(0, total_swaps))]
    types_new = _apply_swap(state.types, i, j)
    candidates_swap_y = _swap_candidates(adj, types_new, valence_by_type)
    total_swaps_y = len(candidates_swap_y)
    if total_swaps_y == 0:
        return state, 0.0, False
    log_qratio = math.log(float(total_swaps) / float(total_swaps_y))
    return HeteroState(n=state.n, edges=state.edges, types=types_new), float(log_qratio), True


def run_hetero_mcmc(
    *,
    init: HeteroState,
    steps: int,
    burnin: int,
    thin: int,
    beta: float,
    rng_seed: int,
    energy_fn: Callable[[HeteroState], float],
    p_rewire: float,
    valence_by_type: Dict[int, int],
) -> Tuple[List[Dict[str, object]], HeteroMCMCSummary]:
    state = init
    rng = np.random.default_rng(int(rng_seed))
    energy_x = float(energy_fn(state))
    samples: List[Dict[str, object]] = []
    accepted = 0
    proposals = 0
    for step in range(int(steps)):
        new_state, log_qratio, moved = step_hetero_mcmc(
            state,
            rng=rng,
            p_rewire=p_rewire,
            valence_by_type=valence_by_type,
        )
        if not moved:
            continue
        energy_y = float(energy_fn(new_state))
        proposals += 1
        log_alpha = -float(beta) * (energy_y - energy_x) + float(log_qratio)
        if log_alpha >= 0 or float(rng.random()) < math.exp(log_alpha):
            state = new_state
            energy_x = energy_y
            accepted += 1

        if step >= burnin and ((step - burnin) % max(1, thin) == 0):
            edges_can, types_can, state_id = canonicalize_hetero_state(
                state.n, list(state.edges), list(state.types)
            )
            samples.append(
                {
                    "state_id": state_id,
                    "edges": edges_can,
                    "types": types_can,
                    "energy": float(energy_x),
                }
            )

    summary = HeteroMCMCSummary(
        steps=int(steps),
        burnin=int(burnin),
        thin=int(thin),
        accepted=int(accepted),
        proposals=int(proposals),
        p_accept=(float(accepted) / float(proposals)) if proposals > 0 else 0.0,
    )
    return samples, summary
