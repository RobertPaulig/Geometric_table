from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import numpy as np

from analysis.chem.core2_fit import compute_p_pred
from core.complexity import compute_complexity_features_v2


Edge = Tuple[int, int]


_GLOBAL_ENERGY_CACHE_BY_TOPOLOGY: Dict[Tuple[str, str], float] = {}


def _canonical_edge(i: int, j: int) -> Edge:
    a, b = (int(i), int(j))
    return (a, b) if a < b else (b, a)


def edges_to_adj(n: int, edges: Sequence[Edge]) -> np.ndarray:
    adj = np.zeros((n, n), dtype=float)
    for i, j in edges:
        if i == j:
            continue
        a, b = _canonical_edge(i, j)
        adj[a, b] = 1.0
        adj[b, a] = 1.0
    return adj


def degrees(n: int, edges: Sequence[Edge]) -> List[int]:
    deg = [0] * n
    for i, j in edges:
        deg[int(i)] += 1
        deg[int(j)] += 1
    return deg


def is_tree(n: int, edges: Sequence[Edge]) -> bool:
    if n <= 0:
        return False
    if len(edges) != n - 1:
        return False
    # connectivity via BFS
    adj_list: List[List[int]] = [[] for _ in range(n)]
    for i, j in edges:
        a, b = _canonical_edge(i, j)
        adj_list[a].append(b)
        adj_list[b].append(a)
    seen = [False] * n
    stack = [0]
    seen[0] = True
    while stack:
        u = stack.pop()
        for v in adj_list[u]:
            if not seen[v]:
                seen[v] = True
                stack.append(v)
    return all(seen)


@dataclass(frozen=True)
class LeafRewireMove:
    leaf: int
    old_parent: int
    new_parent: int


def enumerate_leaf_rewire_moves(
    n: int,
    edges: Sequence[Edge],
    *,
    max_valence: int = 4,
) -> List[LeafRewireMove]:
    deg = degrees(n, edges)
    # build parent mapping for leaves (tree)
    adj_list: List[List[int]] = [[] for _ in range(n)]
    edge_set = {_canonical_edge(i, j) for (i, j) in edges}
    for a, b in edge_set:
        adj_list[a].append(b)
        adj_list[b].append(a)

    leaves = [u for u in range(n) if deg[u] == 1]
    moves: List[LeafRewireMove] = []
    for u in leaves:
        old_parent = adj_list[u][0]
        for v in range(n):
            if v == u or v == old_parent:
                continue
            if deg[v] >= max_valence:
                continue
            moves.append(LeafRewireMove(leaf=u, old_parent=old_parent, new_parent=v))
    return moves


def apply_leaf_rewire_move(edges: Sequence[Edge], move: LeafRewireMove) -> List[Edge]:
    edge_set = {_canonical_edge(i, j) for (i, j) in edges}
    edge_set.discard(_canonical_edge(move.leaf, move.old_parent))
    edge_set.add(_canonical_edge(move.leaf, move.new_parent))
    return sorted(edge_set)


def classify_tree_topology_by_deg_sorted(deg_sorted: Sequence[int]) -> str:
    seq = tuple(int(x) for x in deg_sorted)
    if seq == (1, 1, 2, 2):
        return "n_butane"
    if seq == (1, 1, 1, 3):
        return "isobutane"
    if seq == (1, 1, 2, 2, 2):
        return "n_pentane"
    if seq == (1, 1, 1, 2, 3):
        return "isopentane"
    if seq == (1, 1, 1, 1, 4):
        return "neopentane"
    return "other"


def compute_energy_for_tree(edges: Sequence[Edge], *, backend: str) -> float:
    n = 0
    for i, j in edges:
        n = max(n, int(i) + 1, int(j) + 1)
    adj = edges_to_adj(n, edges)
    feats = compute_complexity_features_v2(adj, backend=backend)
    return float(feats.total)


@dataclass
class MCMCSummary:
    n: int
    steps: int
    burnin: int
    thin: int
    accepted: int
    proposals: int
    mean_moves: float
    p90_moves: float
    mean_log_qratio: float
    p90_log_qratio: float
    p_topology: Dict[str, float]
    energy_cache_hits: int = 0
    energy_cache_misses: int = 0
    steps_per_sec: float = 0.0

    @property
    def energy_cache_hit_rate(self) -> float:
        total = int(self.energy_cache_hits) + int(self.energy_cache_misses)
        return (float(self.energy_cache_hits) / float(total)) if total > 0 else 0.0


def run_fixed_n_tree_mcmc(
    *,
    n: int,
    steps: int,
    burnin: int,
    thin: int,
    backend: str,
    lam: float,
    temperature_T: float,
    seed: int,
    max_valence: int = 4,
    topology_classifier: Optional[Callable[[np.ndarray], str]] = None,
    start_edges: Optional[Sequence[Edge]] = None,
    energy_cache: Optional[MutableMapping[Tuple[str, str], float]] = None,
    progress: Optional[callable] = None,
) -> Tuple[List[Dict[str, object]], MCMCSummary]:
    """
    Fixed-N MCMC on labeled trees using reversible leaf-rewire moves with Hastings correction.

    Target: π(G) ∝ exp(-lam * E(G) / T)
    """
    rng = np.random.default_rng(int(seed))

    # Start from a path tree unless an explicit start state is provided.
    if start_edges is None:
        edges: List[Edge] = [(_canonical_edge(i, i + 1)) for i in range(n - 1)]
    else:
        edges = [(_canonical_edge(int(i), int(j))) for (i, j) in start_edges]
    assert is_tree(n, edges)

    t0_total = time.perf_counter()

    # Cache energies for speed (two-level):
    # - global cache by (backend, topology) when topology_classifier is provided (label-invariant energy on trees)
    # - fallback per-run cache by labeled edge tuple otherwise
    e_cache_edges: Dict[Tuple[Edge, ...], float] = {}
    topo_cache = energy_cache if energy_cache is not None else _GLOBAL_ENERGY_CACHE_BY_TOPOLOGY
    cache_hits = 0
    cache_misses = 0

    def energy(e: Sequence[Edge]) -> float:
        nonlocal cache_hits, cache_misses
        key_edges = tuple(sorted(_canonical_edge(i, j) for i, j in e))
        if topology_classifier is not None:
            # Topology-keyed cache gives maximal reuse for label-invariant tree energies.
            adj = edges_to_adj(n, key_edges)
            topo = str(topology_classifier(adj))
            key_topo = (str(backend), topo)
            if key_topo in topo_cache:
                cache_hits += 1
                return float(topo_cache[key_topo])
            cache_misses += 1
            val = compute_energy_for_tree(key_edges, backend=backend)
            topo_cache[key_topo] = float(val)
            return float(val)

        if key_edges in e_cache_edges:
            cache_hits += 1
            return float(e_cache_edges[key_edges])
        cache_misses += 1
        val = compute_energy_for_tree(key_edges, backend=backend)
        e_cache_edges[key_edges] = float(val)
        return float(val)

    T = float(temperature_T)
    beta = float(lam) / T if T > 0 else float("inf")

    e_x = energy(edges)
    samples: List[Dict[str, object]] = []
    moves_sizes: List[int] = []
    log_qratios: List[float] = []
    accepted = 0
    proposals = 0

    for step in range(int(steps)):
        moves_x = enumerate_leaf_rewire_moves(n, edges, max_valence=max_valence)
        if not moves_x:
            continue
        move = moves_x[int(rng.integers(0, len(moves_x)))]
        edges_y = apply_leaf_rewire_move(edges, move)
        assert is_tree(n, edges_y)

        moves_y = enumerate_leaf_rewire_moves(n, edges_y, max_valence=max_valence)
        q_x = 1.0 / float(len(moves_x))
        q_y = 1.0 / float(len(moves_y)) if moves_y else 0.0
        if q_y <= 0.0:
            continue
        log_qratio = math.log(q_y / q_x)

        e_y = energy(edges_y)
        proposals += 1
        log_alpha = -beta * float(e_y - e_x) + log_qratio
        alpha = 1.0 if log_alpha >= 0 else math.exp(log_alpha)
        if float(rng.random()) < alpha:
            edges = edges_y
            e_x = e_y
            accepted += 1

        moves_sizes.append(len(moves_x))
        log_qratios.append(log_qratio)

        if progress is not None:
            progress(1)

        if step >= burnin and ((step - burnin) % max(1, thin) == 0):
            if topology_classifier is None:
                deg = degrees(n, edges)
                topo = classify_tree_topology_by_deg_sorted(sorted(deg))
            else:
                adj = edges_to_adj(n, edges)
                topo = str(topology_classifier(adj))
            samples.append(
                {
                    "step": int(step),
                    "topology": topo,
                    "deg_sorted": str(sorted(degrees(n, edges))),
                    "energy": float(e_x),
                    "accepted": int(accepted),
                    "proposals": int(proposals),
                }
            )

    def _pctl(arr: Sequence[float], q: float) -> float:
        if not arr:
            return 0.0
        return float(np.percentile(np.asarray(arr, dtype=float), q))

    topo_counts: Dict[str, int] = {}
    for s in samples:
        topo = str(s["topology"])
        topo_counts[topo] = topo_counts.get(topo, 0) + 1
    total = sum(topo_counts.values()) or 1
    p_topo = {k: (v / float(total)) for k, v in topo_counts.items()}

    summary = MCMCSummary(
        n=int(n),
        steps=int(steps),
        burnin=int(burnin),
        thin=int(thin),
        accepted=int(accepted),
        proposals=int(proposals),
        mean_moves=float(np.mean(moves_sizes)) if moves_sizes else 0.0,
        p90_moves=_pctl(moves_sizes, 90),
        mean_log_qratio=float(np.mean(log_qratios)) if log_qratios else 0.0,
        p90_log_qratio=_pctl(log_qratios, 90),
        p_topology=p_topo,
        energy_cache_hits=int(cache_hits),
        energy_cache_misses=int(cache_misses),
        steps_per_sec=(float(steps) / max(1e-9, (time.perf_counter() - t0_total))),
    )
    return samples, summary
