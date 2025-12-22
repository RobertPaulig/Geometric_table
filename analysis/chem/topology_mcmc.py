from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple, Union

import numpy as np

from analysis.chem.core2_fit import compute_p_pred
from core.complexity import compute_complexity_features_v2
from core.tree_canonical import canonical_tree_permutation, relabel_adj_list


Edge = Tuple[int, int]
TreeTopoKey = Tuple[Edge, ...]


_GLOBAL_ENERGY_CACHE_BY_TOPOLOGY: Dict[Tuple[str, object], float] = {}


def reset_global_energy_cache(*, clear_values: bool = False) -> None:
    """
    Reset global energy cache state used by analysis MCMC.

    Per-run hit/miss counters are local to `run_fixed_n_tree_mcmc`; this helper
    exists for diagnostics/reproducibility.

    If `clear_values=True`, removes all cached energies.
    """
    if clear_values:
        _GLOBAL_ENERGY_CACHE_BY_TOPOLOGY.clear()


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


def edges_to_adj_list(n: int, edges: Sequence[Edge]) -> List[List[int]]:
    adj_list: List[List[int]] = [[] for _ in range(int(n))]
    for i, j in edges:
        a, b = _canonical_edge(i, j)
        adj_list[a].append(b)
        adj_list[b].append(a)
    for u in range(int(n)):
        adj_list[u].sort()
    return adj_list


def tree_topology_edge_key_from_edges(n: int, edges: Sequence[Edge]) -> TreeTopoKey:
    """
    Permutation-invariant tree topology key as canonical edge tuple.

    The tuple is hashable and is the preferred internal key for caching.
    """
    adj_list = edges_to_adj_list(int(n), edges)
    perm = canonical_tree_permutation(adj_list)
    rel = relabel_adj_list(adj_list, perm)
    out_edges: List[Edge] = []
    for u in range(int(n)):
        for v in rel[u]:
            if int(v) > int(u):
                out_edges.append((int(u), int(v)))
    out_edges.sort()
    return tuple(out_edges)


def tree_topology_id_from_edge_key(edge_key: TreeTopoKey) -> str:
    return "tree:" + ",".join(f"{a}-{b}" for a, b in edge_key)


def tree_topology_id_from_edges(n: int, edges: Sequence[Edge]) -> str:
    return tree_topology_id_from_edge_key(tree_topology_edge_key_from_edges(int(n), edges))


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
    t_move_avg: float = 0.0
    t_energy_avg: float = 0.0
    t_canon_avg: float = 0.0

    @property
    def energy_cache_hit_rate(self) -> float:
        total = int(self.energy_cache_hits) + int(self.energy_cache_misses)
        return (float(self.energy_cache_hits) / float(total)) if total > 0 else 0.0


class _ReservoirStats:
    """
    Bounded-memory streaming stats for long runs.

    Tracks mean and an approximate p90 using reservoir sampling.
    """

    def __init__(self, *, reservoir_size: int, seed: int) -> None:
        self._n = 0
        self._mean = 0.0
        self._reservoir_size = int(reservoir_size)
        self._rng = np.random.default_rng(int(seed))
        self._reservoir: List[float] = []

    def add(self, x: float) -> None:
        self._n += 1
        # running mean
        self._mean += (float(x) - self._mean) / float(self._n)
        if self._reservoir_size <= 0:
            return
        if len(self._reservoir) < self._reservoir_size:
            self._reservoir.append(float(x))
            return
        # reservoir sampling
        j = int(self._rng.integers(0, self._n))
        if j < self._reservoir_size:
            self._reservoir[j] = float(x)

    @property
    def count(self) -> int:
        return int(self._n)

    @property
    def mean(self) -> float:
        return float(self._mean) if self._n > 0 else 0.0

    def pctl(self, q: float) -> float:
        if not self._reservoir:
            return 0.0
        return float(np.percentile(np.asarray(self._reservoir, dtype=float), q))


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
    topology_key_fn_edges: Optional[Callable[[int, Sequence[Edge]], object]] = None,
    start_edges: Optional[Sequence[Edge]] = None,
    energy_cache: Optional[MutableMapping[Tuple[str, object], float]] = None,
    progress: Optional[callable] = None,
    profile_every: int = 0,
    step_heartbeat_every: int = 0,
    step_heartbeat: Optional[Callable[[Mapping[str, float]], None]] = None,
    sample_callback: Optional[Callable[[str, float], None]] = None,
    collect_samples: bool = True,
    collect_move_stats: bool = False,
    move_stats_reservoir: int = 10_000,
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
    t0_heartbeat = t0_total

    # Cache energies for speed (two-level):
    # - global cache by (backend, topology) when topology_classifier is provided (label-invariant energy on trees)
    # - fallback per-run cache by labeled edge tuple otherwise
    e_cache_edges: Dict[Tuple[Edge, ...], float] = {}
    topo_cache = energy_cache if energy_cache is not None else _GLOBAL_ENERGY_CACHE_BY_TOPOLOGY
    cache_hits = 0
    cache_misses = 0
    seen_topologies: set[object] = set()
    topo_id_memo: Dict[object, str] = {}
    topo_key_memo: Dict[Tuple[Edge, ...], object] = {}

    profile_this_step = False
    t_move_total = 0.0
    t_energy_total = 0.0
    t_canon_total = 0.0
    profiled_steps = 0
    profiled_energy_calls = 0

    def energy(e: Sequence[Edge]) -> float:
        nonlocal cache_hits, cache_misses
        nonlocal t_energy_total, t_canon_total, profiled_energy_calls, profile_this_step
        key_edges = tuple(sorted(_canonical_edge(i, j) for i, j in e))
        if topology_key_fn_edges is not None:
            if key_edges in topo_key_memo:
                topo_key = topo_key_memo[key_edges]
            else:
                t_c0 = time.perf_counter() if (profile_every > 0 and profile_this_step) else 0.0
                topo_key = topology_key_fn_edges(int(n), key_edges)
                if profile_every > 0 and profile_this_step:
                    t_canon_total += time.perf_counter() - t_c0
                topo_key_memo[key_edges] = topo_key

            # Track per-run "first time seen" regardless of cache warm-up.
            if topo_key in seen_topologies:
                cache_hits += 1
            else:
                cache_misses += 1
                seen_topologies.add(topo_key)

            key_topo = (str(backend), topo_key)
            t_e0 = time.perf_counter() if (profile_every > 0 and profile_this_step) else 0.0
            if key_topo in topo_cache:
                val = float(topo_cache[key_topo])
            else:
                val = compute_energy_for_tree(key_edges, backend=backend)
                topo_cache[key_topo] = float(val)
            if profile_every > 0 and profile_this_step:
                t_energy_total += time.perf_counter() - t_e0
                profiled_energy_calls += 1
            return float(val)

        if topology_classifier is not None:
            t_c0 = time.perf_counter() if (profile_every > 0 and profile_this_step) else 0.0
            # Topology-keyed cache gives maximal reuse for label-invariant tree energies.
            adj = edges_to_adj(n, key_edges)
            topo = str(topology_classifier(adj))
            if profile_every > 0 and profile_this_step:
                t_canon_total += time.perf_counter() - t_c0

            # Track per-run "first time seen" regardless of cache warm-up.
            if topo in seen_topologies:
                cache_hits += 1
            else:
                cache_misses += 1
                seen_topologies.add(topo)

            key_topo = (str(backend), topo)
            t_e0 = time.perf_counter() if (profile_every > 0 and profile_this_step) else 0.0
            if key_topo in topo_cache:
                val = float(topo_cache[key_topo])
            else:
                val = compute_energy_for_tree(key_edges, backend=backend)
                topo_cache[key_topo] = float(val)
            if profile_every > 0 and profile_this_step:
                t_energy_total += time.perf_counter() - t_e0
                profiled_energy_calls += 1
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
    moves_sizes: List[int] = [] if collect_move_stats else []
    log_qratios: List[float] = [] if collect_move_stats else []
    move_size_stats = _ReservoirStats(reservoir_size=int(move_stats_reservoir), seed=int(seed) + 1_001)
    log_qratio_stats = _ReservoirStats(reservoir_size=int(move_stats_reservoir), seed=int(seed) + 2_003)
    accepted = 0
    proposals = 0

    for step in range(int(steps)):
        profile_this_step = bool(profile_every) and (int(step) % int(profile_every) == 0)
        t_m0 = time.perf_counter() if profile_this_step else 0.0

        moves_x = enumerate_leaf_rewire_moves(n, edges, max_valence=max_valence)
        if not moves_x:
            continue
        move = moves_x[int(rng.integers(0, len(moves_x)))]
        edges_y = apply_leaf_rewire_move(edges, move)
        assert is_tree(n, edges_y)

        moves_y = enumerate_leaf_rewire_moves(n, edges_y, max_valence=max_valence)
        if profile_this_step:
            t_move_total += time.perf_counter() - t_m0
            profiled_steps += 1
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

        if collect_move_stats:
            moves_sizes.append(int(len(moves_x)))
            log_qratios.append(float(log_qratio))
        else:
            move_size_stats.add(float(len(moves_x)))
            log_qratio_stats.add(float(log_qratio))

        if progress is not None:
            progress(1)

        if (
            step_heartbeat is not None
            and int(step_heartbeat_every) > 0
            and int(step) > 0
            and (int(step) % int(step_heartbeat_every) == 0)
        ):
            now = time.perf_counter()
            dt = now - t0_heartbeat
            steps_per_sec = (float(step_heartbeat_every) / dt) if dt > 0 else 0.0
            t0_heartbeat = now
            step_heartbeat(
                {
                    "step": float(step),
                    "steps_total": float(steps),
                    "accept_rate": (float(accepted) / float(proposals)) if proposals > 0 else 0.0,
                    "energy_cache_hit_rate": float(cache_hits) / float(cache_hits + cache_misses)
                    if (cache_hits + cache_misses) > 0
                    else 0.0,
                    "energy_cache_misses_seen": float(cache_misses),
                    "heartbeat_steps_per_sec": float(steps_per_sec),
                }
            )

        if step >= burnin and ((step - burnin) % max(1, thin) == 0):
            if topology_key_fn_edges is not None:
                topo_key = topology_key_fn_edges(int(n), edges)
                if topo_key in topo_id_memo:
                    topo = topo_id_memo[topo_key]
                else:
                    if isinstance(topo_key, tuple):
                        topo = tree_topology_id_from_edge_key(topo_key)  # type: ignore[arg-type]
                    else:
                        topo = str(topo_key)
                    topo_id_memo[topo_key] = topo
            elif topology_classifier is None:
                deg = degrees(n, edges)
                topo = classify_tree_topology_by_deg_sorted(sorted(deg))
            else:
                adj = edges_to_adj(n, edges)
                topo = str(topology_classifier(adj))
            if sample_callback is not None:
                sample_callback(topo, float(e_x))
            if collect_samples:
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

    t_move_avg = float(t_move_total) / float(profiled_steps) if profiled_steps > 0 else 0.0
    t_energy_avg = float(t_energy_total) / float(profiled_energy_calls) if profiled_energy_calls > 0 else 0.0
    t_canon_avg = float(t_canon_total) / float(profiled_energy_calls) if profiled_energy_calls > 0 else 0.0

    summary = MCMCSummary(
        n=int(n),
        steps=int(steps),
        burnin=int(burnin),
        thin=int(thin),
        accepted=int(accepted),
        proposals=int(proposals),
        mean_moves=float(np.mean(moves_sizes)) if collect_move_stats and moves_sizes else move_size_stats.mean,
        p90_moves=_pctl(moves_sizes, 90) if collect_move_stats else move_size_stats.pctl(90),
        mean_log_qratio=float(np.mean(log_qratios))
        if collect_move_stats and log_qratios
        else log_qratio_stats.mean,
        p90_log_qratio=_pctl(log_qratios, 90) if collect_move_stats else log_qratio_stats.pctl(90),
        p_topology=p_topo,
        energy_cache_hits=int(cache_hits),
        energy_cache_misses=int(cache_misses),
        steps_per_sec=(float(steps) / max(1e-9, (time.perf_counter() - t0_total))),
        t_move_avg=float(t_move_avg),
        t_energy_avg=float(t_energy_avg),
        t_canon_avg=float(t_canon_avg),
    )
    return samples, summary
