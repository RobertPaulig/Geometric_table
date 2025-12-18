from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np


Edge = Tuple[int, int]


def _adj_list_from_adj(adj: np.ndarray) -> List[List[int]]:
    n = int(adj.shape[0])
    out: List[List[int]] = [[] for _ in range(n)]
    for i in range(n):
        out[i] = [int(x) for x in np.flatnonzero(adj[i] > 0).tolist()]
    return out


def _degrees_from_adj_list(adj_list: Sequence[Sequence[int]]) -> List[int]:
    return [len(nei) for nei in adj_list]


def _is_connected(adj_list: Sequence[Sequence[int]]) -> bool:
    n = int(len(adj_list))
    if n <= 0:
        return False
    seen = [False] * n
    stack = [0]
    seen[0] = True
    while stack:
        u = stack.pop()
        for v in adj_list[u]:
            if not seen[int(v)]:
                seen[int(v)] = True
                stack.append(int(v))
    return all(seen)


@dataclass(frozen=True)
class LeafRewireMove:
    leaf: int
    old_parent: int
    new_parent: int


def leaf_rewire_moves(adj: np.ndarray, *, max_degree: int = 4) -> List[LeafRewireMove]:
    """
    Enumerate reversible leaf-rewire moves on a connected tree.

    Move:
      - pick a leaf u (deg=1)
      - detach from old_parent
      - attach to new_parent v != u, v != old_parent, with deg(v) < max_degree
    """
    if adj.ndim != 2 or adj.shape[0] != adj.shape[1]:
        raise ValueError("adj must be square")
    n = int(adj.shape[0])
    if n <= 2:
        return []

    adj_list = _adj_list_from_adj(adj)
    if not _is_connected(adj_list):
        return []

    m = int(sum(len(nei) for nei in adj_list) // 2)
    if m != n - 1:
        return []

    deg = _degrees_from_adj_list(adj_list)
    leaves = [u for u in range(n) if deg[u] == 1]
    moves: List[LeafRewireMove] = []
    for u in leaves:
        old_parent = int(adj_list[u][0])
        for v in range(n):
            if v == u or v == old_parent:
                continue
            if deg[v] >= int(max_degree):
                continue
            moves.append(LeafRewireMove(leaf=int(u), old_parent=int(old_parent), new_parent=int(v)))
    return moves


def apply_move(adj: np.ndarray, move: LeafRewireMove) -> np.ndarray:
    if adj.ndim != 2 or adj.shape[0] != adj.shape[1]:
        raise ValueError("adj must be square")
    n = int(adj.shape[0])
    u = int(move.leaf)
    a = int(move.old_parent)
    b = int(move.new_parent)
    if not (0 <= u < n and 0 <= a < n and 0 <= b < n):
        raise ValueError("move node out of bounds")
    if u == a or u == b or a == b:
        raise ValueError("invalid move")

    out = np.array(adj, dtype=float, copy=True)
    out[u, a] = 0.0
    out[a, u] = 0.0
    out[u, b] = 1.0
    out[b, u] = 1.0
    return out


@dataclass
class EquilibrateStats:
    steps: int
    proposals: int
    accepted: int
    mean_moves: float
    p90_moves: float
    mean_log_qratio: float
    p90_log_qratio: float

    @property
    def accept_rate(self) -> float:
        return (float(self.accepted) / float(self.proposals)) if self.proposals > 0 else 0.0


def fixedn_mcmc_equilibrate(
    adj0: np.ndarray,
    *,
    energy_fn: Callable[[np.ndarray], float],
    steps: int,
    burnin: int = 0,
    thin: int = 1,
    rng: Optional[np.random.Generator] = None,
    max_degree: int = 4,
    temperature_T: float = 1.0,
    lam: float = 1.0,
    progress_cb: Optional[Callable[[int, int, int], None]] = None,
) -> Tuple[np.ndarray, EquilibrateStats]:
    """
    Fixed-N MCMC equilibration on labeled trees with Hastings correction.

    Target: π(G) ∝ exp(-lam * E(G) / T)
    """
    if rng is None:
        rng = np.random.default_rng()
    steps = int(steps)
    burnin = int(burnin)
    thin = max(1, int(thin))

    if steps <= 0:
        stats = EquilibrateStats(
            steps=0,
            proposals=0,
            accepted=0,
            mean_moves=0.0,
            p90_moves=0.0,
            mean_log_qratio=0.0,
            p90_log_qratio=0.0,
        )
        return np.array(adj0, dtype=float, copy=True), stats

    T = float(temperature_T)
    beta = (float(lam) / T) if T > 0 else float("inf")

    adj = np.array(adj0, dtype=float, copy=True)
    e_x = float(energy_fn(adj))

    proposals = 0
    accepted = 0
    moves_sizes: List[float] = []
    log_qratios: List[float] = []

    # Optionally return a (uniform) random post-burnin sample instead of the last state.
    # This reduces sensitivity to the final step and allows thinning without storing all samples.
    use_reservoir = burnin > 0 or thin > 1
    reservoir_edges: Optional[Tuple[Edge, ...]] = None
    reservoir_n = 0

    def _edges_key(a: np.ndarray) -> Tuple[Edge, ...]:
        n_local = int(a.shape[0])
        edges: List[Edge] = []
        for i in range(n_local):
            for j in range(i + 1, n_local):
                if a[i, j] > 0:
                    edges.append((int(i), int(j)))
        return tuple(edges)

    for step in range(steps):
        moves_x = leaf_rewire_moves(adj, max_degree=max_degree)
        if not moves_x:
            if progress_cb is not None:
                progress_cb(step + 1, proposals, accepted)
            continue
        move = moves_x[int(rng.integers(0, len(moves_x)))]
        adj_y = apply_move(adj, move)
        moves_y = leaf_rewire_moves(adj_y, max_degree=max_degree)
        if not moves_y:
            if progress_cb is not None:
                progress_cb(step + 1, proposals, accepted)
            continue

        q_x = 1.0 / float(len(moves_x))
        q_y = 1.0 / float(len(moves_y))
        log_qratio = math.log(q_y / q_x)

        e_y = float(energy_fn(adj_y))
        proposals += 1
        log_alpha = -beta * float(e_y - e_x) + log_qratio
        alpha = 1.0 if log_alpha >= 0 else math.exp(log_alpha)
        if float(rng.random()) < alpha:
            adj = adj_y
            e_x = e_y
            accepted += 1

        moves_sizes.append(float(len(moves_x)))
        log_qratios.append(float(log_qratio))

        if progress_cb is not None:
            progress_cb(step + 1, proposals, accepted)

        if use_reservoir and step >= burnin and ((step - burnin) % thin == 0):
            reservoir_n += 1
            if reservoir_edges is None or int(rng.integers(0, reservoir_n)) == 0:
                reservoir_edges = _edges_key(adj)

    def _pctl(arr: Sequence[float], q: float) -> float:
        if not arr:
            return 0.0
        return float(np.percentile(np.asarray(arr, dtype=float), q))

    stats = EquilibrateStats(
        steps=int(steps),
        proposals=int(proposals),
        accepted=int(accepted),
        mean_moves=float(np.mean(moves_sizes)) if moves_sizes else 0.0,
        p90_moves=_pctl(moves_sizes, 90),
        mean_log_qratio=float(np.mean(log_qratios)) if log_qratios else 0.0,
        p90_log_qratio=_pctl(log_qratios, 90),
    )
    if reservoir_edges is not None:
        n_local = int(adj.shape[0])
        out_adj = np.zeros((n_local, n_local), dtype=float)
        for a, b in reservoir_edges:
            out_adj[int(a), int(b)] = 1.0
            out_adj[int(b), int(a)] = 1.0
        return out_adj, stats
    return adj, stats
