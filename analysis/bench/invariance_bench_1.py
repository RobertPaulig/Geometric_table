from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np

from analysis.io_utils import results_path
from analysis.growth.reporting import write_growth_txt
from core.complexity_fdm import compute_fdm_complexity
from core.tree_canonical import canonical_relabel_tree


def _edges_from_prufer(seq: Sequence[int], n: int) -> List[Tuple[int, int]]:
    deg = [1] * n
    for x in seq:
        deg[int(x)] += 1
    seq = list(int(x) for x in seq)
    edges: List[Tuple[int, int]] = []
    for x in seq:
        leaf = next(i for i in range(n) if deg[i] == 1)
        edges.append((leaf, x))
        deg[leaf] -= 1
        deg[x] -= 1
    u, v = [i for i in range(n) if deg[i] == 1]
    edges.append((u, v))
    return edges


def _random_tree_adj(n: int, rng: np.random.Generator) -> np.ndarray:
    if n <= 1:
        return np.zeros((n, n), dtype=float)
    seq = rng.integers(0, n, size=max(0, n - 2), dtype=int).tolist()
    edges = _edges_from_prufer(seq, n)
    adj = np.zeros((n, n), dtype=float)
    for a, b in edges:
        adj[a, b] = 1.0
        adj[b, a] = 1.0
    return adj


def _compute_fdm_dense_legacy(adj: np.ndarray) -> float:
    """
    Legacy-style baseline: canonicalize via dense NxN permutation + BFS using np.where.

    This matches the pre INVARIANCE-OPT-1 approach used in invariance_bench_0.
    """
    if adj.size == 0:
        return 0.0
    n = int(adj.shape[0])
    if n <= 1:
        return 0.0
    m = int(np.sum(adj) // 2)
    if m == n - 1:
        adj = canonical_relabel_tree(adj)

    visited = np.zeros(n, dtype=bool)
    parent = np.full(n, -1, dtype=int)
    depth = np.zeros(n, dtype=int)
    root = 0
    visited[root] = True
    from collections import deque

    queue: deque[int] = deque([root])
    children: list[list[int]] = [[] for _ in range(n)]
    while queue:
        u = queue.popleft()
        neighbors = np.where(adj[u] > 0)[0]
        for v in neighbors:
            if not visited[v]:
                visited[v] = True
                parent[v] = u
                depth[v] = depth[u] + 1
                children[u].append(int(v))
                queue.append(int(v))

    subtree_size = np.ones(n, dtype=int)
    order: list[int] = []

    def dfs(u: int) -> None:
        order.append(u)
        for v in children[u]:
            dfs(v)

    dfs(root)
    for u in reversed(order):
        total = 1
        for v in children[u]:
            total += subtree_size[v]
        subtree_size[u] = total

    n_total = float(subtree_size[root])
    if n_total <= 0:
        return 0.0
    mu = subtree_size.astype(float) / n_total
    lambda_weight = 0.72
    q = 1.35
    c2_raw = 0.0
    for u in range(n):
        mu_u = mu[u]
        if mu_u <= 0.0:
            continue
        c2_raw += (lambda_weight ** depth[u]) * (mu_u ** q)
    return float(c2_raw * math.log2(1.0 + n_total))


@dataclass
class BenchRow:
    N: int
    t_fdm_dense_mean: float
    t_fdm_opt_mean: float
    t_fdm_raw_mean: float
    canon_overhead_mean: float
    speedup_mean: float


def main() -> None:
    rng = np.random.default_rng(0)
    n_values = [6, 10, 20, 40, 80, 160]
    n_trees = 200

    rows: List[BenchRow] = []
    lines: List[str] = []
    lines.append("INVARIANCE-BENCH-1: legacy dense vs optimized tree-canon path")
    lines.append(f"n_values={n_values}, n_trees_per_n={n_trees}, seed=0")
    lines.append("")

    import csv

    out_csv = results_path("invariance_bench_1.csv")
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "N",
                "n_samples",
                "t_fdm_dense_mean",
                "t_fdm_opt_mean",
                "t_fdm_raw_mean",
                "canon_overhead_mean",
                "speedup_mean",
            ],
        )
        w.writeheader()

        for n in n_values:
            t_dense: List[float] = []
            t_opt: List[float] = []
            t_raw: List[float] = []
            for _ in range(n_trees):
                adj = _random_tree_adj(int(n), rng)

                t0 = time.perf_counter()
                _ = _compute_fdm_dense_legacy(adj)
                t_dense.append(time.perf_counter() - t0)

                t1 = time.perf_counter()
                _ = compute_fdm_complexity(adj, canonicalize_tree=True)
                t_opt.append(time.perf_counter() - t1)

                t2 = time.perf_counter()
                _ = compute_fdm_complexity(adj, canonicalize_tree=False)
                t_raw.append(time.perf_counter() - t2)

            dense_mean = float(np.mean(np.asarray(t_dense)))
            opt_mean = float(np.mean(np.asarray(t_opt)))
            raw_mean = float(np.mean(np.asarray(t_raw)))
            speedup = (dense_mean / opt_mean) if opt_mean > 0 else float("inf")
            canon_overhead = ((opt_mean - raw_mean) / opt_mean) if opt_mean > 0 else 0.0

            w.writerow(
                {
                    "N": int(n),
                    "n_samples": int(n_trees),
                    "t_fdm_dense_mean": dense_mean,
                    "t_fdm_opt_mean": opt_mean,
                    "t_fdm_raw_mean": raw_mean,
                    "canon_overhead_mean": canon_overhead,
                    "speedup_mean": speedup,
                }
            )

            rows.append(
                BenchRow(
                    N=int(n),
                    t_fdm_dense_mean=dense_mean,
                    t_fdm_opt_mean=opt_mean,
                    t_fdm_raw_mean=raw_mean,
                    canon_overhead_mean=canon_overhead,
                    speedup_mean=speedup,
                )
            )

            lines.append(f"[N={int(n)}]")
            lines.append(f"  dense_mean_sec={dense_mean:.6f}")
            lines.append(f"  opt_mean_sec={opt_mean:.6f}")
            lines.append(f"  raw_mean_sec={raw_mean:.6f}")
            lines.append(f"  canon_overhead_mean={(100.0 * canon_overhead):.2f}%")
            lines.append(f"  speedup_mean={speedup:.2f}x")
            lines.append("")

    write_growth_txt("invariance_bench_1", lines)
    out_txt = results_path("invariance_bench_1.txt")
    print(f"[ANALYSIS-IO] Saved text: {out_txt}")
    print(f"[INVARIANCE-BENCH-1] done.")
    print(f"CSV: {out_csv}")
    print(f"Summary: {out_txt}")


if __name__ == "__main__":
    main()
