from __future__ import annotations

import csv
import time
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

from analysis.chem.exact_trees import edges_to_adj, prufer_to_edges
from analysis.growth.reporting import write_growth_txt
from analysis.io_utils import results_path
from core.complexity import compute_complexity_features_v2
from core.complexity_fdm import compute_fdm_complexity
from core.tree_canonical import canonical_relabel_tree


@dataclass
class BenchConfig:
    n_values: Tuple[int, ...] = (6, 10, 20, 40, 80, 160)
    n_trees_per_n: int = 200
    seed: int = 0


def _sample_random_tree_adj(n: int, rng: np.random.Generator) -> np.ndarray:
    if n == 1:
        return np.zeros((1, 1), dtype=float)
    if n == 2:
        adj = np.zeros((2, 2), dtype=float)
        adj[0, 1] = adj[1, 0] = 1.0
        return adj
    code = rng.integers(0, n, size=(n - 2,), dtype=int).tolist()
    edges = prufer_to_edges(code, n)
    return edges_to_adj(n, edges)


def _pcts(x: List[float]) -> Dict[str, float]:
    arr = np.asarray(x, dtype=float)
    return {
        "p50": float(np.percentile(arr, 50)),
        "p90": float(np.percentile(arr, 90)),
        "p99": float(np.percentile(arr, 99)),
        "mean": float(np.mean(arr)),
    }


def main() -> None:
    cfg = BenchConfig()
    rng = np.random.default_rng(cfg.seed)

    out_csv = results_path("invariance_bench_0.csv")
    out_txt = results_path("invariance_bench_0.txt")
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, object]] = []
    lines: List[str] = []
    lines.append("INVARIANCE-BENCH-0: canonical tree relabel overhead")
    lines.append(f"n_values={list(cfg.n_values)}, n_trees_per_n={cfg.n_trees_per_n}, seed={cfg.seed}")
    lines.append("")

    for n in cfg.n_values:
        t_canon: List[float] = []
        t_fdm: List[float] = []
        t_fdm_raw: List[float] = []

        for _ in range(cfg.n_trees_per_n):
            adj = _sample_random_tree_adj(int(n), rng)

            t0 = time.perf_counter()
            adj_c = canonical_relabel_tree(adj)
            t_canon.append(time.perf_counter() - t0)

            t1 = time.perf_counter()
            feats = compute_complexity_features_v2(adj, backend="fdm")
            _ = float(feats.total)
            t_fdm.append(time.perf_counter() - t1)

            t2 = time.perf_counter()
            _ = float(compute_fdm_complexity(adj, canonicalize_tree=False))
            t_fdm_raw.append(time.perf_counter() - t2)

        canon = _pcts(t_canon)
        fdm = _pcts(t_fdm)
        fdm_raw = _pcts(t_fdm_raw)

        overhead_mean = (canon["mean"] / fdm["mean"]) if fdm["mean"] > 0 else 0.0

        rows.append(
            {
                "N": int(n),
                "n_samples": int(cfg.n_trees_per_n),
                "t_canonicalize_mean": canon["mean"],
                "t_canonicalize_p50": canon["p50"],
                "t_canonicalize_p90": canon["p90"],
                "t_canonicalize_p99": canon["p99"],
                "t_energy_fdm_mean": fdm["mean"],
                "t_energy_fdm_p50": fdm["p50"],
                "t_energy_fdm_p90": fdm["p90"],
                "t_energy_fdm_p99": fdm["p99"],
                "t_energy_fdm_raw_mean": fdm_raw["mean"],
                "t_energy_fdm_raw_p50": fdm_raw["p50"],
                "t_energy_fdm_raw_p90": fdm_raw["p90"],
                "t_energy_fdm_raw_p99": fdm_raw["p99"],
                "overhead_canon_over_fdm_mean": float(overhead_mean),
            }
        )

        lines.append(f"[N={n}]")
        lines.append(f"  t_canonicalize_sec: mean={canon['mean']:.6f}, p50={canon['p50']:.6f}, p90={canon['p90']:.6f}, p99={canon['p99']:.6f}")
        lines.append(f"  t_energy_fdm_sec:   mean={fdm['mean']:.6f}, p50={fdm['p50']:.6f}, p90={fdm['p90']:.6f}, p99={fdm['p99']:.6f}")
        lines.append(f"  t_energy_fdm_raw:   mean={fdm_raw['mean']:.6f}, p50={fdm_raw['p50']:.6f}, p90={fdm_raw['p90']:.6f}, p99={fdm_raw['p99']:.6f}")
        lines.append(f"  overhead_mean = t_canonicalize / t_energy_fdm = {overhead_mean*100:.2f}%")
        lines.append("")

    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["N"])
        w.writeheader()
        for r in rows:
            w.writerow(r)

    out_txt = write_growth_txt("invariance_bench_0", lines)
    print("[INVARIANCE-BENCH-0] done.")
    print(f"CSV: {out_csv}")
    print(f"Summary: {out_txt}")


if __name__ == "__main__":
    main()

