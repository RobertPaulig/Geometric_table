from __future__ import annotations

import csv
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np

from core.complexity import compute_complexity_features_v2
from core.thermo_config import ThermoConfig, override_thermo_config


@dataclass
class PrefilterBenchConfig:
    n_graphs: int = 200
    n_nodes: int = 20


def _random_tree_adj(n: int, rng: np.random.Generator) -> np.ndarray:
    # Simple random tree via random parent assignment
    adj = np.zeros((n, n), dtype=float)
    for i in range(1, n):
        parent = int(rng.integers(0, i))
        adj[i, parent] = 1.0
        adj[parent, i] = 1.0
    return adj


def run_prefilter_benchmark(cfg: PrefilterBenchConfig):
    results_dir = Path("results")
    results_dir.mkdir(parents=True, exist_ok=True)
    csv_path = results_dir / "topo3d_prefilter_bench.csv"
    summary_path = results_dir / "topo3d_prefilter_bench_summary.txt"

    rng = np.random.default_rng(42)
    rows = []

    for idx in range(cfg.n_graphs):
        adj = _random_tree_adj(cfg.n_nodes, rng)

        # Baseline: prefilter off
        thermo_off = ThermoConfig(
            coupling_topo_3d=1.0,
            topo_3d_beta=1.0,
            topo3d_prefilter_tree=False,
        )
        with override_thermo_config(thermo_off):
            t0 = time.perf_counter()
            feats_off = compute_complexity_features_v2(adj, backend="fdm_entanglement")
            t_off = time.perf_counter() - t0

        # Prefilter on
        thermo_on = ThermoConfig(
            coupling_topo_3d=1.0,
            topo_3d_beta=1.0,
            topo3d_prefilter_tree=True,
        )
        with override_thermo_config(thermo_on):
            t1 = time.perf_counter()
            feats_on = compute_complexity_features_v2(adj, backend="fdm_entanglement")
            t_on = time.perf_counter() - t1

        speedup = t_off / t_on if t_on > 0 else float("inf")

        rows.append(
            {
                "idx": idx,
                "n": cfg.n_nodes,
                "cyclomatic": feats_off.cyclomatic,
                "total_off": feats_off.total,
                "total_on": feats_on.total,
                "t_off_sec": t_off,
                "t_on_sec": t_on,
                "speedup": speedup,
            }
        )

    fieldnames = [
        "idx",
        "n",
        "cyclomatic",
        "total_off",
        "total_on",
        "t_off_sec",
        "t_on_sec",
        "speedup",
    ]

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    cycl = [r["cyclomatic"] for r in rows]
    speedups = [r["speedup"] for r in rows]

    n = len(rows)
    n_tree = sum(1 for c in cycl if c == 0)
    median_speedup = float(np.median(speedups)) if speedups else 1.0
    max_speedup = max(speedups) if speedups else 1.0

    with summary_path.open("w", encoding="utf-8") as f:
        f.write("TOPO-PREFILTER-0 benchmark (trees)\n")
        f.write(f"n_graphs={n}, n_nodes_per_graph={cfg.n_nodes}\n")
        f.write(f"n_tree(cyclomatic==0)={n_tree}\n\n")
        for r in rows[:5]:
            f.write(
                f"idx={r['idx']}: cyclomatic={r['cyclomatic']}, "
                f"t_off={r['t_off_sec']:.6f}s, t_on={r['t_on_sec']:.6f}s, "
                f"speedup={r['speedup']:.2f}x\n"
            )
        f.write("\n")
        f.write(f"median_speedup={median_speedup:.2f}x\n")
        f.write(f"max_speedup={max_speedup:.2f}x\n")

    return csv_path, summary_path


def main() -> None:
    cfg = PrefilterBenchConfig()
    csv_path, summary_path = run_prefilter_benchmark(cfg)
    print("[TOPO-PREFILTER-0] benchmark done.")
    print(f"CSV: {csv_path}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
