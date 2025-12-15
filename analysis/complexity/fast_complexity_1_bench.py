from __future__ import annotations

import csv
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from analysis.io_utils import results_path
from core.complexity import compute_complexity_features_v2
from core.thermo_config import ThermoConfig, override_thermo_config


@dataclass
class GraphSpec:
    name: str
    n_min: int
    n_max: int
    cyclic: bool


GRAPH_SPECS: Tuple[GraphSpec, ...] = (
    GraphSpec(name="small_tree", n_min=5, n_max=10, cyclic=False),
    GraphSpec(name="small_cyclic", n_min=5, n_max=10, cyclic=True),
    GraphSpec(name="medium_tree", n_min=20, n_max=40, cyclic=False),
    GraphSpec(name="medium_cyclic", n_min=20, n_max=40, cyclic=True),
    GraphSpec(name="large_tree", n_min=60, n_max=100, cyclic=False),
    GraphSpec(name="large_cyclic", n_min=60, n_max=100, cyclic=True),
)


def _make_random_graph(n: int, cyclic: bool, rng: np.random.Generator) -> np.ndarray:
    """
    Простейший генератор графов:
    - если cyclic=False: случайное дерево (спаннинг-три).
    - если cyclic=True: дерево + несколько случайных дополнительных рёбер.
    """
    adj = np.zeros((n, n), dtype=int)
    # сначала строим дерево
    for i in range(1, n):
        j = int(rng.integers(0, i))
        adj[i, j] = adj[j, i] = 1
    if cyclic:
        extra_edges = max(1, n // 5)
        for _ in range(extra_edges):
            u = int(rng.integers(0, n))
            v = int(rng.integers(0, n))
            if u == v:
                continue
            adj[u, v] = adj[v, u] = 1
    return adj


def _run_batch_for_spec(
    spec: GraphSpec,
    mode: str,
    n_graphs: int,
    rng: np.random.Generator,
) -> List[Dict[str, float]]:
    rows: List[Dict[str, float]] = []
    for _ in range(n_graphs):
        n = int(rng.integers(spec.n_min, spec.n_max + 1))
        adj = _make_random_graph(n, cyclic=spec.cyclic, rng=rng)
        t0 = time.perf_counter()
        _ = compute_complexity_features_v2(adj, backend="fdm_entanglement")
        dt = time.perf_counter() - t0
        rows.append(
            {
                "mode": mode,
                "graph_class": spec.name,
                "n_nodes": n,
                "cyclic": int(spec.cyclic),
                "runtime_sec": dt,
            }
        )
    return rows


def run_fast_complexity_1_bench() -> Tuple[Path, Path]:
    """
    Сравнение baseline/optimized для compute_complexity_features_v2(fdm_entanglement)
    на наборе случайных графов.
    """
    results: List[Dict[str, float]] = []
    n_graphs_per_class = 10
    N_MIN_PREFILTER = 10

    # baseline: без size-prefilter, только возможный tree-prefilter (по умолчанию False).
    cfg_base = ThermoConfig(
        coupling_topo_3d=1.0,
        topo_3d_beta=1.0,
        topo3d_prefilter_tree=False,
        topo3d_prefilter_min_n=0,
    )

    # optimized: включён tree-prefilter и size-prefilter (n < N_MIN_PREFILTER).
    cfg_opt = ThermoConfig(
        coupling_topo_3d=1.0,
        topo_3d_beta=1.0,
        topo3d_prefilter_tree=True,
        topo3d_prefilter_min_n=N_MIN_PREFILTER,
    )

    rng = np.random.default_rng(20251216)

    for mode, cfg in (("baseline", cfg_base), ("optimized", cfg_opt)):
        with override_thermo_config(cfg):
            for spec in GRAPH_SPECS:
                results.extend(
                    _run_batch_for_spec(
                        spec=spec,
                        mode=mode,
                        n_graphs=n_graphs_per_class,
                        rng=rng,
                    )
                )

    csv_path = results_path("fast_complexity_1_bench.csv")
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = ["mode", "graph_class", "n_nodes", "cyclic", "runtime_sec"]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow(r)

    # summary txt: p50/p90/p99 по классам и speedup optimized vs baseline
    lines: List[str] = []
    lines.append("FAST-COMPLEXITY-1 micro-benchmark for fdm_entanglement")
    lines.append(f"N_MIN_PREFILTER={N_MIN_PREFILTER}")
    lines.append("")

    def _percentiles(vals: List[float]) -> Tuple[float, float, float]:
        arr = np.array(vals, dtype=float)
        return (
            float(np.percentile(arr, 50)),
            float(np.percentile(arr, 90)),
            float(np.percentile(arr, 99)),
        )

    for spec in GRAPH_SPECS:
        times_base = [r["runtime_sec"] for r in results if r["mode"] == "baseline" and r["graph_class"] == spec.name]
        times_opt = [r["runtime_sec"] for r in results if r["mode"] == "optimized" and r["graph_class"] == spec.name]
        if not times_base or not times_opt:
            continue
        p50_b, p90_b, p99_b = _percentiles(times_base)
        p50_o, p90_o, p99_o = _percentiles(times_opt)
        speedup_p50 = p50_b / p50_o if p50_o > 0 else 1.0
        speedup_p90 = p90_b / p90_o if p90_o > 0 else 1.0
        speedup_p99 = p99_b / p99_o if p99_o > 0 else 1.0
        lines.append(
            f"{spec.name}: "
            f"baseline_p50={p50_b:.6e}s, optimized_p50={p50_o:.6e}s, speedup_p50={speedup_p50:.2f}x; "
            f"baseline_p90={p90_b:.6e}s, optimized_p90={p90_o:.6e}s, speedup_p90={speedup_p90:.2f}x; "
            f"baseline_p99={p99_b:.6e}s, optimized_p99={p99_o:.6e}s, speedup_p99={speedup_p99:.2f}x"
        )

    summary_path = results_path("fast_complexity_1_bench.txt")
    with summary_path.open("w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    return csv_path, summary_path


def main() -> None:
    csv_path, summary_path = run_fast_complexity_1_bench()
    print("[FAST-COMPLEXITY-1] micro-benchmark done.")
    print(f"CSV: {csv_path}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
