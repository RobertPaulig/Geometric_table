from __future__ import annotations

import csv
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from analysis.growth.rng import make_rng
from analysis.io_utils import results_path
from core.complexity import compute_complexity_features_v2
from core.thermo_config import ThermoConfig, override_thermo_config


@dataclass(frozen=True)
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
    - cyclic=False: случайное дерево (спаннинг-три)
    - cyclic=True: дерево + несколько дополнительных рёбер
    """
    adj = np.zeros((n, n), dtype=int)

    # дерево
    for i in range(1, n):
        j = int(rng.integers(0, i))
        adj[i, j] = adj[j, i] = 1

    # циклы
    if cyclic:
        extra_edges = max(1, n // 5)
        for _ in range(extra_edges):
            u = int(rng.integers(0, n))
            v = int(rng.integers(0, n))
            if u == v:
                continue
            adj[u, v] = adj[v, u] = 1
    return adj


def run_fast_complexity_2_bench() -> Tuple[Path, Path]:
    """
    FAST-COMPLEXITY-2:
    - Paired microbench: baseline/optimized прогоняются на ОДНОМ И ТОМ ЖЕ наборе графов.
    - n_graphs_per_class=50
    - baseline: topo3d_prefilter_tree=False, topo3d_prefilter_min_n=0
    - optimized: topo3d_prefilter_tree=True, topo3d_prefilter_min_n=10
    """
    n_graphs_per_class = 50
    N_MIN_PREFILTER = 10

    rng = make_rng("fast_complexity_2")

    # 1) фиксируем набор графов (PAIRED)
    graphs_by_class: Dict[str, List[Tuple[int, np.ndarray]]] = {spec.name: [] for spec in GRAPH_SPECS}
    for spec in GRAPH_SPECS:
        for _ in range(n_graphs_per_class):
            n = int(rng.integers(spec.n_min, spec.n_max + 1))
            adj = _make_random_graph(n, cyclic=spec.cyclic, rng=rng)
            graphs_by_class[spec.name].append((n, adj))

    cfg_base = ThermoConfig(
        coupling_topo_3d=1.0,
        topo_3d_beta=1.0,
        topo3d_prefilter_tree=False,
        topo3d_prefilter_min_n=0,
    )
    cfg_opt = ThermoConfig(
        coupling_topo_3d=1.0,
        topo_3d_beta=1.0,
        topo3d_prefilter_tree=True,
        topo3d_prefilter_min_n=N_MIN_PREFILTER,
    )

    results: List[Dict[str, float]] = []
    times: Dict[Tuple[str, str], List[float]] = {}

    for mode, cfg in (("baseline", cfg_base), ("optimized", cfg_opt)):
        with override_thermo_config(cfg):
            for spec in GRAPH_SPECS:
                key = (mode, spec.name)
                times.setdefault(key, [])
                adj_list = graphs_by_class[spec.name]
                for graph_id, (n, adj) in enumerate(adj_list):
                    t0 = time.perf_counter()
                    _ = compute_complexity_features_v2(adj, backend="fdm_entanglement")
                    dt = time.perf_counter() - t0
                    times[key].append(float(dt))
                    results.append(
                        {
                            "mode": mode,
                            "graph_class": spec.name,
                            "graph_id": float(graph_id),
                            "n_nodes": float(n),
                            "cyclic": float(int(spec.cyclic)),
                            "runtime_sec": float(dt),
                        }
                    )

    # 2) CSV (игнорится git по /results/*.csv)
    csv_path = results_path("fast_complexity_2_bench.csv")
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["mode", "graph_class", "graph_id", "n_nodes", "cyclic", "runtime_sec"]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in results:
            w.writerow(r)

    # 3) Summary (paired speedup)
    def _pct(arr: np.ndarray) -> Tuple[float, float, float]:
        return (
            float(np.percentile(arr, 50)),
            float(np.percentile(arr, 90)),
            float(np.percentile(arr, 99)),
        )

    lines: List[str] = []
    lines.append("FAST-COMPLEXITY-2 paired micro-benchmark for fdm_entanglement")
    lines.append(f"n_graphs_per_class={n_graphs_per_class}")
    lines.append(f"N_MIN_PREFILTER={N_MIN_PREFILTER}")
    lines.append("")

    for spec in GRAPH_SPECS:
        base = np.array(times[("baseline", spec.name)], dtype=float)
        opt = np.array(times[("optimized", spec.name)], dtype=float)

        p50_b, p90_b, p99_b = _pct(base)
        p50_o, p90_o, p99_o = _pct(opt)

        speedup_p50 = (p50_b / p50_o) if p50_o > 0 else 1.0
        speedup_p90 = (p90_b / p90_o) if p90_o > 0 else 1.0
        speedup_p99 = (p99_b / p99_o) if p99_o > 0 else 1.0

        # paired ratios per-graph
        ratios = (base / np.maximum(opt, 1e-30))
        r50, r90, r99 = _pct(ratios)

        lines.append(
            f"{spec.name}: "
            f"baseline_p50={p50_b:.6e}s, optimized_p50={p50_o:.6e}s, speedup_p50={speedup_p50:.2f}x; "
            f"baseline_p90={p90_b:.6e}s, optimized_p90={p90_o:.6e}s, speedup_p90={speedup_p90:.2f}x; "
            f"baseline_p99={p99_b:.6e}s, optimized_p99={p99_o:.6e}s, speedup_p99={speedup_p99:.2f}x; "
            f"paired_speedup_ratio_p50={r50:.2f}x, p90={r90:.2f}x, p99={r99:.2f}x"
        )

    summary_path = results_path("fast_complexity_2_bench.txt")
    with summary_path.open("w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    return csv_path, summary_path


def main() -> None:
    csv_path, summary_path = run_fast_complexity_2_bench()
    print("[FAST-COMPLEXITY-2] micro-benchmark done.")
    print(f"CSV: {csv_path}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()

