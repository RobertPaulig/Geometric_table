from __future__ import annotations

import csv
import time
from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple

import numpy as np

from analysis.io_utils import results_path
from analysis.utils.timing import now_iso
from core.tree_canonical import canonical_relabel_tree
from analysis.chem.topology_mcmc import edges_to_adj, tree_topology_edge_key_from_edges


Edge = Tuple[int, int]


def prufer_tree_edges(n: int, rng: np.random.Generator) -> List[Edge]:
    n = int(n)
    if n <= 1:
        return []
    if n == 2:
        return [(0, 1)]
    seq = rng.integers(0, n, size=n - 2, dtype=int).tolist()
    deg = [1] * n
    for x in seq:
        deg[int(x)] += 1
    edges: List[Edge] = []
    for x in seq:
        leaf = next(i for i in range(n) if deg[i] == 1)
        edges.append((int(leaf), int(x)))
        deg[leaf] -= 1
        deg[int(x)] -= 1
    u, v = [i for i in range(n) if deg[i] == 1]
    edges.append((int(u), int(v)))
    return edges


def _pctl(vals: Sequence[float], q: float) -> float:
    if not vals:
        return 0.0
    return float(np.percentile(np.asarray(vals, dtype=float), q))


@dataclass
class BenchCfg:
    ns: Tuple[int, ...] = (40, 80, 160, 320)
    n_trees: int = 200
    seed: int = 0


def bench_one_n(n: int, *, n_trees: int, rng: np.random.Generator) -> dict:
    t_opt1: List[float] = []
    t_opt2: List[float] = []

    for _ in range(int(n_trees)):
        edges = prufer_tree_edges(int(n), rng)
        adj = edges_to_adj(int(n), edges)

        t0 = time.perf_counter()
        _ = canonical_relabel_tree(adj)
        t_opt1.append(time.perf_counter() - t0)

        t1 = time.perf_counter()
        _ = tree_topology_edge_key_from_edges(int(n), edges)
        t_opt2.append(time.perf_counter() - t1)

    mean1 = float(np.mean(np.asarray(t_opt1, dtype=float)))
    mean2 = float(np.mean(np.asarray(t_opt2, dtype=float)))
    return {
        "N": int(n),
        "opt1_mean_sec": mean1,
        "opt1_p90_sec": _pctl(t_opt1, 90),
        "opt2_mean_sec": mean2,
        "opt2_p90_sec": _pctl(t_opt2, 90),
        "speedup_opt2_vs_opt1": (mean1 / mean2) if mean2 > 0 else float("inf"),
        "canon_overhead_opt2": (mean2 / mean1) if mean1 > 0 else 0.0,
    }


def main() -> None:
    cfg = BenchCfg()
    rng = np.random.default_rng(int(cfg.seed))

    start_ts = now_iso()
    t0 = time.perf_counter()

    rows: List[dict] = []
    for n in cfg.ns:
        rows.append(bench_one_n(int(n), n_trees=int(cfg.n_trees), rng=rng))

    elapsed = time.perf_counter() - t0
    end_ts = now_iso()

    out_csv = results_path("invariance_bench_2.csv")
    out_txt = results_path("invariance_bench_2.txt")

    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)

    lines: List[str] = []
    lines.append("INVARIANCE-BENCH-2: OPT-2 vs OPT-1 canonicalization timing")
    lines.append(f"config: ns={list(cfg.ns)}, n_trees={cfg.n_trees}, seed={cfg.seed}")
    lines.append("")
    for r in rows:
        lines.append(
            f"N={r['N']}: opt1_mean={r['opt1_mean_sec']:.6g}s opt2_mean={r['opt2_mean_sec']:.6g}s "
            f"speedup={r['speedup_opt2_vs_opt1']:.3f} overhead_opt2={r['canon_overhead_opt2']:.3f}"
        )
    lines.append("")
    lines.append(f"START_TS={start_ts}")
    lines.append(f"END_TS={end_ts}")
    lines.append(f"ELAPSED_TOTAL_SEC={elapsed:.6f}")

    out_txt.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[BENCH] wrote {out_csv}")
    print(f"[BENCH] wrote {out_txt}")
    print(f"Wall-clock: start={start_ts} end={end_ts} elapsed_total_sec={elapsed:.3f}")


if __name__ == "__main__":
    main()

