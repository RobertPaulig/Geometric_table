from __future__ import annotations

import argparse
import csv
import math
import time
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from analysis.io_utils import results_path
from analysis.utils.timing import now_iso
from analysis.growth.reporting import write_growth_txt


def _parse_tree_topology_edges(topo: str) -> List[Tuple[int, int]]:
    if not topo.startswith("tree:"):
        raise ValueError(f"Expected topology key 'tree:<...>', got: {topo!r}")
    s = topo[len("tree:") :]
    if not s:
        return []
    edges: List[Tuple[int, int]] = []
    for part in s.split(","):
        a_str, b_str = part.split("-", 1)
        a = int(a_str)
        b = int(b_str)
        edges.append((a, b) if a < b else (b, a))
    edges.sort()
    return edges


def _edges_to_adj_list(n: int, edges: Sequence[Tuple[int, int]]) -> List[List[int]]:
    adj: List[List[int]] = [[] for _ in range(int(n))]
    for a, b in edges:
        adj[int(a)].append(int(b))
        adj[int(b)].append(int(a))
    return adj


def _degrees(adj_list: Sequence[Sequence[int]]) -> List[int]:
    return [len(list(nei)) for nei in adj_list]


def _bfs_dists(adj_list: Sequence[Sequence[int]], start: int) -> List[int]:
    n = len(adj_list)
    dist = [-1] * n
    dist[int(start)] = 0
    q = [int(start)]
    qi = 0
    while qi < len(q):
        u = q[qi]
        qi += 1
        for v in adj_list[u]:
            if dist[v] < 0:
                dist[v] = dist[u] + 1
                q.append(v)
    return dist


def tree_diameter(adj_list: Sequence[Sequence[int]]) -> int:
    # two BFS trick
    d0 = _bfs_dists(adj_list, 0)
    far = int(max(range(len(d0)), key=lambda i: d0[i]))
    d1 = _bfs_dists(adj_list, far)
    return int(max(d1))


def wiener_index(adj_list: Sequence[Sequence[int]]) -> int:
    n = len(adj_list)
    total = 0
    for i in range(n):
        di = _bfs_dists(adj_list, i)
        for j in range(i + 1, n):
            total += int(di[j])
    return int(total)


def spearman_rho(x: Sequence[float], y: Sequence[float]) -> float:
    """
    Spearman rho with average ranks for ties (no scipy dependency).
    """

    def rankdata(a: Sequence[float]) -> np.ndarray:
        arr = np.asarray(list(a), dtype=float)
        order = np.argsort(arr, kind="mergesort")
        ranks = np.empty_like(order, dtype=float)
        i = 0
        while i < len(arr):
            j = i + 1
            while j < len(arr) and arr[order[j]] == arr[order[i]]:
                j += 1
            r = 0.5 * (i + j - 1) + 1.0
            for k in range(i, j):
                ranks[order[k]] = r
            i = j
        return ranks

    rx = rankdata(x)
    ry = rankdata(y)
    rxm = float(rx.mean())
    rym = float(ry.mean())
    num = float(np.sum((rx - rxm) * (ry - rym)))
    den = math.sqrt(float(np.sum((rx - rxm) ** 2)) * float(np.sum((ry - rym) ** 2)))
    return float(num / den) if den > 0 else 0.0


def _load_p_exact(n: int, mode: str) -> Dict[str, float]:
    path = results_path(f"alkane_exact_1_N{int(n)}.csv")
    if not path.exists():
        raise RuntimeError(f"Missing exact baseline: {path}")
    out: Dict[str, float] = {}
    with path.open("r", newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            if int(row.get("N", 0)) != int(n):
                continue
            if str(row.get("mode", "")).upper() != str(mode).upper():
                continue
            topo = str(row.get("topology", ""))
            out[topo] = float(row.get("p_exact", 0.0))
    return out


def _load_p_growth(txt_name: str, mode: str) -> Dict[str, float]:
    text = results_path(txt_name).read_text(encoding="utf-8", errors="replace").splitlines()
    in_mode = False
    in_block = False
    counts: Dict[str, int] = {}
    for line in text:
        s = line.strip()
        if s == f"[Mode {mode}]":
            in_mode = True
            in_block = False
            continue
        if in_mode and s.startswith("[Mode ") and s.endswith("]") and s != f"[Mode {mode}]":
            break
        if in_mode and s == "P_growth(topology):":
            in_block = True
            continue
        if in_mode and in_block and s.startswith("mass_topK_growth="):
            break
        if in_mode and in_block and s.startswith("tree:") and "count=" in s:
            topo = s.split("=", 1)[0].strip()
            cnt = int(s.split("count=")[1].split(")")[0])
            counts[topo] = counts.get(topo, 0) + int(cnt)
    total = float(sum(counts.values()))
    return {k: (v / total if total > 0 else 0.0) for k, v in counts.items()}


@dataclass
class Config:
    N: int = 7
    mode: str = "A"
    top_k: int = 10


def _parse_args(argv: Optional[Sequence[str]] = None) -> Config:
    ap = argparse.ArgumentParser(description="GROWTH-KERNEL-BIAS-1: bias vector and correlations vs invariants.")
    ap.add_argument("--N", type=int, required=True, choices=[7, 8])
    ap.add_argument("--mode", type=str, default="A", choices=["A", "B", "C"])
    ap.add_argument("--top_k", type=int, default=10)
    args = ap.parse_args(argv)
    return Config(N=int(args.N), mode=str(args.mode).upper(), top_k=int(args.top_k))


def main(argv: Optional[Sequence[str]] = None) -> None:
    cfg = _parse_args(argv)
    start_ts = now_iso()
    t0_total = time.perf_counter()

    if cfg.N == 7:
        growth_txt = "chem_validation_1c_heptane.txt"
    else:
        growth_txt = "chem_validation_1d_octane.txt"

    p_exact = _load_p_exact(cfg.N, cfg.mode)
    p_growth = _load_p_growth(growth_txt, cfg.mode)

    eps = 1e-12
    topo_keys = sorted(set(p_exact.keys()) | set(p_growth.keys()))
    bias = {k: float(math.log((p_growth.get(k, 0.0) + eps) / (p_exact.get(k, 0.0) + eps))) for k in topo_keys}

    # Invariants per topology
    inv: Dict[str, Dict[str, float]] = {}
    for topo in topo_keys:
        edges = _parse_tree_topology_edges(topo)
        adj_list = _edges_to_adj_list(cfg.N, edges)
        deg = _degrees(adj_list)
        inv[topo] = {
            "diameter": float(tree_diameter(adj_list)),
            "max_degree": float(max(deg) if deg else 0),
            "n_leaves": float(sum(1 for d in deg if d == 1)),
            "wiener": float(wiener_index(adj_list)),
        }

    # Correlations
    bias_vals = [bias[k] for k in topo_keys]
    lines: List[str] = []
    lines.append(f"GROWTH-KERNEL-BIAS-1 (N={cfg.N}, mode={cfg.mode})")
    lines.append(f"Uses P_exact from results/alkane_exact_1_N{cfg.N}.csv and P_growth from results/{growth_txt}")
    lines.append("")
    lines.append("Bias vector per topology: bias = log(P_growth/P_exact)")
    for topo, b in sorted(bias.items(), key=lambda kv: (-abs(kv[1]), kv[0])):
        lines.append(f"  {topo}: bias={b:+.6f} P_growth={p_growth.get(topo, 0.0):.6f} P_exact={p_exact.get(topo, 0.0):.6f}")
    lines.append("")
    lines.append(f"Top-|bias| (K={cfg.top_k})")
    for topo, b in sorted(bias.items(), key=lambda kv: (-abs(kv[1]), kv[0]))[: max(1, int(cfg.top_k))]:
        lines.append(
            f"  {topo}: bias={b:+.6f} diameter={inv[topo]['diameter']:.0f} max_deg={inv[topo]['max_degree']:.0f} "
            f"leaves={inv[topo]['n_leaves']:.0f} wiener={inv[topo]['wiener']:.0f}"
        )
    lines.append("")

    lines.append("Spearman correlations with bias (signed):")
    for name in ["diameter", "max_degree", "n_leaves", "wiener"]:
        vals = [float(inv[k][name]) for k in topo_keys]
        lines.append(f"  rho(bias, {name}) = {spearman_rho(bias_vals, vals):+.4f}")

    elapsed_total = time.perf_counter() - t0_total
    end_ts = now_iso()
    lines.append("")
    lines.append("TIMING")
    lines.append(f"START_TS={start_ts}")
    lines.append(f"END_TS={end_ts}")
    lines.append(f"ELAPSED_TOTAL_SEC={elapsed_total:.6f}")

    out_name = f"growth_kernel_bias_1_N{cfg.N}_mode{cfg.mode}"
    write_growth_txt(out_name, lines)
    print(f"[GROWTH-KERNEL-BIAS-1] wrote {results_path(out_name + '.txt')}")
    print(f"Wall-clock: start={start_ts} end={end_ts} elapsed_total_sec={elapsed_total:.3f}")


if __name__ == "__main__":
    main()

