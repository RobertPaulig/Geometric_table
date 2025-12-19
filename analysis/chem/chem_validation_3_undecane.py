from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np

from analysis.chem.chem_validation_2_common import (
    EqRunConfig,
    edges_to_adj,
    parse_tree_topology_edges,
    p_pred_from_energy,
    run_equilibrium_with_guardrail,
    write_report_and_csv,
)


def _wiener_index_tree(adj: np.ndarray) -> int:
    n = int(adj.shape[0])
    adj_list: List[List[int]] = [[] for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            if float(adj[i, j]) > 0:
                adj_list[i].append(j)
                adj_list[j].append(i)

    total = 0
    for src in range(n):
        dist = [-1] * n
        dist[src] = 0
        q = [src]
        qi = 0
        while qi < len(q):
            u = q[qi]
            qi += 1
            for v in adj_list[u]:
                if dist[v] < 0:
                    dist[v] = dist[u] + 1
                    q.append(v)
        for dst in range(src + 1, n):
            total += int(dist[dst])
    return int(total)


def _rankdata(a: Sequence[float]) -> List[float]:
    # Average ranks for ties (1-based ranks).
    idx = sorted(range(len(a)), key=lambda i: (float(a[i]), i))
    ranks = [0.0] * len(a)
    i = 0
    while i < len(idx):
        j = i + 1
        while j < len(idx) and float(a[idx[j]]) == float(a[idx[i]]):
            j += 1
        r = 0.5 * (i + 1 + j)
        for k in range(i, j):
            ranks[idx[k]] = float(r)
        i = j
    return ranks


def _spearman_rho(x: Sequence[float], y: Sequence[float]) -> float:
    if len(x) != len(y) or not x:
        return float("nan")
    rx = _rankdata(x)
    ry = _rankdata(y)
    mx = float(np.mean(np.asarray(rx, dtype=float)))
    my = float(np.mean(np.asarray(ry, dtype=float)))
    num = 0.0
    denx = 0.0
    deny = 0.0
    for a, b in zip(rx, ry):
        dx = float(a) - mx
        dy = float(b) - my
        num += dx * dy
        denx += dx * dx
        deny += dy * dy
    den = math.sqrt(denx * deny)
    return float(num / den) if den > 0 else float("nan")


@dataclass
class Config:
    mode: str = "A"
    lam: float = 1.0
    temperature_T: float = 1.0
    steps: int = 100_000
    burnin: int = 10_000
    thin: int = 10
    chains: int = 5
    start_specs: Tuple[str, ...] = ("path", "max_branch")
    seed: int = 0
    progress: bool = True
    top_k: int = 20
    profile_every: int = 100


def _parse_args(argv: Optional[Sequence[str]] = None) -> Config:
    ap = argparse.ArgumentParser(
        description="CHEM-VALIDATION-3: C11 (undecane) equilibrium-first + self-consistency + Wiener sanity."
    )
    ap.add_argument("--mode", type=str, default="A", choices=["A", "B", "C"])
    ap.add_argument("--lambda", dest="lam", type=float, default=1.0)
    ap.add_argument("--T", dest="temperature_T", type=float, default=1.0)
    ap.add_argument("--steps", type=int, default=100_000)
    ap.add_argument("--burnin", type=int, default=10_000)
    ap.add_argument("--thin", type=int, default=10)
    ap.add_argument("--chains", type=int, default=5)
    ap.add_argument("--start_specs", type=str, nargs="*", default=["path", "max_branch"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--progress", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--top_k", type=int, default=20)
    ap.add_argument("--profile_every", type=int, default=100)
    args = ap.parse_args(argv)
    return Config(
        mode=str(args.mode).upper(),
        lam=float(args.lam),
        temperature_T=float(args.temperature_T),
        steps=int(args.steps),
        burnin=int(args.burnin),
        thin=int(args.thin),
        chains=int(args.chains),
        start_specs=tuple(str(x) for x in args.start_specs),
        seed=int(args.seed),
        progress=bool(args.progress),
        top_k=int(args.top_k),
        profile_every=int(args.profile_every),
    )


def main(argv: Optional[Sequence[str]] = None) -> None:
    cfg0 = _parse_args(argv)
    cfg = EqRunConfig(
        n=11,
        expected_n_topologies=159,
        mode=cfg0.mode,
        backend={"A": "fdm", "B": "fdm_entanglement", "C": "fdm_entanglement"}[cfg0.mode],
        lam=cfg0.lam,
        temperature_T=cfg0.temperature_T,
        steps=cfg0.steps,
        burnin=cfg0.burnin,
        thin=cfg0.thin,
        chains=cfg0.chains,
        start_specs=cfg0.start_specs,
        seed=cfg0.seed,
        progress=cfg0.progress,
        top_k=cfg0.top_k,
        max_attempts=6,
        guardrail_kl_max_target=0.005,
        profile_every=cfg0.profile_every,
    )

    p_eq, meta = run_equilibrium_with_guardrail(cfg)

    p_pred, energies, aut_sizes, g_vals = p_pred_from_energy(
        n=int(cfg.n),
        backend=str(cfg.backend),
        topo_keys=sorted(p_eq.keys()),
        lam=float(cfg.lam),
        temperature_T=float(cfg.temperature_T),
    )

    # Physical sanity: Wiener index vs energy correlation (Spearman).
    wieners: List[float] = []
    ens: List[float] = []
    for topo in sorted(p_eq.keys()):
        edges = parse_tree_topology_edges(topo)
        adj = edges_to_adj(int(cfg.n), edges)
        w = float(_wiener_index_tree(adj))
        e = float(energies.get(topo, 0.0))
        wieners.append(w)
        ens.append(e)
    rho = _spearman_rho(ens, wieners)

    extra = [f"Sanity: SpearmanCorr(Energy, Wiener)={rho:.6f}  (expect strong positive correlation)"]
    write_report_and_csv(
        out_stub="chem_validation_3_undecane",
        cfg=cfg,
        p_eq=p_eq,
        meta=meta,
        p_pred=p_pred,
        energies=energies,
        aut_sizes=aut_sizes,
        g_vals=g_vals,
        extra_summary_lines=extra,
    )


if __name__ == "__main__":
    main()
