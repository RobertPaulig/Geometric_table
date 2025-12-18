from __future__ import annotations

import argparse
import csv
import math
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from analysis.chem.core2_fit import kl_divergence
from analysis.chem.topology_mcmc import Edge, edges_to_adj, run_fixed_n_tree_mcmc
from analysis.io_utils import results_path
from analysis.utils.progress import progress_iter
from analysis.utils.timing import now_iso
from analysis.growth.reporting import write_growth_txt


def _make_path_edges(n: int) -> List[Edge]:
    return [(int(i), int(i + 1)) for i in range(int(n) - 1)]


def _make_max_branch_edges(n: int) -> List[Edge]:
    """
    Construct a highly branched alkane-valid tree (deg<=4) deterministically.

    Strategy:
    - create a center node 0 with degree 4 as early as possible
    - attach remaining nodes as a short chain off one of the leaves
    """
    n = int(n)
    if n < 2:
        return []
    if n <= 5:
        # For small N, a star is alkane-valid.
        return [(0, i) for i in range(1, n)]

    edges: List[Edge] = [(0, 1), (0, 2), (0, 3), (0, 4)]
    last = 4
    for v in range(5, n):
        edges.append((last, v))
        last = v
    return edges


def _deg_sorted_key(adj: np.ndarray) -> str:
    deg = [int(x) for x in np.asarray(adj, dtype=float).sum(axis=1).tolist()]
    deg_sorted = ",".join(str(x) for x in sorted(deg))
    return f"deg:{deg_sorted}"


@dataclass
class ScanConfig:
    N: int = 7
    mode: str = "A"
    steps_list: Tuple[int, ...] = (2_000, 5_000, 10_000, 20_000)
    burnin_frac: float = 0.1
    thin: int = 10
    chains: int = 3
    seed: int = 0
    lam: float = 1.0
    start_specs: Tuple[str, ...] = ("path", "max_branch")
    progress: bool = True


def _parse_args(argv: Optional[Sequence[str]] = None) -> ScanConfig:
    ap = argparse.ArgumentParser(
        description="EQ-TARGET-2: fixed-N MCMC mixing budget for N=7/8 (guardrail via bad starts)."
    )
    ap.add_argument("--N", type=int, required=True, choices=[7, 8])
    ap.add_argument("--mode", type=str, default="A", choices=["A", "B", "C"])
    ap.add_argument("--steps", type=int, nargs="*", default=[2000, 5000, 10000, 20000])
    ap.add_argument("--include_50k", action="store_true", help="Append 50000 steps to the scan grid.")
    ap.add_argument("--burnin_frac", type=float, default=0.1)
    ap.add_argument("--thin", type=int, default=10)
    ap.add_argument("--chains", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--lambda", dest="lam", type=float, default=1.0)
    ap.add_argument(
        "--start_topologies",
        type=str,
        nargs="*",
        default=["path", "max_branch"],
        help="Start specs for guardrail; for N>=7 defaults to ['path','max_branch'].",
    )
    ap.add_argument("--progress", action=argparse.BooleanOptionalAction, default=True)
    args = ap.parse_args(argv)
    steps_list = tuple(int(x) for x in args.steps) + ((50_000,) if bool(args.include_50k) else tuple())
    return ScanConfig(
        N=int(args.N),
        mode=str(args.mode).upper(),
        steps_list=steps_list,
        burnin_frac=float(args.burnin_frac),
        thin=int(args.thin),
        chains=int(args.chains),
        seed=int(args.seed),
        lam=float(args.lam),
        start_specs=tuple(str(x) for x in args.start_topologies),
        progress=bool(args.progress),
    )


def _backend_for_mode(mode: str) -> str:
    mode = mode.upper()
    if mode == "A":
        return "fdm"
    if mode == "B":
        return "fdm_entanglement"
    if mode == "C":
        return "fdm_entanglement"
    raise ValueError(f"Unknown mode: {mode!r}")


def _start_edges_for_spec(n: int, spec: str) -> List[Edge]:
    spec = str(spec).strip()
    if spec == "path":
        return _make_path_edges(n)
    if spec == "max_branch":
        return _make_max_branch_edges(n)
    raise ValueError(f"Unknown start spec for N={n}: {spec!r} (expected 'path' or 'max_branch').")


def _mean_ci(values: Sequence[float]) -> Tuple[float, float]:
    arr = np.asarray(list(values), dtype=float)
    if arr.size <= 1:
        return float(arr.mean()) if arr.size else 0.0, 0.0
    se = float(arr.std(ddof=1)) / math.sqrt(float(arr.size))
    return float(arr.mean()), 1.96 * se


def _kl_pairwise_max_mean(p_by_start: List[Tuple[str, Dict[str, float]]]) -> Tuple[float, float]:
    if len(p_by_start) < 2:
        return float("nan"), float("nan")
    kls: List[float] = []
    for i in range(len(p_by_start)):
        for j in range(len(p_by_start)):
            if i == j:
                continue
            p_i = p_by_start[i][1]
            p_j = p_by_start[j][1]
            kls.append(float(kl_divergence(p_i, p_j)))
    return float(max(kls)), float(np.mean(np.asarray(kls, dtype=float)))


def main(argv: Optional[Sequence[str]] = None) -> None:
    cfg = _parse_args(argv)

    start_ts = now_iso()
    t0_total = time.perf_counter()

    backend = _backend_for_mode(cfg.mode)
    temperature_T = 1.0

    out_stub = f"eq_target_2_N{cfg.N}_mode{cfg.mode}"
    out_csv = results_path(f"{out_stub}.csv")
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, object]] = []
    lines: List[str] = []
    lines.append(f"EQ-TARGET-2: steps -> KL guardrail via bad starts (N={cfg.N}, mode {cfg.mode})")
    lines.append(
        f"steps_list={list(cfg.steps_list)}, chains={cfg.chains}, thin={cfg.thin}, burnin_frac={cfg.burnin_frac}, "
        f"lambda={cfg.lam}, starts={list(cfg.start_specs)}"
    )
    lines.append("")

    for steps in cfg.steps_list:
        burnin = int(max(0, round(float(cfg.burnin_frac) * float(steps))))
        point_t0 = time.perf_counter()

        per_start_probs: List[Tuple[str, Dict[str, float]]] = []
        per_start_elapsed: Dict[str, float] = {}
        per_start_cache_hit_rate: Dict[str, float] = {}
        per_start_steps_per_sec: Dict[str, float] = {}

        total_steps_done = 0
        total_cache_hits = 0
        total_cache_misses = 0
        total_proposals = 0
        total_accepted = 0

        for start_spec in cfg.start_specs:
            chain_probs: List[Dict[str, float]] = []
            chain_cache_rates: List[float] = []
            chain_steps_per_sec: List[float] = []
            start_t0 = time.perf_counter()

            for chain_idx in progress_iter(
                range(int(cfg.chains)),
                total=int(cfg.chains),
                desc=f"EQ-TARGET-2 N{cfg.N} mode{cfg.mode} steps={steps} start={start_spec}",
                enabled=bool(cfg.progress),
            ):
                edges0 = _start_edges_for_spec(cfg.N, start_spec)
                _, summary = run_fixed_n_tree_mcmc(
                    n=int(cfg.N),
                    steps=int(steps),
                    burnin=int(burnin),
                    thin=int(cfg.thin),
                    backend=str(backend),
                    lam=float(cfg.lam),
                    temperature_T=float(temperature_T),
                    seed=int(cfg.seed) + 10_000 * int(steps) + 101 * int(chain_idx),
                    max_valence=4,
                    topology_classifier=lambda adj: _deg_sorted_key(adj),
                    start_edges=edges0,
                    progress=None,
                )

                chain_probs.append(dict(summary.p_topology))
                chain_cache_rates.append(float(summary.energy_cache_hit_rate))
                chain_steps_per_sec.append(float(summary.steps_per_sec))

                total_steps_done += int(steps)
                total_cache_hits += int(summary.energy_cache_hits)
                total_cache_misses += int(summary.energy_cache_misses)
                total_proposals += int(summary.proposals)
                total_accepted += int(summary.accepted)

            # mean P across chains
            keys: List[str] = []
            for p in chain_probs:
                for k in p.keys():
                    if k not in keys:
                        keys.append(k)
            p_mean: Dict[str, float] = {}
            for k in keys:
                vals = [float(p.get(k, 0.0)) for p in chain_probs]
                p_mean[k] = float(np.mean(np.asarray(vals, dtype=float)))
            s = float(sum(p_mean.values()))
            if s > 0:
                p_mean = {k: float(v) / s for k, v in p_mean.items()}

            per_start_probs.append((str(start_spec), p_mean))
            per_start_elapsed[str(start_spec)] = float(time.perf_counter() - start_t0)
            per_start_cache_hit_rate[str(start_spec)] = float(np.mean(np.asarray(chain_cache_rates, dtype=float)))
            per_start_steps_per_sec[str(start_spec)] = float(np.mean(np.asarray(chain_steps_per_sec, dtype=float)))

        point_elapsed = float(time.perf_counter() - point_t0)
        kl_max, kl_mean = _kl_pairwise_max_mean(per_start_probs)

        hit_rate_total = (
            float(total_cache_hits) / float(total_cache_hits + total_cache_misses)
            if (total_cache_hits + total_cache_misses) > 0
            else 0.0
        )
        steps_total = int(steps) * int(cfg.chains) * max(1, len(cfg.start_specs))
        steps_per_sec_total = float(steps_total) / float(point_elapsed) if point_elapsed > 0 else 0.0
        accept_rate_total = float(total_accepted) / float(total_proposals) if total_proposals > 0 else 0.0

        row = {
            "N": int(cfg.N),
            "mode": str(cfg.mode),
            "backend": str(backend),
            "lambda": float(cfg.lam),
            "steps": int(steps),
            "burnin": int(burnin),
            "thin": int(cfg.thin),
            "chains": int(cfg.chains),
            "n_starts": int(len(cfg.start_specs)),
            "elapsed_total_sec": float(point_elapsed),
            "steps_total": int(steps_total),
            "steps_per_sec_total": float(steps_per_sec_total),
            "energy_cache_hit_rate": float(hit_rate_total),
            "accept_rate": float(accept_rate_total),
            "kl_max_pairwise": float(kl_max),
            "kl_mean_pairwise": float(kl_mean),
        }
        rows.append(row)

        lines.append(
            f"steps={steps:6d} burnin={burnin:5d} elapsed={point_elapsed:7.3f}s "
            f"steps_total={steps_total:7d} steps/sec={steps_per_sec_total:8.1f} "
            f"hit_rate={hit_rate_total:6.3f} acc={accept_rate_total:6.3f} "
            f"KL_max={kl_max:.6f} KL_mean={kl_mean:.6f}"
        )
        for st, _ in per_start_probs:
            lines.append(
                f"  start={st}: elapsed={per_start_elapsed[st]:.3f}s "
                f"cache_hit_rate={per_start_cache_hit_rate[st]:.3f} steps/sec={per_start_steps_per_sec[st]:.1f}"
            )

    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["steps"])
        w.writeheader()
        for r in rows:
            w.writerow(r)

    end_ts = now_iso()
    elapsed_total = time.perf_counter() - t0_total
    lines.append("")
    lines.append("TIMING")
    lines.append(f"START_TS={start_ts}")
    lines.append(f"END_TS={end_ts}")
    lines.append(f"ELAPSED_TOTAL_SEC={elapsed_total:.6f}")

    out_txt = write_growth_txt(out_stub, lines)
    print("[EQ-TARGET-2] done.")
    print(f"CSV: {out_csv}")
    print(f"Summary: {out_txt}")
    print(f"Wall-clock: start={start_ts} end={end_ts} elapsed_total_sec={elapsed_total:.3f}")


if __name__ == "__main__":
    main()

