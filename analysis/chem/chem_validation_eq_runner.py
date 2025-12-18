from __future__ import annotations

import argparse
import math
import re
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from analysis.chem.chem_validation_1b_hexane import HEXANE_DEGENERACY, classify_hexane_topology, _make_reference_adjs
from analysis.chem.core2_fit import kl_divergence
from analysis.chem.topology_mcmc import run_fixed_n_tree_mcmc
from analysis.io_utils import results_path
from analysis.utils.progress import progress_iter
from analysis.utils.timing import format_sec, now_iso
from analysis.growth.reporting import write_growth_txt


def _load_p_exact_hexane_mode_a() -> Optional[Dict[str, float]]:
    try:
        txt = results_path("mh_kernel_3_c6_exact_modeA.txt").read_text(encoding="utf-8", errors="replace")
    except Exception:
        return None
    if "P_exact(topology):" not in txt:
        return None
    block = txt.split("P_exact(topology):", 1)[1].split("\n\n", 1)[0]
    out: Dict[str, float] = {}
    for line in block.splitlines():
        line = line.strip()
        if not line or "=" not in line:
            continue
        k, v = [x.strip() for x in line.split("=", 1)]
        try:
            out[k] = float(v)
        except Exception:
            continue
    return out or None


def _load_lambda_star_from_hexane(mode: str) -> Optional[float]:
    mode = mode.upper()
    path = results_path("chem_validation_1b_hexane.txt")
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return None
    m_core = re.search(r"CORE-2: Fit lambda \(degeneracy-aware\)(.*)\Z", text, flags=re.S)
    if not m_core:
        return None
    tail = m_core.group(1)
    m_mode = re.search(rf"\[Mode {re.escape(mode)}\].*?lambda\*=([0-9.]+)", tail, flags=re.S)
    if not m_mode:
        return None
    try:
        return float(m_mode.group(1))
    except Exception:
        return None


def _mean_ci(values: Sequence[float]) -> Tuple[float, float]:
    arr = np.asarray(list(values), dtype=float)
    if arr.size <= 1:
        return float(arr.mean()) if arr.size else 0.0, 0.0
    se = float(arr.std(ddof=1)) / math.sqrt(float(arr.size))
    return float(arr.mean()), 1.96 * se


@dataclass
class RunnerConfig:
    N: int = 6
    mode: str = "A"
    steps: int = 20_000
    burnin: int = 2_000
    thin: int = 10
    chains: int = 3
    seed: int = 0
    lam: Optional[float] = None
    start_topology: str = "n_hexane"
    start_topologies: Optional[Tuple[str, ...]] = None
    progress: bool = True


def _parse_args(argv: Optional[Sequence[str]] = None) -> RunnerConfig:
    ap = argparse.ArgumentParser(description="CHEM-VALIDATION-EQ runner: fixed-N MCMC equilibrium on trees.")
    ap.add_argument("--N", type=int, default=6)
    ap.add_argument("--mode", type=str, default="A", choices=["A", "B", "C"])
    ap.add_argument("--steps", type=int, default=20_000)
    ap.add_argument("--burnin", type=int, default=2_000)
    ap.add_argument("--thin", type=int, default=10)
    ap.add_argument("--chains", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--lambda", dest="lam", type=float, default=None)
    ap.add_argument(
        "--start_topology",
        type=str,
        default="n_hexane",
        choices=["n_hexane", "2_methylpentane", "3_methylpentane", "2,2_dimethylbutane", "2,3_dimethylbutane"],
        help="Alias for a single start topology (sanity against optimistic starts).",
    )
    ap.add_argument(
        "--start_topologies",
        type=str,
        nargs="*",
        default=None,
        help="If provided, run a separate multi-chain estimate per start and report KL_max/KL_mean across starts.",
    )
    ap.add_argument(
        "--progress",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Show progress output.",
    )
    args = ap.parse_args(argv)
    start_topologies = tuple(str(x) for x in args.start_topologies) if args.start_topologies is not None else None
    if start_topologies is None:
        start_topologies = (str(args.start_topology),)
    return RunnerConfig(
        N=int(args.N),
        mode=str(args.mode).upper(),
        steps=int(args.steps),
        burnin=int(args.burnin),
        thin=int(args.thin),
        chains=int(args.chains),
        seed=int(args.seed),
        lam=float(args.lam) if args.lam is not None else None,
        start_topology=str(args.start_topology),
        start_topologies=start_topologies,
        progress=bool(args.progress),
    )


def run_runner(cfg: RunnerConfig) -> Tuple[Dict[str, float], Path]:
    start_ts = now_iso()
    t0_total = time.perf_counter()

    if cfg.N != 6:
        raise ValueError("This runner currently supports N=6 only (CHEM-VALIDATION-1B scope).")

    mode = cfg.mode.upper()
    lam = float(cfg.lam) if cfg.lam is not None else None
    if lam is None:
        lam = _load_lambda_star_from_hexane(mode) or 1.0

    backend = "fdm_entanglement"
    temperature_T = 1.0

    refs = _make_reference_adjs()
    start_list = (
        list(cfg.start_topologies)
        if cfg.start_topologies is not None and len(cfg.start_topologies) > 0
        else [cfg.start_topology]
    )
    for st in start_list:
        if st not in refs:
            raise ValueError(f"Unknown start_topology: {st!r}")

    # For each start topology, run multiple independent chains and aggregate.
    start_results: List[Tuple[str, Dict[str, float], Dict[str, float], List[Dict[str, float]]]] = []

    for start_topology in start_list:
        chain_probs: List[Dict[str, float]] = []
        chain_summaries: List[Dict[str, float]] = []

        for chain_idx in progress_iter(
            range(int(cfg.chains)),
            total=int(cfg.chains),
            desc=f"CHEM-EQ N{cfg.N} mode{mode} start={start_topology}",
            enabled=bool(cfg.progress),
        ):
            adj0 = np.asarray(refs[start_topology], dtype=float)
            edges0: List[Tuple[int, int]] = []
            for a in range(int(adj0.shape[0])):
                for b in range(a + 1, int(adj0.shape[0])):
                    if adj0[a, b] > 0:
                        edges0.append((int(a), int(b)))

            _, summary = run_fixed_n_tree_mcmc(
                n=int(cfg.N),
                steps=int(cfg.steps),
                burnin=int(cfg.burnin),
                thin=int(cfg.thin),
                backend=backend,
                lam=float(lam),
                temperature_T=float(temperature_T),
                seed=int(cfg.seed) + 101 * int(chain_idx),
                max_valence=4,
                topology_classifier=classify_hexane_topology,
                start_edges=edges0,
                progress=None,
            )
            chain_probs.append(dict(summary.p_topology))
            chain_summaries.append(
                {
                    "accept_rate": (float(summary.accepted) / float(summary.proposals)) if summary.proposals > 0 else 0.0,
                    "moves_mean": float(summary.mean_moves),
                    "cache_hit_rate": float(getattr(summary, "energy_cache_hit_rate", 0.0)),
                    "steps_per_sec": float(getattr(summary, "steps_per_sec", 0.0)),
                }
            )

        topologies = list(HEXANE_DEGENERACY.keys())
        p_mean: Dict[str, float] = {}
        p_ci: Dict[str, float] = {}
        for topo in topologies:
            vals = [float(p.get(topo, 0.0)) for p in chain_probs]
            mu, ci = _mean_ci(vals)
            p_mean[topo] = mu
            p_ci[topo] = ci
        s = float(sum(p_mean.values()))
        if s > 0:
            p_mean = {k: float(v) / s for k, v in p_mean.items()}

        start_results.append((str(start_topology), p_mean, p_ci, chain_summaries))

    # Primary P_eq output: first start (stable default).
    p_mean = start_results[0][1]
    p_ci = start_results[0][2]

    p_exact = _load_p_exact_hexane_mode_a() if mode == "A" else None
    topologies = list(HEXANE_DEGENERACY.keys())
    kl_by_start: List[Tuple[str, float]] = []
    if p_exact is not None:
        p_exact_vec = {k: float(p_exact.get(k, 0.0)) for k in topologies}
        for st, p_st, _, _ in start_results:
            kl_by_start.append((st, float(kl_divergence(p_st, p_exact_vec))))

    elapsed_total = time.perf_counter() - t0_total
    end_ts = now_iso()
    steps_total = int(cfg.steps) * int(cfg.chains)
    steps_per_sec_total = (float(steps_total) / float(elapsed_total)) if elapsed_total > 0 else 0.0

    out_name = f"chem_eq_N{cfg.N}_mode{mode}"
    out_txt = results_path(f"{out_name}.txt")

    lines: List[str] = []
    lines.append("CHEM-VALIDATION-EQ-2: equilibrium runner (fixed-N MCMC on trees)")
    lines.append(f"N={cfg.N}, mode={mode}, backend={backend}, lambda={lam:.6g}, T={temperature_T:.3f}")
    lines.append(f"steps={cfg.steps}, burnin={cfg.burnin}, thin={cfg.thin}, chains={cfg.chains}, seed={cfg.seed}")
    lines.append(f"start_topologies={start_list}")
    lines.append("")
    for st, p_st, ci_st, chain_summaries in start_results:
        lines.append(f"[Start {st}]")
        lines.append("  Per-chain summary:")
        for i, ssum in enumerate(chain_summaries):
            lines.append(
                f"    chain={i}: accept_rate={ssum['accept_rate']:.4f}, moves_mean={ssum['moves_mean']:.3f}, "
                f"energy_cache_hit_rate={ssum['cache_hit_rate']:.3f}, steps_per_sec={ssum['steps_per_sec']:.1f}"
            )
        lines.append("  P_eq(topology): mean ± 95% CI across chains")
        for topo in topologies:
            lines.append(f"    {topo} = {p_st[topo]:.6f} ± {ci_st[topo]:.6f}")
        if p_exact is not None:
            p_exact_vec = {k: float(p_exact.get(k, 0.0)) for k in topologies}
            lines.append(f"  KL(P_eq||P_exact) = {float(kl_divergence(p_st, p_exact_vec)):.6f}")
        lines.append("")

    if kl_by_start:
        vals = [x[1] for x in kl_by_start]
        lines.append(f"KL_max(P_eq||P_exact) = {float(max(vals)):.6f}")
        lines.append(f"KL_mean(P_eq||P_exact) = {float(np.mean(np.asarray(vals, dtype=float))):.6f}")
    lines.append("")
    lines.append("TIMING")
    lines.append(f"START_TS={start_ts}")
    lines.append(f"END_TS={end_ts}")
    lines.append(f"ELAPSED_TOTAL_SEC={elapsed_total:.6f}")
    lines.append(f"STEPS_TOTAL={steps_total}")
    # Backward-compatible + explicit alias for downstream parsers.
    lines.append(f"STEPS_PER_SEC={steps_per_sec_total:.1f}")
    lines.append(f"STEPS_PER_SEC_TOTAL={steps_per_sec_total:.1f}")

    write_growth_txt(out_name, lines)
    print(f"[CHEM-EQ] wrote {out_txt}")
    print(
        f"Wall-clock: start={start_ts} end={end_ts} elapsed_total_sec={elapsed_total:.3f} "
        f"steps_total={steps_total} steps_per_sec_total={steps_per_sec_total:.1f}"
    )
    return p_mean, out_txt


def main(argv: Optional[Sequence[str]] = None) -> None:
    cfg = _parse_args(argv)
    run_runner(cfg)


if __name__ == "__main__":
    main()
