from __future__ import annotations

import argparse
import csv
import time
from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from analysis.chem.mixing_diagnostics_1 import compute_mixing_diagnostics
from analysis.chem.topology_mcmc import Edge, run_fixed_n_tree_mcmc, tree_topology_edge_key_from_edges
from analysis.io_utils import results_path
from analysis.utils.progress import progress_iter
from analysis.utils.timing import now_iso


def _start_edges_for_spec(n: int, spec: str) -> List[Edge]:
    spec = str(spec)
    if spec == "path":
        return [(int(i), int(i + 1)) for i in range(int(n) - 1)]
    if spec == "max_branch":
        if int(n) <= 5:
            return [(0, i) for i in range(1, int(n))]
        edges: List[Edge] = [(0, 1), (0, 2), (0, 3), (0, 4)]
        last = 4
        for v in range(5, int(n)):
            edges.append((last, v))
            last = v
        return edges
    raise ValueError(f"Unknown start_spec: {spec!r}")


@dataclass
class Cfg:
    n: int = 15
    steps_grid: Tuple[int, ...] = (2_000_000, 4_000_000, 8_000_000, 16_000_000)
    chains: int = 3
    thin: int = 10
    start_specs: Tuple[str, ...] = ("path", "max_branch")
    seed: int = 0
    progress: bool = True
    step_progress: bool = True
    step_heartbeat_every: int = 200_000


def _parse_args(argv: Optional[Sequence[str]] = None) -> Cfg:
    ap = argparse.ArgumentParser(description="EQ-TARGET-3: scan steps for arbitrary N using mixing diagnostics (Mode A).")
    ap.add_argument("--N", type=int, required=True)
    ap.add_argument("--steps_grid", type=int, nargs="*", default=[2_000_000, 4_000_000, 8_000_000, 16_000_000])
    ap.add_argument("--chains", type=int, default=3)
    ap.add_argument("--thin", type=int, default=10)
    ap.add_argument("--start_specs", type=str, nargs="*", default=["path", "max_branch"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--progress", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--step_progress", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--step_heartbeat_every", type=int, default=200_000)
    args = ap.parse_args(argv)
    return Cfg(
        n=int(args.N),
        steps_grid=tuple(int(x) for x in args.steps_grid),
        chains=int(args.chains),
        thin=int(args.thin),
        start_specs=tuple(str(x) for x in args.start_specs),
        seed=int(args.seed),
        progress=bool(args.progress),
        step_progress=bool(args.step_progress),
        step_heartbeat_every=int(args.step_heartbeat_every),
    )


def main(argv: Optional[Sequence[str]] = None) -> None:
    cfg = _parse_args(argv)
    out_stub = f"eq_target_3_N{cfg.n}_modeA"
    out_csv = results_path(f"{out_stub}.csv")
    out_txt = results_path(f"{out_stub}.txt")

    start_ts = now_iso()
    t0 = time.perf_counter()

    rows: List[Dict[str, object]] = []
    lines: List[str] = []
    lines.append(f"EQ-TARGET-3: N={cfg.n}, mode=A, chains={cfg.chains}, thin={cfg.thin}, starts={list(cfg.start_specs)}")
    lines.append(f"steps_grid={list(cfg.steps_grid)}")
    lines.append("")

    for steps_total_budget in cfg.steps_grid:
        burnin = int(max(0, round(0.1 * float(steps_total_budget))))

        per_start_diags = {}
        per_start_elapsed = {}
        total_elapsed = 0.0
        total_steps_done = 0
        cache_hits_total = 0
        cache_misses_total = 0

        for start_spec in cfg.start_specs:
            seqs: List[List[str]] = []
            energies: List[List[float]] = []
            t_start = time.perf_counter()

            # If step_progress is enabled, the outer "chains" progress is redundant (and produces 0.0 it/s).
            chains_iter = progress_iter(
                range(int(cfg.chains)),
                total=int(cfg.chains),
                desc=f"EQ-TARGET-3 N{cfg.n} steps={steps_total_budget} start={start_spec}",
                enabled=bool(cfg.progress) and not bool(cfg.step_progress),
            )
            for chain_idx in chains_iter:
                edges0 = _start_edges_for_spec(cfg.n, start_spec)

                pbar = None
                hb_every = max(1, int(cfg.step_heartbeat_every))
                if bool(cfg.progress) and bool(cfg.step_progress):
                    try:
                        from tqdm import tqdm  # type: ignore

                        pbar = tqdm(
                            total=int(steps_total_budget),
                            desc=f"EQ N{cfg.n} start={start_spec} chain={int(chain_idx)}",
                            unit="step",
                            mininterval=1.0,
                            smoothing=0.1,
                        )
                    except Exception:
                        pbar = None

                last_hb_step = 0

                def _heartbeat(info: Mapping[str, float]) -> None:
                    nonlocal last_hb_step
                    step = int(info.get("step", 0))
                    if pbar is not None:
                        delta = max(0, step - last_hb_step)
                        last_hb_step = step
                        pbar.update(delta)
                        pbar.set_postfix(
                            steps_per_sec=f"{info.get('heartbeat_steps_per_sec', 0.0):.0f}",
                            acc=f"{info.get('accept_rate', 0.0):.3f}",
                            hit=f"{info.get('energy_cache_hit_rate', 0.0):.3f}",
                            misses=str(int(info.get("energy_cache_misses_seen", 0.0))),
                        )
                    else:
                        # Fallback heartbeat print.
                        print(
                            f"[EQ-TARGET-3 N={cfg.n} start={start_spec} chain={int(chain_idx)}] "
                            f"step={step}/{int(steps_total_budget)} "
                            f"steps/sec={info.get('heartbeat_steps_per_sec', 0.0):.0f} "
                            f"acc={info.get('accept_rate', 0.0):.3f} "
                            f"hit={info.get('energy_cache_hit_rate', 0.0):.3f} "
                            f"misses_seen={int(info.get('energy_cache_misses_seen', 0.0))}"
                        )

                try:
                    samples, summary = run_fixed_n_tree_mcmc(
                        n=int(cfg.n),
                        steps=int(steps_total_budget),
                        burnin=int(burnin),
                        thin=int(cfg.thin),
                        backend="fdm",
                        lam=1.0,
                        temperature_T=1.0,
                        seed=int(cfg.seed) + 101 * int(chain_idx) + 10_000 * int(steps_total_budget),
                        max_valence=4,
                        topology_key_fn_edges=tree_topology_edge_key_from_edges,
                        start_edges=edges0,
                        progress=None,
                        step_heartbeat_every=hb_every if bool(cfg.progress) else 0,
                        step_heartbeat=_heartbeat if bool(cfg.progress) else None,
                    )
                finally:
                    if pbar is not None:
                        try:
                            # Ensure the bar reaches total, then close.
                            if last_hb_step < int(steps_total_budget):
                                pbar.update(int(steps_total_budget) - int(last_hb_step))
                        finally:
                            pbar.close()
                seqs.append([str(s["topology"]) for s in samples])
                energies.append([float(s["energy"]) for s in samples])
                cache_hits_total += int(summary.energy_cache_hits)
                cache_misses_total += int(summary.energy_cache_misses)
                total_steps_done += int(summary.steps)

            dt = time.perf_counter() - t_start
            total_elapsed += dt
            per_start_elapsed[str(start_spec)] = float(dt)
            per_start_diags[str(start_spec)] = compute_mixing_diagnostics(
                n=int(cfg.n),
                steps=int(steps_total_budget),
                burnin=int(burnin),
                thin=int(cfg.thin),
                start_spec=str(start_spec),
                topology_sequences_by_chain=seqs,
                energy_traces_by_chain=energies,
            )

        kl_max = float(np.nanmax(np.asarray([d.kl_pairwise_max for d in per_start_diags.values()], dtype=float)))
        kl_split_max = float(np.nanmax(np.asarray([d.kl_split_max for d in per_start_diags.values()], dtype=float)))
        rhat_max = float(np.nanmax(np.asarray([d.rhat_energy for d in per_start_diags.values()], dtype=float)))
        ess_min = float(np.nanmin(np.asarray([d.ess_energy_min for d in per_start_diags.values()], dtype=float)))
        steps_per_sec_total = float(total_steps_done) / float(total_elapsed) if total_elapsed > 0 else 0.0
        cache_total = cache_hits_total + cache_misses_total
        cache_hit_rate = float(cache_hits_total) / float(cache_total) if cache_total > 0 else 0.0

        row = {
            "N": int(cfg.n),
            "steps": int(steps_total_budget),
            "burnin": int(burnin),
            "thin": int(cfg.thin),
            "chains": int(cfg.chains),
            "starts": "|".join(cfg.start_specs),
            "elapsed_sec": float(total_elapsed),
            "steps_total": int(total_steps_done),
            "steps_per_sec_total": float(steps_per_sec_total),
            "energy_cache_hit_rate": float(cache_hit_rate),
            "KL_max_pairwise": float(kl_max),
            "KL_split_max": float(kl_split_max),
            "Rhat_energy_max": float(rhat_max),
            "ESS_energy_min": float(ess_min),
        }
        rows.append(row)
        line = (
            f"steps={int(steps_total_budget)}: KL_max={kl_max:.6g} KL_split_max={kl_split_max:.6g} "
            f"Rhat_max={rhat_max:.4g} ESS_min={ess_min:.1f} "
            f"elapsed={total_elapsed:.1f}s steps/s={steps_per_sec_total:.0f} hit={cache_hit_rate:.3f}"
        )
        lines.append(line)
        print(f"[EQ-TARGET-3] {line}")
        print(
            f"[EQ-TARGET-3] STEPS_TOTAL={int(total_steps_done)} "
            f"STEPS_PER_SEC_TOTAL={steps_per_sec_total:.1f} "
            f"ELAPSED_SEC={total_elapsed:.3f}"
        )

    end_ts = now_iso()
    elapsed_total = time.perf_counter() - t0

    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)

    lines.append("")
    lines.append(f"START_TS={start_ts}")
    lines.append(f"END_TS={end_ts}")
    lines.append(f"ELAPSED_TOTAL_SEC={elapsed_total:.6f}")
    out_txt.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[EQ-TARGET-3] wrote {out_csv}")
    print(f"[EQ-TARGET-3] wrote {out_txt}")
    print(f"Wall-clock: start={start_ts} end={end_ts} elapsed_total_sec={elapsed_total:.3f}")


if __name__ == "__main__":
    main()
