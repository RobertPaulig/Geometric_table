from __future__ import annotations

import argparse
import math
import re
import time
from pathlib import Path
from typing import Dict, Optional

from analysis.chem.topology_mcmc import run_fixed_n_tree_mcmc
from analysis.chem.chem_validation_1b_hexane import classify_hexane_topology
from analysis.io_utils import results_path
from analysis.growth.reporting import write_growth_txt
from analysis.utils.timing import now_iso


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


def _summarize_counts(samples: list[dict[str, object]]) -> Dict[str, float]:
    counts: Dict[str, int] = {}
    for row in samples:
        topo = str(row.get("topology"))
        counts[topo] = counts.get(topo, 0) + 1
    total = float(sum(counts.values())) if counts else 0.0
    if total <= 0:
        return {}
    return {k: v / total for k, v in sorted(counts.items(), key=lambda kv: kv[0])}


def main() -> None:
    start_ts = now_iso()
    t0_total = time.perf_counter()
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="A", choices=["A", "B", "C"])
    ap.add_argument("--steps", type=int, default=200_000)
    ap.add_argument("--burnin", type=int, default=20_000)
    ap.add_argument("--thin", type=int, default=10)
    ap.add_argument("--lambda", dest="lam", type=float, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--progress", action=argparse.BooleanOptionalAction, default=True)
    args = ap.parse_args()

    mode = str(args.mode).upper()
    lam = float(args.lam) if args.lam is not None else None
    if lam is None:
        lam = _load_lambda_star_from_hexane(mode) or 1.0

    backend = "fdm_entanglement"
    temperature_T = 1.0
    n = 6

    t0 = time.perf_counter()
    # Simple progress (tqdm if available, otherwise periodic stderr prints).
    enabled_progress = bool(args.progress)
    try:
        from tqdm import tqdm  # type: ignore

        pbar = tqdm(total=int(args.steps), desc=f"[MH-KERNEL-3:C6:MCMC:{mode}]", leave=True) if enabled_progress else None

        def _progress(n: int = 1) -> None:
            if pbar is not None:
                pbar.update(int(n))

    except Exception:
        import sys

        total = int(args.steps)
        every = 1000
        counter = {"n": 0}
        start = time.perf_counter()

        def _progress(n: int = 1) -> None:
            if not enabled_progress:
                return
            counter["n"] += int(n)
            k = counter["n"]
            if k == 1 or k % every == 0 or k >= total:
                elapsed = time.perf_counter() - start
                rate = (k / elapsed) if elapsed > 0 else 0.0
                print(f"[MH-KERNEL-3:C6:MCMC:{mode}] {k}/{total} ({rate:.1f} it/s, {elapsed:.1f}s)", file=sys.stderr, flush=True)

    samples, summary = run_fixed_n_tree_mcmc(
        n=n,
        steps=int(args.steps),
        burnin=int(args.burnin),
        thin=int(args.thin),
        backend=backend,
        lam=float(lam),
        temperature_T=float(temperature_T),
        seed=int(args.seed),
        max_valence=4,
        topology_classifier=classify_hexane_topology,
        progress=_progress,
    )
    if "pbar" in locals() and locals().get("pbar") is not None:
        try:
            locals()["pbar"].close()
        except Exception:
            pass
    dt = time.perf_counter() - t0
    elapsed_total = time.perf_counter() - t0_total
    end_ts = now_iso()

    p_mcmc = _summarize_counts(samples)
    out_name = f"mh_kernel_3_c6_mcmc_mode{mode}"
    out_csv = results_path(f"{out_name}.csv")

    # CSV: minimal samples, keep it small (already thinned).
    import csv

    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=sorted(samples[0].keys()) if samples else ["step"])
        w.writeheader()
        for row in samples:
            w.writerow(row)

    lines: list[str] = []
    lines.append("MH-KERNEL-3: C6 fixed-N MCMC (deg<=4)")
    lines.append(f"mode={mode}, backend={backend}, lambda={lam:.6g}, T={temperature_T:.3f}")
    lines.append(f"steps={int(args.steps)}, burnin={int(args.burnin)}, thin={int(args.thin)}, seed={int(args.seed)}")
    lines.append(f"elapsed_sec={dt:.3f}")
    lines.append("")
    lines.append("TIMING")
    lines.append(f"START_TS={start_ts}")
    lines.append(f"END_TS={end_ts}")
    lines.append(f"ELAPSED_TOTAL_SEC={elapsed_total:.6f}")
    lines.append("")
    lines.append("MCMC acceptance:")
    accept_rate = (float(summary.accepted) / float(summary.proposals)) if summary.proposals > 0 else 0.0
    lines.append(f"  proposals={summary.proposals}, accepted={summary.accepted}, rate={accept_rate:.6f}")
    lines.append(f"  moves_size_mean={summary.mean_moves:.3f}, moves_size_p90={summary.p90_moves:.3f}")
    lines.append(f"  log_qratio_mean={summary.mean_log_qratio:.6f}, log_qratio_p90={summary.p90_log_qratio:.6f}")
    lines.append("")
    lines.append("P_mcmc(topology):")
    for k, v in p_mcmc.items():
        lines.append(f"  {k} = {v:.6f}")

    write_growth_txt(out_name, lines)
    print(f"[MH-KERNEL-3:C6:MCMC] wrote {results_path(out_name + '.txt')}")
    print(f"[MH-KERNEL-3:C6:MCMC] wrote {out_csv}")
    print(f"Wall-clock: start={start_ts} end={end_ts} elapsed_total_sec={elapsed_total:.3f}")


if __name__ == "__main__":
    main()
