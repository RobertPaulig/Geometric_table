from __future__ import annotations

import argparse
import csv
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from analysis.chem.chem_validation_eq_runner import RunnerConfig, run_runner
from analysis.chem.core2_fit import kl_divergence
from analysis.io_utils import results_path
from analysis.utils.timing import now_iso
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


def _load_lambda_from_exact_mode_a() -> Optional[float]:
    try:
        txt = results_path("mh_kernel_3_c6_exact_modeA.txt").read_text(encoding="utf-8", errors="replace")
    except Exception:
        return None
    for line in txt.splitlines():
        if line.startswith("mode=") and "lambda=" in line:
            # mode=A, lambda=0.949, T=1.000
            try:
                parts = [p.strip() for p in line.split(",")]
                for p in parts:
                    if p.startswith("lambda="):
                        return float(p.split("=", 1)[1])
            except Exception:
                return None
    return None


@dataclass
class ScanConfig:
    N: int = 6
    mode: str = "A"
    steps_list: Tuple[int, ...] = (2_000, 5_000, 10_000, 20_000)
    burnin_frac: float = 0.1
    thin: int = 10
    chains: int = 3
    seed: int = 0
    lam: Optional[float] = None
    progress: bool = True


def _parse_args(argv: Optional[Sequence[str]] = None) -> ScanConfig:
    ap = argparse.ArgumentParser(description="EQ-TARGET-1: steps->KL curve for fixed-N MCMC equilibrium runner.")
    ap.add_argument("--N", type=int, default=6)
    ap.add_argument("--mode", type=str, default="A", choices=["A", "B", "C"])
    ap.add_argument("--steps", type=int, nargs="*", default=[2000, 5000, 10000, 20000])
    ap.add_argument("--burnin_frac", type=float, default=0.1)
    ap.add_argument("--thin", type=int, default=10)
    ap.add_argument("--chains", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--lambda", dest="lam", type=float, default=None)
    ap.add_argument("--progress", action=argparse.BooleanOptionalAction, default=True)
    args = ap.parse_args(argv)
    return ScanConfig(
        N=int(args.N),
        mode=str(args.mode).upper(),
        steps_list=tuple(int(x) for x in args.steps),
        burnin_frac=float(args.burnin_frac),
        thin=int(args.thin),
        chains=int(args.chains),
        seed=int(args.seed),
        lam=float(args.lam) if args.lam is not None else None,
        progress=bool(args.progress),
    )


def main(argv: Optional[Sequence[str]] = None) -> None:
    cfg = _parse_args(argv)

    start_ts = now_iso()
    t0_total = time.perf_counter()

    if cfg.N != 6 or cfg.mode.upper() != "A":
        raise ValueError("EQ-TARGET-1 currently supports N=6 mode A (baseline vs exact).")

    p_exact = _load_p_exact_hexane_mode_a()
    if p_exact is None:
        raise RuntimeError("Missing exact baseline: results/mh_kernel_3_c6_exact_modeA.txt")
    keys = list(sorted(p_exact.keys()))
    p_exact_vec = {k: float(p_exact.get(k, 0.0)) for k in keys}

    lam = float(cfg.lam) if cfg.lam is not None else None
    if lam is None:
        lam = _load_lambda_from_exact_mode_a()

    rows: List[Dict[str, object]] = []
    lines: List[str] = []
    lines.append("EQ-TARGET-1: steps -> KL(P_eq || P_exact) curve (N=6, mode A)")
    lines.append(
        f"steps_list={list(cfg.steps_list)}, chains={cfg.chains}, thin={cfg.thin}, burnin_frac={cfg.burnin_frac}, lambda={lam}"
    )
    lines.append("")

    out_csv = results_path("eq_target_1_N6_modeA.csv")
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    for steps in cfg.steps_list:
        burnin = int(max(0, round(float(cfg.burnin_frac) * float(steps))))
        runner_cfg = RunnerConfig(
            N=int(cfg.N),
            mode=str(cfg.mode).upper(),
            steps=int(steps),
            burnin=int(burnin),
            thin=int(cfg.thin),
            chains=int(cfg.chains),
            seed=int(cfg.seed),
            lam=float(lam) if lam is not None else None,
            progress=bool(cfg.progress),
        )

        t0 = time.perf_counter()
        p_eq, _ = run_runner(runner_cfg)
        elapsed = time.perf_counter() - t0

        p_eq_vec = {k: float(p_eq.get(k, 0.0)) for k in keys}
        kl = float(kl_divergence(p_eq_vec, p_exact_vec))

        row = {
            "steps": int(steps),
            "burnin": int(burnin),
            "thin": int(cfg.thin),
            "chains": int(cfg.chains),
            "elapsed_sec": float(elapsed),
            "steps_per_sec": float((int(steps) * int(cfg.chains)) / max(1e-9, elapsed)),
            "kl_mcmc_exact": float(kl),
        }
        rows.append(row)
        lines.append(
            f"steps={steps:6d} burnin={burnin:5d} elapsed={elapsed:7.3f}s "
            f"steps/sec={row['steps_per_sec']:.1f} KL={kl:.6f}"
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

    out_txt = write_growth_txt("eq_target_1_N6_modeA", lines)
    print(f"[EQ-TARGET-1] done.")
    print(f"CSV: {out_csv}")
    print(f"Summary: {out_txt}")
    print(f"Wall-clock: start={start_ts} end={end_ts} elapsed_total_sec={elapsed_total:.3f}")


if __name__ == "__main__":
    main()
