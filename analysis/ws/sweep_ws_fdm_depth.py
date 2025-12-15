from __future__ import annotations

import csv
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np

from core.shape_observables import get_shape_observables, thermo_fingerprint_for_shape
from core.thermo_config import ThermoConfig, override_thermo_config


Z_VALUES: List[int] = [1, 6, 8, 14, 26]
BASE_VALUES: List[int] = [2, 3, 4]
DEPTH_VALUES: List[int] = [5, 6, 7, 8, 9, 10]


@dataclass
class SweepConfig:
    repeats: int = 5
    warmup: int = 2


def _kurt_and_time(Z: int, thermo: ThermoConfig) -> Tuple[float, float]:
    with override_thermo_config(thermo):
        fp = thermo_fingerprint_for_shape(thermo)
        t0 = time.perf_counter()
        obs = get_shape_observables(Z, fp)
        dt = time.perf_counter() - t0
    return float(obs.kurt_ws), dt


def run_sweep(cfg: SweepConfig):
    results_dir = Path("results")
    results_dir.mkdir(parents=True, exist_ok=True)
    csv_path = results_dir / "ws_fdm_sweep.csv"
    summary_path = results_dir / "ws_fdm_sweep_summary.txt"

    rows = []

    # Baseline trapz timings and kurtosis
    baseline = {}
    for Z in Z_VALUES:
        thermo_trapz = ThermoConfig(ws_integrator="trapz")
        # warmup
        for _ in range(cfg.warmup):
            _kurt_and_time(Z, thermo_trapz)
        times = []
        kurt = None
        for _ in range(cfg.repeats):
            # очищаем кэш перед замером
            get_shape_observables.cache_clear()
            k, dt = _kurt_and_time(Z, thermo_trapz)
            times.append(dt)
            kurt = k
        t_trapz = float(np.median(times))
        baseline[Z] = (kurt, t_trapz)

    # Sweep over base, depth
    for base in BASE_VALUES:
        for depth in DEPTH_VALUES:
            for Z in Z_VALUES:
                kurt_trapz, t_trapz = baseline[Z]
                thermo_fdm = ThermoConfig(
                    ws_integrator="fdm",
                    ws_fdm_base=base,
                    ws_fdm_depth=depth,
                )
                # warmup
                for _ in range(cfg.warmup):
                    _kurt_and_time(Z, thermo_fdm)
                times_fdm = []
                kurt_fdm = None
                for _ in range(cfg.repeats):
                    get_shape_observables.cache_clear()
                    k_fdm, dt_fdm = _kurt_and_time(Z, thermo_fdm)
                    times_fdm.append(dt_fdm)
                    kurt_fdm = k_fdm
                t_fdm = float(np.median(times_fdm))
                abs_err = abs(kurt_fdm - kurt_trapz)
                speedup = t_trapz / t_fdm if t_fdm > 0 else float("inf")

                rows.append(
                    {
                        "base": base,
                        "depth": depth,
                        "Z": Z,
                        "kurt_trapz": kurt_trapz,
                        "kurt_fdm": kurt_fdm,
                        "abs_err": abs_err,
                        "t_trapz_sec": t_trapz,
                        "t_fdm_sec": t_fdm,
                        "speedup": speedup,
                    }
                )

    fieldnames = [
        "base",
        "depth",
        "Z",
        "kurt_trapz",
        "kurt_fdm",
        "abs_err",
        "t_trapz_sec",
        "t_fdm_sec",
        "speedup",
    ]

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    # Aggregate per (base, depth)
    summary = {}
    for r in rows:
        key = (r["base"], r["depth"])
        summary.setdefault(key, {"abs_errs": [], "speedups": []})
        summary[key]["abs_errs"].append(r["abs_err"])
        summary[key]["speedups"].append(r["speedup"])

    with summary_path.open("w", encoding="utf-8") as f:
        f.write("FAST-SPECTRUM-1 WS FDM depth/base sweep\n")
        f.write(f"Z_values={Z_VALUES}\n")
        f.write(f"BASE_VALUES={BASE_VALUES}\n")
        f.write(f"DEPTH_VALUES={DEPTH_VALUES}\n")
        f.write(f"repeats={cfg.repeats}, warmup={cfg.warmup}\n\n")

        best = None
        for (base, depth), stats in sorted(summary.items()):
            abs_errs = np.asarray(stats["abs_errs"], dtype=float)
            speedups = np.asarray(stats["speedups"], dtype=float)
            max_err = float(abs_errs.max())
            median_speedup = float(np.median(speedups))
            summary[(base, depth)]["max_err"] = max_err
            summary[(base, depth)]["median_speedup"] = median_speedup

            f.write(
                f"base={base}, depth={depth}: "
                f"max_abs_err_over_Z={max_err:.5f}, "
                f"median_speedup_over_Z={median_speedup:.2f}x\n"
            )

            # Select best candidate: passes DoD and maximizes median_speedup,
            # tie-breaker: smaller N = base^depth
            if max_err <= 0.05 and median_speedup >= 1.8:
                N = base ** depth
                if best is None:
                    best = (base, depth, median_speedup, max_err, N)
                else:
                    _, _, best_speedup, _, best_N = best
                    if median_speedup > best_speedup or (
                        abs(median_speedup - best_speedup) < 1e-9 and N < best_N
                    ):
                        best = (base, depth, median_speedup, max_err, N)

        f.write("\n")
        if best is not None:
            base_star, depth_star, med_sp, max_err = best[0], best[1], best[2], best[3]
            f.write(
                "SELECTED base*, depth* = "
                f"({base_star}, {depth_star}) "
                f"because max_abs_err={max_err:.5f} <= 0.05 "
                f"and median_speedup={med_sp:.2f}x >= 1.8x\n"
            )
        else:
            f.write("NO candidate satisfies DoD (max_abs_err<=0.05 and median_speedup>=1.8x).\n")

    return csv_path, summary_path


def main() -> None:
    cfg = SweepConfig()
    csv_path, summary_path = run_sweep(cfg)
    print("[FAST-SPECTRUM-1] WS FDM depth/base sweep done.")
    print(f"CSV: {csv_path}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()

