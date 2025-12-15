from __future__ import annotations

import csv
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List

from core.shape_observables import get_shape_observables, thermo_fingerprint_for_shape
from core.thermo_config import ThermoConfig, override_thermo_config


@dataclass
class BenchConfig:
    z_values: List[int] = None
    ws_fdm_depth: int = 9
    ws_fdm_base: int = 2

    def __post_init__(self):
        if self.z_values is None:
            self.z_values = [1, 6, 8, 14, 26]


def _kurt_with_cfg(Z: int, cfg: ThermoConfig) -> float:
    with override_thermo_config(cfg):
        fp = thermo_fingerprint_for_shape(cfg)
        obs = get_shape_observables(Z, fp)
    return float(obs.kurt_ws)


def run_benchmark(cfg: BenchConfig):
    results_dir = Path("results")
    results_dir.mkdir(parents=True, exist_ok=True)
    csv_path = results_dir / "ws_integrator_bench.csv"
    summary_path = results_dir / "ws_integrator_bench_summary.txt"

    rows = []

    for Z in cfg.z_values:
        # trapz baseline
        cfg_trapz = ThermoConfig(ws_integrator="trapz")
        t0 = time.perf_counter()
        k_trapz = _kurt_with_cfg(Z, cfg_trapz)
        t_trapz = time.perf_counter() - t0

        # fdm
        cfg_fdm = ThermoConfig(
            ws_integrator="fdm",
            ws_fdm_depth=cfg.ws_fdm_depth,
            ws_fdm_base=cfg.ws_fdm_base,
        )
        t1 = time.perf_counter()
        k_fdm = _kurt_with_cfg(Z, cfg_fdm)
        t_fdm = time.perf_counter() - t1

        abs_err = abs(k_fdm - k_trapz)
        speedup = t_trapz / t_fdm if t_fdm > 0 else float("inf")

        rows.append(
            {
                "Z": Z,
                "kurt_trapz": k_trapz,
                "kurt_fdm": k_fdm,
                "abs_err": abs_err,
                "t_trapz_sec": t_trapz,
                "t_fdm_sec": t_fdm,
                "speedup": speedup,
            }
        )

    fieldnames = [
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

    # summary
    import numpy as np

    abs_errs = [r["abs_err"] for r in rows]
    speedups = [r["speedup"] for r in rows]

    max_err = max(abs_errs) if abs_errs else 0.0
    med_err = float(np.median(abs_errs)) if abs_errs else 0.0
    med_speedup = float(np.median(speedups)) if speedups else 1.0

    with summary_path.open("w", encoding="utf-8") as f:
        f.write("FAST-SPECTRUM-0 WS integrator benchmark\n")
        f.write(f"Z_values={cfg.z_values}\n")
        f.write(f"ws_fdm_depth={cfg.ws_fdm_depth}, ws_fdm_base={cfg.ws_fdm_base}\n\n")
        for r in rows:
            f.write(
                f"Z={r['Z']}: kurt_trapz={r['kurt_trapz']:.4f}, "
                f"kurt_fdm={r['kurt_fdm']:.4f}, abs_err={r['abs_err']:.4f}, "
                f"t_trapz={r['t_trapz_sec']:.4f}s, t_fdm={r['t_fdm_sec']:.4f}s, "
                f"speedup={r['speedup']:.2f}x\n"
            )
        f.write("\n")
        f.write(f"max_abs_err={max_err:.4f}\n")
        f.write(f"median_abs_err={med_err:.4f}\n")
        f.write(f"median_speedup={med_speedup:.2f}x\n")

    return csv_path, summary_path


def main() -> None:
    cfg = BenchConfig()
    csv_path, summary_path = run_benchmark(cfg)
    print("[FAST-SPECTRUM-0] WS integrator benchmark done.")
    print(f"CSV: {csv_path}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()

