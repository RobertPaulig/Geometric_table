from __future__ import annotations

"""
scan_temperature_effects.py — TEMP-1:
скан влияния температуры на рост и циклы.
"""

import argparse
from dataclasses import replace
from typing import Dict, List

import numpy as np
import pandas as pd

from analysis.growth_cli import make_growth_params_from_config_path
from analysis.io_utils import results_path
from analysis.seeds import GROWTH_SEEDS
from core.complexity import compute_complexity_features, compute_complexity_features_v2
from core.grower import grow_molecule_christmas_tree


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(
        description="TEMP-1: scan temperature effects on growth and cycles."
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to growth config YAML (CY-1-A/B, baseline, etc.).",
    )
    parser.add_argument(
        "--num-runs",
        type=int,
        default=200,
        help="Number of growth runs per (seed, T)",
    )
    args = parser.parse_args(argv)

    base_params = make_growth_params_from_config_path(args.config)

    T_grid = [0.5, 1.0, 2.0, 3.0]
    seeds = [s for s in GROWTH_SEEDS if s in ["C", "Si", "O", "S"]]
    num_runs = args.num_runs
    rng = np.random.default_rng(123456)

    rows: List[Dict] = []

    for T in T_grid:
        params_T = replace(base_params, temperature=T)
        for seed in seeds:
            n_vals: List[int] = []
            cycl_vals: List[int] = []
            cycle_load_vals: List[float] = []
            C_norm_fdm_vals: List[float] = []

            for _ in range(num_runs):
                mol = grow_molecule_christmas_tree(seed, params_T, rng=rng)
                adj = mol.adjacency_matrix()

                feats = compute_complexity_features(adj)
                n = feats.n
                cycl = feats.cyclomatic
                n_vals.append(n)
                cycl_vals.append(cycl)
                cycle_load_vals.append(cycl / n if n > 0 else 0.0)

                feats_fdm = compute_complexity_features_v2(adj, backend="fdm")
                # Нормировка как в других скриптах: total / (n * log2(1+n))
                if n > 0:
                    denom = n * np.log2(1.0 + n)
                    C_norm_fdm_vals.append(float(feats_fdm.total) / denom)
                else:
                    C_norm_fdm_vals.append(0.0)

            n_arr = np.array(n_vals, dtype=float)
            cycl_arr = np.array(cycl_vals, dtype=float)
            cycle_load_arr = np.array(cycle_load_vals, dtype=float)
            C_norm_arr = np.array(C_norm_fdm_vals, dtype=float)

            frac_cycles = float(np.mean(cycl_arr > 0))

            rows.append(
                {
                    "T": T,
                    "seed": seed,
                    "n_mean": float(n_arr.mean()),
                    "n_std": float(n_arr.std()),
                    "frac_cycles": frac_cycles,
                    "cycle_load_mean": float(cycle_load_arr.mean()),
                    "C_norm_fdm_mean": float(C_norm_arr.mean()),
                    "num_runs": num_runs,
                }
            )

    df = pd.DataFrame(rows)
    csv_path = results_path("temperature_scan_growth.csv")
    txt_path = results_path("temperature_scan_growth.txt")
    df.to_csv(csv_path, index=False)

    lines: List[str] = []
    lines.append("# TEMP-1: temperature_scan_growth")
    lines.append("")

    for seed in seeds:
        sub = df[df["seed"] == seed]
        if sub.empty:
            continue
        lines.append(f"Seed: {seed}")
        for _, row in sub.sort_values("T").iterrows():
            lines.append(
                f"  T={row['T']:.2f}: n_mean={row['n_mean']:.2f}, "
                f"frac_cycles={row['frac_cycles']:.3f}, "
                f"cycle_load_mean={row['cycle_load_mean']:.3f}, "
                f"C_norm_fdm_mean={row['C_norm_fdm_mean']:.3f}"
            )
        lines.append("")

    txt_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {csv_path} and {txt_path}")


if __name__ == "__main__":
    main()
