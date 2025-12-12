from __future__ import annotations

"""
analyze_loopy_modes.py — CY-1/step2: анализ режимов loopy-growth.

Измеряет frac_cycles, max_cyclomatic, n_mean и cycle_load_mean
для двух эталонных режимов CY-1-A и CY-1-B.
"""

from typing import Iterable, List

import numpy as np

from analysis.io_utils import results_path
from core.grower import GrowthParams, grow_molecule_christmas_tree
from analysis.seeds import GROWTH_SEEDS
from core.complexity import compute_complexity_features
from core.growth_config import load_growth_config


def run_regime(name: str, params: GrowthParams, seeds: Iterable[str], num_runs: int = 500) -> None:
    lines: List[str] = []

    for seed in seeds:
        cyclomatic_vals = []
        n_vals = []

        rng = np.random.default_rng(1234)
        for _ in range(num_runs):
            mol = grow_molecule_christmas_tree(seed, params=params, rng=rng)
            adj = mol.adjacency_matrix()
            feats = compute_complexity_features(adj)
            cyclomatic_vals.append(float(feats.cyclomatic))
            # Если в features нет явного n, используем размер матрицы.
            n_vals.append(float(getattr(feats, "n", adj.shape[0])))

        arr_c = np.array(cyclomatic_vals, dtype=float)
        arr_n = np.array(n_vals, dtype=float)

        frac_cycles = float(np.mean(arr_c > 0))
        max_cyclomatic = int(arr_c.max())
        n_mean = float(arr_n.mean())
        with np.errstate(divide="ignore", invalid="ignore"):
            cycle_load = arr_c / np.maximum(arr_n, 1.0)
        cycle_load_mean = float(np.nanmean(cycle_load))

        lines.append(
            f"{name}/{seed}: frac_cycles={frac_cycles:.3f}, "
            f"max_cyclomatic={max_cyclomatic}, "
            f"n_mean={n_mean:.2f}, "
            f"cycle_load_mean={cycle_load_mean:.3f}"
        )

    out_path = results_path(f"loopy_modes_{name}.txt")
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {out_path}")


def main() -> None:
    # Режимы CY-1-A и CY-1-B берутся из конфигов growth_cy1a/growth_cy1b.
    params_A = load_growth_config("configs/growth_cy1a.yaml").to_growth_params()
    params_B = load_growth_config("configs/growth_cy1b.yaml").to_growth_params()

    run_regime("CY1A", params_A, GROWTH_SEEDS)
    run_regime("CY1B", params_B, GROWTH_SEEDS)


if __name__ == "__main__":
    main()
