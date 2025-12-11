from __future__ import annotations

"""
analyze_loopy_modes.py — CY-1/step2: анализ режимов loopy-growth.

Измеряет frac_cycles, max_cyclomatic, n_mean и cycle_load_mean
для двух эталонных режимов CY-1-A и CY-1-B.
"""

from pathlib import Path
from typing import Iterable, List

import numpy as np

from core.grower import GrowthParams, grow_molecule_christmas_tree
from core.complexity import compute_complexity_features


RESULTS_DIR = Path("results")


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

    RESULTS_DIR.mkdir(exist_ok=True)
    out_path = RESULTS_DIR / f"loopy_modes_{name}.txt"
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {out_path}")


def main() -> None:
    seeds = ["Li", "Na", "K", "Be", "Mg", "Ca", "C", "N", "O", "Si", "P", "S"]

    # Режимы CY-1-A и CY-1-B выбраны из results/cycle_param_scan.csv
    # как конфигурации с умеренной и более высокой долей циклов.
    params_A = GrowthParams(
        max_depth=4,
        max_atoms=25,
        p_continue_base=0.5,
        chi_sensitivity=0.3,
        role_bonus_hub=0.4,
        role_penalty_terminator=-0.6,
        temperature=1.0,
        allow_cycles=True,
        max_extra_bonds=3,
        p_extra_bond=0.3,
    )
    params_B = GrowthParams(
        max_depth=4,
        max_atoms=25,
        p_continue_base=0.9,
        chi_sensitivity=0.3,
        role_bonus_hub=0.2,
        role_penalty_terminator=-0.6,
        temperature=1.0,
        allow_cycles=True,
        max_extra_bonds=3,
        p_extra_bond=0.3,
    )

    run_regime("CY1A", params_A, seeds)
    run_regime("CY1B", params_B, seeds)


if __name__ == "__main__":
    main()

