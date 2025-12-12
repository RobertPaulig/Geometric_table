from __future__ import annotations

import argparse
import numpy as np

from analysis.cli_common import script_banner
from analysis.growth_cli import make_growth_params_from_config_path
from analysis.growth.reporting import write_growth_txt
from analysis.seeds import GROWTH_SEEDS
from core.grower import grow_molecule_christmas_tree
from core.complexity import compute_complexity_features


def main(argv=None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        required=False,
        help="Путь к YAML/JSON-конфигу роста (например, configs/growth_baseline_v5.yaml)",
    )
    args = parser.parse_args(argv)

    with script_banner("analyze_cycle_stats"):
        # QSG v5.0 baseline: деревьевый режим grower'а.
        # Эти параметры считаются «зелёной зоной» и не меняются
        # в рамках CY-1 R&D-экспериментов.
        seeds = GROWTH_SEEDS
        params = make_growth_params_from_config_path(args.config)

        num_runs = 1000

        lines = []
        for seed in seeds:
            cyclomatic_vals = []
            for _ in range(num_runs):
                mol = grow_molecule_christmas_tree(seed, params=params)
                adj = mol.adjacency_matrix()
                feats = compute_complexity_features(adj)
                cyclomatic_vals.append(feats.cyclomatic)

            cyclomatic_arr = np.array(cyclomatic_vals, dtype=int)
            frac_trees = float(np.mean(cyclomatic_arr == 0))
            frac_cycles = float(np.mean(cyclomatic_arr > 0))
            max_cyclomatic = int(cyclomatic_arr.max()) if cyclomatic_arr.size > 0 else 0

            lines.append(
                f"{seed}: frac_trees={frac_trees:.3f}, "
                f"frac_cycles={frac_cycles:.3f}, "
                f"max_cyclomatic={max_cyclomatic}"
            )

        write_growth_txt(
            name="cycle_stats",
            lines=lines,
            header="[CYCLE STATS]",
        )


if __name__ == "__main__":
    main()
