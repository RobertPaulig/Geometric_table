from __future__ import annotations

from pathlib import Path
import argparse

import numpy as np

from core.grower import GrowthParams, grow_molecule_christmas_tree
from core.complexity import compute_complexity_features
from core.growth_config import load_growth_config


RESULTS_DIR = Path("results")


def main(argv=None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        required=False,
        help="Путь к YAML/JSON-конфигу роста (например, configs/growth_baseline_v5.yaml)",
    )
    args = parser.parse_args(argv)

    # QSG v5.0 baseline: деревьевый режим grower'а.
    # Эти параметры считаются «зелёной зоной» и не меняются
    # в рамках CY-1 R&D-экспериментов.
    seeds = ["Li", "Na", "K", "Be", "Mg", "Ca", "C", "N", "O", "Si", "P", "S"]

    if args.config:
        cfg = load_growth_config(args.config)
        params = cfg.to_growth_params()
    else:
        params = GrowthParams(max_depth=4, max_atoms=25)

    num_runs = 1000

    RESULTS_DIR.mkdir(exist_ok=True)

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

    out = RESULTS_DIR / "cycle_stats.txt"
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
