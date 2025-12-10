from __future__ import annotations

from math import log2
from pathlib import Path

import numpy as np

from core.grower import GrowthParams, grow_molecule_christmas_tree
from core.complexity import compute_complexity_features
from core.geom_atoms import get_atom


RESULTS_DIR = Path("results")


def run_for_seed(seed: str, params: GrowthParams, n_trials: int = 100) -> dict:
    """Run grower multiple times for a given seed and collect tree stats."""
    n_list = []
    c_list = []
    c_norm_list = []
    frac_trees_count = 0

    for _ in range(n_trials):
        mol = grow_molecule_christmas_tree(seed, params)
        adj = mol.adjacency_matrix()
        feats = compute_complexity_features(adj)

        n = feats.n
        cycl = feats.cyclomatic
        c = feats.total

        if n <= 1:
            continue

        if cycl == 0:
            frac_trees_count += 1
            n_list.append(n)
            c_list.append(c)
            c_norm = c / (n * log2(1.0 + n))
            c_norm_list.append(c_norm)

    if not n_list:
        return {
            "seed": seed,
            "role": get_atom(seed).role if get_atom(seed) else "",
            "n_trials": n_trials,
            "n_trees": 0,
            "frac_trees": 0.0,
            "n_mean": np.nan,
            "C_mean": np.nan,
            "C_norm_mean": np.nan,
            "C_norm_std": np.nan,
        }

    n_arr = np.array(n_list, dtype=float)
    c_arr = np.array(c_list, dtype=float)
    c_norm_arr = np.array(c_norm_list, dtype=float)

    return {
        "seed": seed,
        "role": get_atom(seed).role if get_atom(seed) else "",
        "n_trials": n_trials,
        "n_trees": len(n_list),
        "frac_trees": len(n_list) / float(n_trials),
        "n_mean": float(n_arr.mean()),
        "C_mean": float(c_arr.mean()),
        "C_norm_mean": float(c_norm_arr.mean()),
        "C_norm_std": float(c_norm_arr.std()),
    }


def main() -> None:
    import pandas as pd

    RESULTS_DIR.mkdir(exist_ok=True)

    params = GrowthParams(max_depth=4, max_atoms=25)
    n_trials = 100

    seeds = ["Li", "Na", "K", "Be", "Mg", "Ca", "C", "N", "O", "Si", "P", "S"]

    rows = [run_for_seed(seed, params, n_trials=n_trials) for seed in seeds]

    df = pd.DataFrame(rows)
    out_path = RESULTS_DIR / "tree_growth_law_stats.txt"
    df.to_csv(out_path, index=False)

    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()

