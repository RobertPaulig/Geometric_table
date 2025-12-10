from __future__ import annotations

from math import log2
from pathlib import Path

import numpy as np
import pandas as pd

from core.grower import GrowthParams, grow_molecule_christmas_tree
from core.complexity import compute_complexity_features, compute_complexity_features_v2
from core.complexity_fdm import LAMBDA_FDM_DEFAULT, Q_FDM_DEFAULT
from core.geom_atoms import get_atom


RESULTS_DIR = Path("results")


def run_for_seed_fdm(seed: str, params: GrowthParams, n_trials: int = 100) -> dict:
    """
    Вариант закона роста деревьев, сравнивающий:
    - C1: heuristic complexity (v1),
    - C2: FDM-based complexity (v2 backend="fdm").
    Рассматриваются только деревья (cyclomatic == 0).
    """
    n_list = []
    c1_list = []
    c2_list = []
    c1_norm_list = []
    c2_norm_list = []
    frac_trees_count = 0

    for _ in range(n_trials):
        mol = grow_molecule_christmas_tree(seed, params)
        adj = mol.adjacency_matrix()

        feats_v1 = compute_complexity_features(adj)
        feats_v2 = compute_complexity_features_v2(adj, backend="fdm")

        n = feats_v1.n
        cycl = feats_v1.cyclomatic
        c1 = feats_v1.total
        c2 = feats_v2.total

        if n <= 1:
            continue

        if cycl == 0:
            frac_trees_count += 1
            n_list.append(n)
            c1_list.append(c1)
            c2_list.append(c2)

            denom = n * log2(1.0 + n)
            c1_norm_list.append(c1 / denom)
            c2_norm_list.append(c2 / denom)

    if not n_list:
        role = get_atom(seed).role if get_atom(seed) else ""
        return {
            "seed": seed,
            "role": role,
            "n_trials": n_trials,
            "n_trees": 0,
            "frac_trees": 0.0,
            "n_mean": np.nan,
            "C1_mean": np.nan,
            "C2_mean": np.nan,
            "C1_norm_mean": np.nan,
            "C1_norm_std": np.nan,
            "C2_norm_mean": np.nan,
            "C2_norm_std": np.nan,
        }

    n_arr = np.array(n_list, dtype=float)
    c1_arr = np.array(c1_list, dtype=float)
    c2_arr = np.array(c2_list, dtype=float)
    c1_norm_arr = np.array(c1_norm_list, dtype=float)
    c2_norm_arr = np.array(c2_norm_list, dtype=float)

    role = get_atom(seed).role if get_atom(seed) else ""

    return {
        "seed": seed,
        "role": role,
        "n_trials": n_trials,
        "n_trees": len(n_list),
        "frac_trees": len(n_list) / float(n_trials),
        "n_mean": float(n_arr.mean()),
        "C1_mean": float(c1_arr.mean()),
        "C2_mean": float(c2_arr.mean()),
        "lambda_fdm": LAMBDA_FDM_DEFAULT,
        "q_fdm": Q_FDM_DEFAULT,
        "C1_norm_mean": float(c1_norm_arr.mean()),
        "C1_norm_std": float(c1_norm_arr.std()),
        "C2_norm_mean": float(c2_norm_arr.mean()),
        "C2_norm_std": float(c2_norm_arr.std()),
    }


def main() -> None:
    RESULTS_DIR.mkdir(exist_ok=True)

    params = GrowthParams(max_depth=4, max_atoms=25)
    n_trials = 100

    seeds = ["Li", "Na", "K", "Be", "Mg", "Ca", "C", "N", "O", "Si", "P", "S"]

    rows = [run_for_seed_fdm(seed, params, n_trials=n_trials) for seed in seeds]

    df = pd.DataFrame(rows)
    out_path = RESULTS_DIR / "tree_growth_law_fdm_stats.txt"
    df.to_csv(out_path, index=False)

    # Summary block for quick R&D inspection
    with out_path.open("a", encoding="utf-8") as f:
        f.write("\n\nSummary:\n")
        f.write(
            f"- FDM parameters: lambda={LAMBDA_FDM_DEFAULT}, q={Q_FDM_DEFAULT}\n"
        )
        # Simple role-level aggregation for normalized complexities
        by_role = df.groupby("role")[["C1_norm_mean", "C2_norm_mean"]].mean()
        for role, row in by_role.iterrows():
            f.write(
                f"- role={role}: C1_norm_mean={row['C1_norm_mean']:.4f}, "
                f"C2_norm_mean={row['C2_norm_mean']:.4f}\n"
            )

    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
