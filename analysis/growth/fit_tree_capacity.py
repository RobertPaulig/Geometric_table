from __future__ import annotations

import numpy as np
import pandas as pd

from analysis.growth.reporting import write_growth_txt
from analysis.io_utils import results_path


def main() -> None:
    stats_path = results_path("tree_growth_law_fdm_stats.txt")
    if not stats_path.exists():
        raise SystemExit(
            f"{stats_path} not found. Run analysis.test_tree_growth_law_fdm first."
        )

    df = pd.read_csv(stats_path)
    # Use rows with at least some trees
    mask = df["n_trees"] > 0
    df = df[mask].copy()
    if df.empty:
        raise SystemExit("No tree data found in tree_growth_law_fdm_stats.txt")

    n = df["n_mean"].values.astype(float)
    C1 = df["C1_mean"].values.astype(float)
    C2 = df["C2_mean"].values.astype(float)

    denom = np.sum(n * np.log1p(n))
    if denom <= 0.0:
        raise SystemExit("Non-positive denominator in alpha_tree fit.")

    alpha_v1 = float(np.sum(C1) / denom)
    alpha_fdm = float(np.sum(C2) / denom)

    header = (
        "Tree capacity fit (C ~ alpha * n log(1+n))\n"
        "========================================="
    )
    lines = [
        f"alpha_tree_v1  = {alpha_v1:.4f}",
        f"alpha_tree_fdm = {alpha_fdm:.4f}",
        f"n_mean range   = [{n.min():.2f}, {n.max():.2f}]",
        f"max C1_mean    = {C1.max():.2f}",
        f"max C2_mean    = {C2.max():.2f}",
    ]

    out_path = write_growth_txt(
        name="tree_capacity_fit",
        lines=lines,
        header=header,
    )

    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
