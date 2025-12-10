from __future__ import annotations

from collections import defaultdict
from itertools import product
from math import log2
from pathlib import Path

import numpy as np
import pandas as pd

from core.grower import GrowthParams, grow_molecule_christmas_tree
from core.complexity import compute_complexity_features_v2
from core.geom_atoms import get_atom
import core.complexity_fdm as cfdm


RESULTS_DIR = Path("results")


def run_for_seed_with_params(
    seed: str,
    params: GrowthParams,
    n_trials: int,
    lambda_weight: float,
    q: float,
) -> list[dict]:
    rows: list[dict] = []

    for _ in range(n_trials):
        mol = grow_molecule_christmas_tree(seed, params)
        adj = mol.adjacency_matrix()
        n = adj.shape[0]
        if n <= 1:
            continue

        feats_v1 = compute_complexity_features_v2(adj, backend="heuristic")
        feats_v2 = compute_complexity_features_v2(adj, backend="fdm")

        if feats_v1.cyclomatic != 0:
            # интересуют только деревья
            continue

        C1 = feats_v1.total
        C2 = feats_v2.total
        denom = n * log2(1.0 + n)
        C1_norm = C1 / denom
        C2_norm = C2 / denom

        rows.append(
            {
                "seed": seed,
                "role": get_atom(seed).role if get_atom(seed) else "",
                "n": n,
                "C1": C1,
                "C2": C2,
                "C1_norm": C1_norm,
                "C2_norm": C2_norm,
                "lambda_fdm": lambda_weight,
                "q_fdm": q,
            }
        )

    return rows


def main() -> None:
    RESULTS_DIR.mkdir(exist_ok=True)

    params = GrowthParams(max_depth=4, max_atoms=25)
    n_trials = 100
    seeds = ["Li", "Na", "K", "Be", "Mg", "Ca", "C", "N", "O", "Si", "P", "S"]

    lambda_grid = [0.3, 0.4, 0.5, 0.6, 0.7]
    q_grid = [1.0, 1.5, 2.0, 2.5]

    all_rows: list[dict] = []

    for lambda_weight, q in product(lambda_grid, q_grid):
        # установить параметры для текущего прогона
        cfdm.LAMBDA_FDM_DEFAULT = lambda_weight
        cfdm.Q_FDM_DEFAULT = q

        for seed in seeds:
            rows = run_for_seed_with_params(seed, params, n_trials, lambda_weight, q)
            all_rows.extend(rows)

    if not all_rows:
        print("No data collected.")
        return

    df = pd.DataFrame(all_rows)
    grid_path = RESULTS_DIR / "fdm_calibration_grid.csv"
    df.to_csv(grid_path, index=False)

    # агрегируем по (lambda, q, role)
    group_cols = ["lambda_fdm", "q_fdm", "role"]
    agg = (
        df.groupby(group_cols)
        .agg(
            C2_norm_mean=("C2_norm", "mean"),
            C2_norm_std=("C2_norm", "std"),
            n_samples=("C2_norm", "size"),
        )
        .reset_index()
    )

    agg_path = RESULTS_DIR / "fdm_calibration_summary.csv"
    agg.to_csv(agg_path, index=False)

    # строим метрику качества по (lambda, q)
    summary_rows: list[dict] = []
    for (lam, q), sub in agg.groupby(["lambda_fdm", "q_fdm"]):
        df_role = sub.set_index("role")
        roles = ["terminator", "bridge", "hub"]
        if not all(r in df_role.index for r in roles):
            order_ok = False
            delta_tb = float("nan")
            delta_bh = float("nan")
            min_gap = float("nan")
            sigma_max = float("nan")
            score = float("-inf")
        else:
            m_term = float(df_role.loc["terminator", "C2_norm_mean"])
            m_bridge = float(df_role.loc["bridge", "C2_norm_mean"])
            m_hub = float(df_role.loc["hub", "C2_norm_mean"])

            s_term = float(df_role.loc["terminator", "C2_norm_std"])
            s_bridge = float(df_role.loc["bridge", "C2_norm_std"])
            s_hub = float(df_role.loc["hub", "C2_norm_std"])

            # Закон FDM по деревьям: terminator > bridge > hub по C2_norm_mean
            delta_tb = m_term - m_bridge      # зазор T → B
            delta_bh = m_bridge - m_hub       # зазор B → H
            min_gap = min(delta_tb, delta_bh)
            sigma_max = max(s_term, s_bridge, s_hub)

            # порядок считаем корректным, если оба зазора положительные
            order_ok = (delta_tb > 0.0) and (delta_bh > 0.0)

            beta = 0.7
            score = min_gap - beta * sigma_max

        summary_rows.append(
            {
                "lambda_fdm": lam,
                "q_fdm": q,
                "order_ok": order_ok,
                "delta_tb": delta_tb,
                "delta_bh": delta_bh,
                "min_gap": min_gap,
                "max_C2_norm_std": sigma_max,
                "score": score,
            }
        )

    summary_df = pd.DataFrame(summary_rows)
    summary_df = summary_df.sort_values(
        by=["score"],
        ascending=[False],
    )

    summary_txt = RESULTS_DIR / "fdm_calibration_summary.txt"
    with summary_txt.open("w", encoding="utf-8") as f:
        f.write("FDM calibration summary (tree growth law)\n")
        f.write("=========================================\n\n")
        for _, row in summary_df.iterrows():
            f.write(
                f"lambda={row['lambda_fdm']:.2f}, q={row['q_fdm']:.2f}, "
                f"order_ok={row['order_ok']}, "
                f"min_gap={row['min_gap']:.4f}, "
                f"max_C2_norm_std={row['max_C2_norm_std']:.4f}, "
                f"score={row['score']:.4f}\n"
            )

        # Top-10 по score
        top = summary_df.head(10)
        f.write("\nTop 10 parameter pairs by score (higher is better):\n")
        for _, row in top.iterrows():
            f.write(
                f"- lambda={row['lambda_fdm']:.3f}, q={row['q_fdm']:.3f}, "
                f"score={row['score']:.4f}, min_gap={row['min_gap']:.4f}, "
                f"max_std={row['max_C2_norm_std']:.4f}, "
                f"order_ok={row['order_ok']}\n"
            )

    print(f"Wrote {grid_path}")
    print(f"Wrote {agg_path}")
    print(f"Wrote {summary_txt}")


if __name__ == "__main__":
    main()
