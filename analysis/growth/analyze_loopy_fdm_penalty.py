from __future__ import annotations

"""
analyze_loopy_fdm_penalty.py — CY-1/step3:
сравнение FDM и FDM_loopy в режимах с циклами.
"""

import argparse
import math
from pathlib import Path
from typing import Iterable, List

import numpy as np
import pandas as pd

from analysis.seeds import GROWTH_SEEDS
from core.grower import GrowthParams, grow_molecule_christmas_tree
from core.complexity import compute_complexity_features, compute_complexity_features_v2
from core.growth_config import load_growth_config
from core.complexity_config import load_complexity_penalties, set_current_penalties


RESULTS_DIR = Path("results")
SEEDS = GROWTH_SEEDS


def norm_C(total: float, n: int) -> float:
    if n <= 0:
        return 0.0
    return float(total) / (n * math.log2(1.0 + n))


def make_params_cy1a() -> GrowthParams:
    return load_growth_config("configs/growth_cy1a.yaml").to_growth_params()


def make_params_cy1b() -> GrowthParams:
    return load_growth_config("configs/growth_cy1b.yaml").to_growth_params()


def run_loopy_fdm_stats(label: str, params: GrowthParams, num_runs: int = 200) -> Path:
    rows: List[dict] = []
    rng = np.random.default_rng(12345)

    for seed in SEEDS:
        for _ in range(num_runs):
            mol = grow_molecule_christmas_tree(seed, params=params, rng=rng)
            adj = mol.adjacency_matrix()

            feats = compute_complexity_features(adj)
            feats_fdm = compute_complexity_features_v2(adj, backend="fdm")
            feats_loopy = compute_complexity_features_v2(adj, backend="fdm_loopy")

            n = feats.n
            cyclomatic = feats.cyclomatic
            cycle_load = cyclomatic / n if n > 0 else 0.0

            C_norm_fdm = norm_C(feats_fdm.total, n)
            C_norm_fdm_loopy = norm_C(feats_loopy.total, n)

            penalty_factor = (
                C_norm_fdm_loopy / C_norm_fdm if C_norm_fdm > 0 else 1.0
            )

            rows.append(
                {
                    "seed": seed,
                    "n": n,
                    "cyclomatic": cyclomatic,
                    "cycle_load": cycle_load,
                    "C_norm_fdm": C_norm_fdm,
                    "C_norm_fdm_loopy": C_norm_fdm_loopy,
                    "penalty_factor": penalty_factor,
                    "mode": label,
                }
            )

    RESULTS_DIR.mkdir(exist_ok=True)
    out_csv = RESULTS_DIR / f"loopy_fdm_penalty_{label}.csv"
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    return out_csv


def summarize_penalty(csv_paths: Iterable[Path]) -> None:
    lines: List[str] = []
    all_df = []
    for p in csv_paths:
        df = pd.read_csv(p)
        all_df.append(df)
    if not all_df:
        return

    df_all = pd.concat(all_df, ignore_index=True)

    # Группы по seed
    lines.append("# Loopy FDM penalty summary (CY-1/step3)")
    lines.append("")

    def add_block(title: str, mask):
        sub = df_all[mask].copy()
        if sub.empty:
            lines.append(f"{title}: no data")
            lines.append("")
            return
        lines.append(f"{title}: n={len(sub)}")
        for col in ["C_norm_fdm", "C_norm_fdm_loopy", "penalty_factor", "cycle_load"]:
            lines.append(
                f"  {col}: mean={sub[col].mean():.4f}, std={sub[col].std():.4f}"
            )
        lines.append("")

    hubs = {"C", "N", "Si", "P"}
    add_block("Hubs (C,N,Si,P)", df_all["seed"].isin(hubs))

    terms = {"Li", "Na", "K", "Be", "Mg", "Ca", "O", "S"}
    add_block("Terminators/bridges rough set", df_all["seed"].isin(terms))

    # Корреляция penalty vs cycle_load
    mask_nonzero = df_all["cycle_load"] > 0
    sub = df_all[mask_nonzero]
    if len(sub) >= 2:
        corr = sub["penalty_factor"].corr(sub["cycle_load"])
        lines.append(
            f"Correlation(penalty_factor, cycle_load) over loopy samples: r={corr:.4f}"
        )

    # сколько раз penalty_factor > порогов
    for thr in [1.1, 1.2, 1.5]:
        frac = float((df_all["penalty_factor"] > thr).mean())
        lines.append(f"Frac(penalty_factor > {thr:.1f}) = {frac:.3f}")

    out_txt = RESULTS_DIR / "loopy_fdm_penalty_summary.txt"
    out_txt.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {out_txt}")


def main(argv=None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--complexity-config",
        type=str,
        required=False,
        help=(
            "YAML/JSON-конфиг штрафов FDM (alpha_cycle, alpha_load, "
            "beta_cross, max_cross_n); по умолчанию используются "
            "дефолтные значения ядра."
        ),
    )
    args = parser.parse_args(argv)

    if args.complexity_config:
        cfg_pen = load_complexity_penalties(args.complexity_config)
        set_current_penalties(cfg_pen)

    cy1a_csv = run_loopy_fdm_stats("CY1A", make_params_cy1a())
    cy1b_csv = run_loopy_fdm_stats("CY1B", make_params_cy1b())
    summarize_penalty([cy1a_csv, cy1b_csv])


if __name__ == "__main__":
    main()
