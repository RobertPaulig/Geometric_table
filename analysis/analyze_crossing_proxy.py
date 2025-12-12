from __future__ import annotations

"""
analyze_crossing_proxy.py — CY-1/step4:
сравнение toy crossing-number с proxy (cyclomatic, cycle_load, fdm_loopy).
"""

import math
from pathlib import Path
from typing import Iterable, List

import numpy as np
import pandas as pd

from analysis.io_utils import results_path
from analysis.seeds import GROWTH_SEEDS
from core.grower import GrowthParams, grow_molecule_christmas_tree
from core.complexity import compute_complexity_features, compute_complexity_features_v2
from core.crossing import estimate_crossing_number_circle
from core.growth_config import load_growth_config
SEEDS = GROWTH_SEEDS


def norm_C(total: float, n: int) -> float:
    if n <= 0:
        return 0.0
    return float(total) / (n * math.log2(1.0 + n))


def make_params_cy1a() -> GrowthParams:
    return load_growth_config("configs/growth_cy1a.yaml").to_growth_params()


def make_params_cy1b() -> GrowthParams:
    return load_growth_config("configs/growth_cy1b.yaml").to_growth_params()


def _collect_for_mode(label: str, params: GrowthParams, num_runs: int = 300) -> Path:
    rows: List[dict] = []
    rng = np.random.default_rng(2025)

    for seed in SEEDS:
        for _ in range(num_runs):
            mol = grow_molecule_christmas_tree(seed, params=params, rng=rng)
            adj = mol.adjacency_matrix()

            feats = compute_complexity_features(adj)
            n = feats.n
            if n <= 1 or n > 8:
                # Для step4 считаем crossing только для малых графов (точный режим).
                continue

            cyclomatic = feats.cyclomatic
            m = feats.m
            cycle_load = cyclomatic / n if n > 0 else 0.0

            feats_fdm = compute_complexity_features_v2(adj, backend="fdm")
            feats_loopy = compute_complexity_features_v2(adj, backend="fdm_loopy")
            C_norm_fdm = norm_C(feats_fdm.total, n)
            C_norm_fdm_loopy = norm_C(feats_loopy.total, n)
            penalty_factor = (
                C_norm_fdm_loopy / C_norm_fdm if C_norm_fdm > 0 else 1.0
            )

            crossing, is_exact = estimate_crossing_number_circle(adj, max_exact_n=8)
            crossing_density = crossing / max(1, m)

            rows.append(
                {
                    "seed": seed,
                    "mode": label,
                    "n": n,
                    "m": m,
                    "cyclomatic": cyclomatic,
                    "cycle_load": cycle_load,
                    "crossing": crossing,
                    "crossing_density": crossing_density,
                    "C_norm_fdm": C_norm_fdm,
                    "C_norm_fdm_loopy": C_norm_fdm_loopy,
                    "penalty_factor": penalty_factor,
                    "is_exact": bool(is_exact),
                }
            )

    out_csv = results_path(f"crossing_proxy_{label}.csv")
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    return out_csv


def summarize_crossing(csv_paths: Iterable[Path]) -> None:
    dfs = [pd.read_csv(p) for p in csv_paths]
    if not dfs:
        return
    df_all = pd.concat(dfs, ignore_index=True)

    lines: List[str] = []
    lines.append("# CY-1/step4: crossing-number vs proxy summary")
    lines.append("")
    lines.append(f"Total samples (n<=8): {len(df_all)}")
    lines.append("")

    def add_corr_block(title: str, sub):
        if len(sub) < 2:
            lines.append(f"{title}: not enough data")
            lines.append("")
            return
        lines.append(f"{title}: n={len(sub)}")
        for (x, y) in [
            ("crossing", "cyclomatic"),
            ("crossing_density", "cycle_load"),
            ("crossing", "penalty_factor"),
            ("crossing_density", "penalty_factor"),
        ]:
            corr = sub[x].corr(sub[y])
            lines.append(f"  corr({x}, {y}) = {corr:.4f}")
        lines.append("")

    # Глобальные корреляции
    add_corr_block("Global", df_all)

    hubs = {"C", "N", "Si", "P"}
    terms = {"Li", "Na", "K", "Be", "Mg", "Ca", "O", "S"}

    add_corr_block("Hubs (C,N,Si,P)", df_all[df_all["seed"].isin(hubs)])
    add_corr_block("Terminators/bridges rough set", df_all[df_all["seed"].isin(terms)])

    # Доли случаев с crossing>0 и малыми cyclomatic
    sub_loopy = df_all[df_all["cyclomatic"] > 0].copy()
    if not sub_loopy.empty:
        for mu in [1, 2]:
            mask = (sub_loopy["cyclomatic"] == mu) & (sub_loopy["crossing"] > 0)
            frac = float(mask.mean())
            lines.append(
                f"Frac(crossing>0 | cyclomatic={mu}) over loopy graphs: {frac:.3f}"
            )

    out_txt = results_path("crossing_proxy_summary.txt")
    out_txt.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {out_txt}")


def main() -> None:
    csv_a = _collect_for_mode("CY1A", make_params_cy1a())
    csv_b = _collect_for_mode("CY1B", make_params_cy1b())
    summarize_crossing([csv_a, csv_b])


if __name__ == "__main__":
    main()
