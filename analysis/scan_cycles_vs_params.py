from __future__ import annotations

"""
scan_cycles_vs_params.py — CY-1 / QSG v6.x R&D стенд циклов.

Сканирует область параметров GrowthParams и измеряет:
- долю деревьев / графов с циклами,
- максимальное и среднее цикломатическое число,
- размер графов и «циклонагрузку» как простой crossing-proxy.

Результаты:
- results/cycle_param_scan.csv
- results/cycle_param_scan.txt
"""

import argparse
from dataclasses import asdict
from pathlib import Path
from typing import Iterable, List, Tuple

import numpy as np
import pandas as pd

from analysis.growth_cli import make_growth_params_from_config_path
from analysis.seeds import GROWTH_SEEDS
from core.complexity import compute_complexity_features
from core.grower import GrowthParams, grow_molecule_christmas_tree


RESULTS_DIR = Path("results")


def iter_param_grid(base: GrowthParams) -> Iterable[Tuple[GrowthParams, dict]]:
    """
    Генерирует сетку параметров роста для CY-1.

    Мы варьируем несколько параметров вокруг базового режима,
    заданного base (по умолчанию v5.0), не меняя его напрямую.
    """

    p_continue_values = [0.5, 0.7, 0.9]
    role_bonus_hub_values = [0.0, 0.2, 0.4]
    role_penalty_terminator_values = [-0.6, -0.4, -0.2]

    # Для CY-1/step2 включаем loopy-overlay с фиксированными параметрами.
    max_extra_bonds = 3
    p_extra_bond = 0.3

    for p_c in p_continue_values:
        for bonus in role_bonus_hub_values:
            for penalty in role_penalty_terminator_values:
                params = GrowthParams(
                    max_depth=base.max_depth,
                    max_atoms=base.max_atoms,
                    p_continue_base=p_c,
                    chi_sensitivity=base.chi_sensitivity,
                    role_bonus_hub=bonus,
                    role_penalty_terminator=penalty,
                    temperature=base.temperature,
                    allow_cycles=True,
                    max_extra_bonds=max_extra_bonds,
                    p_extra_bond=p_extra_bond,
                )
                meta = {
                    "p_continue_base": p_c,
                    "role_bonus_hub": bonus,
                    "role_penalty_terminator": penalty,
                    "allow_cycles": True,
                    "max_extra_bonds": max_extra_bonds,
                    "p_extra_bond": p_extra_bond,
                }
                yield params, meta


def run_cycle_stats_for_params(
    seeds: List[str],
    params: GrowthParams,
    num_runs: int,
    rng: np.random.Generator,
) -> dict:
    """
    Прогоняет grower для заданных параметров и возвращает агрегированные метрики.
    """
    cyclomatic_all: List[int] = []
    size_all: List[int] = []

    for seed in seeds:
        for _ in range(num_runs):
            mol = grow_molecule_christmas_tree(seed, params=params, rng=rng)
            adj = mol.adjacency_matrix()
            feats = compute_complexity_features(adj)
            cyclomatic_all.append(int(feats.cyclomatic))
            size_all.append(int(adj.shape[0]))

    cyclomatic_arr = np.array(cyclomatic_all, dtype=int)
    size_arr = np.array(size_all, dtype=int)

    if cyclomatic_arr.size == 0:
        return {
            "n_samples": 0,
            "frac_trees": np.nan,
            "frac_cycles": np.nan,
            "max_cyclomatic": 0,
            "cyclomatic_mean": np.nan,
            "cyclomatic_std": np.nan,
            "n_mean": np.nan,
            "n_std": np.nan,
            "cycle_load_mean": np.nan,
            "cycle_load_std": np.nan,
        }

    frac_trees = float(np.mean(cyclomatic_arr == 0))
    frac_cycles = float(np.mean(cyclomatic_arr > 0))
    max_cyclomatic = int(cyclomatic_arr.max())

    # crossing-proxy: циклонагрузка на вершину
    with np.errstate(divide="ignore", invalid="ignore"):
        cycle_load = np.where(size_arr > 0, cyclomatic_arr / size_arr, np.nan)

    return {
        "n_samples": int(cyclomatic_arr.size),
        "frac_trees": frac_trees,
        "frac_cycles": frac_cycles,
        "max_cyclomatic": max_cyclomatic,
        "cyclomatic_mean": float(cyclomatic_arr.mean()),
        "cyclomatic_std": float(cyclomatic_arr.std()),
        "n_mean": float(size_arr.mean()),
        "n_std": float(size_arr.std()),
        "cycle_load_mean": float(np.nanmean(cycle_load)),
        "cycle_load_std": float(np.nanstd(cycle_load)),
    }


def main(argv=None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        required=False,
        help="Базовый YAML/JSON-конфиг роста для сетки (по умолчанию v5.0 baseline).",
    )
    args = parser.parse_args(argv)

    seeds = GROWTH_SEEDS
    num_runs = 200  # R&D: достаточно для первого скана

    RESULTS_DIR.mkdir(exist_ok=True)

    rng = np.random.default_rng(12345)
    base_params = make_growth_params_from_config_path(args.config)

    rows = []
    for params, meta in iter_param_grid(base_params):
        stats = run_cycle_stats_for_params(seeds, params, num_runs, rng=rng)
        row = {
            **meta,
            **stats,
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    csv_path = RESULTS_DIR / "cycle_param_scan.csv"
    txt_path = RESULTS_DIR / "cycle_param_scan.txt"

    df.to_csv(csv_path, index=False)

    # Краткий текстовый отчёт по интересным режимам:
    # берём точки, где есть заметная доля циклов и умеренный размер графа.
    lines = []
    lines.append("# CY-1 / QSG v6.x — cycle_param_scan")
    lines.append("# Фильтр: frac_cycles in [0.1, 0.5], n_mean in [5, 30]")
    lines.append("")

    mask = (
        (df["n_samples"] > 0)
        & (df["frac_cycles"].between(0.1, 0.5))
        & (df["n_mean"].between(5.0, 30.0))
    )
    df_sel = df[mask].copy()
    df_sel = df_sel.sort_values(by="cycle_load_mean", ascending=False)

    if df_sel.empty:
        lines.append("No parameter regimes matched the filter.")
    else:
        for _, row in df_sel.iterrows():
            lines.append(
                (
                    f"p_continue={row['p_continue_base']:.2f}, "
                    f"role_bonus_hub={row['role_bonus_hub']:.2f}, "
                    f"role_penalty_terminator={row['role_penalty_terminator']:.2f} "
                    f"=> frac_cycles={row['frac_cycles']:.3f}, "
                    f"n_mean={row['n_mean']:.2f}, "
                    f"cycle_load_mean={row['cycle_load_mean']:.4f}, "
                    f"max_cyclomatic={int(row['max_cyclomatic'])}"
                )
            )

    Path(txt_path).write_text("\n".join(lines), encoding="utf-8")

    print(f"Wrote {csv_path} and {txt_path}")


if __name__ == "__main__":
    main()
