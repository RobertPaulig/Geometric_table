from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict

import numpy as np
import yaml

from analysis.io_utils import results_path, write_text_result
from analysis.nuclear_cli import apply_nuclear_config_if_provided
from analysis.nuclear.tune_metrics import TARGET_MAGIC, cost_magic_l2
from core.nuclear_magic import set_magic_mode
from core.nuclear_spectrum_ws import collect_levels_ws


def get_toy_magic_ws(
    R_max: float,
    R0: float,
    a: float,
    V0: float,
    N_grid: int = 220,
    L_max: int = 5,
    levels_per_l: int = 12,
    energy_cut: float = 0.0,
    n_prefix: int = 40,
    top_k_gaps: int = 8,
):
    levels = collect_levels_ws(
        R_max=R_max,
        R0=R0,
        a=a,
        V0=V0,
        N_grid=N_grid,
        L_max=L_max,
        levels_per_l=levels_per_l,
        energy_cut=energy_cut,
    )
    if not levels:
        return []

    energies = np.array([E for (E, g, ell) in levels])
    g = np.array([g for (E, g, ell) in levels], dtype=int)
    N_cum = np.cumsum(g)

    n_use = min(n_prefix, len(levels))
    E_sub = energies[:n_use]
    N_sub = N_cum[:n_use]

    dE = np.diff(E_sub)
    if dE.size == 0:
        return []

    gap_idx_sorted = np.argsort(dE)[::-1][:top_k_gaps]
    gap_idx_unique = sorted(set(gap_idx_sorted.tolist()))

    toy_magic = [int(N_sub[idx]) for idx in gap_idx_unique]
    toy_magic = sorted(set(toy_magic))
    return toy_magic


def load_ws_tune_config(path: str | None) -> Dict[str, Any]:
    if not path:
        return {}
    cfg_path = Path(path)
    if not cfg_path.exists():
        raise FileNotFoundError(f"WS-tune config not found: {cfg_path}")
    with cfg_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"WS-tune config must be a mapping, got {type(data)!r}")
    return data


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(
        description="Scan WS spectrum and tune magic numbers."
    )
    parser.add_argument(
        "--nuclear-config",
        type=str,
        default=None,
        help="Path to nuclear shell config (YAML/JSON).",
    )
    parser.add_argument(
        "--magic",
        type=str,
        choices=["legacy", "ws"],
        default="ws",
        help="Magic-number mode for nuclear_magic: legacy or ws.",
    )
    parser.add_argument(
        "--ws-config",
        type=str,
        default=None,
        help="YAML-конфиг с сетками параметров WS-тюнинга (опционально).",
    )

    args = parser.parse_args(argv)

    apply_nuclear_config_if_provided(args.nuclear_config)
    set_magic_mode(args.magic)

    cfg = load_ws_tune_config(args.ws_config)

    R0_grid = np.array(cfg.get("R0_values", np.linspace(4.0, 7.0, 4)), dtype=float)
    a_grid = np.array(cfg.get("a_values", np.linspace(0.4, 1.0, 4)), dtype=float)
    V0_grid = np.array(cfg.get("V0_values", np.linspace(40.0, 80.0, 5)), dtype=float)

    R_max = float(cfg.get("R_max", 12.0))
    N_grid = int(cfg.get("N_grid", 220))
    L_max = int(cfg.get("L_max", 5))

    results = []

    print("Scanning WS parameter grid for toy magic numbers...")
    print(f"R0 in {R0_grid}, a in {a_grid}, V0 in {V0_grid}")

    for R0 in R0_grid:
        for a in a_grid:
            for V0 in V0_grid:
                toy_magic = get_toy_magic_ws(
                    R_max=R_max,
                    R0=R0,
                    a=a,
                    V0=V0,
                    N_grid=N_grid,
                    L_max=L_max,
                    levels_per_l=12,
                    energy_cut=0.0,
                    n_prefix=int(cfg.get("n_prefix", 40)),
                    top_k_gaps=int(cfg.get("top_k_gaps", 10)),
                )
                c = cost_magic_l2(toy_magic, TARGET_MAGIC, n_compare=int(cfg.get("n_compare", 4)))
                results.append((c, R0, a, V0, toy_magic))

    results.sort(key=lambda t: t[0])

    lines = []
    lines.append("=== Top 10 parameter sets by cost (WS) ===")
    for i, (c, R0, a, V0, toy_magic) in enumerate(results[:10], start=1):
        line = (
            f"{i:2d}) R0={R0:.2f}, a={a:.2f}, V0={V0:.1f}, "
            f"cost={c:.3f}, magic_N={toy_magic}"
        )
        print(line)
        lines.append(line)

    out_csv = results_path("ws_magic_tuning_results.csv")
    out_txt = results_path("ws_magic_tuning_summary.txt")

    with out_csv.open("w", newline="", encoding="utf-8") as f:
        f.write("R0,a,V0,cost,magic_N\n")
        for c, R0, a, V0, toy_magic in results:
            magic_str = " ".join(str(n) for n in toy_magic)
            f.write(f"{R0:.6g},{a:.6g},{V0:.6g},{c:.6g},{magic_str}\n")

    write_text_result("\n".join(lines) + "\n", out_txt)

    print(f"\nSaved full scan to {out_csv}")


if __name__ == "__main__":
    main()
