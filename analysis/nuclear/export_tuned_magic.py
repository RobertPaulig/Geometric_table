from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import pandas as pd
import yaml

from analysis.io_utils import results_path


def export_best_magic(csv_path: str | None, out_path: str | None, top_k: int = 1) -> None:
    if csv_path is None:
        csv_path = str(results_path("ws_magic_tuning_results.csv"))

    df = pd.read_csv(csv_path)
    if df.empty:
        raise SystemExit(f"No rows in tuning results: {csv_path}")

    df_sorted = df.sort_values("cost", ascending=True).head(top_k)

    # Берём первый как основной набор
    best = df_sorted.iloc[0]
    magic_str = str(best["magic_N"])
    magic_list: List[int] = [int(x) for x in str(magic_str).split() if x.strip()]

    if not magic_list:
        raise SystemExit("Best row has empty magic_N list.")

    data = {
        "Z": [2, 8, 20, 28, 50, 82],
        "N": magic_list,
    }

    if out_path is None:
        out_path = "configs/nuclear_magic_ws_tuned.yaml"
    p = Path(out_path)
    with p.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False)

    print(f"Exported tuned magic numbers to {p}")


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(
        description="Export tuned WS magic numbers from CSV to YAML MagicSet."
    )
    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help="Path to ws_magic_tuning_results.csv (default: results/...).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output YAML path for MagicSet (default: configs/nuclear_magic_ws_tuned.yaml).",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=1,
        help="How many best rows to consider (currently only first is exported).",
    )
    args = parser.parse_args(argv)

    export_best_magic(args.input, args.output, top_k=args.top_k)


if __name__ == "__main__":
    main()

