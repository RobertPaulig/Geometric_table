from __future__ import annotations

from pathlib import Path

import pandas as pd

from export_geom_periodic_table import classify_geom_row


def main() -> None:
    base = Path(".")
    data_dir = base / "data"
    results_dir = base / "results"
    results_dir.mkdir(exist_ok=True)

    idx_path = data_dir / "element_indices_with_dblock.csv"
    summary_path = data_dir / "geom_nuclear_complexity_summary.csv"

    df_idx = pd.read_csv(idx_path).rename(columns={"El": "symbol"})
    if "geom_class" not in df_idx.columns:
        df_idx["geom_class"] = df_idx.apply(classify_geom_row, axis=1)

    df_sum = pd.read_csv(summary_path)

    df = df_idx.merge(
        df_sum[["Z", "symbol", "Avg_Complexity", "Max_Complexity"]],
        on=["Z", "symbol"],
        how="left",
    )

    df_d = df[df["geom_class"] == "d_octa"].copy()
    if df_d.empty:
        print("No d_octa elements found; nothing to analyze.")
        return

    by_period = (
        df_d.groupby("period", dropna=False)[
            ["D_index", "A_index", "Avg_Complexity", "Max_Complexity", "Z"]
        ]
        .agg(
            mean_D=("D_index", "mean"),
            std_D=("D_index", "std"),
            mean_A=("A_index", "mean"),
            std_A=("A_index", "std"),
            mean_AvgC=("Avg_Complexity", "mean"),
            std_AvgC=("Avg_Complexity", "std"),
            mean_MaxC=("Max_Complexity", "mean"),
            std_MaxC=("Max_Complexity", "std"),
            n=("Z", "count"),
        )
        .reset_index()
    )

    global_stats = pd.DataFrame(
        {
            "mean_D": [df_d["D_index"].mean()],
            "std_D": [df_d["D_index"].std()],
            "mean_A": [df_d["A_index"].mean()],
            "std_A": [df_d["A_index"].std()],
            "mean_AvgC": [df_d["Avg_Complexity"].mean()],
            "std_AvgC": [df_d["Avg_Complexity"].std()],
            "mean_MaxC": [df_d["Max_Complexity"].mean()],
            "std_MaxC": [df_d["Max_Complexity"].std()],
        }
    )

    out_path = results_dir / "d_layer_extended_stats.txt"
    with out_path.open("w", encoding="utf-8") as f:
        f.write("D-layer extended stats (Hypothesis D-layer-extended)\n\n")
        f.write("[By period]\n")
        f.write(by_period.to_string(index=False))
        f.write("\n\n[Global]\n")
        f.write(global_stats.to_string(index=False))
        f.write("\n")

    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()

