from __future__ import annotations

from pathlib import Path

import pandas as pd

from export_geom_periodic_table import classify_geom_row


def main() -> None:
    base = Path(".")
    data_dir = base / "data"
    results_dir = base / "results"
    results_dir.mkdir(exist_ok=True)

    indices_path = data_dir / "element_indices_with_dblock.csv"
    summary_path = data_dir / "geom_nuclear_complexity_summary.csv"

    df_idx = pd.read_csv(indices_path)
    if "geom_class" not in df_idx.columns:
        df_idx["geom_class"] = df_idx.apply(classify_geom_row, axis=1)

    df_summary = pd.read_csv(summary_path)

    # Соединяем по Z, El/symbol
    df_idx = df_idx.rename(columns={"El": "symbol"})
    df = df_idx.merge(
        df_summary[["Z", "symbol", "Avg_Complexity", "Max_Complexity"]],
        on=["Z", "symbol"],
        how="left",
    )

    df_d = df[df["geom_class"] == "d_octa"].copy()

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

    out_path = results_dir / "d_block_universality_stats.txt"
    with out_path.open("w", encoding="utf-8") as f:
        f.write("D-block universality stats (Hypothesis D-layer)\n\n")
        f.write("[By period]\n")
        f.write(by_period.to_string(index=False))
        f.write("\n\n[Global]\n")
        f.write(global_stats.to_string(index=False))
        f.write("\n\n[Summary]\n")
        for _, row in by_period.iterrows():
            period = int(row["period"])
            f.write(
                f"period={period}: D={row['mean_D']:.6f}±{row['std_D']:.6f}, "
                f"A={row['mean_A']:.6f}±{row['std_A']:.6f}, "
                f"AvgC={row['mean_AvgC']:.3f}±{row['std_AvgC']:.3f}, "
                f"MaxC={row['mean_MaxC']:.3f}±{row['std_MaxC']:.3f}\n"
            )

    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
