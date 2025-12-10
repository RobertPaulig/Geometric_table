from __future__ import annotations

from pathlib import Path

import pandas as pd

from export_geom_periodic_table import classify_geom_row


def main() -> None:
    base = Path(".")
    data_path = base / "data" / "element_indices_with_dblock.csv"
    results_dir = base / "results"
    results_dir.mkdir(exist_ok=True)

    df = pd.read_csv(data_path)
    if "geom_class" not in df.columns:
        df["geom_class"] = df.apply(classify_geom_row, axis=1)

    focus_classes = ["s_donor", "s_bridge", "p_semihub", "p_acceptor", "inert"]
    df_focus = df[df["geom_class"].isin(focus_classes)].copy()

    # Статистика по (geom_class, period)
    by_class_period = (
        df_focus.groupby(["geom_class", "period"], dropna=False)
        .agg(
            mean_D=("D_index", "mean"),
            std_D=("D_index", "std"),
            mean_A=("A_index", "mean"),
            std_A=("A_index", "std"),
            n=("Z", "count"),
        )
        .reset_index()
    )

    # Глобальная статистика по geom_class
    by_class = (
        df_focus.groupby("geom_class", dropna=False)
        .agg(
            mean_D=("D_index", "mean"),
            std_D=("D_index", "std"),
            mean_A=("A_index", "mean"),
            std_A=("A_index", "std"),
            n=("Z", "count"),
        )
        .reset_index()
    )

    out_path = results_dir / "geom_periodicity_sp_stats.txt"
    with out_path.open("w", encoding="utf-8") as f:
        f.write("Geom periodicity stats for s/p blocks (Hypothesis S/P)\n\n")
        f.write("[By geom_class, period]\n")
        f.write(by_class_period.to_string(index=False))
        f.write("\n\n[By geom_class]\n")
        f.write(by_class.to_string(index=False))
        f.write("\n\n[Summary]\n")
        for _, row in by_class.iterrows():
            gc = row["geom_class"]
            mean_D = row["mean_D"]
            std_D = row["std_D"]
            mean_A = row["mean_A"]
            std_A = row["std_A"]
            f.write(
                f"{gc:10s}: mean_D={mean_D:8.6f}, std_D={std_D:8.6f}, "
                f"mean_A={mean_A:8.6f}, std_A={std_A:8.6f}\n"
            )

    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()

