from __future__ import annotations

from pathlib import Path

import pandas as pd

from export_geom_periodic_table import classify_geom_row


# Z новых тяжёлых s/p-элементов, которые появятся при дальнейшем расширении
NEW_SP_ELEMENTS = {
    # Cs, Ba, Tl, Pb, Bi, Po, At, Rn
    "Cs",
    "Ba",
    "Tl",
    "Pb",
    "Bi",
    "Po",
    "At",
    "Rn",
}


def main() -> None:
    base = Path(".")
    data_path = base / "data" / "element_indices_with_dblock.csv"
    results_dir = base / "results"
    results_dir.mkdir(exist_ok=True)

    df = pd.read_csv(data_path)
    if "symbol" not in df.columns and "El" in df.columns:
        df = df.rename(columns={"El": "symbol"})
    if "geom_class" not in df.columns:
        df["geom_class"] = df.apply(classify_geom_row, axis=1)

    focus_classes = ["s_donor", "s_bridge", "p_semihub", "p_acceptor", "inert"]
    df_focus = df[df["geom_class"].isin(focus_classes)].copy()

    df_focus["is_new"] = df_focus["symbol"].isin(NEW_SP_ELEMENTS)

    old = df_focus[~df_focus["is_new"]]
    new = df_focus[df_focus["is_new"]]

    def stats_block(sub: pd.DataFrame, label: str) -> pd.DataFrame:
        if sub.empty:
            return pd.DataFrame(
                columns=[
                    "geom_class",
                    f"mean_D_{label}",
                    f"std_D_{label}",
                    f"mean_A_{label}",
                    f"std_A_{label}",
                    f"n_{label}",
                ]
            )
        grouped = (
            sub.groupby("geom_class", dropna=False)[["D_index", "A_index", "Z"]]
            .agg(
                **{
                    f"mean_D_{label}": ("D_index", "mean"),
                    f"std_D_{label}": ("D_index", "std"),
                    f"mean_A_{label}": ("A_index", "mean"),
                    f"std_A_{label}": ("A_index", "std"),
                    f"n_{label}": ("Z", "count"),
                }
            )
            .reset_index()
        )
        return grouped

    old_stats = stats_block(old, "old")
    new_stats = stats_block(new, "new")

    merged = pd.merge(old_stats, new_stats, on="geom_class", how="outer").fillna(pd.NA)

    out_path = results_dir / "geom_sp_extended_stats.txt"
    with out_path.open("w", encoding="utf-8") as f:
        f.write("Geom s/p extension stats (Hypothesis SP-extended)\n\n")
        f.write(merged.to_string(index=False))
        f.write("\n")

    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
