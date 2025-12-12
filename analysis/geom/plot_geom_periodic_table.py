from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from analysis.io_utils import results_path


def main() -> None:
    csv_path = results_path("geom_periodic_table_v1.csv")
    if not csv_path.exists():
        raise SystemExit(
            "geom_periodic_table_v1.csv not found in results/. "
            "Run export_geom_periodic_table.py first."
        )

    df = pd.read_csv(csv_path)

    # Ensure expected columns
    if "symbol" not in df.columns and "El" in df.columns:
        df = df.rename(columns={"El": "symbol"})

    geom_order = [
        "s_donor",
        "s_bridge",
        "p_semihub",
        "p_acceptor",
        "inert",
        "d_octa",
        "other",
    ]
    periods = sorted(df["period"].unique())

    fig, ax = plt.subplots(figsize=(len(geom_order) * 2.5, len(periods) * 1.8))

    geom_colors = {
        "s_donor": "#f4d03f",
        "s_bridge": "#f5b041",
        "p_semihub": "#5dade2",
        "p_acceptor": "#1f618d",
        "inert": "#bdc3c7",
        "d_octa": "#58d68d",
        "other": "#e5e7e9",
    }

    n_periods = len(periods)
    for i_p, period in enumerate(periods):
        for j_g, g in enumerate(geom_order):
            cell = df[(df["period"] == period) & (df["geom_class"] == g)]
            x, y = j_g, n_periods - 1 - i_p

            rect = plt.Rectangle(
                (x, y),
                1,
                1,
                facecolor=geom_colors.get(g, "#ffffff"),
                edgecolor="black",
            )
            ax.add_patch(rect)

            if not cell.empty:
                symbols = ", ".join(cell["symbol"].astype(str).tolist())
                ax.text(
                    x + 0.5,
                    y + 0.55,
                    symbols,
                    ha="center",
                    va="center",
                    fontsize=9,
                    wrap=True,
                )

    ax.set_xlim(0, len(geom_order))
    ax.set_ylim(0, n_periods)
    ax.set_xticks([i + 0.5 for i in range(len(geom_order))])
    ax.set_xticklabels(geom_order, rotation=45, ha="right")
    ax.set_yticks([i + 0.5 for i in range(n_periods)])
    ax.set_yticklabels(list(reversed(periods)))
    ax.invert_yaxis()
    ax.set_xlabel("geom_class")
    ax.set_ylabel("period")
    fig.tight_layout()

    out_path = results_path("geom_periodic_table_v1.png")
    fig.savefig(out_path, dpi=200)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
