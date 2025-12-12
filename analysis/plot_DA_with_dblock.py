from __future__ import annotations

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


def main() -> None:
    path = Path("data") / "element_indices_with_dblock.csv"
    df = pd.read_csv(path)

    donors = df[(df["D_index"] > 1e-6) & (df["A_index"].abs() < 1e-6)]
    strong_acc = df[(df["A_index"] > 0.8)]
    metallic = df[
        (df["Z"].between(21, 30))
        & (df["A_index"] > 0.02)
        & (df["A_index"] < 0.25)
        & (df["D_index"] < 0.05)
    ]

    used_idx = donors.index.union(strong_acc.index).union(metallic.index)
    others = df[~df.index.isin(used_idx)]

    plt.figure(figsize=(6, 5))

    if not others.empty:
        plt.scatter(
            others["D_index"],
            others["A_index"],
            s=20,
            alpha=0.4,
            label="other",
        )
    if not donors.empty:
        plt.scatter(
            donors["D_index"],
            donors["A_index"],
            s=40,
            marker="o",
            label="donors",
        )
    if not strong_acc.empty:
        plt.scatter(
            strong_acc["D_index"],
            strong_acc["A_index"],
            s=40,
            marker="s",
            label="strong acceptors",
        )
    if not metallic.empty:
        plt.scatter(
            metallic["D_index"],
            metallic["A_index"],
            s=60,
            marker="^",
            label="d-block metallic",
        )
        for _, row in metallic.iterrows():
            plt.text(
                row["D_index"] + 0.002,
                row["A_index"] + 0.002,
                row["El"],
                fontsize=8,
            )

    plt.xlabel("D_index")
    plt.ylabel("A_index")
    plt.title("D/A map with d-block metallic plateau")
    plt.legend()
    plt.tight_layout()

    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    out_path = results_dir / "DA_scatter_with_dblock.png"
    plt.savefig(out_path, dpi=200)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
