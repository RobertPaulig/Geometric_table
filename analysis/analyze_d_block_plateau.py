from __future__ import annotations

from pathlib import Path
import pandas as pd

from analysis.io_utils import read_data_csv


def main() -> None:
    df = read_data_csv(
        "element_indices_with_dblock.csv",
        required=True,
        expected_columns=["Z", "El", "role", "chi_spec", "D_index", "A_index"],
    )

    d_block = df[(df["Z"] >= 21) & (df["Z"] <= 30)].copy()
    donors = df[df["role"] == "donor"]
    hubs = df[df["role"] == "hub"]

    print("=== D-BLOCK (first row) ===")
    print(d_block[["Z", "El", "role", "chi_spec", "D_index", "A_index"]])

    print("\n[STATS]")

    def stats(label: str, sub: pd.DataFrame) -> None:
        if len(sub) == 0:
            print(f"{label}: n=0")
            return
        print(
            f"{label}: n={len(sub)}, "
            f"avg D={sub['D_index'].mean():.3f}, "
            f"avg A={sub['A_index'].mean():.3f}"
        )

    stats("donors", donors)
    stats("hubs", hubs)
    stats("d-block", d_block)

    print("\n[RANGES]")
    if len(d_block):
        print(
            f"d-block D: [{d_block['D_index'].min():.3f}, "
            f"{d_block['D_index'].max():.3f}]"
        )
        print(
            f"d-block A: [{d_block['A_index'].min():.3f}, "
            f"{d_block['A_index'].max():.3f}]"
        )


if __name__ == "__main__":
    main()
