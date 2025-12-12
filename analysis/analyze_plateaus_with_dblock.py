from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd


def main() -> None:
    path = Path("data") / "element_indices_with_dblock.csv"
    df = pd.read_csv(path)

    donors = df[(df["D_index"] > 1e-6) & (df["A_index"].abs() < 1e-6)].copy()
    donor_groups = donors.groupby("D_index")

    print("=== DONOR PLATEAUS (D_index > 0, A_index ~ 0) ===")
    for D, g in donor_groups:
        elems = ", ".join(f"{int(Z)}{El}" for Z, El in zip(g["Z"], g["El"]))
        print(f"  D={D:.4f} : {elems}")
    print()

    strong_acc = df[np.isclose(df["A_index"], 1.237, atol=1e-3)].copy()
    print("=== STRONG ACCEPTOR PLATEAU (A_index ~ 1.237) ===")
    elems = ", ".join(f"{int(Z)}{El}" for Z, El in zip(strong_acc["Z"], strong_acc["El"]))
    print(f"  A ~ 1.237 : {elems}")
    print()

    metallic = df[
        (df["Z"].between(21, 30))
        & (df["A_index"] > 0.02)
        & (df["A_index"] < 0.25)
        & (df["D_index"] < 0.05)
    ].copy()

    print("=== METALLIC PLATEAU (d-block, low-A hubs) ===")
    if metallic.empty:
        print("  (no elements matched metallic criteria)")
    else:
        for _, row in metallic.sort_values("Z").iterrows():
            print(
                f"  Z={int(row['Z']):2d} {row['El']:>2s} "
                f"role={row['role']:<5s} "
                f"D={row['D_index']:.4f} A={row['A_index']:.4f}"
            )

        print()
        print("[METALLIC PLATEAU STATS]")
        print(
            f"  n = {len(metallic)}, "
            f"avg A = {metallic['A_index'].mean():.4f}, "
            f"range A = [{metallic['A_index'].min():.4f}, {metallic['A_index'].max():.4f}], "
            f"avg D = {metallic['D_index'].mean():.4f}"
        )


if __name__ == "__main__":
    main()
