from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"


def classify_geom_row(row: pd.Series) -> str:
    """Rough geometric classification for Geom-Mendeleev v1."""
    D = float(row.get("D_index", 0.0))
    A = float(row.get("A_index", 0.0))
    role = row.get("role", "")
    Z = int(row.get("Z", 0))
    period = int(row.get("period", 0))

    # s-layer donors (group 1) and bridges (group 2) with A ≈ 0, D > 0
    if abs(A) < 1e-3 and D > 0:
        if Z in (3, 11, 19, 37):  # Li, Na, K, Rb
            return "s_donor"
        if Z in (4, 12, 20, 38):  # Be, Mg, Ca, Sr
            return "s_bridge"

    # p-layer semi-hubs: small positive A ~ 0.12, near amphoteric line
    if abs(D) < 1e-3 and 0.10 <= A <= 0.16:
        return "p_semihub"

    # p-layer strong acceptors: A ≈ 1.237
    if abs(D) < 1e-3 and 1.1 <= A <= 1.3:
        return "p_acceptor"

    # Inert / noble-like: D≈0, A≈0, inert role
    if abs(D) < 1e-3 and abs(A) < 1e-3 and role == "inert":
        return "inert"

    # d-block: octahedral 6-port metals (here just by Z range)
    if 21 <= Z <= 30:
        return "d_octa"

    return "other"


def main() -> None:
    RESULTS_DIR.mkdir(exist_ok=True)

    src = DATA_DIR / "element_indices_with_dblock.csv"
    if not src.exists():
        raise SystemExit(
            "element_indices_with_dblock.csv not found. "
            "Run run_pipeline.py (or extend_d_block_from_pauling.py) first."
        )

    df = pd.read_csv(src)
    df["geom_class"] = df.apply(classify_geom_row, axis=1)

    # Save raw Geom periodic table v1
    out_csv = RESULTS_DIR / "geom_periodic_table_v1.csv"
    df.to_csv(out_csv, index=False)

    # Periodicity statistics for main classes
    focus_classes = ["s_donor", "s_bridge", "p_semihub", "p_acceptor", "inert"]
    mask = df["geom_class"].isin(focus_classes)
    group_stats = (
        df[mask]
        .groupby(["geom_class", "role"], dropna=False)
        .agg(
            mean_D=("D_index", "mean"),
            std_D=("D_index", "std"),
            mean_A=("A_index", "mean"),
            std_A=("A_index", "std"),
            n=("Z", "count"),
        )
        .reset_index()
    )

    stats_path = RESULTS_DIR / "geom_periodic_law_stats.txt"
    with stats_path.open("w", encoding="utf-8") as f:
        f.write("Geom periodic law stats (D/A by class):\n\n")
        f.write(group_stats.to_string(index=False))
        f.write("\n")

    print(f"Wrote {out_csv} and {stats_path}")


if __name__ == "__main__":
    main()
