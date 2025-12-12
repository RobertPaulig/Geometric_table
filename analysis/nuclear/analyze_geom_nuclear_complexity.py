from __future__ import annotations

import sys
import pandas as pd
from pathlib import Path

from analysis.io_utils import (
    data_path,
    write_text_result,
)


def _find_csv(base: Path, name: str) -> Path | None:
    """
    Ищет CSV сначала в data/, затем в корне репозитория.
    Возвращает Path или None, если файл не найден.
    """
    candidates = [base / "data" / name, base / name]
    for p in candidates:
        if p.exists():
            return p
    return None


def load_data() -> pd.DataFrame:
    base = Path(".")

    # Geometric layer
    geom_path = base / "data" / "element_indices_with_dblock.csv"
    df_geom = pd.read_csv(geom_path)
    df_geom = df_geom.rename(columns={"El": "symbol"})

    # Complexity layer
    comp_path = base / "data" / "complexity_summary.csv"
    df_comp = pd.read_csv(comp_path)
    df_comp = df_comp.rename(columns={"Element": "symbol", "D": "D_index", "A": "A_index", "Role": "role"})

    # Optional nuclear layers
    iso_path = _find_csv(base, "geom_isotope_bands.csv")
    map_path = _find_csv(base, "geom_nuclear_map.csv")

    df_iso = pd.read_csv(iso_path) if iso_path is not None else None
    df_map = pd.read_csv(map_path) if map_path is not None else None

    # Start from geometric dataframe and merge others
    df = df_geom.copy()

    comp_cols = ["symbol", "Avg_Complexity", "Max_Complexity", "Avg_Size"]
    # add FDM-related columns if present
    for col in [
        "C_total_v1_mean",
        "C_total_v1_std",
        "C_total_fdm_mean",
        "C_total_fdm_std",
        "C_norm_v1_mean",
        "C_norm_v1_std",
        "C_norm_fdm_mean",
        "C_norm_fdm_std",
    ]:
        if col in df_comp.columns:
            comp_cols.append(col)

    df = df.merge(df_comp[comp_cols], on="symbol", how="left")

    if df_iso is not None:
        # Try to standardize column names for isotope bands
        col_renames = {}
        if "El" in df_iso.columns:
            col_renames["El"] = "symbol"
        if "Z" not in df_iso.columns and "Z_atomic" in df_iso.columns:
            col_renames["Z_atomic"] = "Z"
        df_iso = df_iso.rename(columns=col_renames)

        band_cols = [
            "Z",
            "symbol",
            "N_min",
            "N_max",
            "N_best",
            "band_width",
        ]
        band_cols = [c for c in band_cols if c in df_iso.columns]
        df = df.merge(df_iso[band_cols], on=["Z", "symbol"], how="left")

    if df_map is not None:
        col_renames = {}
        if "El" in df_map.columns:
            col_renames["El"] = "symbol"
        if "Z" not in df_map.columns and "Z_atomic" in df_map.columns:
            col_renames["Z_atomic"] = "Z"
        df_map = df_map.rename(columns=col_renames)

        map_cols = ["Z", "symbol", "N_best", "F_min"]
        map_cols = [c for c in map_cols if c in df_map.columns]
        df = df.merge(df_map[map_cols], on=["Z", "symbol"], how="left", suffixes=("", "_map"))

        if "N_best_map" in df.columns:
            df["N_best"] = df["N_best"].fillna(df["N_best_map"])

    # One row per element (Z, symbol)
    df_all = df.drop_duplicates(subset=["Z", "symbol"]).reset_index(drop=True)

    # Derived nuclear columns
    if "band_width" not in df_all.columns and {"N_min", "N_max"} <= set(df_all.columns):
        df_all["band_width"] = df_all["N_max"] - df_all["N_min"]

    if "N_best" in df_all.columns:
        has_nbest = df_all["N_best"].notna() & (df_all["Z"] > 0)
        df_all.loc[has_nbest, "Nbest_over_Z"] = df_all.loc[has_nbest, "N_best"] / df_all.loc[has_nbest, "Z"]
        df_all.loc[has_nbest, "neutron_excess"] = df_all.loc[has_nbest, "N_best"] - df_all.loc[has_nbest, "Z"]
    else:
        df_all["Nbest_over_Z"] = pd.NA
        df_all["neutron_excess"] = pd.NA

    # Flags / groups
    LIVING_HUBS = ["C", "N", "Si", "P", "Ge", "As", "Sn", "Sb"]
    df_all["is_living_hub"] = df_all["symbol"].isin(LIVING_HUBS)
    df_all["is_d_block"] = df_all["Z"].between(21, 30)
    df_all["is_donor_plateau"] = (df_all["D_index"] > 0) & (df_all["A_index"].abs() < 1e-3)
    df_all["is_strong_acceptor_plateau"] = df_all["A_index"].between(1.1, 1.3)

    return df_all


def summarize_group(df: pd.DataFrame, name: str, out) -> None:
    cols = ["Avg_Complexity", "Max_Complexity", "band_width", "Nbest_over_Z", "neutron_excess"]
    cols_present = [c for c in cols if c in df.columns]
    sub = df[cols_present]
    print(f"[GROUP] {name:15s}: n={len(sub):2d}", file=out)
    for c in cols:
        if c in sub.columns and not sub[c].dropna().empty:
            print(
                f"  {c:15s}: mean={sub[c].mean():8.3f}, std={sub[c].std():8.3f}",
                file=out,
            )
        else:
            print(f"  {c:15s}: mean=    nan, std=    nan", file=out)
    print(file=out)


def main() -> None:
    df_all = load_data()

    # Save combined table
    out_csv = data_path("geom_nuclear_complexity_summary.csv")
    df_all.to_csv(out_csv, index=False)

    # Prepare text stats
    lines: list[str] = []

    def writeln(s: str = "") -> None:
        lines.append(s + "\n")

    from io import StringIO

    buf = StringIO()
    summarize_group(df_all[df_all["role"] == "terminator"], "role=terminator", buf)
    summarize_group(df_all[df_all["role"] == "bridge"], "role=bridge", buf)
    summarize_group(df_all[df_all["role"] == "hub"], "role=hub", buf)
    summarize_group(df_all[df_all["role"] == "inert"], "role=inert", buf)
    summarize_group(df_all[df_all["is_d_block"]], "d_block", buf)
    summarize_group(df_all[df_all["is_living_hub"]], "living_hubs", buf)
    summarize_group(df_all[~df_all["is_living_hub"]], "non_living", buf)

    report = buf.getvalue()
    out_txt_path = write_text_result(report, "geom_nuclear_complexity_stats.txt")

    print(f"[ANALYSIS-IO] Saved complexity summary: {out_csv}")
    print(f"[ANALYSIS-IO] Saved geom–nuclear stats to {out_txt_path}")


if __name__ == "__main__":
    sys.exit(main())
