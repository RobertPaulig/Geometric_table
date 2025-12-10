from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


def main() -> None:
    data_path = Path("data") / "geom_nuclear_complexity_summary.csv"
    if not data_path.exists():
        raise SystemExit(f"Missing {data_path}, run analyze_geom_nuclear_complexity first.")

    df = pd.read_csv(data_path)

    if "C_norm_fdm_mean" not in df.columns:
        raise SystemExit("C_norm_fdm_mean not found in geom_nuclear_complexity_summary.csv")

    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    out_txt = results_dir / "fdm_vs_nucleus_stats.txt"

    with out_txt.open("w", encoding="utf-8") as f:
        f.write("FDM vs nuclear metrics (geom_nuclear_complexity_summary)\n")
        f.write("========================================================\n\n")

        def corr_report(name: str, x: pd.Series, y: pd.Series | None) -> None:
            if y is None:
                f.write(f"{name}: missing column\n")
                return
            mask = x.notna() & y.notna()
            if mask.sum() < 3:
                f.write(f"{name}: not enough data (n={mask.sum()})\n")
                return
            r, p = stats.spearmanr(x[mask], y[mask])
            f.write(f"{name}: Spearman r={r:.4f}, p={p:.6g}, n={mask.sum()}\n")

        # Global correlations
        f.write("[GLOBAL]\n")
        corr_report("C_norm_fdm_mean vs band_width", df["C_norm_fdm_mean"], df.get("band_width"))
        corr_report("C_norm_fdm_mean vs Nbest_over_Z", df["C_norm_fdm_mean"], df.get("Nbest_over_Z"))
        corr_report("C_norm_fdm_mean vs neutron_excess", df["C_norm_fdm_mean"], df.get("neutron_excess"))
        f.write("\n")

        # By role
        if "role" in df.columns:
            for role in ["terminator", "bridge", "hub"]:
                sub = df[df["role"] == role]
                if sub.empty:
                    continue
                f.write(f"[ROLE={role}]\n")
                corr_report(
                    "C_norm_fdm_mean vs band_width", sub["C_norm_fdm_mean"], sub.get("band_width")
                )
                corr_report(
                    "C_norm_fdm_mean vs Nbest_over_Z", sub["C_norm_fdm_mean"], sub.get("Nbest_over_Z")
                )
                corr_report(
                    "C_norm_fdm_mean vs neutron_excess",
                    sub["C_norm_fdm_mean"],
                    sub.get("neutron_excess"),
                )
                f.write("\n")

        # d-block and living hubs groups if available
        for flag_col, label in [
            ("is_d_block", "d_block"),
            ("is_living_hub", "living_hubs"),
        ]:
            if flag_col in df.columns:
                sub = df[df[flag_col]]
                if sub.empty:
                    continue
                f.write(f"[GROUP={label}]\n")
                corr_report(
                    "C_norm_fdm_mean vs band_width", sub["C_norm_fdm_mean"], sub.get("band_width")
                )
                corr_report(
                    "C_norm_fdm_mean vs Nbest_over_Z", sub["C_norm_fdm_mean"], sub.get("Nbest_over_Z")
                )
                corr_report(
                    "C_norm_fdm_mean vs neutron_excess",
                    sub["C_norm_fdm_mean"],
                    sub.get("neutron_excess"),
                )
                f.write("\n")

    print(f"Wrote {out_txt}")


if __name__ == "__main__":
    main()
