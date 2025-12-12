from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from analysis.io_utils import (
    read_data_csv,
    MissingDataError,
    write_text_result,
)


def main() -> None:
    try:
        df = read_data_csv(
            "geom_nuclear_complexity_summary.csv",
            required=True,
            expected_columns=["Z", "symbol", "C_norm_fdm_mean"],
        )
    except MissingDataError as e:
        raise SystemExit(str(e))

    if "C_norm_fdm_mean" not in df.columns:
        raise SystemExit("C_norm_fdm_mean not found in geom_nuclear_complexity_summary.csv")

    lines: list[str] = []

    def writeln(s: str = "") -> None:
        lines.append(s + "\n")

    writeln("FDM vs nuclear metrics (geom_nuclear_complexity_summary)")
    writeln("========================================================")
    writeln()

    def corr_report(name: str, x: pd.Series, y: pd.Series | None) -> None:
        if y is None:
            writeln(f"{name}: missing column")
            return
        mask = x.notna() & y.notna()
        if mask.sum() < 3:
            writeln(f"{name}: not enough data (n={mask.sum()})")
            return
        r, p = stats.spearmanr(x[mask], y[mask])
        writeln(f"{name}: Spearman r={r:.4f}, p={p:.6g}, n={mask.sum()}")

    # Global correlations
    writeln("[GLOBAL]")
    corr_report("C_norm_fdm_mean vs band_width", df["C_norm_fdm_mean"], df.get("band_width"))
    corr_report("C_norm_fdm_mean vs Nbest_over_Z", df["C_norm_fdm_mean"], df.get("Nbest_over_Z"))
    corr_report("C_norm_fdm_mean vs neutron_excess", df["C_norm_fdm_mean"], df.get("neutron_excess"))
    writeln()

    # By role
    if "role" in df.columns:
        for role in ["terminator", "bridge", "hub"]:
            sub = df[df["role"] == role]
            if sub.empty:
                continue
            writeln(f"[ROLE={role}]")
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
            writeln()

    # d-block and living hubs groups if available
    for flag_col, label in [
        ("is_d_block", "d_block"),
        ("is_living_hub", "living_hubs"),
    ]:
        if flag_col in df.columns:
            sub = df[df[flag_col]]
            if sub.empty:
                continue
            writeln(f"[GROUP={label}]")
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
            writeln()

    report = "".join(lines)
    out_txt = write_text_result(report, "fdm_vs_nucleus_stats.txt")
    print(f"[ANALYSIS-IO] Saved FDM vs nucleus stats to {out_txt}")


if __name__ == "__main__":
    main()
