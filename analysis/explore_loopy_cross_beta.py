from __future__ import annotations

"""
explore_loopy_cross_beta.py — CY-1/step5:
скан по β_cross для crossing-aware штрафа поверх уже посчитанных proxy.
"""

from pathlib import Path
from typing import Iterable, List

import numpy as np
import pandas as pd


RESULTS_DIR = Path("results")


def load_crossing_data() -> pd.DataFrame:
    paths = [
        RESULTS_DIR / "crossing_proxy_CY1A.csv",
        RESULTS_DIR / "crossing_proxy_CY1B.csv",
    ]
    frames: List[pd.DataFrame] = []
    for p in paths:
        if p.exists():
            frames.append(pd.read_csv(p))
    if not frames:
        raise FileNotFoundError("crossing_proxy_CY1A/B.csv not found in results/")
    df = pd.concat(frames, ignore_index=True)
    return df


def scan_beta_cross(df: pd.DataFrame, betas: Iterable[float]) -> List[str]:
    lines: List[str] = []
    hubs = {"C", "N", "Si", "P"}
    terms = {"Li", "Na", "K", "Be", "Mg", "Ca", "O", "S"}

    for beta in betas:
        cross_penalty = 1.0 + beta * df["crossing_density"]
        penalty_total = df["penalty_factor"] * cross_penalty

        corr = penalty_total.corr(df["crossing_density"])

        hubs_mask = df["seed"].isin(hubs)
        terms_mask = df["seed"].isin(terms)

        hubs_mean = float(penalty_total[hubs_mask].mean()) if hubs_mask.any() else float("nan")
        terms_mean = float(penalty_total[terms_mask].mean()) if terms_mask.any() else float("nan")

        line = (
            f"beta={beta:.2f}: "
            f"corr(penalty_total, crossing_density)={corr:.4f}, "
            f"hubs_mean={hubs_mean:.3f}, "
            f"term_bridge_mean={terms_mean:.3f}"
        )
        lines.append(line)

    # Простое текстовое резюме по диапазону
    lines.append("")
    lines.append("# Summary (manual inspection recommended):")
    lines.append("# look for beta where corr stays high and means are moderate.")

    return lines


def main() -> None:
    df = load_crossing_data()
    beta_grid = np.linspace(0.0, 3.0, 31)  # 0.0, 0.1, ..., 3.0
    lines = scan_beta_cross(df, beta_grid)
    out_txt = RESULTS_DIR / "loopy_cross_beta_scan.txt"
    out_txt.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {out_txt}")


if __name__ == "__main__":
    main()

