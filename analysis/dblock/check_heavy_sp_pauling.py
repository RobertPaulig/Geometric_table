from __future__ import annotations

import numpy as np
import pandas as pd

from analysis.io_utils import data_path, results_path
from fit_chi_to_pauling import fit_linear


HEAVY_SP = ["Cs", "Ba", "Tl", "Pb", "Bi", "Po", "At"]


def main() -> None:
    # 1. Текущая линейная подгонка chi_Pauling ≈ a * chi_spec + b
    a, b, X_calib, Y_calib = fit_linear()
    pred_calib = a * X_calib + b
    rmse_calib = float(np.sqrt(np.mean((pred_calib - Y_calib) ** 2)))

    # 2. Загружаем geom-индексы (включая chi_spec)
    df_idx = pd.read_csv(data_path("element_indices_with_dblock.csv"))

    if "symbol" in df_idx.columns:
        sym_col = "symbol"
    elif "El" in df_idx.columns:
        sym_col = "El"
    else:
        raise RuntimeError("element_indices_with_dblock.csv: no 'symbol' or 'El' column found")

    df_idx = df_idx.rename(columns={sym_col: "symbol"})

    if "chi_spec" not in df_idx.columns:
        raise RuntimeError("element_indices_with_dblock.csv must contain 'chi_spec' column")

    # 3. Референсные χ_Pauling для тяжёлых s/p
    ref = pd.read_csv(data_path("pauling_heavy_sp_reference.csv"))
    ref = ref[ref["symbol"].isin(HEAVY_SP)]

    df_heavy = pd.merge(df_idx, ref, on="symbol", how="inner")
    if df_heavy.empty:
        out_path = results_path("heavy_sp_pauling_fit.txt")
        with out_path.open("w", encoding="utf-8") as f:
            f.write(
                "Heavy s/p Pauling-fit check (geom-spec v4.x)\n"
                "============================================\n\n"
                "No heavy s/p elements from HEAVY_SP found in element_indices_with_dblock.csv.\n"
                "Add Cs/Ba/Tl/Pb/... to index CSV before running this check.\n"
            )
        print(f"Written {out_path} (no heavy elements present yet)")
        return

    # 4. Модельная χ_Pauling и дельты
    df_heavy["chi_model"] = a * df_heavy["chi_spec"] + b
    df_heavy["delta"] = df_heavy["chi_model"] - df_heavy["chi_pauling"]

    rmse_heavy = float(np.sqrt(np.mean(df_heavy["delta"] ** 2)))
    max_abs_heavy = float(df_heavy["delta"].abs().max())

    out_path = results_path("heavy_sp_pauling_fit.txt")
    with out_path.open("w", encoding="utf-8") as f:
        f.write("Heavy s/p Pauling-fit check (geom-spec v4.x)\n")
        f.write("============================================\n\n")
        f.write(f"Linear mapping (current): chi_Pauling ≈ {a:.3f} * chi_spec + {b:.3f}\n")
        f.write(f"RMSE on calibration set (fit_chi_to_pauling.py): {rmse_calib:.3f}\n\n")

        f.write("Per-element heavy s/p comparison:\n\n")
        f.write(
            df_heavy[["symbol", "chi_spec", "chi_pauling", "chi_model", "delta"]]
            .sort_values("symbol")
            .to_string(index=False)
        )
        f.write("\n\n")
        f.write(f"Heavy s/p RMSE   : {rmse_heavy:.3f}\n")
        f.write(f"Heavy s/p max|Δχ|: {max_abs_heavy:.3f}\n")

    print(f"Written {out_path}")


if __name__ == "__main__":
    main()
