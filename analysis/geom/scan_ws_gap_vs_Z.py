from __future__ import annotations

from pathlib import Path
import argparse

import pandas as pd

from core.spectral_density_ws import WSRadialParams
from core.port_geometry_spectral import ws_sp_gap
from analysis.thermo_cli import add_thermo_args, apply_thermo_from_args


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Scan WS s-p gap vs Z with thermo-configurable WS parameters."
    )
    add_thermo_args(parser)
    parser.add_argument(
        "--out",
        type=str,
        default="results/ws_sp_gap_scan_Z1_30.csv",
        help="Output CSV path for WS s-p gap scan.",
    )
    args = parser.parse_args()
    apply_thermo_from_args(args)

    params = WSRadialParams()
    rows = []
    for Z in range(1, 31):
        gap = ws_sp_gap(Z, params)
        rows.append({"Z": Z, "gap_sp": gap})

    df = pd.DataFrame(rows)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"Saved WS s-p gap scan to {out_path}")


if __name__ == "__main__":
    main()
