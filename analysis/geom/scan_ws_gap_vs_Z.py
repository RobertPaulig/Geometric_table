from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from core.spectral_density_ws import WSRadialParams
from core.port_geometry_spectral import ws_sp_gap


def main() -> None:
    params = WSRadialParams()
    rows = []
    for Z in range(1, 31):
        gap = ws_sp_gap(Z, params)
        rows.append({"Z": Z, "gap_sp": gap})

    df = pd.DataFrame(rows)
    out_dir = Path("results")
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "ws_sp_gap_scan_Z1_30.csv"
    df.to_csv(csv_path, index=False)
    print(f"Saved WS s-p gap scan to {csv_path}")


if __name__ == "__main__":
    main()

