from __future__ import annotations

from pathlib import Path

import pandas as pd


def test_element_indices_v4_schema() -> None:
    path = Path("data") / "element_indices_v4.csv"
    df = pd.read_csv(path)

    required = {
        "Z",
        "El",
        "role",
        "period",
        "chi_spec",
        "E_port",
        "D_index",
        "A_index",
    }
    assert required.issubset(df.columns), f"Missing columns: {required - set(df.columns)}"

    assert df["Z"].notna().all()
    assert df["El"].notna().all()
    assert not df["D_index"].isna().any()
    assert not df["A_index"].isna().any()

