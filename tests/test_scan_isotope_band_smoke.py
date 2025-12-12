from __future__ import annotations

from pathlib import Path

import pandas as pd

from analysis.nuclear.scan_isotope_band import main as scan_isotope_band_main


def test_scan_isotope_band_small_range(tmp_path, monkeypatch) -> None:
    """
    Быстрый смоук-тест: узкий диапазон Z, чтобы не гонять всё.
    """
    scan_isotope_band_main(["--z-min", "6", "--z-max", "8"])

    path = Path("data") / "geom_isotope_bands.csv"
    assert path.exists()

    df = pd.read_csv(path)
    assert (df["Z"].between(6, 8)).all()
    assert df["N_best"].notna().any()

