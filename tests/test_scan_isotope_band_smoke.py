from __future__ import annotations

from pathlib import Path

from analysis.nuclear.scan_isotope_band import main as scan_isotope_band_main


def test_scan_isotope_band_small_range(tmp_path, monkeypatch) -> None:
    """
    Быстрый смоук-тест: узкий диапазон Z, чтобы не гонять всё.
    """
    scan_isotope_band_main(
        [
            "--z-min",
            "5",
            "--z-max",
            "10",
        ]
    )

    path = Path("data") / "geom_isotope_bands.csv"
    assert path.exists()

    lines = path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) >= 3
