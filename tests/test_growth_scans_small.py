from __future__ import annotations

from pathlib import Path

import pytest

from analysis.growth.scan_cycles_vs_params import main as scan_cycles_main
from analysis.growth.scan_temperature_effects import main as scan_temp_main


@pytest.mark.slow
def test_scan_cycles_vs_params_small() -> None:
    scan_cycles_main(
        [
            "--config",
            "configs/growth_cy1a.yaml",
            "--num-runs",
            "20",
        ]
    )
    path = Path("results") / "cycle_param_scan.csv"
    assert path.exists()


@pytest.mark.slow
def test_scan_temperature_effects_small() -> None:
    scan_temp_main(
        [
            "--config",
            "configs/growth_cy1a.yaml",
            "--num-runs",
            "20",
        ]
    )
    path = Path("results") / "temperature_scan_growth.csv"
    assert path.exists()

