from __future__ import annotations

from core.nuclear_config import load_nuclear_config, set_current_nuclear_config
from core.nuclear_island import nuclear_functional


def test_nuclear_functional_baseline_smoke() -> None:
    cfg = load_nuclear_config("configs/nuclear_shell_baseline.yaml")
    set_current_nuclear_config(cfg)
    F = nuclear_functional(6, 8)
    assert isinstance(F, float)


