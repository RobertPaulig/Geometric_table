from __future__ import annotations

from analysis.chem.chem_validation_1b_hexane import HEXANE_DEGENERACY


def test_hexane_degeneracy_sum() -> None:
    assert sum(int(v) for v in HEXANE_DEGENERACY.values()) == 1290

