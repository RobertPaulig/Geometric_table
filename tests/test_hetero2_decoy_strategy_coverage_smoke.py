import pytest

pytest.importorskip("rdkit")

from hetero2.decoy_strategy import DECOY_STRATEGY_FALLBACK, DECOY_STRATEGY_STRICT, generate_decoys_v1


def _marker_value(warnings: list[str], prefix: str) -> str:
    matches = [w for w in warnings if w.startswith(prefix)]
    assert len(matches) == 1, f"Expected exactly 1 marker {prefix!r}, got: {matches!r}"
    return matches[0].split(":", 1)[1]


@pytest.mark.parametrize(
    ("smiles", "expect_not_fallback_aromatic_as_single"),
    [
        ("C1CCCCC1", True),  # cyclohexane
        ("CCc1ccccc1", False),  # ethylbenzene
    ],
)
def test_decoy_strategy_coverage_smoke_basic(smiles: str, expect_not_fallback_aromatic_as_single: bool) -> None:
    decoys_result, strategy = generate_decoys_v1(smiles, k=3, seed=0, max_attempts=200)
    assert len(decoys_result.decoys) >= 1

    if expect_not_fallback_aromatic_as_single:
        assert strategy.strategy_id != DECOY_STRATEGY_FALLBACK
        assert "decoy_fallback_aromatic_as_single:1" not in decoys_result.warnings

    assert strategy.strategy_id != DECOY_STRATEGY_STRICT

    assert int(_marker_value(decoys_result.warnings, "candidates_strict:")) >= 0
    assert int(_marker_value(decoys_result.warnings, "candidates_relax_a:")) >= 0
    assert int(_marker_value(decoys_result.warnings, "candidates_relax_b:")) >= 0
    assert _marker_value(decoys_result.warnings, "decoy_strategy_used:")


def test_decoy_strategy_coverage_smoke_caffeine_is_explainable() -> None:
    smiles = "Cn1cnc2n(C)c(=O)n(C)c(=O)c12"
    decoys_result, _ = generate_decoys_v1(smiles, k=3, seed=0, max_attempts=200)

    if len(decoys_result.decoys) >= 1:
        return

    assert int(_marker_value(decoys_result.warnings, "candidates_strict:")) >= 0
    assert int(_marker_value(decoys_result.warnings, "candidates_relax_a:")) >= 0
    assert int(_marker_value(decoys_result.warnings, "candidates_relax_b:")) >= 0
    assert _marker_value(decoys_result.warnings, "decoy_strategy_used:")

