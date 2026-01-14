import pytest

pytest.importorskip("rdkit")

from hetero2.decoy_strategy import DECOY_STRATEGY_FALLBACK, DECOY_STRATEGY_SCHEMA, generate_decoys_v1


def test_decoy_strategy_fallback_generates_decoys_for_benzene() -> None:
    decoys_result, strategy = generate_decoys_v1("c1ccccc1", k=2, seed=0, max_attempts=None)
    assert strategy.schema_version == DECOY_STRATEGY_SCHEMA
    assert strategy.strategy_id == DECOY_STRATEGY_FALLBACK
    assert len(decoys_result.decoys) > 0

