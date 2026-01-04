from __future__ import annotations

import pytest

from hetero2.guardrails import preflight_smiles
from hetero2.pipeline import run_pipeline_v2


def test_preflight_invalid_and_too_many_atoms() -> None:
    pytest.importorskip("rdkit")
    invalid = preflight_smiles("C1")
    assert invalid.skip_reason == "invalid_smiles"
    assert invalid.warnings and invalid.warnings == sorted(set(invalid.warnings))

    large = preflight_smiles("CCCCCC", max_atoms=2)
    assert not large.ok
    assert large.skip_reason == "too_many_atoms"
    assert any("skip:too_large" in w for w in large.warnings)


def test_pipeline_guardrails_disconnected_and_invalid(tmp_path) -> None:
    pytest.importorskip("rdkit")
    ts = "2026-01-02T00:00:00+00:00"

    disconnected = run_pipeline_v2("CC.O", k_decoys=2, seed=0, timestamp=ts)
    assert disconnected.get("schema_version") == "hetero2_pipeline.v1"
    assert disconnected.get("skip", {}).get("reason") == "disconnected"
    assert disconnected.get("audit", {}).get("neg_controls", {}).get("verdict") == "SKIP"
    assert disconnected.get("warnings", []) == sorted(set(disconnected.get("warnings", [])))

    invalid = run_pipeline_v2("C1", k_decoys=2, seed=0, timestamp=ts)
    assert invalid.get("skip", {}).get("reason") == "invalid_smiles"
    assert invalid.get("decoys") == []
    assert invalid.get("audit", {}).get("neg_controls", {}).get("verdict") == "SKIP"


def test_pipeline_default_mock_and_guardrail_limit() -> None:
    pytest.importorskip("rdkit")
    ts = "2026-01-02T00:00:00+00:00"

    ok_payload = run_pipeline_v2("CC(=O)OC1=CC=CC=C1C(=O)O", k_decoys=2, seed=0, timestamp=ts)
    assert "skip" not in ok_payload
    assert ok_payload.get("score_mode") == "mock"

    limited = run_pipeline_v2("CCO", k_decoys=1, seed=0, timestamp=ts, guardrails_max_atoms=1)
    assert limited.get("skip", {}).get("reason") == "too_many_atoms"
    assert any("too_large" in w for w in limited.get("warnings", []))
