import json
from pathlib import Path

import pytest

pytest.importorskip("rdkit")

from hetero2.decoys_rewire import DecoyResult
from hetero2.pipeline import run_pipeline_v2


def test_pipeline_skips_when_no_decoys(monkeypatch):
    def _fake_generate_rewire_decoys(*args, **kwargs):
        return DecoyResult(decoys=[], warnings=["test:no_decoys"], stats={"attempts": 0})

    monkeypatch.setattr("hetero2.pipeline.generate_rewire_decoys", _fake_generate_rewire_decoys)
    out = run_pipeline_v2("CC", k_decoys=2, seed=0, timestamp="t0")

    assert out.get("skip", {}).get("reason") == "no_decoys_generated"
    assert out.get("audit", {}).get("neg_controls", {}).get("verdict") == "SKIP"
    assert out.get("decoys") == []
    assert "skip:no_decoys_generated" in out.get("warnings", [])


def test_pipeline_external_scores_missing_all_decoys_skips_audit(tmp_path: Path, monkeypatch):
    def _fake_generate_rewire_decoys(*args, **kwargs):
        return DecoyResult(
            decoys=[{"smiles": "CC", "hash": "deadbeef", "ring_info": {}, "physchem": {}}],
            warnings=[],
            stats={"attempts": 1},
        )

    scores_path = tmp_path / "scores.json"
    scores_payload = {"schema_version": "hetero_scores.v1", "original": {"score": 1.0, "weight": 1.0}, "decoys": {}}
    scores_path.write_text(json.dumps(scores_payload), encoding="utf-8")

    monkeypatch.setattr("hetero2.pipeline.generate_rewire_decoys", _fake_generate_rewire_decoys)
    out = run_pipeline_v2(
        "CC",
        k_decoys=1,
        seed=0,
        timestamp="t0",
        score_mode="external_scores",
        scores_input=str(scores_path),
    )

    assert out.get("score_mode") == "external_scores"
    assert out.get("audit", {}).get("neg_controls", {}).get("verdict") == "SKIP"
    assert "skip:missing_scores_for_all_decoys" in out.get("warnings", [])
