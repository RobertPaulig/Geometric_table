import pytest

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
