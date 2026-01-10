import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hetero1a.schemas import SCORES_SCHEMA  # noqa: E402
from hetero2.pipeline import run_pipeline_v2  # noqa: E402


def test_hetero_scores_v1_fixture_is_accepted(tmp_path: Path) -> None:
    pytest.importorskip("rdkit")
    fixture_path = Path(__file__).resolve().parent / "fixtures" / "hetero_scores.v1.sample.json"
    payload = json.loads(fixture_path.read_text(encoding="utf-8"))
    assert payload.get("schema_version") == SCORES_SCHEMA

    scores_path = tmp_path / "scores.json"
    scores_path.write_text(json.dumps(payload), encoding="utf-8")

    smiles = "CC(=O)OC1=CC=CC=C1C(=O)O"  # aspirin, expected to generate decoys under rdkit
    out = run_pipeline_v2(smiles, score_mode="external_scores", scores_input=str(scores_path), k_decoys=2, seed=0)
    assert out.get("score_mode") == "external_scores"
    assert out.get("scores_provenance", {}).get("scores_schema_version") == SCORES_SCHEMA


def test_hetero_scores_schema_version_must_match(tmp_path: Path) -> None:
    pytest.importorskip("rdkit")
    scores_path = tmp_path / "scores.json"
    scores_path.write_text(
        json.dumps({"schema_version": "hetero_scores.v0", "original": {"score": 1.0, "weight": 1.0}, "decoys": {}}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError):
        run_pipeline_v2("CC", score_mode="external_scores", scores_input=str(scores_path))

