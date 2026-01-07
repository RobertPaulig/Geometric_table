import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hetero1a.schemas import SCORES_SCHEMA  # noqa: E402
from hetero2.batch import run_batch  # noqa: E402
from hetero2.decoys_rewire import generate_rewire_decoys  # noqa: E402
from hetero2.pipeline import run_pipeline_v2  # noqa: E402


def _write_scores(path: Path, *, decoy_hash: str, schema_version: str) -> None:
    payload = {
        "schema_version": schema_version,
        "original": {"score": 1.0, "weight": 1.0},
        "decoys": {decoy_hash: {"score": 0.1, "weight": 1.0}},
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_external_scores_schema_validation(tmp_path: Path) -> None:
    pytest.importorskip("rdkit")
    scores_path = tmp_path / "scores.json"
    _write_scores(scores_path, decoy_hash="deadbeef", schema_version="bad_schema.v0")

    with pytest.raises(ValueError):
        run_pipeline_v2("CC", score_mode="external_scores", scores_input=str(scores_path))


def test_external_scores_manifest_provenance(tmp_path: Path) -> None:
    pytest.importorskip("rdkit")
    smiles = "CC(=O)OC1=CC=CC=C1C(=O)O"
    decoys = generate_rewire_decoys(smiles, k=1, seed=0, max_attempts=None, lock_aromatic=True, allow_ring_bonds=False)
    assert decoys.decoys
    decoy_hash = str(decoys.decoys[0]["hash"])

    scores_path = tmp_path / "scores.json"
    _write_scores(scores_path, decoy_hash=decoy_hash, schema_version=SCORES_SCHEMA)

    input_csv = tmp_path / "input.csv"
    input_csv.write_text("id,smiles\nm1,CC\n", encoding="utf-8")

    out_dir = tmp_path / "out"
    run_batch(
        input_csv=input_csv,
        out_dir=out_dir,
        artifacts="light",
        score_mode="external_scores",
        scores_input=str(scores_path),
        k_decoys=1,
        workers=1,
    )

    manifest = json.loads((out_dir / "manifest.json").read_text(encoding="utf-8"))
    prov = manifest.get("config", {}).get("scores_provenance", {})
    assert prov.get("scores_input_id") == "scores.json"
    assert prov.get("scores_schema_version") == SCORES_SCHEMA
    assert isinstance(prov.get("scores_input_sha256"), str) and len(prov["scores_input_sha256"]) == 64
