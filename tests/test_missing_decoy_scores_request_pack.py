import csv
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest


def test_batch_writes_missing_decoy_scores_request_pack(tmp_path: Path) -> None:
    pytest.importorskip("rdkit")

    scores_path = tmp_path / "scores.json"
    scores_payload = {
        "schema_version": "hetero_scores.v1",
        "original": {"score": 1.0, "weight": 1.0},
        "decoys": {},
    }
    scores_path.write_text(json.dumps(scores_payload), encoding="utf-8")

    input_csv = tmp_path / "input.csv"
    rows = [{"id": "aspirin", "smiles": "CC(=O)OC1=CC=CC=C1C(=O)O"}]
    with input_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "smiles"])
        writer.writeheader()
        writer.writerows(rows)

    out_dir = tmp_path / "out_scores"
    cmd = [
        sys.executable,
        "-c",
        (
            "from hetero2.cli import main_batch; import sys; "
            f"sys.exit(main_batch(['--input','{input_csv.as_posix()}','--out_dir','{out_dir.as_posix()}',"
            f"'--score_mode','external_scores','--scores_input','{scores_path.as_posix()}',"
            "'--timestamp','2026-01-02T00:00:00+00:00','--seed','0']))"
        ),
    ]
    subprocess.run(cmd, check=True)

    request_path = out_dir / "missing_decoy_scores.csv"
    assert request_path.exists()
    request_rows = list(csv.DictReader(request_path.read_text(encoding="utf-8").splitlines()))
    assert request_rows

    # With a single input row, each missing decoy hash affects exactly 1 row.
    assert all(int(r["count_rows_affected"]) == 1 for r in request_rows)

    # Validate decoy_hash matches sha256(decoy_smiles).
    for r in request_rows[:5]:
        decoy_hash = str(r["decoy_hash"]).strip().lower()
        decoy_smiles = str(r["decoy_smiles"]).strip()
        assert len(decoy_hash) == 64
        assert decoy_smiles
        assert hashlib.sha256(decoy_smiles.encode("utf-8")).hexdigest() == decoy_hash

    metrics = json.loads((out_dir / "metrics.json").read_text(encoding="utf-8"))
    cov = metrics.get("scores_coverage", {})
    assert cov.get("unique_missing_decoy_hashes") == len(request_rows)

    top10 = cov.get("missing_decoy_hashes_top10")
    assert isinstance(top10, list)
    assert top10
    assert "decoy_hash" in top10[0]
    assert "count_rows_affected" in top10[0]
