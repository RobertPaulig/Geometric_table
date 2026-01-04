import csv
import json
import sys
from pathlib import Path

import pytest


def test_hetero2_batch_creates_artifacts(tmp_path: Path) -> None:
    pytest.importorskip("rdkit")
    input_csv = tmp_path / "input.csv"
    rows = [
        {"id": "aspirin", "smiles": "CC(=O)OC1=CC=CC=C1C(=O)O"},
        {"id": "empty", "smiles": ""},
    ]
    with input_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "smiles"])
        writer.writeheader()
        writer.writerows(rows)

    out_dir = tmp_path / "out"
    cmd = [
        sys.executable,
        "-c",
        (
            "from hetero2.cli import main_batch; import sys; "
            f"sys.exit(main_batch(['--input','{input_csv.as_posix()}','--out_dir','{out_dir.as_posix()}','--score_mode','mock']))"
        ),
    ]
    import subprocess

    subprocess.run(cmd, check=True)

    summary = out_dir / "summary.csv"
    assert summary.exists()
    data = list(csv.DictReader(summary.read_text(encoding="utf-8").splitlines()))
    assert len(data) == 2
    assert data[0]["id"] == "aspirin"
    assert data[0]["status"] == "OK"
    assert data[1]["id"] == "empty"
    assert data[1]["status"] in {"SKIP", "ERROR"}
    assert data[1]["reason"]
    assert (out_dir / "aspirin.pipeline.json").exists()
    assert (out_dir / "aspirin.report.md").exists()
    assets = out_dir / "aspirin_assets"
    assert assets.exists()


def test_batch_respects_score_mode(tmp_path: Path) -> None:
    pytest.importorskip("rdkit")
    dummy_scores = tmp_path / "scores.json"
    dummy_scores.write_text("{}", encoding="utf-8")

    # Run with score_mode=mock and provided scores_input -> warning but still OK.
    input_csv = tmp_path / "input_mock.csv"
    rows = [{"id": "mock_with_scores", "smiles": "CC(=O)OC1=CC=CC=C1C(=O)O", "scores_input": dummy_scores.as_posix()}]
    with input_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "smiles", "scores_input"])
        writer.writeheader()
        writer.writerows(rows)
    out_dir = tmp_path / "out_mock"
    import subprocess

    cmd = [
        sys.executable,
        "-c",
        (
            "from hetero2.cli import main_batch; import sys; "
            f"sys.exit(main_batch(['--input','{input_csv.as_posix()}','--out_dir','{out_dir.as_posix()}',"
            "'--score_mode','mock','--timestamp','2026-01-02T00:00:00+00:00','--seed','0']))"
        ),
    ]
    subprocess.run(cmd, check=True)
    data = list(csv.DictReader((out_dir / "summary.csv").read_text(encoding="utf-8").splitlines()))
    assert data[0]["status"] == "OK"
    payload_mock = json.loads((out_dir / "mock_with_scores.pipeline.json").read_text(encoding="utf-8"))
    assert "scores_input_ignored_in_mock_mode" in payload_mock.get("warnings", [])

    # Run with score_mode=external_scores but no scores_input -> SKIP with reason.
    input_csv2 = tmp_path / "input_external.csv"
    rows2 = [{"id": "external_missing", "smiles": "CC(=O)OC1=CC=CC=C1C(=O)O"}]
    with input_csv2.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "smiles"])
        writer.writeheader()
        writer.writerows(rows2)
    out_dir2 = tmp_path / "out_external"
    cmd2 = [
        sys.executable,
        "-c",
        (
            "from hetero2.cli import main_batch; import sys; "
            f"sys.exit(main_batch(['--input','{input_csv2.as_posix()}','--out_dir','{out_dir2.as_posix()}',"
            "'--score_mode','external_scores','--timestamp','2026-01-02T00:00:00+00:00','--seed','0']))"
        ),
    ]
    subprocess.run(cmd2, check=True)
    data2 = list(csv.DictReader((out_dir2 / "summary.csv").read_text(encoding="utf-8").splitlines()))
    assert data2[0]["status"] == "SKIP"
    assert data2[0]["reason"] == "missing_scores_input"
    payload_missing = json.loads((out_dir2 / "external_missing.pipeline.json").read_text(encoding="utf-8"))
    assert payload_missing.get("skip", {}).get("reason") == "missing_scores_input"
