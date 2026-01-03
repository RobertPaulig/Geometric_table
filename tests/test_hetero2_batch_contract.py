import csv
import sys
from pathlib import Path

import pytest


def test_hetero2_batch_creates_artifacts(tmp_path: Path) -> None:
    pytest.importorskip("rdkit")
    input_csv = tmp_path / "input.csv"
    rows = [{"id": "aspirin", "smiles": "CC(=O)OC1=CC=CC=C1C(=O)O"}]
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
    assert data and data[0]["id"] == "aspirin"
    assert (out_dir / "aspirin.pipeline.json").exists()
    assert (out_dir / "aspirin.report.md").exists()
    assets = out_dir / "aspirin_assets"
    assert assets.exists()
