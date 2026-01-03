import json
import subprocess
import sys
from pathlib import Path

import pytest


def test_hetero2_pipeline_and_report_cli(tmp_path: Path) -> None:
    pytest.importorskip("rdkit")
    pipeline_path = tmp_path / "pipeline.json"
    report_path = tmp_path / "report.md"
    assets_dir = tmp_path / "report_assets"
    smiles = "CC(=O)OC1=CC=CC=C1C(=O)O"

    cmd_pipeline = [
        sys.executable,
        "-c",
        (
            "from hetero2.cli import main_pipeline; import sys; "
            f"sys.exit(main_pipeline(['--smiles','{smiles}','--out','{pipeline_path.as_posix()}','--score_mode','mock']))"
        ),
    ]
    subprocess.run(cmd_pipeline, check=True)
    assert pipeline_path.exists()
    data = json.loads(pipeline_path.read_text(encoding="utf-8"))
    assert data.get("schema_version") == "hetero2_pipeline.v1"

    cmd_report = [
        sys.executable,
        "-c",
        (
            "from hetero2.cli import main_report; import sys; "
            f"sys.exit(main_report(['--input','{pipeline_path.as_posix()}','--out','{report_path.as_posix()}','--assets_dir','{assets_dir.as_posix()}']))"
        ),
    ]
    subprocess.run(cmd_report, check=True)
    assert report_path.exists()
    assert assets_dir.exists()
    assert list(assets_dir.glob("*.png"))
