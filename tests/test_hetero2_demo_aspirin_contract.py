import json
import subprocess
import sys
from pathlib import Path

import pytest


def test_hetero2_demo_creates_report_and_assets(tmp_path: Path) -> None:
    pytest.importorskip("rdkit")
    out_dir = tmp_path / "demo"
    out_dir.mkdir()
    cmd = [
        sys.executable,
        "-c",
        (
            "from hetero2.cli import main_demo_aspirin; import sys; "
            f"sys.exit(main_demo_aspirin(['--out_dir','{out_dir.as_posix()}','--stem','aspirin']))"
        ),
    ]
    subprocess.run(cmd, check=True)
    report = out_dir / "aspirin_report.md"
    assets = out_dir / "aspirin_assets"
    assert report.exists()
    assert assets.exists()
    pngs = list(assets.glob("*.png"))
    assert pngs, "expected at least one PNG asset"
    text = report.read_text(encoding="utf-8")
    for section in ["Summary", "Rings", "PhysChem", "Hardness", "Repro"]:
        assert section in text
    assert "![" in text  # images referenced
