import csv
import json
import sys
from pathlib import Path

import pytest


def test_batch_emits_index_and_manifest(tmp_path: Path) -> None:
    pytest.importorskip("rdkit")
    input_csv = tmp_path / "input.csv"
    rows = [
        {"id": "mol_a", "smiles": "CC(=O)OC1=CC=CC=C1C(=O)O"},
        {"id": "mol_b", "smiles": "CC(=O)OC1=CC=CC=C1C(=O)O"},
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
            f"sys.exit(main_batch(['--input','{input_csv.as_posix()}','--out_dir','{out_dir.as_posix()}',"
            "'--score_mode','mock','--seed','0']))"
        ),
    ]
    import subprocess

    subprocess.run(cmd, check=True)

    summary = out_dir / "summary.csv"
    index_md = out_dir / "index.md"
    manifest = out_dir / "manifest.json"

    assert summary.exists()
    assert index_md.exists()
    assert manifest.exists()

    index_text = index_md.read_text(encoding="utf-8")
    assert "mol_a" in index_text and "mol_b" in index_text
    assert "./mol_a.report.md" in index_text
    assert "./mol_b.report.md" in index_text

    manifest_data = json.loads(manifest.read_text(encoding="utf-8"))
    assert "tool_version" in manifest_data
    assert "config" in manifest_data
    cfg = manifest_data["config"]
    assert cfg.get("seed_strategy") == "global"
    assert "guardrails_max_atoms" in cfg
    assert "score_mode" in cfg
    assert cfg.get("potential_unit_model") == "dimensionless"
    assert float(cfg.get("potential_scale_gamma")) == 1.0
