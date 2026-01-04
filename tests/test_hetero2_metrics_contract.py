import csv
import json
import sys
from pathlib import Path

import pytest


def test_metrics_manifest_and_index(tmp_path: Path) -> None:
    pytest.importorskip("rdkit")
    input_csv = tmp_path / "input.csv"
    rows = [{"id": "mol_a", "smiles": "CC(=O)OC1=CC=CC=C1C(=O)O"}]
    with input_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "smiles"])
        writer.writeheader()
        writer.writerows(rows)

    out_dir = tmp_path / "out"
    import subprocess

    cmd = [
        sys.executable,
        "-c",
        (
            "from hetero2.cli import main_batch; import sys; "
            f"sys.exit(main_batch(['--input','{input_csv.as_posix()}','--out_dir','{out_dir.as_posix()}',"
            "'--score_mode','mock']))"
        ),
    ]
    subprocess.run(cmd, check=True)

    metrics = out_dir / "metrics.json"
    manifest = out_dir / "manifest.json"
    index_md = out_dir / "index.md"
    assert metrics.exists() and manifest.exists() and index_md.exists()

    metrics_data = json.loads(metrics.read_text(encoding="utf-8"))
    assert "counts" in metrics_data and "config" in metrics_data

    manifest_data = json.loads(manifest.read_text(encoding="utf-8"))
    files = {f["path"] for f in manifest_data.get("files", []) if isinstance(f, dict)}
    assert "./metrics.json" in files
    assert "./index.md" in files
