import csv
import json
import sys
import zipfile
from pathlib import Path

import pytest


def test_evidence_pack_has_checksums_and_manifest_files(tmp_path: Path) -> None:
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
    import subprocess

    cmd = [
        sys.executable,
        "-c",
        (
            "from hetero2.cli import main_batch; import sys; "
            f"sys.exit(main_batch(['--input','{input_csv.as_posix()}','--out_dir','{out_dir.as_posix()}',"
            "'--score_mode','mock','--zip_pack']))"
        ),
    ]
    subprocess.run(cmd, check=True)

    checksums = out_dir / "checksums.sha256"
    manifest_path = out_dir / "manifest.json"
    index_path = out_dir / "index.md"
    zip_path = out_dir / "evidence_pack.zip"

    assert checksums.exists()
    assert manifest_path.exists()
    assert index_path.exists()
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    files = {f["path"] for f in data.get("files", []) if isinstance(f, dict)}
    assert "./summary.csv" in files
    assert "./index.md" in files
    assert "./manifest.json" in files or any("manifest" in f for f in files)

    assert zip_path.exists()
    with zipfile.ZipFile(zip_path, "r") as zf:
        names = set(zf.namelist())
        assert "index.md" in names
        assert "manifest.json" in names
