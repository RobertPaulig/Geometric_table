import csv
import sys
from pathlib import Path

import pytest


def test_batch_resume_skips_done_ids(tmp_path: Path) -> None:
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
            "'--score_mode','mock','--workers','1']))"
        ),
    ]
    subprocess.run(cmd, check=True)

    cmd_resume = [
        sys.executable,
        "-c",
        (
            "from hetero2.cli import main_batch; import sys; "
            f"sys.exit(main_batch(['--input','{input_csv.as_posix()}','--out_dir','{out_dir.as_posix()}',"
            "'--score_mode','mock','--workers','1','--resume']))"
        ),
    ]
    subprocess.run(cmd_resume, check=True)

    summary = out_dir / "summary.csv"
    data = list(csv.DictReader(summary.read_text(encoding="utf-8").splitlines()))
    ids = [row["id"] for row in data]
    assert len(ids) == 2
    assert set(ids) == {"mol_a", "mol_b"}
