import csv
import json
import zipfile
from pathlib import Path
from typing import Iterable

import pytest

from hetero2.batch import run_batch
from hetero2.physics_operator import DOS_ETA_DEFAULT, DOS_GRID_N_DEFAULT, DOS_LDOS_SCHEMA


def _write_csv(path: Path, *, rows: Iterable[dict[str, str]]) -> None:
    path.write_text("", encoding="utf-8")
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "smiles"])
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def test_batch_writes_dos_ldos_artifacts_and_includes_them_in_zip(tmp_path: Path) -> None:
    pytest.importorskip("rdkit")

    input_csv = tmp_path / "input.csv"
    out_dir = tmp_path / "out"
    _write_csv(
        input_csv,
        rows=[
            {"id": "m0", "smiles": "c1ccccc1"},
        ],
    )

    summary_path = run_batch(
        input_csv=input_csv,
        out_dir=out_dir,
        artifacts="light",
        seed=0,
        timestamp="test",
        k_decoys=3,
        score_mode="mock",
        scores_input=None,
        physics_mode="both",
        edge_weight_mode="unweighted",
        seed_strategy="global",
        no_index=True,
        no_manifest=False,
        zip_pack=True,
        workers=1,
        timeout_s=30.0,
        resume=False,
        overwrite=True,
    )

    assert summary_path.exists()

    dos_curve_csv = out_dir / "dos_curve.csv"
    ldos_summary_csv = out_dir / "ldos_summary.csv"
    summary_metadata_json = out_dir / "summary_metadata.json"
    zip_path = out_dir / "evidence_pack.zip"

    assert dos_curve_csv.exists()
    assert ldos_summary_csv.exists()
    assert summary_metadata_json.exists()
    assert zip_path.exists()

    with dos_curve_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        assert reader.fieldnames == ["energy", "dos_L", "dos_H", "dos_WH"]

    with ldos_summary_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        assert reader.fieldnames == [
            "id",
            "H_atom_idx",
            "H_atomic_number",
            "H_ldos_peak_energy",
            "H_ldos_peak_value",
            "H_ldos_entropy",
            "WH_atom_idx",
            "WH_atomic_number",
            "WH_ldos_peak_energy",
            "WH_ldos_peak_value",
            "WH_ldos_entropy",
        ]

    meta = json.loads(summary_metadata_json.read_text(encoding="utf-8"))
    assert meta["dos_ldos_schema"] == DOS_LDOS_SCHEMA
    assert int(meta["dos_grid_n"]) == int(DOS_GRID_N_DEFAULT)
    assert float(meta["dos_eta"]) == float(DOS_ETA_DEFAULT)

    with zipfile.ZipFile(zip_path, "r") as zf:
        names = set(zf.namelist())
        assert "dos_curve.csv" in names
        assert "ldos_summary.csv" in names
        assert "summary_metadata.json" in names
