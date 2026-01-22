import csv
import json
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def test_build_isomer_truth_v1_generates_canonical_csv(tmp_path: Path) -> None:
    raw_csv = tmp_path / "raw.csv"
    rows = [
        {"id": "mol_1", "group_id": "G1", "smiles": "CCO", "energy_rel_kcalmol": "0.0"},
        {"id": "mol_2", "group_id": "G1", "smiles": "CCN", "energy_rel_kcalmol": "0.1"},
        {"id": "mol_3", "group_id": "G2", "smiles": "c1ccccc1", "energy_rel_kcalmol": "0.0"},
        {"id": "mol_4", "group_id": "G2", "smiles": "c1ccncc1", "energy_rel_kcalmol": "0.2"},
    ]
    with raw_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["id", "group_id", "smiles", "energy_rel_kcalmol"], lineterminator="\n")
        w.writeheader()
        w.writerows(rows)

    out_csv = tmp_path / "isomer_truth.v1.csv"
    cmd = [sys.executable, "scripts/build_isomer_truth_v1.py", "--raw_csv", raw_csv.as_posix(), "--out_csv", out_csv.as_posix()]
    subprocess.run(cmd, cwd=_repo_root(), check=True)

    assert out_csv.exists()
    out_rows = list(csv.DictReader(out_csv.read_text(encoding="utf-8").splitlines()))
    assert len(out_rows) == 4
    assert {"id", "group_id", "smiles", "energy_rel_kcalmol", "truth_source", "truth_version"}.issubset(set(out_rows[0].keys()))
    assert {r["truth_version"] for r in out_rows} == {"isomer_truth.v1"}
    assert {r["truth_source"] for r in out_rows} == {"spice2_0_1_isomers_v2"}
    assert len({r["id"] for r in out_rows}) == len(out_rows)
    assert all((r["group_id"] or "").strip() for r in out_rows)
    for r in out_rows:
        float(r["energy_rel_kcalmol"])

    counts: dict[str, int] = {}
    for r in out_rows:
        counts[r["group_id"]] = counts.get(r["group_id"], 0) + 1
    assert all(n >= 2 for n in counts.values())


def test_accuracy_a1_isomers_run_produces_evidence_pack(tmp_path: Path) -> None:
    pytest.importorskip("rdkit")
    input_csv = tmp_path / "isomer_truth.v1.csv"
    rows = [
        {"id": "mol_1", "group_id": "G1", "smiles": "CCO", "energy_rel_kcalmol": "0.0"},
        {"id": "mol_2", "group_id": "G1", "smiles": "CCN", "energy_rel_kcalmol": "0.1"},
        {"id": "mol_3", "group_id": "G2", "smiles": "c1ccccc1", "energy_rel_kcalmol": "0.0"},
        {"id": "mol_4", "group_id": "G2", "smiles": "c1ccncc1", "energy_rel_kcalmol": "0.2"},
    ]
    with input_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["id", "group_id", "smiles", "energy_rel_kcalmol"], lineterminator="\n")
        w.writeheader()
        w.writerows(rows)

    out_dir = tmp_path / "out"
    cmd = [
        sys.executable,
        "scripts/accuracy_a1_isomers_run.py",
        "--input_csv",
        input_csv.as_posix(),
        "--out_dir",
        out_dir.as_posix(),
        "--potential_scale_gamma",
        "1.0",
    ]
    subprocess.run(cmd, cwd=_repo_root(), check=True)

    expected = [
        out_dir / "summary.csv",
        out_dir / "metrics.json",
        out_dir / "index.md",
        out_dir / "manifest.json",
        out_dir / "checksums.sha256",
        out_dir / "evidence_pack.zip",
    ]
    for p in expected:
        assert p.exists()

    metrics = json.loads((out_dir / "metrics.json").read_text(encoding="utf-8"))
    assert {"config", "dataset", "metrics", "per_group"}.issubset(set(metrics.keys()))
    assert metrics["dataset"]["rows_total"] == 4
    assert metrics["dataset"]["groups_total"] == 2
    assert metrics["config"]["pred_proxy"] == "H_trace"

    manifest = json.loads((out_dir / "manifest.json").read_text(encoding="utf-8"))
    files = {f["path"] for f in manifest.get("files", []) if isinstance(f, dict)}
    assert "./summary.csv" in files
    assert "./metrics.json" in files
    assert "./index.md" in files
    assert "./manifest.json" in files or any("manifest" in f for f in files)

    zip_path = out_dir / "evidence_pack.zip"
    with zipfile.ZipFile(zip_path, "r") as zf:
        names = set(zf.namelist())
        assert "summary.csv" in names
        assert "metrics.json" in names
        assert "index.md" in names
        assert "manifest.json" in names
        assert "checksums.sha256" in names
