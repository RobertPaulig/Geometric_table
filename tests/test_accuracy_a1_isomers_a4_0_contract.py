import csv
import json
import os
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest


RUN_A4 = os.environ.get("RUN_A4_TESTS") == "1"
pytestmark = pytest.mark.skipif(not RUN_A4, reason="A4.0 contract tests are opt-in; set RUN_A4_TESTS=1")


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def test_accuracy_a1_isomers_a4_0_signed_edge_contract(tmp_path: Path) -> None:
    pytest.importorskip("rdkit")
    repo_root = _repo_root()
    input_csv = repo_root / "data/accuracy/isomer_truth.v1.csv"
    assert input_csv.exists()

    out_dir = tmp_path / "out"
    cmd = [
        sys.executable,
        "scripts/accuracy_a1_isomers_a4_0_signed_edge.py",
        "--experiment_id",
        "ACCURACY-A4.0",
        "--input_csv",
        input_csv.as_posix(),
        "--out_dir",
        out_dir.as_posix(),
        "--seed",
        "0",
    ]
    subprocess.run(cmd, cwd=repo_root, check=True)

    expected = [
        out_dir / "predictions.csv",
        out_dir / "summary.csv",
        out_dir / "fold_metrics.csv",
        out_dir / "group_metrics.csv",
        out_dir / "edge_score_by_molecule.csv",
        out_dir / "edge_contrib_topk.csv",
        out_dir / "metrics.json",
        out_dir / "index.md",
        out_dir / "provenance.json",
        out_dir / "manifest.json",
        out_dir / "checksums.sha256",
        out_dir / "evidence_pack.zip",
    ]
    for p in expected:
        assert p.exists()

    preds = list(csv.DictReader((out_dir / "predictions.csv").read_text(encoding="utf-8").splitlines()))
    assert len(preds) == 35
    assert {
        "fold_id",
        "id",
        "group_id",
        "smiles",
        "truth_rel_kcalmol",
        "pred_raw",
        "pred_rel",
        "pred_rank",
        "variant",
        "phi_fixed",
    }.issubset(set(preds[0].keys()))

    metrics = json.loads((out_dir / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["schema_version"] == "accuracy_a1_isomers_a4_0.v1"
    assert metrics["variant"] == "B"
    assert "kpi" in metrics
    assert "metrics_loocv_test_functional_only" in metrics
    assert metrics["files"]["predictions_csv"] == "predictions.csv"
    assert metrics["files"]["edge_score_by_molecule_csv"] == "edge_score_by_molecule.csv"
    assert metrics["files"]["edge_contrib_topk_csv"] == "edge_contrib_topk.csv"

    zip_path = out_dir / "evidence_pack.zip"
    with zipfile.ZipFile(zip_path, "r") as zf:
        names = set(zf.namelist())
        for required in [
            "predictions.csv",
            "summary.csv",
            "fold_metrics.csv",
            "group_metrics.csv",
            "edge_score_by_molecule.csv",
            "edge_contrib_topk.csv",
            "metrics.json",
            "index.md",
            "provenance.json",
            "manifest.json",
            "checksums.sha256",
            "data/accuracy/isomer_truth.v1.csv",
            "docs/contracts/isomer_truth.v1.md",
            "data/accuracy/raw/dft_golden_isomers_v2_spice2_0_1.csv",
            "data/accuracy/raw/dft_golden_isomers_v2_spice2_0_1.csv.sha256",
            "data/atoms_db_v1.json",
            "docs/specs/accuracy_a4_0_signed_edge_observable.md",
        ]:
            assert required in names

