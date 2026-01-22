import csv
import json
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def test_accuracy_a1_isomers_a1_4_feature_upgrade_contract(tmp_path: Path) -> None:
    pytest.importorskip("rdkit")
    repo_root = _repo_root()
    input_csv = repo_root / "data/accuracy/isomer_truth.v1.csv"
    assert input_csv.exists()

    out_dir = tmp_path / "out"
    cmd = [
        sys.executable,
        "scripts/accuracy_a1_isomers_feature_upgrade.py",
        "--experiment_id",
        "ACCURACY-A1.4",
        "--input_csv",
        input_csv.as_posix(),
        "--out_dir",
        out_dir.as_posix(),
        "--seed",
        "0",
        "--n_train_groups",
        "7",
        "--gamma",
        "0.28",
    ]
    subprocess.run(cmd, cwd=repo_root, check=True)

    expected = [
        out_dir / "predictions.csv",
        out_dir / "summary.csv",
        out_dir / "group_metrics.csv",
        out_dir / "best_config.json",
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
    assert {"id", "group_id", "split", "smiles", "truth_rel_kcalmol", "pred_raw", "pred_rel"}.issubset(set(preds[0].keys()))

    groups = {str(r["group_id"]) for r in preds}
    splits = {str(r["split"]) for r in preds}
    assert groups
    assert splits.issubset({"train", "test"})

    metrics = json.loads((out_dir / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["schema_version"] == "accuracy_a1_isomers_a1_4.v1"
    assert metrics["dataset"]["rows_total"] == 35
    assert metrics["dataset"]["groups_total"] == 10
    assert metrics["dataset"]["n_train_groups"] == 7
    assert metrics["dataset"]["n_test_groups"] == 3
    assert set(metrics["split"]["train_groups"]).isdisjoint(set(metrics["split"]["test_groups"]))
    assert set(metrics["split"]["train_groups"]) | set(metrics["split"]["test_groups"]) == groups
    assert {"overall", "train", "test"} == set(metrics["metrics"].keys())
    assert "kpi" in metrics

    group_rows = list(csv.DictReader((out_dir / "group_metrics.csv").read_text(encoding="utf-8").splitlines()))
    assert len(group_rows) == 10
    assert {"group_id", "split", "n", "spearman_pred_vs_truth", "pairwise_order_accuracy", "top1_accuracy"}.issubset(set(group_rows[0].keys()))

    zip_path = out_dir / "evidence_pack.zip"
    with zipfile.ZipFile(zip_path, "r") as zf:
        names = set(zf.namelist())
        for required in [
            "predictions.csv",
            "group_metrics.csv",
            "metrics.json",
            "index.md",
            "best_config.json",
            "provenance.json",
            "manifest.json",
            "checksums.sha256",
            "data/accuracy/isomer_truth.v1.csv",
            "docs/contracts/isomer_truth.v1.md",
            "data/accuracy/raw/dft_golden_isomers_v2_spice2_0_1.csv",
            "data/accuracy/raw/dft_golden_isomers_v2_spice2_0_1.csv.sha256",
            "data/atoms_db_v1.json",
        ]:
            assert required in names

