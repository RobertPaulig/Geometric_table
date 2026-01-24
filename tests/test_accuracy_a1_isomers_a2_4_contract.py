import csv
import json
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def test_accuracy_a1_isomers_a2_4_soft_occupancy_ldos_contract(tmp_path: Path) -> None:
    pytest.importorskip("rdkit")
    repo_root = _repo_root()
    input_csv = repo_root / "data/accuracy/isomer_truth.v1.csv"
    assert input_csv.exists()

    out_dir = tmp_path / "out"
    cmd = [
        sys.executable,
        "scripts/accuracy_a1_isomers_a2_self_consistent.py",
        "--a2_variant",
        "full_functional_v1_a2_4",
        "--experiment_id",
        "ACCURACY-A2.4",
        "--input_csv",
        input_csv.as_posix(),
        "--out_dir",
        out_dir.as_posix(),
        "--seed",
        "0",
        "--gamma",
        "0.28",
        "--potential_variant",
        "epsilon_z",
        "--edge_weight_mode",
        "bond_order_delta_chi",
        "--calibrator_ridge_lambda",
        "1e-3",
        "--kpi_mean_spearman_by_group_test_min",
        "-1.0",
        "--kpi_median_spearman_by_group_test_min",
        "0.45",
        "--kpi_pairwise_order_accuracy_overall_test_min",
        "0.60",
        "--kpi_top1_accuracy_mean_test_min",
        "0.30",
    ]
    subprocess.run(cmd, cwd=repo_root, check=True)

    expected = [
        out_dir / "predictions.csv",
        out_dir / "summary.csv",
        out_dir / "fold_metrics.csv",
        out_dir / "group_metrics.csv",
        out_dir / "diagnostics.csv",
        out_dir / "rho_compare.csv",
        out_dir / "search_results.csv",
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
    assert {
        "fold_id",
        "selected_config_id",
        "rho_mode",
        "id",
        "group_id",
        "smiles",
        "truth_rel_kcalmol",
        "pred_raw",
        "pred_rel",
    }.issubset(set(preds[0].keys()))

    rho_cmp = list(csv.DictReader((out_dir / "rho_compare.csv").read_text(encoding="utf-8").splitlines()))
    assert len(rho_cmp) == 35
    assert {
        "rho_mode",
        "rho_entropy",
        "rho_trace_norm",
        "rho_sum",
        "rho_renorm_applied",
        "rho_renorm_delta",
        "rho_ldos_k_eff",
        "rho_ldos_deg_tol",
        "rho_ldos_degeneracy_guard_applied",
        "lambda_min",
        "lambda_max",
        "lambda_min_window",
        "lambda_gap",
        "trace_weights",
    }.issubset(set(rho_cmp[0].keys()))

    metrics = json.loads((out_dir / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["schema_version"] == "accuracy_a1_isomers_a2_4.v1"
    assert metrics["dataset"]["rows_total"] == 35
    assert metrics["dataset"]["groups_total"] == 10
    assert "metrics_loocv_test_functional_only" in metrics
    assert "metrics_loocv_test_calibrated_linear" in metrics
    assert "metrics_loocv_test_functional_only_by_rho_mode" in metrics
    assert "negative_spearman_groups_test_by_rho_mode" in metrics
    assert "kpi" in metrics
    assert "worst_groups" in metrics
    assert "rho_compare_csv" in metrics["files"]
    assert metrics["files"]["rho_compare_csv"] == "rho_compare.csv"

    zip_path = out_dir / "evidence_pack.zip"
    with zipfile.ZipFile(zip_path, "r") as zf:
        names = set(zf.namelist())
        for required in [
            "predictions.csv",
            "fold_metrics.csv",
            "group_metrics.csv",
            "diagnostics.csv",
            "rho_compare.csv",
            "search_results.csv",
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
