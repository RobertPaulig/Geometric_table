import csv
import json
import os
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest


RUN_A3 = os.environ.get("RUN_A3_TESTS") == "1"
pytestmark = pytest.mark.skipif(not RUN_A3, reason="A3.3 contract tests are opt-in; set RUN_A3_TESTS=1")


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def test_accuracy_a1_isomers_a3_3_phase_rho_pivot_contract(tmp_path: Path) -> None:
    pytest.importorskip("rdkit")
    repo_root = _repo_root()
    input_csv = repo_root / "data/accuracy/isomer_truth.v1.csv"
    assert input_csv.exists()

    out_dir = tmp_path / "out"
    cmd = [
        sys.executable,
        "scripts/accuracy_a1_isomers_a3_3_phase_rho_pivot.py",
        "--experiment_id",
        "ACCURACY-A3.3",
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
        out_dir / "phase_summary.csv",
        out_dir / "rho_compare.csv",
        out_dir / "search_results.csv",
        out_dir / "phi_sweep_sensitivity.csv",
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
        "rho_mode",
        "phi_selected",
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
        "phi_selected",
        "rho_sum",
        "rho_trace_norm",
        "rho_imag_max",
        "rho_entropy",
        "rho_floor_rate",
        "rho_renorm_applied",
        "rho_renorm_delta",
        "parity_rho_max_abs",
        "parity_phi_max_abs",
        "parity_pass",
    }.issubset(set(rho_cmp[0].keys()))

    phi_sweep = list(csv.DictReader((out_dir / "phi_sweep_sensitivity.csv").read_text(encoding="utf-8").splitlines()))
    assert len(phi_sweep) == 35 * 4
    assert {
        "fold_id",
        "id",
        "group_id",
        "phi_value",
        "delta_rho_L1",
        "delta_rho_Linf",
        "delta_entropy",
        "n_rings",
    }.issubset(set(phi_sweep[0].keys()))

    metrics = json.loads((out_dir / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["schema_version"] == "accuracy_a1_isomers_a3_3.v1"
    assert metrics["dataset"]["rows_total"] == 35
    assert metrics["dataset"]["groups_total"] == 10
    assert "metrics_loocv_test_functional_only" in metrics
    assert "best_config" in metrics
    assert metrics["best_config"]["nested_selection"] is True
    assert metrics["best_config"]["search_space_size"] == 4
    assert "selected_phi_by_outer_fold" in metrics["best_config"]
    assert metrics["best_config"]["inner_selection_metric_tiebreaker"] == "mean_delta_rho_L1_train"
    assert "rho_imag_max_max" in metrics
    assert "kpi" in metrics
    assert "phase_summary_csv" in metrics["files"]
    assert metrics["files"]["rho_compare_csv"] == "rho_compare.csv"
    assert metrics["files"]["phi_sweep_sensitivity_csv"] == "phi_sweep_sensitivity.csv"

    search_results = list(csv.DictReader((out_dir / "search_results.csv").read_text(encoding="utf-8").splitlines()))
    assert len(search_results) == 10 * 4
    assert "mean_delta_rho_L1_train" in search_results[0]

    zip_path = out_dir / "evidence_pack.zip"
    with zipfile.ZipFile(zip_path, "r") as zf:
        names = set(zf.namelist())
        for required in [
            "predictions.csv",
            "fold_metrics.csv",
            "group_metrics.csv",
            "phase_summary.csv",
            "rho_compare.csv",
            "search_results.csv",
            "phi_sweep_sensitivity.csv",
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
            "docs/specs/accuracy_a3_3_phase_rho_pivot.md",
        ]:
            assert required in names

