import csv
import json
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def test_accuracy_a1_isomers_a1_3_produces_group_aware_metrics_and_full_grid(tmp_path: Path) -> None:
    pytest.importorskip("rdkit")
    repo_root = _repo_root()
    input_csv = repo_root / "data/accuracy/isomer_truth.v1.csv"
    assert input_csv.exists()

    out_dir = tmp_path / "out"
    cmd = [
        sys.executable,
        "scripts/accuracy_a1_isomers_sweep.py",
        "--experiment_id",
        "ACCURACY-A1.3",
        "--input_csv",
        input_csv.as_posix(),
        "--out_dir",
        out_dir.as_posix(),
        "--edge_weight_modes",
        "unweighted",
        "--potential_modes",
        "static",
        "--gammas",
        "0.24,0.25,0.26",
        "--predictors",
        "logdet_shifted_eps",
        "--eps_values",
        "1e-6,1e-4",
        "--shift_values",
        "0.0,1e-3",
        "--kpi_mean_spearman_by_group_min",
        "0.55",
        "--kpi_median_spearman_by_group_min",
        "0.50",
    ]
    subprocess.run(cmd, cwd=repo_root, check=True)

    expected = [
        out_dir / "sweep_results.csv",
        out_dir / "best_config.json",
        out_dir / "summary.csv",
        out_dir / "metrics.json",
        out_dir / "index.md",
        out_dir / "provenance.json",
        out_dir / "manifest.json",
        out_dir / "checksums.sha256",
        out_dir / "evidence_pack.zip",
    ]
    for p in expected:
        assert p.exists()

    sweep = list(csv.DictReader((out_dir / "sweep_results.csv").read_text(encoding="utf-8").splitlines()))
    assert len(sweep) == 12  # gammas(3) * eps(2) * shift(2); beta unused for logdet predictor
    required_cols = {
        "predictor",
        "edge_weight_mode",
        "potential_mode",
        "gamma",
        "eps",
        "shift",
        "mean_spearman_pred_vs_truth",
        "pairwise_order_accuracy_overall",
        "top1_accuracy_mean",
    }
    assert required_cols.issubset(set(sweep[0].keys()))

    best_cfg = json.loads((out_dir / "best_config.json").read_text(encoding="utf-8"))
    assert {"predictor", "edge_weight_mode", "potential_mode", "potential_scale_gamma", "eps", "shift"}.issubset(set(best_cfg.keys()))
    assert any(
        r.get("predictor") == best_cfg.get("predictor")
        and r.get("edge_weight_mode") == best_cfg.get("edge_weight_mode")
        and r.get("potential_mode") == best_cfg.get("potential_mode")
        and float(r.get("gamma") or 0.0) == float(best_cfg.get("potential_scale_gamma") or 0.0)
        and float(r.get("eps") or 0.0) == float(best_cfg.get("eps") or 0.0)
        and float(r.get("shift") or 0.0) == float(best_cfg.get("shift") or 0.0)
        for r in sweep
    )

    metrics = json.loads((out_dir / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["dataset"]["rows_total"] == 35
    assert metrics["dataset"]["groups_total"] == 10
    assert "kpi" in metrics

    best = metrics["best"]
    best_metrics = best["metrics"]
    for k in [
        "spearman_by_group",
        "mean_spearman_by_group",
        "median_spearman_by_group",
        "pairwise_order_accuracy_by_group_mean",
        "top1_accuracy_mean",
    ]:
        assert k in best_metrics
    assert isinstance(best_metrics["spearman_by_group"], dict)
    assert len(best_metrics["spearman_by_group"]) == 10

    zip_path = out_dir / "evidence_pack.zip"
    with zipfile.ZipFile(zip_path, "r") as zf:
        names = set(zf.namelist())
        for required in [
            "summary.csv",
            "metrics.json",
            "index.md",
            "provenance.json",
            "manifest.json",
            "checksums.sha256",
            "sweep_results.csv",
            "best_config.json",
        ]:
            assert required in names

