import csv
import json
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def test_accuracy_a1_isomers_sweep_produces_evidence_pack(tmp_path: Path) -> None:
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
        "scripts/accuracy_a1_isomers_sweep.py",
        "--input_csv",
        input_csv.as_posix(),
        "--out_dir",
        out_dir.as_posix(),
        "--edge_weight_modes",
        "unweighted,bond_order",
        "--potential_modes",
        "static",
        "--gammas",
        "0.0,1.0",
        "--predictors",
        "free_energy_beta,logdet_shifted_eps",
        "--betas",
        "1.0",
    ]
    subprocess.run(cmd, cwd=_repo_root(), check=True)

    expected = [
        out_dir / "sweep_results.csv",
        out_dir / "best_config.json",
        out_dir / "summary.csv",
        out_dir / "metrics.json",
        out_dir / "index.md",
        out_dir / "manifest.json",
        out_dir / "checksums.sha256",
        out_dir / "evidence_pack.zip",
    ]
    for p in expected:
        assert p.exists()

    sweep = list(csv.DictReader((out_dir / "sweep_results.csv").read_text(encoding="utf-8").splitlines()))
    assert len(sweep) > 0
    required_cols = {
        "predictor",
        "edge_weight_mode",
        "potential_mode",
        "gamma",
        "beta",
        "mean_spearman_pred_vs_truth",
        "pairwise_order_accuracy_overall",
        "top1_accuracy_mean",
    }
    assert required_cols.issubset(set(sweep[0].keys()))

    best_cfg = json.loads((out_dir / "best_config.json").read_text(encoding="utf-8"))
    assert {"predictor", "edge_weight_mode", "potential_mode", "potential_scale_gamma"}.issubset(set(best_cfg.keys()))
    beta_val = best_cfg.get("beta")
    beta_str = "" if beta_val is None else str(float(beta_val))
    assert any(
        r.get("predictor") == best_cfg.get("predictor")
        and r.get("edge_weight_mode") == best_cfg.get("edge_weight_mode")
        and r.get("potential_mode") == best_cfg.get("potential_mode")
        and float(r.get("gamma") or 0.0) == float(best_cfg.get("potential_scale_gamma") or 0.0)
        and str(r.get("beta") or "") == beta_str
        for r in sweep
    )

    metrics = json.loads((out_dir / "metrics.json").read_text(encoding="utf-8"))
    assert {"baseline", "best", "dataset", "config", "verdict"}.issubset(set(metrics.keys()))
    assert metrics["dataset"]["rows_total"] == 4
    assert metrics["dataset"]["groups_total"] == 2
    assert "metrics" in metrics["baseline"]
    assert "metrics" in metrics["best"]

    zip_path = out_dir / "evidence_pack.zip"
    with zipfile.ZipFile(zip_path, "r") as zf:
        names = set(zf.namelist())
        assert "summary.csv" in names
        assert "metrics.json" in names
        assert "index.md" in names
        assert "manifest.json" in names
        assert "checksums.sha256" in names
        assert "sweep_results.csv" in names
        assert "best_config.json" in names

