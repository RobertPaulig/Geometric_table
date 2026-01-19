import csv
import json
import math
from pathlib import Path
from typing import Iterable

import pytest

from hetero2.batch import run_batch


def _write_csv(path: Path, *, rows: Iterable[dict[str, str]]) -> None:
    path.write_text("", encoding="utf-8")
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "smiles"])
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def test_integrator_eval_ratio_median_positive_when_ok_rows_exist(tmp_path: Path) -> None:
    pytest.importorskip("rdkit")

    input_csv = tmp_path / "input.csv"
    out_dir = tmp_path / "out"
    _write_csv(input_csv, rows=[{"id": "m0", "smiles": "c1ccccc1"}])

    run_batch(
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
        integrator_mode="both",
        integrator_energy_points=128,
        integrator_eps_abs=1e-6,
        integrator_eps_rel=1e-4,
        integrator_subdomains_max=16,
        integrator_poly_degree_max=8,
        integrator_quad_order_max=16,
        integrator_eval_budget_max=0,  # auto
        integrator_split_criterion="max_abs_error",
        seed_strategy="global",
        no_index=True,
        no_manifest=False,
        zip_pack=True,
        workers=1,
        timeout_s=30.0,
        resume=False,
        overwrite=True,
    )

    meta = json.loads((out_dir / "summary_metadata.json").read_text(encoding="utf-8"))
    assert float(meta["integrator_valid_row_fraction"]) >= 0.95

    speed_rows = _read_csv(out_dir / "integration_speed_profile.csv")
    ok_rows = [r for r in speed_rows if str(r.get("row_verdict", "")) == "OK"]

    eval_ratio_median = float(meta["integrator_eval_ratio_median"])
    speedup_median = float(meta["integrator_speedup_median"])

    if ok_rows:
        assert eval_ratio_median > 0.0
        assert speedup_median > 0.0
    else:
        assert math.isnan(eval_ratio_median)
        assert math.isnan(speedup_median)


def test_integrator_verdict_marks_metrics_invalid_when_curves_missing(tmp_path: Path) -> None:
    pytest.importorskip("rdkit")

    input_csv = tmp_path / "input.csv"
    out_dir = tmp_path / "out"
    # Y (Z=39) is not present in data/atoms_db_v1.json; SCF/physics modes should mark missing params.
    _write_csv(input_csv, rows=[{"id": "m0", "smiles": "[Y]"}])

    run_batch(
        input_csv=input_csv,
        out_dir=out_dir,
        artifacts="light",
        seed=0,
        timestamp="test",
        k_decoys=1,
        score_mode="mock",
        scores_input=None,
        physics_mode="both",
        edge_weight_mode="unweighted",
        integrator_mode="both",
        integrator_energy_points=32,
        integrator_eps_abs=1e-6,
        integrator_eps_rel=1e-4,
        integrator_subdomains_max=8,
        integrator_poly_degree_max=6,
        integrator_quad_order_max=8,
        integrator_eval_budget_max=0,
        integrator_split_criterion="max_abs_error",
        seed_strategy="global",
        no_index=True,
        no_manifest=False,
        zip_pack=True,
        workers=1,
        timeout_s=30.0,
        resume=False,
        overwrite=True,
    )

    meta = json.loads((out_dir / "summary_metadata.json").read_text(encoding="utf-8"))
    assert str(meta["integrator_verdict"]) == "INCONCLUSIVE_METRICS_INVALID"
    assert float(meta["integrator_valid_row_fraction"]) < 0.95
