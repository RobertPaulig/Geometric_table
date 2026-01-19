import csv
import json
import zipfile
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


def test_batch_writes_adaptive_integration_artifacts_in_both_mode(tmp_path: Path) -> None:
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
        integrator_eps_abs=1e-6,
        integrator_eps_rel=1e-4,
        integrator_subdomains_max=16,
        integrator_poly_degree_max=8,
        integrator_quad_order_max=16,
        integrator_eval_budget_max=4096,
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

    adaptive_trace_csv = out_dir / "adaptive_integration_trace.csv"
    adaptive_summary_json = out_dir / "adaptive_integration_summary.json"
    compare_csv = out_dir / "integration_compare.csv"
    speed_profile_csv = out_dir / "integration_speed_profile.csv"
    zip_path = out_dir / "evidence_pack.zip"
    meta_path = out_dir / "summary_metadata.json"

    assert adaptive_trace_csv.exists()
    assert adaptive_summary_json.exists()
    assert compare_csv.exists()
    assert speed_profile_csv.exists()
    assert zip_path.exists()
    assert meta_path.exists()

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    assert meta["integrator_mode"] == "both"
    assert "integrator_eps_abs" in meta
    assert "integrator_eps_rel" in meta
    assert "integrator_correctness_pass_rate" in meta
    assert "integrator_speedup_median" in meta
    assert "integrator_speedup_verdict" in meta
    assert "integrator_verdict" in meta
    assert "integrator_eval_ratio_median" in meta
    assert "integrator_cache_hit_rate_median" in meta

    with compare_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        assert reader.fieldnames is not None
        for key in [
            "molecule_id",
            "curve_id",
            "baseline_value",
            "adaptive_value",
            "abs_err",
            "rel_err",
            "pass_tolerance",
            "baseline_walltime_ms",
            "adaptive_walltime_ms",
            "speedup",
            "adaptive_verdict_row",
        ]:
            assert key in reader.fieldnames
        rows = list(reader)
        assert rows

    with speed_profile_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        assert reader.fieldnames is not None
        for key in [
            "molecule_id",
            "curve_id",
            "baseline_points",
            "adaptive_evals_total",
            "eval_ratio",
            "baseline_walltime_ms",
            "adaptive_walltime_ms",
            "speedup",
            "cache_hit_rate",
            "segments_used",
            "adaptive_verdict_row",
        ]:
            assert key in reader.fieldnames
        rows = list(reader)
        assert rows

    with zipfile.ZipFile(zip_path, "r") as zf:
        names = set(zf.namelist())
        assert "adaptive_integration_trace.csv" in names
        assert "adaptive_integration_summary.json" in names
        assert "integration_compare.csv" in names
        assert "integration_speed_profile.csv" in names

