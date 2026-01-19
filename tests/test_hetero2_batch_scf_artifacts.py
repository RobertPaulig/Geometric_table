import csv
import json
import zipfile
from pathlib import Path

import pytest

from hetero2.batch import run_batch
from hetero2.physics_operator import SCF_SCHEMA


def _write_csv(path: Path, *, rows: list[dict[str, str]]) -> None:
    path.write_text("", encoding="utf-8")
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "smiles"])
        writer.writeheader()
        writer.writerows(rows)


def test_batch_emits_scf_artifacts_and_includes_them_in_zip(tmp_path: Path) -> None:
    pytest.importorskip("rdkit")

    input_csv = tmp_path / "input.csv"
    out_dir = tmp_path / "out"
    _write_csv(input_csv, rows=[{"id": "m0", "smiles": "CC(=O)OC1=CC=CC=C1C(=O)O"}])

    run_batch(
        input_csv=input_csv,
        out_dir=out_dir,
        artifacts="light",
        seed=0,
        timestamp="test",
        k_decoys=3,
        score_mode="mock",
        physics_mode="both",
        edge_weight_mode="bond_order_delta_chi",
        potential_mode="both",
        scf_max_iter=50,
        scf_tol=1e-6,
        scf_damping=0.5,
        scf_occ_k=5,
        scf_tau=1.0,
        zip_pack=True,
        workers=1,
    )

    scf_trace_csv = out_dir / "scf_trace.csv"
    potential_vectors_csv = out_dir / "potential_vectors.csv"
    scf_summary_json = out_dir / "scf_summary.json"
    summary_metadata_json = out_dir / "summary_metadata.json"
    zip_path = out_dir / "evidence_pack.zip"

    assert scf_trace_csv.exists()
    assert potential_vectors_csv.exists()
    assert scf_summary_json.exists()
    assert summary_metadata_json.exists()
    assert zip_path.exists()

    trace_rows = list(csv.DictReader(scf_trace_csv.read_text(encoding="utf-8").splitlines()))
    assert len(trace_rows) >= 1
    assert trace_rows[0]["row_id"] == "m0"
    assert trace_rows[0]["mol_id"] == "m0"
    assert "residual" in trace_rows[0]
    assert "delta_V_inf" in trace_rows[0]
    assert "delta_rho_l1" in trace_rows[0]
    assert "gamma" in trace_rows[0]
    assert "status" in trace_rows[0]
    assert "stop_reason" in trace_rows[0]

    vec_rows = list(csv.DictReader(potential_vectors_csv.read_text(encoding="utf-8").splitlines()))
    assert len(vec_rows) >= 1
    assert vec_rows[0]["id"] == "m0"
    assert vec_rows[0]["atom_Z"]
    assert vec_rows[0]["V0"] != ""
    assert vec_rows[0]["V_scaled"] != ""
    assert vec_rows[0]["gamma"] != ""

    meta = json.loads(summary_metadata_json.read_text(encoding="utf-8"))
    assert meta["potential_mode"] == "both"
    assert meta["potential_unit_model"] == "dimensionless"
    assert float(meta["potential_scale_gamma"]) == 1.0
    assert meta["scf_schema"] == SCF_SCHEMA
    assert bool(meta["scf_enabled"]) is True
    assert meta["scf_status"] in {"CONVERGED", "MAX_ITER"}
    assert int(meta["scf_rows_total"]) == 1
    assert meta["scf_audit_verdict"] in {"SUCCESS", "INCONCLUSIVE", "TRIVIAL_SCF", "NONCONVERGENT"}
    assert isinstance(meta["scf_audit_reason"], str)

    scf_sum = json.loads(scf_summary_json.read_text(encoding="utf-8"))
    assert scf_sum["schema_version"] == "hetero2_scf_summary.v1"
    assert bool(scf_sum["scf_enabled"]) is True
    assert scf_sum["scf_status"] in {"CONVERGED", "MAX_ITER"}
    assert int(scf_sum["rows_total"]) == 1
    assert scf_sum["audit_verdict"] in {"SUCCESS", "INCONCLUSIVE", "TRIVIAL_SCF", "NONCONVERGENT"}
    assert "triviality_flags" in scf_sum
    assert "stats_all" in scf_sum
    assert "stats_asym_fixture" in scf_sum

    with zipfile.ZipFile(zip_path, "r") as zf:
        names = set(zf.namelist())
        assert "scf_trace.csv" in names
        assert "potential_vectors.csv" in names
        assert "scf_summary.json" in names


def test_batch_marks_inconclusive_when_scf_does_not_converge(tmp_path: Path) -> None:
    pytest.importorskip("rdkit")

    input_csv = tmp_path / "input.csv"
    out_dir = tmp_path / "out"
    _write_csv(input_csv, rows=[{"id": "m0", "smiles": "CC(=O)OC1=CC=CC=C1C(=O)O"}])

    run_batch(
        input_csv=input_csv,
        out_dir=out_dir,
        artifacts="light",
        seed=0,
        timestamp="test",
        k_decoys=3,
        score_mode="mock",
        physics_mode="both",
        edge_weight_mode="bond_order_delta_chi",
        potential_mode="self_consistent",
        scf_max_iter=1,
        scf_tol=1e-30,
        scf_damping=1.0,
        scf_occ_k=3,
        scf_tau=1.0,
        zip_pack=True,
        workers=1,
    )

    summary_rows = list(csv.DictReader((out_dir / "summary.csv").read_text(encoding="utf-8").splitlines()))
    assert summary_rows[0]["status"] == "OK"
    assert summary_rows[0]["outcome_verdict"] == "INCONCLUSIVE_SCF_NOT_CONVERGED"
    assert summary_rows[0]["outcome_reason"] == "scf_not_converged"

    meta = json.loads((out_dir / "summary_metadata.json").read_text(encoding="utf-8"))
    assert meta["scf_status"] == "MAX_ITER"

    scf_sum = json.loads((out_dir / "scf_summary.json").read_text(encoding="utf-8"))
    assert bool(scf_sum["scf_enabled"]) is True
    assert scf_sum["scf_status"] == "MAX_ITER"

