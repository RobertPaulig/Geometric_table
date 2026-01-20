import json
import zipfile
from pathlib import Path

from hetero2.scale_proof import P5Config, write_p5_evidence_pack


def _build_pack(
    tmp_path: Path,
    *,
    correctness_gate_rate: float,
    min_scale_samples: int,
) -> Path:
    out_dir = tmp_path / "out"
    cfg = P5Config(
        n_atoms_bins=(20, 200),
        samples_per_bin=1,
        seed=0,
        curve_id="dos_H",
        energy_points=32,
        dos_eta=0.05,
        potential_scale_gamma=1.0,
        edge_weight_mode="bond_order_delta_chi",
        integrator_eps_abs=1e-6,
        integrator_eps_rel=1e-4,
        integrator_subdomains_max=16,
        integrator_poly_degree_max=8,
        integrator_quad_order_max=16,
        integrator_eval_budget_max=256,
        integrator_split_criterion="max_abs_error",
        overhead_region_n_max=50,
        gate_n_min=200,
        correctness_gate_rate=float(correctness_gate_rate),
        min_scale_samples=int(min_scale_samples),
        speedup_gate_break_even=1.0,
        speedup_gate_strong=2.0,
    )
    return write_p5_evidence_pack(out_dir=out_dir, cfg=cfg)


def test_p5_1_contract_zip_contains_required_canonical_files(tmp_path: Path) -> None:
    zip_path = _build_pack(tmp_path, correctness_gate_rate=0.0, min_scale_samples=5)
    assert zip_path.exists()

    with zipfile.ZipFile(zip_path, "r") as zf:
        names = set(zf.namelist())
        for required in [
            "summary.csv",
            "summary_metadata.json",
            "manifest.json",
            "checksums.sha256",
            "integration_compare.csv",
            "integration_speed_profile.csv",
            "adaptive_integration_trace.csv",
            "adaptive_integration_summary.json",
            "fixtures_polymer_scale.csv",
            "speedup_vs_n.csv",
        ]:
            assert required in names


def test_p5_1_contract_summary_metadata_required_fields_and_verdict_logic(tmp_path: Path) -> None:
    inconclusive_zip = _build_pack(tmp_path / "inconclusive", correctness_gate_rate=0.0, min_scale_samples=5)
    with zipfile.ZipFile(inconclusive_zip, "r") as zf:
        meta = json.loads(zf.read("summary_metadata.json").decode("utf-8"))
        assert meta["law_ref"]["contract_path"] == "docs/contracts/INTEGRATION_SCALE_CONTRACT.md"
        assert meta["law_ref"]["contract_version"] == "p5.1.v1"
        assert str(meta["law_ref"]["contract_commit"])

        for k in [
            "integrator_mode",
            "integrator_energy_min",
            "integrator_energy_max",
            "integrator_energy_points",
            "integrator_eta",
            "integrator_eps_abs",
            "integrator_eps_rel",
            "gate_n_min",
            "correctness_gate_rate",
            "min_scale_samples",
            "integrator_valid_row_fraction",
            "integrator_correctness_pass_rate_at_scale",
            "integrator_speedup_median_at_scale",
            "integrator_eval_ratio_median_at_scale",
            "integrator_correctness_verdict",
            "integrator_speedup_verdict",
            "integrator_verdict_reason",
            "potential_unit_model",
            "potential_scale_gamma",
        ]:
            assert k in meta

        assert str(meta["integrator_correctness_verdict"]) == "PASS_CORRECTNESS_AT_SCALE"
        assert str(meta["integrator_speedup_verdict"]) == "INCONCLUSIVE_NOT_ENOUGH_SCALE_SAMPLES"

    fail_zip = _build_pack(tmp_path / "fail", correctness_gate_rate=1.1, min_scale_samples=1)
    with zipfile.ZipFile(fail_zip, "r") as zf:
        meta = json.loads(zf.read("summary_metadata.json").decode("utf-8"))
        assert str(meta["integrator_correctness_verdict"]) == "FAIL_CORRECTNESS_AT_SCALE"
        assert str(meta["integrator_speedup_verdict"]) == "NOT_VALID_DUE_TO_CORRECTNESS"

