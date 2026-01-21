import json
import math
import zipfile
from pathlib import Path

from hetero2.scale_proof import P5Config, write_p5_evidence_pack


def _expected_ring_speedup_verdict(meta: dict) -> str:
    gate_speedup = float(meta.get("scale_speedup_gate_break_even", 1.0))
    correctness_gate_rate = float(meta.get("correctness_gate_rate", 1.0))
    min_scale_samples = int(meta.get("min_scale_samples", 0) or 0)

    ring_n = int(meta.get("cost_scale_samples_ring", 0) or 0)
    ring_pass_rate = float(meta.get("ring_correctness_pass_rate_at_scale", float("nan")))
    ring_speed_med = float(meta.get("ring_speedup_median_at_scale", float("nan")))
    poly_speed_med = float(meta.get("speedup_median_at_scale_polymer", float("nan")))

    if ring_n > 0 and math.isfinite(ring_pass_rate) and ring_pass_rate < correctness_gate_rate:
        return "NOT_VALID_DUE_TO_CORRECTNESS"
    if ring_n < min_scale_samples:
        return "NO_SPEEDUP_YET"
    if math.isfinite(ring_speed_med) and ring_speed_med >= gate_speedup:
        return "PASS_RING_SPEEDUP_AT_SCALE"
    if math.isfinite(ring_speed_med) and ring_speed_med < gate_speedup and math.isfinite(poly_speed_med) and poly_speed_med >= gate_speedup:
        return "FAIL_RING_SPEEDUP_AT_SCALE"
    return "NO_SPEEDUP_YET"


def test_p5_6_ring_speedup_contract_fields_and_verdict_logic(tmp_path: Path) -> None:
    out_dir = tmp_path / "out"
    cfg = P5Config(
        n_atoms_bins=(20, 200),
        samples_per_bin=2,
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
        correctness_gate_rate=1.0,
        min_scale_samples=2,
        speedup_gate_break_even=1.0,
        speedup_gate_strong=2.0,
    )
    zip_path = write_p5_evidence_pack(out_dir=out_dir, cfg=cfg)
    assert zip_path.exists()

    with zipfile.ZipFile(zip_path, "r") as zf:
        meta = json.loads(zf.read("summary_metadata.json").decode("utf-8"))

    for k in [
        "ring_speedup_median_at_scale",
        "ring_eval_ratio_median_at_scale",
        "ring_correctness_pass_rate_at_scale",
        "ring_speedup_verdict_at_scale",
        "ring_speedup_verdict_reason_at_scale",
        "topology_ring_cost_gap_verdict_at_scale",
    ]:
        assert k in meta

    assert meta["ring_speedup_verdict_at_scale"] in {
        "PASS_RING_SPEEDUP_AT_SCALE",
        "FAIL_RING_SPEEDUP_AT_SCALE",
        "NO_SPEEDUP_YET",
        "NOT_VALID_DUE_TO_CORRECTNESS",
    }
    assert str(meta.get("ring_speedup_verdict_reason_at_scale") or "").strip()

    assert meta["ring_speedup_verdict_at_scale"] == _expected_ring_speedup_verdict(meta)


def test_p5_6_ring_speedup_not_valid_if_correctness_fails(tmp_path: Path) -> None:
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
        correctness_gate_rate=1.1,  # >1.0 forces correctness FAIL regardless of pass_rate
        min_scale_samples=1,
        speedup_gate_break_even=1.0,
        speedup_gate_strong=2.0,
    )
    zip_path = write_p5_evidence_pack(out_dir=out_dir, cfg=cfg)
    assert zip_path.exists()

    with zipfile.ZipFile(zip_path, "r") as zf:
        meta = json.loads(zf.read("summary_metadata.json").decode("utf-8"))

    assert meta["integrator_correctness_verdict"] == "FAIL_CORRECTNESS_AT_SCALE"
    assert meta["ring_speedup_verdict_at_scale"] == "NOT_VALID_DUE_TO_CORRECTNESS"
