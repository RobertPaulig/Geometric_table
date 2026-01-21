import csv
import io
import json
import zipfile
from pathlib import Path

from hetero2.scale_proof import P5Config, generate_polymer_scale_fixtures, generate_ring_scale_fixtures, write_p5_evidence_pack


def test_p5_fixtures_deterministic_and_has_hetero_atoms() -> None:
    fixtures1 = generate_polymer_scale_fixtures(n_atoms_bins=[20, 50], samples_per_bin=3, seed=0)
    fixtures2 = generate_polymer_scale_fixtures(n_atoms_bins=[20, 50], samples_per_bin=3, seed=0)
    assert fixtures1 == fixtures2

    assert len(fixtures1) == 2 * 3
    for fx in fixtures1:
        assert fx.n_atoms == len(fx.types_z)
        assert fx.n_atoms in {20, 50}
        assert fx.n_hetero > 0
        assert fx.types_z[0] == 6
        assert fx.types_z[-1] == 6


def test_p5_ring_fixtures_deterministic_and_has_hetero_atoms() -> None:
    fixtures1 = generate_ring_scale_fixtures(n_atoms_bins=[20, 50], samples_per_bin=3, seed=0)
    fixtures2 = generate_ring_scale_fixtures(n_atoms_bins=[20, 50], samples_per_bin=3, seed=0)
    assert fixtures1 == fixtures2

    assert len(fixtures1) == 2 * 3
    for fx in fixtures1:
        assert fx.n_atoms == len(fx.types_z)
        assert fx.n_atoms in {20, 50}
        assert fx.n_hetero > 0
        assert fx.types_z[0] == 6
        assert fx.types_z[-1] == 6
        assert any(z == 7 for z in fx.types_z)
        assert any(z == 8 for z in fx.types_z)


def test_p5_evidence_pack_contains_required_files_and_metadata(tmp_path: Path) -> None:
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
        names = set(zf.namelist())
        for required in [
            "fixtures_polymer_scale.csv",
            "fixtures_ring_scale.csv",
            "speedup_vs_n.csv",
            "speedup_vs_n_by_family.csv",
            "speedup_vs_n.md",
            "timing_breakdown.csv",
            "timing_breakdown_by_family.csv",
            "summary.csv",
            "summary_metadata.json",
            "metrics.json",
            "index.md",
            "manifest.json",
            "checksums.sha256",
        ]:
            assert required in names

        meta = json.loads(zf.read("summary_metadata.json").decode("utf-8"))
        assert meta["schema_version"] == "hetero2_scale_speedup_metadata.v1"
        assert meta["law_ref"]["contract_path"] == "docs/contracts/INTEGRATION_SCALE_CONTRACT.md"
        assert meta["law_ref"]["contract_version"] == "p5.1.v1"
        assert meta["curve_id"] == "dos_H"
        assert meta["integrator_mode"] == "both"
        assert meta["scale_n_atoms_min"] == 20
        assert meta["scale_n_atoms_max"] == 200
        assert meta["scale_gate_n_min"] == 200
        assert meta["potential_unit_model"] == "dimensionless"
        assert meta["scale_speedup_verdict"] in {"PASS_BREAK_EVEN", "PASS_SPEEDUP", "FAIL_SPEEDUP_AT_SCALE", "FAIL_CORRECTNESS_AT_SCALE"}

        assert meta["cost_bottleneck_verdict_at_scale"] in {
            "BOTTLENECK_IS_DOS_LDOS",
            "BOTTLENECK_IS_INTEGRATOR",
            "BOTTLENECK_IS_IO",
            "MIXED",
        }
        assert meta["cost_integration_logic_opt_verdict_at_scale"] in {"PASS", "FAIL", "INCONCLUSIVE"}
        for k in [
            "cost_median_integration_logic_ms_at_scale_before",
            "cost_median_integration_logic_ms_at_scale_after",
            "cost_integration_logic_speedup_at_scale",
        ]:
            assert k in meta

        assert meta["topology_families"] == ["polymer", "ring"]
        assert meta["topology_gate_n_min"] == 200
        assert meta["speedup_verdict_at_scale_polymer"] in {
            "PASS_SPEEDUP_AT_SCALE",
            "FAIL_SPEEDUP_AT_SCALE",
            "INCONCLUSIVE_NOT_ENOUGH_SCALE_SAMPLES",
            "FAIL_CORRECTNESS_AT_SCALE",
            "NOT_VALID_DUE_TO_CORRECTNESS",
        }
        assert meta["speedup_verdict_at_scale_ring"] in {
            "PASS_SPEEDUP_AT_SCALE",
            "FAIL_SPEEDUP_AT_SCALE",
            "INCONCLUSIVE_NOT_ENOUGH_SCALE_SAMPLES",
            "FAIL_CORRECTNESS_AT_SCALE",
            "NOT_VALID_DUE_TO_CORRECTNESS",
        }
        assert meta["topology_hardness_verdict"] in {
            "NOT_VALID_DUE_TO_CORRECTNESS",
            "ILLUSION_CONFIRMED_TOPOLOGY_DEPENDENT",
            "SUCCESS_TOPOLOGY_ROBUST",
            "NO_SPEEDUP_YET",
        }
        assert meta["topology_ring_cost_gap_verdict_at_scale"] in {
            "INCONCLUSIVE_NOT_ENOUGH_SCALE_SAMPLES",
            "RING_NOT_SLOWER_THAN_POLYMER",
            "RING_SLOWER_DUE_TO_BUILD_OPERATOR",
            "RING_SLOWER_DUE_TO_DOS_LDOS_EVAL",
            "RING_SLOWER_DUE_TO_INTEGRATION_LOGIC",
            "RING_SLOWER_DUE_TO_IO",
            "RING_SLOWER_MIXED",
        }
        assert str(meta.get("topology_ring_cost_gap_reason_at_scale") or "")
        for k in [
            "cost_median_build_operator_ms_at_scale_polymer",
            "cost_median_dos_ldos_eval_ms_at_scale_polymer",
            "cost_median_integration_logic_ms_at_scale_polymer",
            "cost_median_total_ms_at_scale_polymer_estimate",
            "cost_median_build_operator_ms_at_scale_ring",
            "cost_median_dos_ldos_eval_ms_at_scale_ring",
            "cost_median_integration_logic_ms_at_scale_ring",
            "cost_median_total_ms_at_scale_ring_estimate",
        ]:
            assert k in meta
        for k in [
            "cost_ratio_ring_vs_polymer_build_operator_ms_at_scale",
            "cost_ratio_ring_vs_polymer_dos_ldos_eval_ms_at_scale",
            "cost_ratio_ring_vs_polymer_integration_logic_ms_at_scale",
            "cost_ratio_ring_vs_polymer_total_ms_at_scale_estimate",
        ]:
            assert k in meta

        by_family_text = zf.read("speedup_vs_n_by_family.csv").decode("utf-8")
        r_family = csv.DictReader(io.StringIO(by_family_text))
        rows_family = list(r_family)
        families = {str(row.get("family") or "") for row in rows_family}
        assert "polymer" in families
        assert "ring" in families

        timing_text = zf.read("timing_breakdown.csv").decode("utf-8")
        r = csv.DictReader(io.StringIO(timing_text))
        assert r.fieldnames is not None
        for col in ["build_operator_ms", "dos_ldos_eval_ms", "integration_logic_ms", "io_ms", "total_ms"]:
            assert col in r.fieldnames

        rows = list(r)
        assert any(row.get("row_kind") == "sample" for row in rows)
        assert any(row.get("row_kind") == "bin" for row in rows)

        timing_by_family_text = zf.read("timing_breakdown_by_family.csv").decode("utf-8")
        r_timing_family = csv.DictReader(io.StringIO(timing_by_family_text))
        assert r_timing_family.fieldnames is not None
        for col in [
            "family",
            "n_atoms",
            "n_samples",
            "median_build_operator_ms",
            "median_dos_ldos_eval_ms",
            "median_integration_logic_ms",
            "median_io_ms",
            "median_total_ms",
        ]:
            assert col in r_timing_family.fieldnames
        timing_rows_family = list(r_timing_family)
        timing_families = {str(row.get("family") or "") for row in timing_rows_family}
        assert "polymer" in timing_families
        assert "ring" in timing_families


def test_p5_correctness_passes_at_scale_for_eta_0_2(tmp_path: Path) -> None:
    out_dir = tmp_path / "out"
    cfg = P5Config(
        n_atoms_bins=(20, 200),
        samples_per_bin=1,
        seed=0,
        curve_id="dos_H",
        energy_points=32,
        dos_eta=0.2,
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
        min_scale_samples=1,
        speedup_gate_break_even=1.0,
        speedup_gate_strong=2.0,
    )
    zip_path = write_p5_evidence_pack(out_dir=out_dir, cfg=cfg)
    assert zip_path.exists()

    with zipfile.ZipFile(zip_path, "r") as zf:
        meta = json.loads(zf.read("summary_metadata.json").decode("utf-8"))
        assert meta["integrator_correctness_verdict"] == "PASS_CORRECTNESS_AT_SCALE"

        assert meta["integrator_correctness_verdict"] == "PASS_CORRECTNESS_AT_SCALE"


def test_p5_4_topology_hardness_not_valid_if_correctness_fails(tmp_path: Path) -> None:
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
        correctness_gate_rate=1.1,  # >1.0 forces FAIL regardless of pass_rate
        min_scale_samples=1,
        speedup_gate_break_even=1.0,
        speedup_gate_strong=2.0,
    )
    zip_path = write_p5_evidence_pack(out_dir=out_dir, cfg=cfg)
    assert zip_path.exists()

    with zipfile.ZipFile(zip_path, "r") as zf:
        meta = json.loads(zf.read("summary_metadata.json").decode("utf-8"))
        assert meta["integrator_correctness_verdict"] == "FAIL_CORRECTNESS_AT_SCALE"
        assert meta["topology_hardness_verdict"] == "NOT_VALID_DUE_TO_CORRECTNESS"

