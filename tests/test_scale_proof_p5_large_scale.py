import csv
import io
import json
import zipfile
from pathlib import Path

from hetero2.scale_proof import P5Config, generate_polymer_scale_fixtures, write_p5_evidence_pack


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
            "speedup_vs_n.csv",
            "speedup_vs_n.md",
            "timing_breakdown.csv",
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

        timing_text = zf.read("timing_breakdown.csv").decode("utf-8")
        r = csv.DictReader(io.StringIO(timing_text))
        assert r.fieldnames is not None
        for col in ["build_operator_ms", "dos_ldos_eval_ms", "integration_logic_ms", "io_ms", "total_ms"]:
            assert col in r.fieldnames

        rows = list(r)
        assert any(row.get("row_kind") == "sample" for row in rows)
        assert any(row.get("row_kind") == "bin" for row in rows)


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

