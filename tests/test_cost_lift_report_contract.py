import csv
import math
from pathlib import Path

import pytest

from scripts.cost_lift import generate_cost_lift_report


def test_cost_lift_report_contract(tmp_path: Path) -> None:
    summary_csv = tmp_path / "summary.csv"
    summary_rows = [
        # PASS verdicts (slack >= 0), but only some are true PASS in expensive truth.
        {"id": "m1", "status": "OK", "verdict": "PASS", "gate": "0.6", "slack": "0.4"},
        {"id": "m2", "status": "OK", "verdict": "PASS", "gate": "0.6", "slack": "0.3"},
        {"id": "m3", "status": "OK", "verdict": "PASS", "gate": "0.6", "slack": "0.05"},
        # High AUC but FAIL due to high gate -> should be excluded by verdict filter.
        {"id": "m4", "status": "OK", "verdict": "FAIL", "gate": "0.95", "slack": "-0.05"},
        {"id": "m5", "status": "OK", "verdict": "FAIL", "gate": "0.95", "slack": "-0.45"},
    ]
    with summary_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    truth_csv = tmp_path / "truth.csv"
    truth_rows = [
        {"molecule_id": "m1", "expensive_label": "PASS", "truth_source": "proxy_rule_v1", "truth_version": "customer_truth.v1"},
        {"molecule_id": "m2", "expensive_label": "PASS", "truth_source": "proxy_rule_v1", "truth_version": "customer_truth.v1"},
        {"molecule_id": "m3", "expensive_label": "PASS", "truth_source": "proxy_rule_v1", "truth_version": "customer_truth.v1"},
        {"molecule_id": "m4", "expensive_label": "FAIL", "truth_source": "proxy_rule_v1", "truth_version": "customer_truth.v1"},
        {"molecule_id": "m5", "expensive_label": "FAIL", "truth_source": "proxy_rule_v1", "truth_version": "customer_truth.v1"},
    ]
    with truth_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(truth_rows[0].keys()))
        writer.writeheader()
        writer.writerows(truth_rows)

    report = generate_cost_lift_report(
        summary_csv=summary_csv,
        truth_csv=truth_csv,
        k=3,
        seed=0,
        skip_policy="unknown_bucket",
        n_bootstrap=100,
    )

    assert report["report_schema"] == "cost_lift.v1"
    assert report["truth_schema"] == "customer_truth.v1"
    assert report["K_requested"] == 3
    assert report["K_effective"] == 3

    for key in ["N_total", "N_ok", "N_skip", "N_error", "N_with_truth", "truth_coverage_rate", "unknown_bucket_rate"]:
        assert key in report

    eligibility = report["eligibility"]
    for key in [
        "rows_total",
        "rows_ok",
        "rows_verdict_pass_fail",
        "rows_truth_known",
        "rows_scores_present",
        "rows_eligible_for_cost_lift",
        "K_effective_reason_top",
    ]:
        assert key in eligibility

    assert eligibility["rows_total"] == report["N_total"]
    assert eligibility["rows_ok"] == report["N_ok"]
    assert eligibility["rows_truth_known"] == report["N_with_truth"]
    assert eligibility["rows_verdict_pass_fail"] == eligibility["rows_eligible_for_cost_lift"]
    assert 0 <= eligibility["rows_eligible_for_cost_lift"] <= eligibility["rows_truth_known"] <= eligibility["rows_ok"] <= eligibility["rows_total"]
    assert 0 <= eligibility["rows_scores_present"] <= eligibility["rows_total"]
    assert isinstance(eligibility["K_effective_reason_top"], list)

    methods = report["methods"]
    for name in ["baseline_random", "baseline_score_only_topk", "filtered_score_plus_audit_topk"]:
        assert name in methods
        for key in ["k_effective", "hits", "hit_rate", "ci_low", "ci_high"]:
            assert key in methods[name]
        assert methods[name]["k_effective"] > 0
        assert 0.0 <= methods[name]["hit_rate"] <= 1.0

    uplift_audit_vs_score = float(report["uplift_score_plus_audit_vs_score_only"])
    uplift_audit_vs_random = float(report["uplift_score_plus_audit_vs_random"])
    assert math.isfinite(uplift_audit_vs_score)
    assert math.isfinite(uplift_audit_vs_random)


def test_cost_lift_report_fails_without_truth(tmp_path: Path) -> None:
    summary_csv = tmp_path / "summary.csv"
    summary_rows = [{"id": "m1", "status": "OK", "verdict": "PASS", "gate": "0.6", "slack": "0.4"}]
    with summary_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    truth_csv = tmp_path / "truth.csv"
    truth_rows = [{"molecule_id": "m1", "expensive_label": "PASS", "truth_source": "proxy_rule_v1", "truth_version": "wrong.v0"}]
    with truth_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(truth_rows[0].keys()))
        writer.writeheader()
        writer.writerows(truth_rows)

    with pytest.raises(ValueError):
        generate_cost_lift_report(summary_csv=summary_csv, truth_csv=truth_csv, k=1, seed=0, skip_policy="unknown_bucket", n_bootstrap=10)


def test_cost_lift_report_ignores_ok_rows_with_verdict_skip(tmp_path: Path) -> None:
    summary_csv = tmp_path / "summary.csv"
    summary_rows = [
        {"id": "m1", "status": "OK", "verdict": "SKIP", "gate": "", "slack": ""},
        {"id": "m2", "status": "OK", "verdict": "PASS", "gate": "0.6", "slack": "0.4"},
    ]
    with summary_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    truth_csv = tmp_path / "truth.csv"
    truth_rows = [
        {"molecule_id": "m1", "expensive_label": "FAIL", "truth_source": "proxy_rule_v1", "truth_version": "customer_truth.v1"},
        {"molecule_id": "m2", "expensive_label": "PASS", "truth_source": "proxy_rule_v1", "truth_version": "customer_truth.v1"},
    ]
    with truth_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(truth_rows[0].keys()))
        writer.writeheader()
        writer.writerows(truth_rows)

    report = generate_cost_lift_report(
        summary_csv=summary_csv,
        truth_csv=truth_csv,
        k=1,
        seed=0,
        skip_policy="unknown_bucket",
        n_bootstrap=50,
    )

    assert report["N_ok"] == 2
    assert report["N_with_truth"] == 2
    assert report["K_effective"] == 1

    eligibility = report["eligibility"]
    assert eligibility["rows_total"] == 2
    assert eligibility["rows_ok"] == 2
    assert eligibility["rows_truth_known"] == 2
    assert eligibility["rows_verdict_pass_fail"] == 1
    assert eligibility["rows_eligible_for_cost_lift"] == 1

    top = eligibility["K_effective_reason_top"]
    assert top and top[0]["reason"] == "verdict_not_pass_fail"
    assert int(top[0]["count"]) == 1

