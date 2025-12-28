import argparse

from analysis.chem.hetero_calibration_loop import _evaluate_trial


def test_calibrator_neg_control_gate_any_triggers() -> None:
    args = argparse.Namespace(
        coverage_min=1.0,
        kl_max=0.05,
        coll_cross_pairs_strict_max=0,
        fp_auc_min=0.85,
        fp_auc_gap_min=0.02,
        min_nontrivial_rows=1,
        lambda_other=0.0,
        lambda_collision=0.0,
        use_neg_controls=True,
        neg_auc_max=0.60,
        neg_control_seed=0,
        neg_control_reps=50,
        neg_control_quantile=0.95,
    )
    rows = [
        {
            "formula": "C4H10O",
            "coverage_unique_eq": 1.0,
            "kl_exact_emp": 0.001,
            "coll_cross_pairs": 0,
            "coll_cross": 0.0,
            "coll_cross_pairs_strict": 0,
            "fp_best_is_trivial": False,
            "fp_best_auc_best": 0.95,
            "fp_best_auc_gap": 0.05,
            "fp_neg_auc_best_perm_labels": 0.75,
            "fp_neg_auc_best_rand_fp": 0.55,
        }
    ]
    status, reason, score, meta, gate_meta = _evaluate_trial(rows, args=args)
    assert status == "fail"
    assert reason == "neg_control"
    assert gate_meta["gate_failed_any"] is True
    assert gate_meta["gate_reason_any"] == "neg_control"
    assert gate_meta["gate_formula_any"] == "C4H10O"
    assert meta["gate_failed_any"] is True
    assert meta["gate_reason_any"] == "neg_control"
    assert meta["gate_formula_any"] == "C4H10O"
    assert meta["max_neg_auc_any"] == 0.75

