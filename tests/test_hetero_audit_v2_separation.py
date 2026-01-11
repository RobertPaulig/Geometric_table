import json

from analysis.chem.audit import run_audit


def test_audit_v2_separates_good_from_bad_constant() -> None:
    # n_pos=1 is the hetero2/common case (one original vs many decoys).
    n_decoys = 20
    q = 0.95
    margin = 0.05

    # BAD: all scores equal -> AUC tie-aware should be ~0.5.
    bad_payload = {
        "dataset_id": "unit:bad-constant",
        "items": [{"label": 1, "score": 0.0, "weight": 1.0}, *[{"label": 0, "score": 0.0, "weight": 1.0} for _ in range(n_decoys)]],
    }
    bad = run_audit(bad_payload, seed=0, timestamp="2026-01-11T00:00:00+00:00", cmd_argv=["unit"], neg_control_quantile=q, neg_auc_margin=margin)
    assert bad["schema_version"] == "hetero_audit.v2"

    # GOOD: original above all decoys -> AUC should be 1.0.
    good_payload = {
        "dataset_id": "unit:good-synthetic",
        "items": [{"label": 1, "score": 1.0, "weight": 1.0}, *[{"label": 0, "score": 0.0, "weight": 1.0} for _ in range(n_decoys)]],
    }
    good = run_audit(good_payload, seed=0, timestamp="2026-01-11T00:00:00+00:00", cmd_argv=["unit"], neg_control_quantile=q, neg_auc_margin=margin)
    assert good["schema_version"] == "hetero_audit.v2"

    bad_auc = float(bad["auc_tie_aware"])
    good_auc = float(good["auc_tie_aware"])
    assert good_auc > bad_auc

    bad_slack = float(bad["neg_controls"]["slack"])
    good_slack = float(good["neg_controls"]["slack"])
    assert good_slack > bad_slack

    assert bad["neg_controls"]["verdict"] == "FAIL"
    assert good["neg_controls"]["verdict"] == "PASS"

    # Extra safety: values are JSON-serializable.
    json.dumps({"bad": bad, "good": good}, ensure_ascii=False, sort_keys=True)

