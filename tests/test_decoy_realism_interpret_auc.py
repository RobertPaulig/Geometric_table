from hetero2.decoy_realism import interpret_auc


def test_interpret_auc_inconclusive_decoys_too_easy() -> None:
    label, reason = interpret_auc(median_tanimoto=0.54, auc_hard=0.9, hard_pairs=100)
    assert label == "INCONCLUSIVE"
    assert reason == "decoys_too_easy"


def test_interpret_auc_illusion_confirmed() -> None:
    label, reason = interpret_auc(median_tanimoto=0.80, auc_hard=0.55, hard_pairs=100)
    assert label == "AUC_ILLUSION_CONFIRMED"
    assert reason == "auc_hard_low"


def test_interpret_auc_ok_signal_on_hard_decoys() -> None:
    label, reason = interpret_auc(median_tanimoto=0.80, auc_hard=0.56, hard_pairs=100)
    assert label == "OK"
    assert reason == "signal_on_hard_decoys"

