from hetero2.decoy_realism import interpret_auc


def test_interpret_auc_inconclusive_decoys_too_easy() -> None:
    label, reason = interpret_auc(median_tanimoto=0.54, auc_hard=0.9, hard_pairs=100)
    assert label == "INCONCLUSIVE_DECOYS_TOO_EASY"
    assert reason == "decoys_too_easy"


def test_interpret_auc_illusion_confirmed() -> None:
    label, reason = interpret_auc(median_tanimoto=0.80, auc_easy=0.80, auc_hard=0.55, hard_pairs=100)
    assert label == "ILLUSION_CONFIRMED"
    assert reason in {"auc_hard_low", "auc_easy_high_but_auc_hard_low"}


def test_interpret_auc_ok_signal_on_hard_decoys() -> None:
    label, reason = interpret_auc(median_tanimoto=0.80, auc_hard=0.56, hard_pairs=100)
    assert label == "SUCCESS_SIGNAL_ON_HARD"
    assert reason == "auc_hard_signal"

