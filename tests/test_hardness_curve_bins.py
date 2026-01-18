from hetero2.decoy_realism import (
    AUC_HARD_MIN_PAIRS_DEFAULT,
    TANIMOTO_EASY_MAX_EXCLUSIVE,
    TANIMOTO_MEDIUM_MAX_EXCLUSIVE,
    interpret_auc,
    tanimoto_bin,
)


def test_tanimoto_bin_thresholds() -> None:
    assert tanimoto_bin(0.0) == "easy"
    assert tanimoto_bin(TANIMOTO_EASY_MAX_EXCLUSIVE - 1e-9) == "easy"
    assert tanimoto_bin(TANIMOTO_EASY_MAX_EXCLUSIVE) == "medium"
    assert tanimoto_bin(TANIMOTO_MEDIUM_MAX_EXCLUSIVE - 1e-9) == "medium"
    assert tanimoto_bin(TANIMOTO_MEDIUM_MAX_EXCLUSIVE) == "hard"


def test_auc_interpretation_inconclusive_if_not_enough_hard_pairs() -> None:
    label, reason = interpret_auc(median_tanimoto=0.80, auc_easy=0.90, auc_hard=0.90, hard_pairs=AUC_HARD_MIN_PAIRS_DEFAULT - 1)
    assert label == "INCONCLUSIVE_DECOYS_TOO_EASY"
    assert reason.startswith("hard_bin_insufficient_pairs<")


def test_auc_interpretation_illusion_confirmed_if_auc_hard_low() -> None:
    label, reason = interpret_auc(median_tanimoto=0.80, auc_easy=0.90, auc_hard=0.55, hard_pairs=AUC_HARD_MIN_PAIRS_DEFAULT)
    assert label == "ILLUSION_CONFIRMED"
    assert reason in {"auc_hard_low", "auc_easy_high_but_auc_hard_low"}


def test_auc_interpretation_success_signal_on_hard() -> None:
    label, reason = interpret_auc(median_tanimoto=0.80, auc_easy=0.60, auc_hard=0.56, hard_pairs=AUC_HARD_MIN_PAIRS_DEFAULT)
    assert label == "SUCCESS_SIGNAL_ON_HARD"
    assert reason == "auc_hard_signal"

