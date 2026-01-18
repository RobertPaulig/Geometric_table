from __future__ import annotations

import math
from typing import Tuple

TANIMOTO_EASY_MAX_EXCLUSIVE = 0.50
TANIMOTO_MEDIUM_MAX_EXCLUSIVE = 0.70

AUC_INTERPRETATION_SCHEMA = "auc_interpretation.v1"
AUC_MEDIAN_TANIMOTO_EASY_THRESHOLD = 0.55
AUC_HARD_MIN_PAIRS_DEFAULT = 50
AUC_EASY_MIN_SIGNAL = 0.70
AUC_HARD_MIN_SIGNAL = 0.55


def tanimoto_bin(sim: float) -> str:
    if not math.isfinite(sim):
        return "unknown"
    if sim < TANIMOTO_EASY_MAX_EXCLUSIVE:
        return "easy"
    if sim < TANIMOTO_MEDIUM_MAX_EXCLUSIVE:
        return "medium"
    return "hard"


def auc_pair_contribution(pos_score: float, neg_score: float, *, eps: float = 1e-12) -> float:
    delta = float(pos_score) - float(neg_score)
    if abs(delta) <= float(eps):
        return 0.5
    return 1.0 if delta > 0 else 0.0


def interpret_auc(
    *,
    median_tanimoto: float,
    auc_easy: float | None = None,
    auc_hard: float | None,
    hard_pairs: int,
    hard_pairs_min: int = AUC_HARD_MIN_PAIRS_DEFAULT,
    median_easy_threshold: float = AUC_MEDIAN_TANIMOTO_EASY_THRESHOLD,
    auc_easy_min_signal: float = AUC_EASY_MIN_SIGNAL,
    auc_hard_min_signal: float = AUC_HARD_MIN_SIGNAL,
) -> Tuple[str, str]:
    if not math.isfinite(float(median_tanimoto)):
        return "INCONCLUSIVE_DECOYS_TOO_EASY", "median_tanimoto_nan"
    if float(median_tanimoto) < float(median_easy_threshold):
        return "INCONCLUSIVE_DECOYS_TOO_EASY", "decoys_too_easy"
    if int(hard_pairs) < int(hard_pairs_min):
        return "INCONCLUSIVE_DECOYS_TOO_EASY", f"hard_bin_insufficient_pairs<{int(hard_pairs_min)}"
    if auc_hard is None or not math.isfinite(float(auc_hard)):
        return "INCONCLUSIVE_DECOYS_TOO_EASY", "auc_hard_nan"

    auc_easy_f = float(auc_easy) if auc_easy is not None and math.isfinite(float(auc_easy)) else None

    if float(auc_hard) <= float(auc_hard_min_signal):
        if auc_easy_f is not None and auc_easy_f > float(auc_easy_min_signal):
            return "ILLUSION_CONFIRMED", "auc_easy_high_but_auc_hard_low"
        return "ILLUSION_CONFIRMED", "auc_hard_low"
    return "SUCCESS_SIGNAL_ON_HARD", "auc_hard_signal"

