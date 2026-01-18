from __future__ import annotations

import math
from typing import Tuple

TANIMOTO_EASY_MAX_EXCLUSIVE = 0.50
TANIMOTO_MEDIUM_MAX_EXCLUSIVE = 0.70

AUC_INTERPRETATION_SCHEMA = "auc_interpretation.v1"
AUC_MEDIAN_TANIMOTO_EASY_THRESHOLD = 0.55
AUC_HARD_MIN_PAIRS_DEFAULT = 50
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
    auc_hard: float | None,
    hard_pairs: int,
    hard_pairs_min: int = AUC_HARD_MIN_PAIRS_DEFAULT,
    median_easy_threshold: float = AUC_MEDIAN_TANIMOTO_EASY_THRESHOLD,
    auc_hard_min_signal: float = AUC_HARD_MIN_SIGNAL,
) -> Tuple[str, str]:
    if not math.isfinite(float(median_tanimoto)):
        return "INCONCLUSIVE", "median_tanimoto_nan"
    if float(median_tanimoto) < float(median_easy_threshold):
        return "INCONCLUSIVE", "decoys_too_easy"
    if int(hard_pairs) < int(hard_pairs_min):
        return "INCONCLUSIVE", f"hard_bin_insufficient_pairs<{int(hard_pairs_min)}"
    if auc_hard is None or not math.isfinite(float(auc_hard)):
        return "INCONCLUSIVE", "auc_hard_nan"
    if float(auc_hard) <= float(auc_hard_min_signal):
        return "AUC_ILLUSION_CONFIRMED", "auc_hard_low"
    return "OK", "signal_on_hard_decoys"

