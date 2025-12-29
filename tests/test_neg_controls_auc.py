import numpy as np

from analysis.chem.hetero_score_utils import _roc_auc
from analysis.chem.neg_control_null_auc import null_auc_quantile


def _roc_auc_o2(scores, labels, weights) -> float:
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=int)
    weights = np.asarray(weights, dtype=float)
    pos_idx = np.where(labels == 1)[0]
    neg_idx = np.where(labels == 0)[0]
    w_pos = float(np.sum(weights[pos_idx]))
    w_neg = float(np.sum(weights[neg_idx]))
    if w_pos == 0 or w_neg == 0:
        return float("nan")
    s = 0.0
    for i in pos_idx:
        for j in neg_idx:
            if scores[i] > scores[j]:
                coef = 1.0
            elif scores[i] == scores[j]:
                coef = 0.5
            else:
                coef = 0.0
            s += float(weights[i]) * float(weights[j]) * coef
    return float(s / (w_pos * w_neg))


def test_roc_auc_tie_invariant_within_blocks():
    scores = np.array([0.0, 0.0, 1.0, 1.0, 1.0], dtype=float)
    labels = np.array([1, 0, 1, 0, 0], dtype=int)
    weights = np.array([1.0, 2.0, 1.5, 3.0, 0.5], dtype=float)
    base = _roc_auc(scores, labels, weights)

    # Permute only within equal-score blocks: indices [0,1] and [2,3,4]
    perm = np.array([1, 0, 4, 2, 3], dtype=int)
    got = _roc_auc(scores[perm], labels[perm], weights[perm])
    assert got == base


def test_roc_auc_matches_bruteforce_formula():
    rng = np.random.default_rng(0)
    for _ in range(50):
        n = 10
        scores = rng.integers(0, 4, size=n).astype(float)  # ties likely
        labels = np.array([1] * (n // 2) + [0] * (n - n // 2), dtype=int)
        rng.shuffle(labels)
        weights = rng.random(size=n).astype(float) + 0.1
        a = _roc_auc(scores, labels, weights)
        b = _roc_auc_o2(scores, labels, weights)
        assert np.isfinite(a) and np.isfinite(b)
        assert abs(a - b) < 1e-12


def test_null_auc_quantile_examples():
    assert null_auc_quantile(4, 3, 0.95) == 11 / 12
    assert null_auc_quantile(5, 5, 0.95) == 0.8
    assert null_auc_quantile(8, 8, 0.95) == 0.75

