from __future__ import annotations

from analysis.nuclear.tune_metrics import TARGET_MAGIC, cost_magic_l2


def test_cost_magic_l2_zero_for_perfect_match() -> None:
    toy = list(int(x) for x in TARGET_MAGIC[:4])
    assert cost_magic_l2(toy, TARGET_MAGIC, n_compare=4) == 0.0


def test_cost_magic_l2_positive_for_shifted() -> None:
    toy = [n + 2 for n in TARGET_MAGIC[:4]]
    c = cost_magic_l2(toy, TARGET_MAGIC, n_compare=4)
    assert c > 0.0
