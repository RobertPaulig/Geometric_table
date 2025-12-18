from __future__ import annotations

from analysis.chem.exact_trees import enumerate_labeled_trees


def test_enumerate_labeled_trees_counts() -> None:
    assert len(enumerate_labeled_trees(4)) == 16
    assert len(enumerate_labeled_trees(5)) == 125

