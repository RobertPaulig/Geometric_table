from analysis.chem.alkane_expected_counts import expected_unique_alkane_tree_topologies


def test_expected_unique_counts_c15_c16():
    assert expected_unique_alkane_tree_topologies(15) == 4347
    assert expected_unique_alkane_tree_topologies(16) == 10359

