from analysis.chem.alkane_expected_counts import expected_unique_alkane_tree_topologies


def test_expected_unique_counts_known_values():
    assert expected_unique_alkane_tree_topologies(11) == 159
    assert expected_unique_alkane_tree_topologies(12) == 355
    assert expected_unique_alkane_tree_topologies(13) == 802
    assert expected_unique_alkane_tree_topologies(14) == 1858


def test_expected_unique_counts_monotone():
    vals = [expected_unique_alkane_tree_topologies(n) for n in range(7, 15)]
    assert vals == sorted(vals)

