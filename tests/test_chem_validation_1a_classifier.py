from analysis.chem.chem_validation_1a_pentane import classify_pentane_topology


def test_classify_pentane_topology() -> None:
    assert classify_pentane_topology([1, 1, 2, 2, 2]) == "n_pentane"
    assert classify_pentane_topology([1, 1, 1, 2, 3]) == "isopentane"
    assert classify_pentane_topology([1, 1, 1, 1, 4]) == "neopentane"
    assert classify_pentane_topology([1, 1, 1, 1, 1]) == "other"

