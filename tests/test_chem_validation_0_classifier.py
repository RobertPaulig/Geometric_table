from analysis.chem.chem_validation_0_butane import (
    classify_butane_topology,
)


def test_classify_butane_topology_n_butane():
    assert classify_butane_topology([1, 1, 2, 2]) == "n_butane"


def test_classify_butane_topology_isobutane():
    assert classify_butane_topology([1, 1, 1, 3]) == "isobutane"


def test_classify_butane_topology_other():
    assert classify_butane_topology([1, 2, 3, 4]) == "other"

