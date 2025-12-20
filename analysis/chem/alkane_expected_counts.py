from __future__ import annotations

from typing import Dict


# Tree-only alkane unlabeled topology counts under max valence 4.
# These are the standard constitutional isomer counts for alkanes C_N H_(2N+2).
ALKANE_TREE_TOPOLOGY_COUNTS: Dict[int, int] = {
    4: 2,
    5: 3,
    6: 5,
    7: 9,
    8: 18,
    9: 35,
    10: 75,
    11: 159,
    12: 355,
    13: 802,
    14: 1858,
}


def expected_unique_alkane_tree_topologies(n: int) -> int:
    n = int(n)
    if n not in ALKANE_TREE_TOPOLOGY_COUNTS:
        raise KeyError(f"Expected alkane topology count is not defined for N={n}")
    return int(ALKANE_TREE_TOPOLOGY_COUNTS[n])

