from __future__ import annotations

from functools import lru_cache
from math import comb
from typing import Dict, Tuple


@lru_cache(maxsize=None)
def null_auc_quantile(m: int, n: int, q: float) -> float:
    """
    Exact q-quantile of null AUC for fixed ranks (Wilcoxon U distribution).

    Model: all binary sequences of length m+n with exactly m positives and n negatives
    are equally likely (equivalent to permuting labels with fixed score ordering).

    For a labeling, AUC = U / (m*n), where U counts (pos,neg) pairs with pos ranked above neg.
    """
    if m <= 0 or n <= 0:
        raise ValueError("m,n must be positive")
    if not (0.0 < q <= 1.0):
        raise ValueError("q must be in (0,1]")

    # dp[(i,j)] -> {U: count} for sequences with i positives and j negatives.
    # Append a positive: U increases by j (it beats all previous negatives).
    dp: Dict[Tuple[int, int], Dict[int, int]] = {(0, 0): {0: 1}}
    for i in range(m + 1):
        for j in range(n + 1):
            if (i, j) == (0, 0):
                continue
            acc: Dict[int, int] = {}
            if i > 0:
                for u, cnt in dp[(i - 1, j)].items():
                    acc[u + j] = acc.get(u + j, 0) + cnt
            if j > 0:
                for u, cnt in dp[(i, j - 1)].items():
                    acc[u] = acc.get(u, 0) + cnt
            dp[(i, j)] = acc

    dist = dp[(m, n)]
    total = comb(m + n, m)
    if sum(dist.values()) != total:
        raise RuntimeError("Internal error: DP mass mismatch")

    target = q * total
    cum = 0
    for u in sorted(dist):
        cum += dist[u]
        if cum >= target:
            return u / (m * n)
    return max(dist) / (m * n)


def main() -> None:
    pairs = [(4, 3), (5, 5), (8, 8)]
    q = 0.95
    print("m,n,q95_null")
    for m, n in pairs:
        print(f"{m},{n},{null_auc_quantile(m, n, q)}")


if __name__ == "__main__":
    main()

