import numpy as np

from analysis.chem.topology_mcmc import edges_to_adj
from core.complexity import compute_complexity_features_v2


def _prufer_edges(n: int, rng: np.random.Generator):
    if n <= 1:
        return []
    if n == 2:
        return [(0, 1)]
    seq = rng.integers(0, n, size=n - 2, dtype=int).tolist()
    deg = [1] * n
    for x in seq:
        deg[int(x)] += 1
    edges = []
    for x in seq:
        leaf = next(i for i in range(n) if deg[i] == 1)
        edges.append((int(leaf), int(x)))
        deg[leaf] -= 1
        deg[int(x)] -= 1
    u, v = [i for i in range(n) if deg[i] == 1]
    edges.append((int(u), int(v)))
    return edges


def test_energy_label_invariance_random_trees_n12():
    rng = np.random.default_rng(0)
    n = 12
    for _ in range(10):
        edges = _prufer_edges(n, rng)
        adj = edges_to_adj(n, edges)
        vals = []
        for _ in range(200):
            perm = rng.permutation(n)
            adjp = adj[np.ix_(perm, perm)]
            feats = compute_complexity_features_v2(adjp, backend="fdm")
            vals.append(float(feats.total))
        assert float(np.std(np.asarray(vals, dtype=float))) <= 1e-6

