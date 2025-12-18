from __future__ import annotations

import numpy as np

from core.complexity import compute_complexity_features_v2
from core.thermo_config import override_thermo_config, ThermoConfig


def _permute_adj(adj: np.ndarray, perm: np.ndarray) -> np.ndarray:
    return adj[np.ix_(perm, perm)]


def _make_adj_n5_path() -> np.ndarray:
    adj = np.zeros((5, 5), dtype=float)
    for a, b in ((0, 1), (1, 2), (2, 3), (3, 4)):
        adj[a, b] = adj[b, a] = 1.0
    return adj


def _make_adj_n5_iso() -> np.ndarray:
    adj = np.zeros((5, 5), dtype=float)
    for a, b in ((1, 0), (1, 2), (1, 4), (2, 3)):
        adj[a, b] = adj[b, a] = 1.0
    return adj


def _make_adj_n5_star() -> np.ndarray:
    adj = np.zeros((5, 5), dtype=float)
    for j in (1, 2, 3, 4):
        adj[0, j] = adj[j, 0] = 1.0
    return adj


def test_energy_label_invariance_mode_a_tree_only() -> None:
    thermo = ThermoConfig(coupling_topo_3d=0.0, topo3d_prefilter_tree=True, topo3d_prefilter_min_n=10)
    rng = np.random.default_rng(0)

    for adj in (_make_adj_n5_path(), _make_adj_n5_iso(), _make_adj_n5_star()):
        values = []
        with override_thermo_config(thermo):
            for _ in range(200):
                perm = rng.permutation(adj.shape[0])
                adj_p = _permute_adj(adj, perm)
                feats = compute_complexity_features_v2(adj_p, backend="fdm")
                values.append(float(feats.total))
        values = np.asarray(values, dtype=float)
        assert float(np.std(values)) <= 1e-6

