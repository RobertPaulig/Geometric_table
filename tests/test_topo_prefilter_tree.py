from __future__ import annotations

import numpy as np
import pytest

from core.complexity import compute_complexity_features_v2
from core.thermo_config import ThermoConfig, override_thermo_config


def _chain_adj(n: int) -> np.ndarray:
    adj = np.zeros((n, n), dtype=float)
    for i in range(n - 1):
        adj[i, i + 1] = 1.0
        adj[i + 1, i] = 1.0
    return adj


def _k4_adj() -> np.ndarray:
    n = 4
    adj = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(i + 1, n):
            adj[i, j] = 1.0
            adj[j, i] = 1.0
    return adj


def test_topo_prefilter_tree_skips_layout_on_chain(monkeypatch):
    adj = _chain_adj(6)

    # Настраиваем thermo так, чтобы entanglement был включён,
    # но prefilter должен вернуть чистый FDM на дереве.
    thermo = ThermoConfig(
        coupling_topo_3d=1.0,
        topo_3d_beta=1.0,
        topo3d_prefilter_tree=True,
    )

    # Monkeypatch: если layout будет вызван, тест должен упасть
    def _boom(*args, **kwargs):
        raise RuntimeError("force_directed_layout_3d should not be called for tree when prefilter is enabled")

    monkeypatch.setattr("core.layout_3d.force_directed_layout_3d", _boom)

    with override_thermo_config(thermo):
        feats = compute_complexity_features_v2(adj, backend="fdm_entanglement")

    # Для дерева с prefilter total_entangled == total_fdm
    # compute_complexity_features_v2 с backend="fdm" даёт базовый FDM
    with override_thermo_config(thermo):
        feats_fdm = compute_complexity_features_v2(adj, backend="fdm")

    assert pytest.approx(feats.total, rel=1e-12, abs=1e-12) == feats_fdm.total


def test_topo_prefilter_tree_does_not_skip_layout_on_k4(monkeypatch):
    adj = _k4_adj()

    thermo = ThermoConfig(
        coupling_topo_3d=1.0,
        topo_3d_beta=1.0,
        topo3d_prefilter_tree=True,
    )

    def _boom(*args, **kwargs):
        raise RuntimeError("force_directed_layout_3d called on non-tree, as expected")

    monkeypatch.setattr("core.layout_3d.force_directed_layout_3d", _boom)

    with override_thermo_config(thermo), pytest.raises(RuntimeError):
        # Для K4 cyclomatic > 0, поэтому prefilter не должен сработать,
        # и layout должен вызываться (что приведёт к нашему RuntimeError).
        compute_complexity_features_v2(adj, backend="fdm_entanglement")

