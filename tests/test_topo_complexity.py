from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from core.complexity import compute_complexity_features_v2
from core.entanglement_3d import entanglement_score
from core.layout_3d import force_directed_layout_3d
from core.thermo_config import ThermoConfig, override_thermo_config


def _adj(n: int, edges):
    a = np.zeros((n, n), dtype=float)
    for i, j in edges:
        a[i, j] = 1.0
        a[j, i] = 1.0
    return a


def test_fdm_entanglement_backend_formula_and_ordering():
    n = 4
    chain_edges = [(0, 1), (1, 2), (2, 3)]
    k4_edges = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]

    cfg = replace(ThermoConfig(), coupling_topo_3d=2.0, topo_3d_beta=10.0)

    with override_thermo_config(cfg):
        pos_chain = force_directed_layout_3d(n, chain_edges, seed=42)
        e_chain = float(entanglement_score(pos_chain, chain_edges))

        pos_k4 = force_directed_layout_3d(n, k4_edges, seed=42)
        e_k4 = float(entanglement_score(pos_k4, k4_edges))

        assert e_k4 > e_chain

        adj_chain = _adj(n, chain_edges)
        adj_k4 = _adj(n, k4_edges)

        fdm_chain = compute_complexity_features_v2(adj_chain, backend="fdm").total
        ent_chain = compute_complexity_features_v2(adj_chain, backend="fdm_entanglement").total

        fdm_k4 = compute_complexity_features_v2(adj_k4, backend="fdm").total
        ent_k4 = compute_complexity_features_v2(adj_k4, backend="fdm_entanglement").total

        expected_chain = 1.0 + cfg.coupling_topo_3d * cfg.topo_3d_beta * e_chain
        expected_k4 = 1.0 + cfg.coupling_topo_3d * cfg.topo_3d_beta * e_k4

        assert ent_chain / fdm_chain == pytest.approx(expected_chain, rel=1e-9, abs=1e-12)
        assert ent_k4 / fdm_k4 == pytest.approx(expected_k4, rel=1e-9, abs=1e-12)

        assert ent_k4 > ent_chain

