import math

import numpy as np

from hetero2.physics_operator import SCF_SCHEMA, compute_operator_payload


def _chain_adjacency(n: int) -> np.ndarray:
    adj = np.zeros((n, n), dtype=float)
    for i in range(int(n) - 1):
        adj[i, i + 1] = 1.0
        adj[i + 1, i] = 1.0
    return adj


def _cycle_adjacency(n: int) -> np.ndarray:
    adj = np.zeros((n, n), dtype=float)
    for i in range(int(n)):
        j = (i + 1) % int(n)
        adj[i, j] = 1.0
        adj[j, i] = 1.0
    return adj


def test_scf_nontriviality_on_asymmetric_graph() -> None:
    adj = _chain_adjacency(4)
    # Same topology, asymmetric types (heteroatom breaks symmetry).
    types = [6, 6, 8, 6]  # C C O C

    op = compute_operator_payload(
        adjacency=adj,
        bonds=None,
        types=types,
        physics_mode="both",
        edge_weight_mode="unweighted",
        potential_mode="both",
        scf_max_iter=50,
        scf_tol=1e-12,
        scf_damping=0.5,
        scf_occ_k=4,
        scf_tau=1.0,
    )

    scf = op.get("scf")
    assert isinstance(scf, dict)
    assert scf["schema_version"] == SCF_SCHEMA

    iters = int(scf.get("scf_iters", 0) or 0)
    assert iters > 1

    trace = scf.get("trace", [])
    assert isinstance(trace, list)
    assert len(trace) == iters

    first = trace[0]
    last = trace[-1]
    assert isinstance(first, dict)
    assert isinstance(last, dict)

    residual_init = float(first.get("residual", first.get("residual_inf", float("nan"))))
    residual_final = float(last.get("residual", last.get("residual_inf", float("nan"))))
    assert math.isfinite(residual_init) and math.isfinite(residual_final)
    assert residual_final < residual_init

    delta_v_vals = []
    for t in trace:
        if not isinstance(t, dict):
            continue
        val = float(t.get("delta_V_inf", t.get("residual_inf", float("nan"))))
        if math.isfinite(val):
            delta_v_vals.append(val)
    assert delta_v_vals and max(delta_v_vals) > 0.0


def test_scf_symmetry_trivial_fixed_point_is_marked() -> None:
    # Symmetric cycle with uniform types can converge in 1 iteration with near-zero update.
    adj = _cycle_adjacency(6)
    types = [6] * 6  # all carbon

    op = compute_operator_payload(
        adjacency=adj,
        bonds=None,
        types=types,
        physics_mode="both",
        edge_weight_mode="unweighted",
        potential_mode="both",
        scf_max_iter=50,
        scf_tol=1e-6,
        scf_damping=0.5,
        scf_occ_k=6,
        scf_tau=1.0,
    )

    scf = op.get("scf")
    assert isinstance(scf, dict)
    assert scf["schema_version"] == SCF_SCHEMA
    assert bool(scf.get("scf_converged", False)) is True
    assert int(scf.get("scf_iters", 0) or 0) == 1

    trace = scf.get("trace", [])
    assert isinstance(trace, list)
    assert len(trace) == 1
    t0 = trace[0]
    assert isinstance(t0, dict)
    assert str(t0.get("stop_reason", "")) == "trivial_fixed_point"

    delta_v = float(t0.get("delta_V_inf", t0.get("residual_inf", float("nan"))))
    assert math.isfinite(delta_v)
    assert abs(delta_v) <= 1e-12

