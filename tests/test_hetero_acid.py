from __future__ import annotations

import math
import random
import pytest

import numpy as np

from analysis.chem.hetero_canonical import canonicalize_hetero_state
from analysis.chem.hetero_exact_small import (
    exact_distribution_for_formula_C2H6O,
    exact_distribution_for_formula_C2H7N,
    exact_distribution_for_formula_C3H8O,
)
from analysis.chem.hetero_mcmc import HeteroState, run_hetero_mcmc, step_hetero_mcmc
from analysis.chem.hetero_operator import build_operator_H, hetero_energy_from_state, hetero_fingerprint

VALENCE = {0: 4, 1: 3, 2: 2}
RHO = {0: 0.0, 1: 0.2, 2: 0.5}


def _default_energy(state: HeteroState) -> float:
    return hetero_energy_from_state(
        state.n,
        state.edges,
        state.types,
        rho_by_type=RHO,
        alpha_H=0.5,
        valence_by_type=VALENCE,
    )


def _make_state() -> HeteroState:
    edges = ((0, 1), (1, 2), (1, 3))
    types = (0, 0, 1, 2)
    return HeteroState(n=4, edges=tuple(sorted(edges)), types=types)


def test_colored_canonical_invariant_under_relabel() -> None:
    base_state = _make_state()
    edges = list(base_state.edges)
    types = list(base_state.types)
    n = base_state.n
    perm = list(range(n))
    random.Random(123).shuffle(perm)
    mapping = {old: perm_idx for perm_idx, old in enumerate(perm)}
    edges_perm = [(mapping[u], mapping[v]) for u, v in edges]
    types_perm = [0] * n
    for old, new in mapping.items():
        types_perm[new] = types[old]
    _, _, id_base = canonicalize_hetero_state(n, edges, types)
    _, _, id_perm = canonicalize_hetero_state(n, edges_perm, types_perm)
    assert id_base == id_perm


def test_colored_canonical_two_nodes_order_by_type() -> None:
    edges = ((0, 1),)
    types = (2, 0)
    _, _, state_id = canonicalize_hetero_state(2, edges, types)
    assert state_id == "het:edges=0-1;types=0,2"


def test_leaf_rewire_respects_valence() -> None:
    state = _make_state()
    rng = np.random.default_rng(42)
    for _ in range(200):
        state, _, moved = step_hetero_mcmc(
            state,
            rng=rng,
            p_rewire=1.0,
            valence_by_type=VALENCE,
        )
        if not moved:
            continue
        deg = [0] * state.n
        for u, v in state.edges:
            deg[int(u)] += 1
            deg[int(v)] += 1
        for idx, d in enumerate(deg):
            assert d <= VALENCE[state.types[idx]]


def test_swap_move_respects_valence() -> None:
    state = _make_state()
    rng = np.random.default_rng(7)
    for _ in range(200):
        state, _, moved = step_hetero_mcmc(
            state,
            rng=rng,
            p_rewire=0.0,
            valence_by_type=VALENCE,
        )
        if not moved:
            continue
        deg = [0] * state.n
        for u, v in state.edges:
            deg[int(u)] += 1
            deg[int(v)] += 1
        for idx, d in enumerate(deg):
            assert d <= VALENCE[state.types[idx]]


def test_mcmc_matches_exact_small_C3H8O() -> None:
    exact = exact_distribution_for_formula_C3H8O(beta=1.0, valence_by_type=VALENCE, rho_by_type=RHO)
    init = HeteroState(n=4, edges=((0, 1), (1, 2), (2, 3)), types=(0, 0, 0, 2))
    samples, _ = run_hetero_mcmc(
        init=init,
        steps=6000,
        burnin=800,
        thin=4,
        beta=1.0,
        rng_seed=0,
        energy_fn=_default_energy,
        p_rewire=0.7,
        valence_by_type=VALENCE,
    )
    counts = {}
    for s in samples:
        counts[s["state_id"]] = counts.get(s["state_id"], 0) + 1
    coverage = len(counts) / len(exact)
    assert math.isclose(coverage, 1.0, rel_tol=0, abs_tol=1e-6)
    total = sum(counts.values())
    emp = {k: v / total for k, v in counts.items()}
    kl = 0.0
    for k in exact.keys():
        p = exact[k]
        q = emp.get(k, 1e-9)
        kl += p * math.log(p / q)
    assert kl < 0.02


def test_energy_invariance_under_permutation() -> None:
    rng = np.random.default_rng(0)
    base = _make_state()
    e0 = hetero_energy_from_state(base.n, base.edges, base.types, rho_by_type=RHO, alpha_H=0.5, valence_by_type=VALENCE)
    H0 = build_operator_H(base.n, base.edges, base.types, rho_by_type=RHO, alpha_H=0.5, valence_by_type=VALENCE)
    fp0 = hetero_fingerprint(H0)
    for _ in range(50):
        perm = list(range(base.n))
        rng.shuffle(perm)
        mapping = {old: perm_idx for perm_idx, old in enumerate(perm)}
        edges = []
        for u, v in base.edges:
            a = mapping[u]
            b = mapping[v]
            if a > b:
                a, b = b, a
            edges.append((a, b))
        edges.sort()
        types = tuple(base.types[perm[new]] for new in range(base.n))
        e = hetero_energy_from_state(
            base.n,
            tuple(edges),
            types,
            rho_by_type=RHO,
            alpha_H=0.5,
            valence_by_type=VALENCE,
        )
        assert pytest.approx(e0, rel=0, abs=1e-9) == e
        H = build_operator_H(base.n, tuple(edges), types, rho_by_type=RHO, alpha_H=0.5, valence_by_type=VALENCE)
        fp = hetero_fingerprint(H)
        assert fp.shape == fp0.shape
        assert np.allclose(fp, fp0, atol=1e-9)


def _state_from_id(state_id: str) -> HeteroState:
    parts = state_id.split(";")
    edges_part = parts[0].split("=")[1]
    types_part = parts[1].split("=")[1]
    edges = []
    if edges_part:
        for item in edges_part.split(","):
            if not item:
                continue
            a, b = item.split("-")
            a_i = int(a)
            b_i = int(b)
            if a_i > b_i:
                a_i, b_i = b_i, a_i
            edges.append((a_i, b_i))
    types = tuple(int(t) for t in types_part.split(",")) if types_part else ()
    return HeteroState(n=len(types), edges=tuple(edges), types=types)


def test_state_graph_connected_for_small_formulas() -> None:
    from analysis.chem.hetero_mcmc import _apply_rewire, _apply_swap, _leaf_rewire_candidates, _swap_candidates

    exact_funcs = {
        "C2H6O": exact_distribution_for_formula_C2H6O,
        "C2H7N": exact_distribution_for_formula_C2H7N,
        "C3H8O": exact_distribution_for_formula_C3H8O,
    }
    for name, func in exact_funcs.items():
        exact = func(beta=1.0, valence_by_type=VALENCE, rho_by_type=RHO)
        states = [(sid, _state_from_id(sid)) for sid in exact.keys()]

        def neighbors(state: HeteroState) -> set[str]:
            neigh_ids = set()
            adj = [[] for _ in range(state.n)]
            for u, v in state.edges:
                adj[u].append(v)
                adj[v].append(u)
            for leaf, old_parent, new_parent in _leaf_rewire_candidates(adj, state.types, VALENCE):
                edges_new, _ = _apply_rewire(state.n, state.edges, leaf, old_parent, new_parent)
                _, _, sid = canonicalize_hetero_state(state.n, edges_new, state.types)
                neigh_ids.add(sid)
            for i, j in _swap_candidates(adj, state.types, VALENCE):
                types_new = _apply_swap(state.types, i, j)
                _, _, sid = canonicalize_hetero_state(state.n, state.edges, types_new)
                neigh_ids.add(sid)
            return neigh_ids

        graph = {sid: neighbors(st) for sid, st in states}
        start = states[0][0]
        seen = set([start])
        stack = [start]
        while stack:
            node = stack.pop()
            for nb in graph[node]:
                if nb not in seen:
                    seen.add(nb)
                    stack.append(nb)
        assert seen == set(exact.keys()), f"{name} graph disconnected"
