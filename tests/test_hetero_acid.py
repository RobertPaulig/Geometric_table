from __future__ import annotations

import math
import random

import numpy as np

from analysis.chem.hetero_canonical import canonicalize_hetero_state
from analysis.chem.hetero_exact_small import exact_distribution_for_formula_C3H8O
from analysis.chem.hetero_mcmc import HeteroState, run_hetero_mcmc, step_hetero_mcmc
from analysis.chem.hetero_operator import hetero_energy_from_state

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
