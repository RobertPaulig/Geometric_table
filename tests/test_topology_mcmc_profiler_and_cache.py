import analysis.chem.topology_mcmc as m


def test_mcmc_cache_misses_not_zero_when_cache_prefilled():
    # Prefill cache for both N=4 topologies to ensure "compute misses" would be zero;
    # we still expect per-chain "first-seen topology" misses to be > 0.
    backend = "fdm"
    cache = {
        (backend, "n_butane"): 0.0,
        (backend, "isobutane"): 0.0,
    }
    _samples, summary = m.run_fixed_n_tree_mcmc(
        n=4,
        steps=200,
        burnin=0,
        thin=1,
        backend=backend,
        lam=1.0,
        temperature_T=1.0,
        seed=0,
        max_valence=4,
        topology_classifier=lambda adj: m.classify_tree_topology_by_deg_sorted(sorted([int(x) for x in adj.sum(axis=0)])),
        energy_cache=cache,
    )
    assert summary.energy_cache_misses > 0


def test_mcmc_profiler_populates_averages():
    _samples, summary = m.run_fixed_n_tree_mcmc(
        n=5,
        steps=200,
        burnin=0,
        thin=1,
        backend="fdm",
        lam=1.0,
        temperature_T=1.0,
        seed=0,
        max_valence=4,
        topology_classifier=lambda adj: m.classify_tree_topology_by_deg_sorted(sorted([int(x) for x in adj.sum(axis=0)])),
        profile_every=1,
    )
    assert summary.t_move_avg > 0.0
    assert summary.t_energy_avg > 0.0
    assert summary.t_canon_avg > 0.0
