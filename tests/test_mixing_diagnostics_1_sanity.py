import numpy as np

from analysis.chem.mixing_diagnostics_1 import compute_mixing_diagnostics


def test_mixing_diagnostics_identical_chains_kl_zeroish():
    seq = ["a", "b", "a", "c"] * 50
    energies = [float(i % 7) for i in range(len(seq))]
    out = compute_mixing_diagnostics(
        n=5,
        steps=1000,
        burnin=0,
        thin=1,
        start_spec="test",
        topology_sequences_by_chain=[seq, list(seq)],
        energy_traces_by_chain=[energies, list(energies)],
    )
    assert out.kl_pairwise_max <= 1e-12
    assert out.kl_pairwise_mean <= 1e-12
    assert np.isfinite(out.kl_split_mean)
    assert np.isfinite(out.rhat_energy) or np.isnan(out.rhat_energy) is False

