from __future__ import annotations

import numpy as np

from core.proposal_policy import sample_child_symbol
from core.thermo_config import ThermoConfig, override_thermo_config
from core.geom_atoms import get_atom


def test_proposal_policy_effect_ports_and_softness_bias():
    rng = np.random.default_rng(123)
    candidate_pool = ["H", "C", "O", "Si"]
    parent_atom = get_atom("C")

    # Uniform baseline
    thermo_uniform = ThermoConfig(
        grower_proposal_policy="uniform",
        proposal_beta=0.0,
        proposal_ports_gamma=0.0,
    )
    with override_thermo_config(thermo_uniform):
        counts_uniform = {sym: 0 for sym in candidate_pool}
        for _ in range(2000):
            sym = sample_child_symbol(rng, candidate_pool, parent_atom)
            counts_uniform[sym] += 1

    rng = np.random.default_rng(123)
    # Biased: поощряем более портовые/жёсткие элементы
    thermo_biased = ThermoConfig(
        grower_proposal_policy="ctt_biased",
        proposal_beta=0.5,
        proposal_ports_gamma=1.0,
    )
    with override_thermo_config(thermo_biased):
        counts_biased = {sym: 0 for sym in candidate_pool}
        for _ in range(2000):
            sym = sample_child_symbol(rng, candidate_pool, parent_atom)
            counts_biased[sym] += 1

    # Грубый smoke: суммарная доля C/Si (многопортовые) должна вырасти
    heavy_uniform = counts_uniform["C"] + counts_uniform["Si"]
    heavy_biased = counts_biased["C"] + counts_biased["Si"]

    assert heavy_biased > heavy_uniform

