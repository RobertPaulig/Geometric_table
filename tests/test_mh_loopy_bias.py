from __future__ import annotations

import numpy as np

from core.grower import grow_molecule_loopy
from core.thermo_config import ThermoConfig, override_thermo_config


def _run_loopy_mh(temperature_T: float, seed: int = 111):
    rng = np.random.default_rng(seed)
    thermo = ThermoConfig(
        grower_use_mh=True,
        coupling_delta_G=1.0,
        temperature_T=temperature_T,
    )
    with override_thermo_config(thermo):
        mol = grow_molecule_loopy("C", rng=rng)
    stats = getattr(mol, "mh_stats", {})
    proposals = int(stats.get("proposals", 0))
    accepted = int(stats.get("accepted", 0))
    rejected = int(stats.get("rejected", 0))
    rate = accepted / proposals if proposals > 0 else 1.0
    return proposals, accepted, rejected, rate


def test_mh_loopy_bias_reduces_acceptance_rate_at_low_T():
    proposals_hi, accepted_hi, rejected_hi, rate_hi = _run_loopy_mh(
        temperature_T=1e9,
        seed=111,
    )
    proposals_lo, accepted_lo, rejected_lo, rate_lo = _run_loopy_mh(
        temperature_T=0.1,
        seed=111,
    )

    # При высокой T MH вырождается в always-accept (rejects ~ 0),
    # при низкой T должны появиться отказы и/или меньший acceptance-rate.
    assert rejected_hi == 0
    assert rejected_lo >= 0
    assert rejected_lo > rejected_hi or rate_lo < rate_hi

