from __future__ import annotations

from typing import Any, List

import numpy as np

from core.thermo_config import ThermoConfig, get_current_thermo_config
from .geom_atoms import get_atom


def _get_thermo(thermo: ThermoConfig | None) -> ThermoConfig:
    return thermo if thermo is not None else get_current_thermo_config()


def sample_child_symbol(
    rng: np.random.Generator,
    candidate_pool: List[str],
    parent_atom: Any,
    thermo: ThermoConfig | None = None,
) -> str:
    """
    Proposal policy for child symbol in grower.

    Policies:
    - uniform (default): exact rng.choice(candidate_pool) for deterministic degeneracy.
    - ctt_biased: weights by ports and softness.
    """
    cfg = _get_thermo(thermo)
    policy = getattr(cfg, "grower_proposal_policy", "uniform")
    beta = float(getattr(cfg, "proposal_beta", 0.0))
    gamma = float(getattr(cfg, "proposal_ports_gamma", 0.0))

    # Strict uniform: preserve RNG path (one rng.choice call).
    if policy == "uniform" or (beta == 0.0 and gamma == 0.0):
        return str(rng.choice(candidate_pool))

    if policy != "ctt_biased":
        # Fallback to uniform for unknown policies
        return str(rng.choice(candidate_pool))

    # ctt_biased: w(child) = (max(1, ports(child)-1)**gamma) * exp(-beta * softness(child))
    weights = []
    for sym in candidate_pool:
        atom = get_atom(sym)
        ports = getattr(atom, "ports", 1)
        ports_term = max(1.0, float(ports - 1))
        if gamma != 0.0:
            ports_weight = ports_term ** gamma
        else:
            ports_weight = 1.0

        softness_val = 0.0
        if hasattr(atom, "effective_softness"):
            try:
                softness_val = float(atom.effective_softness(cfg))
            except Exception:
                softness_val = 0.0

        if beta != 0.0:
            softness_weight = float(np.exp(-beta * softness_val))
        else:
            softness_weight = 1.0

        w = ports_weight * softness_weight
        weights.append(max(w, 1e-12))

    weights_arr = np.asarray(weights, dtype=float)
    weights_arr /= float(weights_arr.sum())

    idx = int(rng.choice(len(candidate_pool), p=weights_arr))
    return str(candidate_pool[idx])

