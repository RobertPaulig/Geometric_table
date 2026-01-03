from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Iterable, List

import numpy as np

from hetero2.chemgraph import ChemGraph


@dataclass
class RayAuditor:
    size: int
    num_rays: int

    def __post_init__(self) -> None:
        self.size = int(self.size)
        self.num_rays = int(self.num_rays)
        self.divisor_sum_profile: Dict[int, int] = {}
        self._build_profile()

    def _build_profile(self) -> None:
        for n in range(1, self.size + 1):
            s = 0
            for d in range(1, min(n, self.num_rays) + 1):
                if n % d == 0:
                    s += n // d
            self.divisor_sum_profile[n] = s

    def sigma(self, n: int) -> int:
        if n < 1 or n > self.size:
            raise ValueError("n out of range")
        return self.divisor_sum_profile[n]

    def check_calibration(self) -> int:
        limit = min(self.size, 100)
        return sum(self.divisor_sum_profile[i] for i in range(1, limit + 1))


def phi_from_eigs(eigs: Iterable[float], scale: int = 100) -> float:
    vals = [float(x) for x in eigs]
    if not vals:
        return float("nan")
    # Simple, deterministic aggregation: scaled L1 norm of eigenvalues.
    acc = 0.0
    for x in vals:
        acc += abs(x) * float(scale)
    return acc


def phi_from_smiles(smiles: str, scale: int = 100) -> float:
    cg = ChemGraph(smiles)
    lap = cg.laplacian()
    eigs = np.linalg.eigvalsh(lap)
    return phi_from_eigs(eigs, scale=scale)
