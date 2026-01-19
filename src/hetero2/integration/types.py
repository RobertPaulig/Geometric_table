from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True, slots=True)
class IntegrationConfig:
    integrator_mode: Literal["baseline"]
    energy_min: float | None
    energy_max: float | None
    energy_points: int
    eta: float
    integrator_eps: float


@dataclass(frozen=True, slots=True)
class IntegrationBenchmarkRow:
    molecule_id: str
    integrator_mode: str
    curve_id: str
    energy_points: int
    n_function_evals: int
    walltime_ms_total: float
    walltime_ms_per_point: float
    result_checksum: str
    eigenvalues_count: int

