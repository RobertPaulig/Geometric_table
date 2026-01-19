from __future__ import annotations

from hetero2.integration.metrics import curve_checksum_sha256
from hetero2.integration.types import IntegrationBenchmarkRow


def build_benchmark_row_baseline_grid(
    *,
    molecule_id: str,
    curve_id: str,
    energy_grid: list[float],
    values: list[float],
    eigenvalues_count: int,
    energy_points: int,
    walltime_ms_total: float,
    integrator_mode: str = "baseline",
) -> IntegrationBenchmarkRow:
    n_points = int(energy_points)
    n_evals = n_points
    per_point = float(walltime_ms_total) / float(n_points) if n_points > 0 else float("nan")
    checksum = curve_checksum_sha256(energy_grid=energy_grid, values=values)
    return IntegrationBenchmarkRow(
        molecule_id=str(molecule_id),
        integrator_mode=str(integrator_mode),
        curve_id=str(curve_id),
        energy_points=n_points,
        n_function_evals=int(n_evals),
        walltime_ms_total=float(walltime_ms_total),
        walltime_ms_per_point=float(per_point),
        result_checksum=str(checksum),
        eigenvalues_count=int(eigenvalues_count),
    )

