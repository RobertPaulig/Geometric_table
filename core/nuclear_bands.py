from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List

from core.nuclear_island import nuclear_functional


@dataclass
class IsotopeBandPoint:
    Z: int
    N: int
    A: int
    F: float


def scan_isotope_band_for_Z(Z: int, N_min: int, N_max: int) -> List[IsotopeBandPoint]:
    points: List[IsotopeBandPoint] = []
    for N in range(N_min, N_max + 1):
        A = Z + N
        F = nuclear_functional(Z, N)
        points.append(IsotopeBandPoint(Z=Z, N=N, A=A, F=F))
    return points


def scan_isotope_bands(
    Z_values: Iterable[int],
    N_min: int,
    N_max: int,
) -> List[IsotopeBandPoint]:
    all_points: List[IsotopeBandPoint] = []
    for Z in Z_values:
        all_points.extend(scan_isotope_band_for_Z(Z, N_min, N_max))
    return all_points

