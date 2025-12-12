from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Tuple

from core.nuclear_island import nuclear_functional


@dataclass
class IsotopeBandPoint:
    Z: int
    N: int
    A: int
    F: float


def make_default_corridor(Z: int, factor: float = 1.7, min_width: int = 1) -> tuple[int, int]:
    """
    Стандартный коридор по N для поиска стабильного изотопа.

    Z        — протонное число.
    factor   — множитель для верхней границы (по умолчанию 1.7).
    min_width — минимальная ширина диапазона по N.
    """
    N_min = max(Z, 1)
    N_max = max(int(factor * Z), N_min + min_width)
    return N_min, N_max


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


def find_best_N_for_Z(Z: int, N_min: int, N_max: int) -> tuple[int, float]:
    """
    Вернуть (N_best, F_min) для данного Z на отрезке [N_min, N_max]
    по текущему NuclearConfig / режиму magic.
    """
    best_N = N_min
    best_F: float | None = None

    for N in range(N_min, N_max + 1):
        F = nuclear_functional(Z, N)
        if best_F is None or F < best_F:
            best_F = F
            best_N = N

    # best_F не может остаться None, так как диапазон включает хотя бы одно N
    return best_N, float(best_F if best_F is not None else 0.0)
