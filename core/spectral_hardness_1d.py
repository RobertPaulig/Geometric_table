from __future__ import annotations

from typing import Final

import numpy as np

from .spectral_lab_1d import Spectrum1D


DEFAULT_K_AVG: Final[int] = 5
DEFAULT_K_TAIL: Final[int] = 10
DEFAULT_P_TAIL: Final[float] = 2.0


def chi_ground(spectrum: Spectrum1D) -> float:
    """
    χ_0: энергия основного состояния (минимальное собственное значение).
    """
    if spectrum.energies.size == 0:
        return 0.0
    return float(np.min(spectrum.energies))


def chi_avg_k(spectrum: Spectrum1D, k: int = DEFAULT_K_AVG) -> float:
    """
    Средняя энергия первых k уровней.
    """
    eigs = np.sort(spectrum.energies)
    if eigs.size == 0:
        return 0.0
    k_eff = min(k, eigs.size)
    return float(np.mean(eigs[:k_eff]))


def chi_tail_power(
    spectrum: Spectrum1D,
    k: int = DEFAULT_K_TAIL,
    p: float = DEFAULT_P_TAIL,
) -> float:
    """
    Энергетический «хвост»: сумма E^p по первым k уровням.
    """
    eigs = np.sort(spectrum.energies)
    if eigs.size == 0:
        return 0.0
    k_eff = min(k, eigs.size)
    return float(np.sum(np.power(eigs[:k_eff], p)))

