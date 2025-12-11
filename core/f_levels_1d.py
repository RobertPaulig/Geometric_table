from __future__ import annotations

from typing import Callable

import numpy as np

from .spectral_lab_1d import Lattice1D, Spectrum1D


def compute_f_levels_from_spectrum(
    spectrum: Spectrum1D,
    weight_fn: Callable[[np.ndarray], np.ndarray],
) -> float:
    """
    Простейший F_levels как сумма весов по собственным значениям:
        F_levels = sum_k w(E_k).
    """
    e = spectrum.energies
    w = weight_fn(e)
    return float(np.sum(w))


def estimate_f_levels_fdm_naive(
    lattice: Lattice1D,
    weight_fn: Callable[[np.ndarray], np.ndarray],
    n_samples: int = 1024,
) -> float:
    """
    Игрушечная FDM-оценка F_levels по решётке (базовый вариант):
    - равномерно выбираем точки на решётке,
    - применяем весовую функцию по «эффективной энергии» V(x),
    - усредняем.
    """
    x = lattice.x
    V = lattice.V
    if x.size == 0:
        return 0.0

    rng = np.random.default_rng()
    idx = rng.integers(0, x.size, size=n_samples)
    V_samples = V[idx]

    w = weight_fn(V_samples)
    return float(np.mean(w))


def estimate_f_levels_fdm_linear(
    lattice: Lattice1D,
    weight_fn: Callable[[np.ndarray], np.ndarray],
    a: float,
    b: float,
    n_samples: int = 1024,
) -> float:
    """
    Линейно калиброванный FDM-прокси для F_levels:
        F_fdm ≈ a * F_naive + b,
    где F_naive — базовая FDM-оценка.
    """
    base = estimate_f_levels_fdm_naive(lattice, weight_fn, n_samples=n_samples)
    return a * base + b
