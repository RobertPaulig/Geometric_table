from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from numpy.typing import NDArray


ArrayF = NDArray[np.float64]


@dataclass
class Lattice1D:
    x: ArrayF  # координаты узлов
    dx: float  # шаг
    V: ArrayF  # потенциал V(x) в узлах


@dataclass
class Spectrum1D:
    energies: ArrayF            # собственные значения H
    states: NDArray[np.complex128]  # собственные векторы (столбцы)
    dos_energies: ArrayF        # центры бинов DOS
    dos_values: ArrayF          # значения DOS
    ldos: ArrayF                # LDOS(x) для выбранного окна энергий


def make_lattice_1d(
    x_min: float,
    x_max: float,
    n_points: int,
    potential_fn: Callable[[ArrayF], ArrayF],
) -> Lattice1D:
    """
    Построить простую 1D-решётку и потенциал V(x).
    """
    x = np.linspace(x_min, x_max, n_points, dtype=float)
    dx = float(x[1] - x[0]) if n_points > 1 else 1.0
    V = potential_fn(x).astype(float)
    return Lattice1D(x=x, dx=dx, V=V)


def build_H_1d(lattice: Lattice1D, mass: float = 1.0) -> ArrayF:
    """
    Собрать дискретный оператор H = - (1/2m) d^2/dx^2 + V(x)
    на равномерной решётке (простая трёхдиагональная схема).
    """
    x, dx, V = lattice.x, lattice.dx, lattice.V
    n = x.size
    H = np.zeros((n, n), dtype=float)

    if n == 1:
        H[0, 0] = V[0]
        return H

    coeff = 1.0 / (2.0 * mass * dx * dx)

    # Внутренние точки: -2 на диагонали, +1 на соседях
    for i in range(n):
        H[i, i] += V[i] + 2.0 * coeff
        if i > 0:
            H[i, i - 1] += -coeff
        if i < n - 1:
            H[i, i + 1] += -coeff

    return H


def solve_spectrum(H: ArrayF, n_dos_bins: int = 100) -> Spectrum1D:
    """
    Решить задачу на собственные значения H и построить простую DOS/LDOS.
    """
    energies, states = np.linalg.eigh(H)

    # Простая дискретная DOS как гистограмма
    e_min = float(energies.min())
    e_max = float(energies.max())
    bins = np.linspace(e_min, e_max, n_dos_bins + 1, dtype=float)
    dos_values, edges = np.histogram(energies, bins=bins, density=True)
    dos_energies = 0.5 * (edges[:-1] + edges[1:])

    # LDOS: сумма |psi_k(x)|^2 по энергии в окне [e_min, e_max]
    # Здесь для простоты берём всё множество уровней
    prob_density = np.abs(states) ** 2  # shape (n_x, n_levels)
    ldos = prob_density.sum(axis=1).astype(float)

    return Spectrum1D(
        energies=energies.astype(float),
        states=states,
        dos_energies=dos_energies,
        dos_values=dos_values.astype(float),
        ldos=ldos,
    )


def potential_box(x: ArrayF, V0: float = 0.0) -> ArrayF:
    """
    Простая бесконечная яма в приближении большого конечного барьера.
    """
    V = np.full_like(x, V0, dtype=float)
    # При желании можно добавить большие барьеры на краях
    return V


def potential_harmonic(x: ArrayF, k: float = 1.0) -> ArrayF:
    """
    Гармонический осциллятор: V(x) = 0.5 * k * x^2.
    """
    return 0.5 * k * x * x

