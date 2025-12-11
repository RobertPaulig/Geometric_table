from __future__ import annotations

from pathlib import Path

import numpy as np

from core.spectral_lab_1d import (
    Lattice1D,
    build_H_1d,
    make_lattice_1d,
    potential_box,
    potential_harmonic,
    solve_spectrum,
)
from core.f_levels_1d import compute_f_levels_from_spectrum, estimate_f_levels_fdm_naive


RESULTS_DIR = Path("results")


def weight_fn_gaussian(energies: np.ndarray) -> np.ndarray:
    """
    Простейшая весовая функция по энергии: гауссиана вокруг E=0.
    """
    sigma = 1.0
    return np.exp(-0.5 * (energies / sigma) ** 2)


def run_case(name: str, lattice: Lattice1D) -> None:
    H = build_H_1d(lattice)
    spec = solve_spectrum(H)

    f_levels_spec = compute_f_levels_from_spectrum(spec, weight_fn_gaussian)
    f_levels_fdm = estimate_f_levels_fdm_naive(lattice, weight_fn_gaussian, n_samples=2048)

    RESULTS_DIR.mkdir(exist_ok=True)
    out_txt = RESULTS_DIR / f"spectral_lab_1d_{name}.txt"

    with out_txt.open("w", encoding="utf-8") as f:
        f.write(f"Spectral Lab 1D case: {name}\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"# grid points: {lattice.x.size}\n")
        f.write(f"x range: [{lattice.x.min():.3f}, {lattice.x.max():.3f}], dx={lattice.dx:.3f}\n\n")

        f.write("Eigenvalues (first 10):\n")
        for e in spec.energies[:10]:
            f.write(f"  {e:.6f}\n")
        f.write("\n")

        f.write(f"F_levels (spectrum-based): {f_levels_spec:.6f}\n")
        f.write(f"F_levels (FDM proxy):      {f_levels_fdm:.6f}\n")

    print(f"Wrote {out_txt}")


def main() -> None:
    # Ящик
    lattice_box = make_lattice_1d(-5.0, 5.0, n_points=200, potential_fn=potential_box)
    run_case("box", lattice_box)

    # Гармонический осциллятор
    lattice_harm = make_lattice_1d(-5.0, 5.0, n_points=200, potential_fn=potential_harmonic)
    run_case("harmonic", lattice_harm)


if __name__ == "__main__":
    main()
