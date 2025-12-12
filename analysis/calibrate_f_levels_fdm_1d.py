from __future__ import annotations

import numpy as np

from analysis.io_utils import results_path
from core.spectral_lab_1d import (
    make_lattice_1d,
    build_H_1d,
    solve_spectrum,
    potential_harmonic,
)
from core.f_levels_1d import (
    compute_f_levels_from_spectrum,
    estimate_f_levels_fdm_naive,
)


def weight_fn_gaussian(energies: np.ndarray) -> np.ndarray:
    """
    Та же гауссова весовая функция по энергии, что и в других Spectral Lab тестах.
    """
    sigma = 1.0
    return np.exp(-0.5 * (energies / sigma) ** 2)


def main() -> None:
    Ns = [100, 200, 400, 800]
    mass = 1.0
    x_min, x_max = -5.0, 5.0

    f_spec_list: list[float] = []
    f_naive_list: list[float] = []

    for n_points in Ns:
        lattice = make_lattice_1d(
            x_min=x_min,
            x_max=x_max,
            n_points=n_points,
            potential_fn=potential_harmonic,
        )
        H = build_H_1d(lattice, mass=mass)
        spectrum = solve_spectrum(H)

        f_spec = compute_f_levels_from_spectrum(
            spectrum,
            weight_fn_gaussian,
        )
        f_naive = estimate_f_levels_fdm_naive(
            lattice,
            weight_fn_gaussian,
            n_samples=2048,
        )

        f_spec_list.append(f_spec)
        f_naive_list.append(f_naive)

    f_spec_arr = np.array(f_spec_list, dtype=float)
    f_naive_arr = np.array(f_naive_list, dtype=float)

    A = np.vstack([f_naive_arr, np.ones_like(f_naive_arr)]).T
    a, b = np.linalg.lstsq(A, f_spec_arr, rcond=None)[0]

    f_fit = a * f_naive_arr + b
    residuals = f_spec_arr - f_fit
    mse = float(np.mean(residuals**2))
    rmse = float(np.sqrt(mse))

    out_path = results_path("f_levels_fdm_1d_calibration.txt")
    with out_path.open("w", encoding="utf-8") as f:
        f.write("Calibration of linear FDM proxy for F_levels (1D harmonic potential)\n")
        f.write("===============================================================\n\n")
        f.write("N, F_spec, F_naive, F_fit, residual\n")
        for n_points, f_spec, f_naive, f_fitted, res in zip(
            Ns, f_spec_arr, f_naive_arr, f_fit, residuals
        ):
            f.write(
                f"{n_points:4d}, {f_spec:.8e}, {f_naive:.8e}, "
                f"{f_fitted:.8e}, {res:.3e}\n"
            )

        f.write("\nFitted parameters (F_spec ≈ a * F_naive + b):\n")
        f.write(f"a = {a:.8e}\n")
        f.write(f"b = {b:.8e}\n")
        f.write(f"RMSE = {rmse:.3e}\n")

    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
