from __future__ import annotations

from pathlib import Path

import numpy as np

from core.spectral_lab_1d import (
    make_lattice_1d,
    build_H_1d,
    solve_spectrum,
)
from core.f_levels_1d import compute_f_levels_from_spectrum
from core.spectral_hardness_1d import (
    chi_ground,
    chi_avg_k,
    chi_tail_power,
)


RESULTS_DIR = Path("results")


def weight_fn_gaussian(energies: np.ndarray) -> np.ndarray:
    """
    Весовая функция по энергии для F_levels: гауссиана вокруг E=0.
    """
    sigma = 1.0
    return np.exp(-0.5 * (energies / sigma) ** 2)


def make_harmonic_potential(lambda_val: float):
    def potential(x: np.ndarray) -> np.ndarray:
        return 0.5 * lambda_val * x**2

    return potential


def main() -> None:
    lambda_values = [0.5, 1.0, 2.0, 4.0, 8.0]
    x_min, x_max = -5.0, 5.0
    n_points = 400
    mass = 1.0

    rows = []

    for lambda_val in lambda_values:
        potential_fn = make_harmonic_potential(lambda_val)
        lattice = make_lattice_1d(
            x_min=x_min,
            x_max=x_max,
            n_points=n_points,
            potential_fn=potential_fn,
        )
        H = build_H_1d(lattice, mass=mass)
        spectrum = solve_spectrum(H)

        chi0 = chi_ground(spectrum)
        chi_avg5 = chi_avg_k(spectrum, k=5)
        chi_avg10 = chi_avg_k(spectrum, k=10)
        chi_tail = chi_tail_power(spectrum, k=10, p=2.0)
        chi_F = compute_f_levels_from_spectrum(
            spectrum,
            weight_fn_gaussian,
        )

        rows.append(
            (
                lambda_val,
                chi0,
                chi_avg5,
                chi_avg10,
                chi_tail,
                chi_F,
            )
        )

    RESULTS_DIR.mkdir(exist_ok=True)
    txt_path = RESULTS_DIR / "toy_chi_spec_1d_harmonic.txt"
    csv_path = RESULTS_DIR / "toy_chi_spec_1d_harmonic.csv"

    # Текстовый отчёт
    lines = []
    lines.append(
        "# lambda, chi0, chi_avg5, chi_avg10, chi_tail_p2_k10, chi_F_spec"
    )
    for (
        lambda_val,
        chi0,
        chi_avg5,
        chi_avg10,
        chi_tail,
        chi_F,
    ) in rows:
        lines.append(
            f"{lambda_val:.3f}, "
            f"{chi0:.8e}, {chi_avg5:.8e}, {chi_avg10:.8e}, "
            f"{chi_tail:.8e}, {chi_F:.8e}"
        )
    txt_path.write_text("\n".join(lines), encoding="utf-8")

    # CSV-таблица
    try:
        import csv

        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "lambda",
                    "chi0",
                    "chi_avg5",
                    "chi_avg10",
                    "chi_tail_p2_k10",
                    "chi_F_spec",
                ]
            )
            for row in rows:
                writer.writerow(row)
    except Exception as exc:
        print(f"Failed to write CSV: {exc}")

    print(f"Wrote {txt_path}")
    print(f"Wrote {csv_path}")


if __name__ == "__main__":
    main()

