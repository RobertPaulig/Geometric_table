from __future__ import annotations

from pathlib import Path
import csv

import numpy as np

from core.spectral_lab_1d import (
    make_lattice_1d,
    build_H_1d,
    solve_spectrum,
    potential_harmonic,
)
from core.f_levels_1d import compute_f_levels_from_spectrum, estimate_f_levels_fdm_naive


RESULTS_DIR = Path("results")


def weight_fn_gaussian(energies: np.ndarray) -> np.ndarray:
    """
    Та же простая гауссова весовая функция, что и в test_spectral_lab_1d.
    """
    sigma = 1.0
    return np.exp(-0.5 * (energies / sigma) ** 2)


def main() -> None:
    Ns = [100, 200, 400, 800]
    mass = 1.0
    x_min, x_max = -5.0, 5.0

    rows = []

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
        f_fdm = estimate_f_levels_fdm_naive(
            lattice,
            weight_fn_gaussian,
            n_samples=2048,
        )

        denom = abs(f_spec) if abs(f_spec) > 0.0 else 1.0
        rel_error = abs(f_spec - f_fdm) / denom

        rows.append((n_points, f_spec, f_fdm, rel_error))

    RESULTS_DIR.mkdir(exist_ok=True)
    txt_path = RESULTS_DIR / "spectral_lab_1d_resolution.txt"
    csv_path = RESULTS_DIR / "spectral_lab_1d_resolution.csv"

    lines = ["# N, F_spec, F_fdm, rel_error"]
    for n_points, f_spec, f_fdm, rel_error in rows:
        lines.append(
            f"{n_points}, {f_spec:.8e}, {f_fdm:.8e}, {rel_error:.3e}"
        )
    txt_path.write_text("\n".join(lines), encoding="utf-8")

    try:
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["N", "F_spec", "F_fdm", "rel_error"])
            for row in rows:
                writer.writerow(row)
    except Exception as exc:
        print(f"Failed to write CSV: {exc}")

    print(f"Wrote {txt_path}")
    print(f"Wrote {csv_path}")


if __name__ == "__main__":
    main()
