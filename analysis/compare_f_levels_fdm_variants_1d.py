from __future__ import annotations

from pathlib import Path

import numpy as np

from core.spectral_lab_1d import (
    make_lattice_1d,
    build_H_1d,
    solve_spectrum,
    potential_harmonic,
    potential_box,
)
from core.f_levels_1d import (
    compute_f_levels_from_spectrum,
    estimate_f_levels_fdm_naive,
    estimate_f_levels_fdm_linear,
)


RESULTS_DIR = Path("results")


def weight_fn_gaussian(energies: np.ndarray) -> np.ndarray:
    sigma = 1.0
    return np.exp(-0.5 * (energies / sigma) ** 2)


def main() -> None:
    # Коэффициенты из калибровочного файла для гармонического потенциала
    a = -6.33234302e-03
    b = 1.25570275e00

    Ns_known = [100, 200, 400, 800]
    Ns_new = [160, 320, 640]
    Ns_all = Ns_known + Ns_new

    potentials = [
        ("harmonic", potential_harmonic),
        ("box", potential_box),
    ]

    mass = 1.0
    x_min, x_max = -5.0, 5.0

    rows = []

    for pot_name, pot_fn in potentials:
        for n_points in Ns_all:
            lattice = make_lattice_1d(
                x_min=x_min,
                x_max=x_max,
                n_points=n_points,
                potential_fn=pot_fn,
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
            f_lin = estimate_f_levels_fdm_linear(
                lattice,
                weight_fn_gaussian,
                a=a,
                b=b,
                n_samples=2048,
            )

            denom = abs(f_spec) if abs(f_spec) > 0.0 else 1.0
            err_naive = abs(f_spec - f_naive) / denom
            err_lin = abs(f_spec - f_lin) / denom

            is_known = n_points in Ns_known

            rows.append(
                (
                    pot_name,
                    n_points,
                    f_spec,
                    f_naive,
                    f_lin,
                    err_naive,
                    err_lin,
                    is_known,
                )
            )

    RESULTS_DIR.mkdir(exist_ok=True)
    txt_path = RESULTS_DIR / "f_levels_fdm_variants_1d.txt"
    csv_path = RESULTS_DIR / "f_levels_fdm_variants_1d.csv"

    lines = []
    lines.append(
        "# potential, N, F_spec, F_naive, F_lin, err_naive, err_lin, known_N"
    )
    for (
        pot_name,
        n_points,
        f_spec,
        f_naive,
        f_lin,
        err_naive,
        err_lin,
        is_known,
    ) in rows:
        lines.append(
            f"{pot_name}, {n_points:4d}, "
            f"{f_spec:.8e}, {f_naive:.8e}, {f_lin:.8e}, "
            f"{err_naive:.3e}, {err_lin:.3e}, {int(is_known)}"
        )
    txt_path.write_text("\n".join(lines), encoding="utf-8")

    try:
        import csv

        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "potential",
                    "N",
                    "F_spec",
                    "F_naive",
                    "F_lin",
                    "err_naive",
                    "err_lin",
                    "known_N",
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

