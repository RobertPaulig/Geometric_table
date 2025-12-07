"""
Least-squares fit of liquid-drop coefficients (a_v, a_s, a_c, a_a)
to a small set of stable nuclei.

After running this script, you can copy the printed coefficients
into liquid_drop_binding() in nuclear_island.py.
"""

from __future__ import annotations

import numpy as np


# label, Z, N, binding energy per nucleon (MeV)
DATA = [
    ("2H", 1, 1, 1.15785),
    ("4He", 2, 2, 7.0739),
    ("12C", 6, 6, 7.6801),
    ("23Na", 11, 12, 8.1117),
    ("32S", 16, 16, 8.4931),
    ("56Fe", 26, 30, 8.7903),
    ("82Kr", 36, 46, 8.6995),
    ("138Ba", 56, 82, 8.3810),
    ("181Ta", 73, 108, 8.0030),
]


def main() -> None:
    rows = []
    targets = []

    for name, Z, N, be_per_A in DATA:
        A = Z + N
        B_exp = be_per_A * A

        X1 = A
        X2 = -A ** (2.0 / 3.0)
        X3 = -Z * (Z - 1) / (A ** (1.0 / 3.0))
        X4 = -(N - Z) ** 2 / A

        rows.append([X1, X2, X3, X4])
        targets.append(B_exp)

    X = np.array(rows, dtype=float)
    y = np.array(targets, dtype=float)

    coef, residuals, rank, s = np.linalg.lstsq(X, y, rcond=None)
    a_v, a_s, a_c, a_a = coef

    def B_model(Z: int, N: int) -> float:
        A = Z + N
        if A <= 1:
            return 0.0
        return (
            a_v * A
            - a_s * A ** (2.0 / 3.0)
            - a_c * Z * (Z - 1) / (A ** (1.0 / 3.0))
            - a_a * (N - Z) ** 2 / A
        )

    errors = []
    for name, Z, N, be_per_A in DATA:
        A = Z + N
        B_exp = be_per_A * A
        errors.append(B_model(Z, N) - B_exp)

    rmse = float(np.sqrt(np.mean(np.square(errors))))

    print("Fitted coefficients (a_v, a_s, a_c, a_a):")
    print(f"a_v = {a_v:.3f}, a_s = {a_s:.3f}, a_c = {a_c:.3f}, a_a = {a_a:.3f}")
    print(f"RMSE on calibration set: {rmse:.3f} MeV")


if __name__ == "__main__":
    main()

