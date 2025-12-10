from __future__ import annotations

import argparse

import numpy as np

from core.geom_atoms import base_atoms, ALPHA_CALIBRATED, EPS_NEUTRAL, PAULING
from optimize.symmetric_newton import symmetric_newton_nd


# Набор элементов для калибровки спектральной χ к шкале Полинга
CALIB_SET = ["H", "Li", "C", "N", "O", "F", "Na", "Cl", "Br", "I"]


def collect_data():
    X = []
    Y = []
    for atom in base_atoms:
        if atom.name not in CALIB_SET:
            continue
        # подписанная спектральная χ с текущими параметрами модели
        chi_spec = atom.chi_geom_signed_spec(
            alpha=ALPHA_CALIBRATED,
            eps_neutral=EPS_NEUTRAL,
        )
        chi_p = PAULING.get(atom.name)
        if chi_spec is None or chi_p is None:
            continue
        X.append(chi_spec)
        Y.append(chi_p)
    X_arr = np.array(X, dtype=float)
    Y_arr = np.array(Y, dtype=float)
    return X_arr, Y_arr


def fit_linear_least_squares():
    X, Y = collect_data()
    if X.size == 0:
        raise RuntimeError("No calibration data collected; check CALIB_SET/PAULING.")
    # аппроксимация chi_Pauling ≈ a * chi_spec + b
    A = np.vstack([X, np.ones_like(X)]).T
    a, b = np.linalg.lstsq(A, Y, rcond=None)[0]
    pred = a * X + b
    rmse = float(np.sqrt(np.mean((pred - Y) ** 2)))
    return a, b, rmse, X, Y


def fit_linear_symmetric_newton():
    X, Y = collect_data()
    if X.size == 0:
        raise RuntimeError("No calibration data collected; check CALIB_SET/PAULING.")

    def F(theta: np.ndarray) -> np.ndarray:
        a, b = theta
        residual = a * X + b - Y
        r1 = np.sum(residual * X)
        r2 = np.sum(residual)
        return np.array([r1, r2], dtype=float)

    a_ls, b_ls, _, _, _ = fit_linear_least_squares()
    theta0 = np.array([a_ls, b_ls], dtype=float)

    res = symmetric_newton_nd(F, theta0)
    a, b = res.x

    pred = a * X + b
    rmse = float(np.sqrt(np.mean((pred - Y) ** 2)))

    return a, b, rmse, res


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fit chi_spec → chi_Pauling mapping."
    )
    parser.add_argument(
        "--method",
        choices=["least_squares", "symmetric_newton"],
        default="least_squares",
    )
    args = parser.parse_args()

    if args.method == "least_squares":
        a, b, rmse, _, _ = fit_linear_least_squares()
        converged = True
        n_iter = 1
        f_norm = rmse
    else:
        a, b, rmse, res = fit_linear_symmetric_newton()
        converged = res.converged
        n_iter = res.n_iter
        f_norm = res.f_norm

    print(f"Method: {args.method}")
    print(f"Best-fit mapping: chi_Pauling ≈ {a:.6f} * chi_spec + {b:.6f}")
    print(f"RMSE on calib set: {rmse:.6f}")
    print(f"Converged: {converged}, n_iter={n_iter}, f_norm={f_norm:.6e}")


if __name__ == "__main__":
    main()
