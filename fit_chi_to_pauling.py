from __future__ import annotations

import numpy as np

from core.geom_atoms import base_atoms, ALPHA_CALIBRATED, EPS_NEUTRAL, PAULING


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
    return np.array(X, dtype=float), np.array(Y, dtype=float)


def fit_linear():
    X, Y = collect_data()
    if X.size == 0:
        raise RuntimeError("No calibration data collected; check CALIB_SET/PAULING.")
    # аппроксимация chi_Pauling ≈ a * chi_spec + b
    A = np.vstack([X, np.ones_like(X)]).T
    a, b = np.linalg.lstsq(A, Y, rcond=None)[0]
    return a, b, X, Y


if __name__ == "__main__":
    a, b, X, Y = fit_linear()
    print(f"Best-fit mapping: chi_Pauling ≈ {a:.3f} * chi_spec + {b:.3f}")
    pred = a * X + b
    rmse = np.sqrt(np.mean((pred - Y) ** 2))
    print(f"RMSE on calib set: {rmse:.3f}")
