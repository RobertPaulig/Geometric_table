from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np


@dataclass
class SymmetricNewtonResult:
    x: np.ndarray
    converged: bool
    n_iter: int
    f_norm: float


def symmetric_newton_1d(
    f: Callable[[float], float],
    x0: float,
    h: float = 1e-5,
    tol: float = 1e-10,
    max_iter: int = 50,
) -> SymmetricNewtonResult:
    """
    Symmetric Newton solver for scalar equations f(x)=0.

    Uses symmetric finite-difference derivative:
        f'(x) ≈ (f(x+h) - f(x-h)) / (2h)
    """
    x = float(x0)

    for k in range(max_iter):
        fx = f(x)
        if abs(fx) < tol:
            return SymmetricNewtonResult(
                x=np.array([x], dtype=float),
                converged=True,
                n_iter=k,
                f_norm=float(abs(fx)),
            )

        fp = f(x + h)
        fm = f(x - h)
        denom = fp - fm
        if denom == 0.0:
            h *= 0.5
            if h < 1e-16:
                break
            continue

        dfdx = denom / (2.0 * h)
        if dfdx == 0.0:
            break

        step = fx / dfdx
        x_new = x - step

        if not np.isfinite(x_new):
            h *= 0.5
            if h < 1e-16:
                break
            continue

        x = x_new

    fx = f(x)
    return SymmetricNewtonResult(
        x=np.array([x], dtype=float),
        converged=False,
        n_iter=max_iter,
        f_norm=float(abs(fx)),
    )


def symmetric_newton_nd(
    F: Callable[[np.ndarray], np.ndarray],
    x0: Sequence[float],
    h: float = 1e-5,
    tol: float = 1e-10,
    max_iter: int = 50,
) -> SymmetricNewtonResult:
    """
    Symmetric Newton for systems F(x)=0 in R^n.

    Jacobian is approximated by symmetric differences:
        J_ij ≈ (F_i(x + h e_j) - F_i(x - h e_j)) / (2h)

    Intended for small n (<= 8).
    """
    x = np.array(x0, dtype=float)
    n = x.size

    for k in range(max_iter):
        Fx = F(x)
        f_norm = float(np.linalg.norm(Fx, ord=2))
        if f_norm < tol:
            return SymmetricNewtonResult(
                x=x,
                converged=True,
                n_iter=k,
                f_norm=f_norm,
            )

        J = np.zeros((n, n), dtype=float)
        for j in range(n):
            e = np.zeros_like(x)
            e[j] = 1.0
            Fp = F(x + h * e)
            Fm = F(x - h * e)
            J[:, j] = (Fp - Fm) / (2.0 * h)

        try:
            delta = np.linalg.solve(J, -Fx)
        except np.linalg.LinAlgError:
            delta, *_ = np.linalg.lstsq(J, -Fx, rcond=None)

        x_new = x + delta

        if not np.all(np.isfinite(x_new)):
            h *= 0.5
            if h < 1e-16:
                break
            continue

        x = x_new

    Fx = F(x)
    return SymmetricNewtonResult(
        x=x,
        converged=False,
        n_iter=max_iter,
        f_norm=float(np.linalg.norm(Fx, ord=2)),
    )

