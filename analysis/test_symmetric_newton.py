from __future__ import annotations

import math

import numpy as np

from optimize.symmetric_newton import (
    symmetric_newton_1d,
    symmetric_newton_nd,
)


def test_scalar_quadratic() -> None:
    def f(x: float) -> float:
        return x * x - 2.0

    res = symmetric_newton_1d(f, x0=1.0)
    root = res.x[0]
    err = abs(root - math.sqrt(2.0))
    assert err < 1e-6


def test_scalar_oscillatory() -> None:
    def f(x: float) -> float:
        return math.sin(100.0 * x) + x

    res = symmetric_newton_1d(f, x0=0.1, h=1e-4)
    assert abs(res.fun) < 1e-6


def test_nd_system() -> None:
    # Simple system:
    # x^2 + y^2 = 1
    # x - y = 0  => x = y = 1/sqrt(2)
    def F(v: np.ndarray) -> np.ndarray:
        x, y = v
        return np.array(
            [
                x * x + y * y - 1.0,
                x - y,
            ]
        )

    res = symmetric_newton_nd(F, x0=[0.5, 0.5])
    x, y = res.x
    assert abs(x - y) < 1e-6
    assert abs(x**2 + y**2 - 1.0) < 1e-6


def main() -> None:
    test_scalar_quadratic()
    test_scalar_oscillatory()
    test_nd_system()


if __name__ == "__main__":
    main()
