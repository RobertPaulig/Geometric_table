import os

import numpy as np
import pytest


RUN_A3 = os.environ.get("RUN_A3_TESTS") == "1"
pytestmark = pytest.mark.skipif(not RUN_A3, reason="A3.0 spec tests are opt-in; set RUN_A3_TESTS=1")


def _magnetic_laplacian(*, n: int, edges: list[tuple[int, int, float]], A: np.ndarray) -> np.ndarray:
    A0 = np.asarray(A, dtype=float)
    if A0.shape != (n, n):
        raise ValueError("A must be (n,n)")

    W = np.zeros((n, n), dtype=np.complex128)
    deg = np.zeros(n, dtype=float)
    for i, j, w in edges:
        if i == j:
            continue
        deg[i] += float(w)
        deg[j] += float(w)
        W[i, j] = float(w) * np.exp(1j * float(A0[i, j]))
        W[j, i] = float(w) * np.exp(1j * float(A0[j, i]))

    D = np.diag(deg.astype(np.complex128))
    return D - W


def _example_graph() -> tuple[int, list[tuple[int, int, float]]]:
    # Small connected graph with one cycle: 0-1-2-0 plus a tail 2-3-4.
    n = 5
    edges = [
        (0, 1, 1.0),
        (1, 2, 2.0),
        (2, 0, 1.5),
        (2, 3, 1.0),
        (3, 4, 1.0),
    ]
    return n, edges


def test_a3_phase_channel_T1_hermiticity() -> None:
    n, edges = _example_graph()

    A = np.zeros((n, n), dtype=float)
    A[0, 1] = 0.3
    A[1, 0] = -0.3
    A[1, 2] = -0.2
    A[2, 1] = 0.2
    A[2, 0] = 0.1
    A[0, 2] = -0.1

    L = _magnetic_laplacian(n=n, edges=edges, A=A)
    assert np.allclose(L, L.conj().T, atol=1e-10)


def test_a3_phase_channel_T2_gauge_invariance_spectrum() -> None:
    n, edges = _example_graph()

    rng = np.random.default_rng(0)
    theta = rng.normal(size=n)

    A = np.zeros((n, n), dtype=float)
    A[0, 1] = 0.25
    A[1, 0] = -0.25
    A[1, 2] = 0.10
    A[2, 1] = -0.10
    A[2, 0] = -0.05
    A[0, 2] = 0.05

    A2 = A.copy()
    for i, j, _w in edges:
        A2[i, j] = A[i, j] + theta[i] - theta[j]
        A2[j, i] = A[j, i] + theta[j] - theta[i]

    L1 = _magnetic_laplacian(n=n, edges=edges, A=A)
    L2 = _magnetic_laplacian(n=n, edges=edges, A=A2)

    ev1 = np.sort(np.linalg.eigvalsh(L1))
    ev2 = np.sort(np.linalg.eigvalsh(L2))
    assert np.max(np.abs(ev1 - ev2)) <= 1e-8


def test_a3_phase_channel_T3_dof_guard_only_eligible_edges() -> None:
    n, edges = _example_graph()
    eligible = {(0, 1), (1, 2), (2, 0)}  # the cycle edges only

    s_ring = 0.4  # single global parameter (P1)
    A = np.zeros((n, n), dtype=float)
    for i, j, _w in edges:
        if (i, j) in eligible or (j, i) in eligible:
            sign = 1.0 if i < j else -1.0
            A[i, j] = s_ring * sign
            A[j, i] = -A[i, j]

    nonzero = {(i, j) for i in range(n) for j in range(n) if abs(float(A[i, j])) > 1e-12}
    for i, j in nonzero:
        assert (i, j) in eligible or (j, i) in eligible

