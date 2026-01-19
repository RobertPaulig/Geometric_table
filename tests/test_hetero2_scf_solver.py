import numpy as np

from hetero2.physics_operator import solve_self_consistent_potential


def _path_laplacian(n: int) -> np.ndarray:
    adj = np.zeros((n, n), dtype=float)
    for i in range(n - 1):
        adj[i, i + 1] = 1.0
        adj[i + 1, i] = 1.0
    deg = np.diag(np.sum(adj, axis=1))
    return deg - adj


def test_scf_solver_is_deterministic_on_small_chain() -> None:
    lap = _path_laplacian(4)
    v0 = np.array([0.1, 0.2, 0.3, 0.4], dtype=float)

    _, v_a, rho_a, trace_a, converged_a, iters_a, residual_a = solve_self_consistent_potential(
        laplacian=lap,
        v0=v0,
        scf_max_iter=50,
        scf_tol=1e-8,
        scf_damping=0.5,
        scf_occ_k=3,
        scf_tau=1.0,
        scf_gamma=0.5,
    )
    _, v_b, rho_b, trace_b, converged_b, iters_b, residual_b = solve_self_consistent_potential(
        laplacian=lap,
        v0=v0,
        scf_max_iter=50,
        scf_tol=1e-8,
        scf_damping=0.5,
        scf_occ_k=3,
        scf_tau=1.0,
        scf_gamma=0.5,
    )

    assert converged_a == converged_b
    assert iters_a == iters_b
    assert np.allclose(v_a, v_b, atol=1e-12, rtol=0.0)
    assert np.allclose(rho_a, rho_b, atol=1e-12, rtol=0.0)
    assert float(residual_a) == float(residual_b)

    resid_a = [float(t["residual_inf"]) for t in trace_a]
    resid_b = [float(t["residual_inf"]) for t in trace_b]
    assert np.allclose(resid_a, resid_b, atol=1e-12, rtol=0.0)


def test_scf_converges_on_small_chain() -> None:
    lap = _path_laplacian(6)
    v0 = np.zeros((6,), dtype=float)

    _, _, _, _, converged, iters, residual = solve_self_consistent_potential(
        laplacian=lap,
        v0=v0,
        scf_max_iter=50,
        scf_tol=1e-6,
        scf_damping=0.5,
        scf_occ_k=4,
        scf_tau=1.0,
        scf_gamma=0.5,
    )
    assert converged is True
    assert 1 <= int(iters) <= 50
    assert float(residual) < 1e-6


def test_scf_nonconvergence_returns_trace_without_crash() -> None:
    lap = _path_laplacian(6)
    v0 = np.zeros((6,), dtype=float)

    _, _, _, trace, converged, iters, residual = solve_self_consistent_potential(
        laplacian=lap,
        v0=v0,
        scf_max_iter=5,
        scf_tol=1e-12,
        scf_damping=1.0,
        scf_occ_k=4,
        scf_tau=1.0,
        scf_gamma=1e6,
    )
    assert converged is False
    assert int(iters) == 5
    assert len(trace) == 5
    assert float(residual) >= 0.0

