from __future__ import annotations

from analysis.chem.core2_fit import compute_p_pred, fit_lambda, kl_divergence


def test_compute_p_pred_equal_energies_reduces_to_g() -> None:
    g = {"a": 12, "b": 4}
    e = {"a": 1.0, "b": 1.0}
    p = compute_p_pred(g, e, T=1.0, lam=3.0)
    assert abs(p["a"] - 0.75) < 1e-12
    assert abs(p["b"] - 0.25) < 1e-12


def test_fit_lambda_prefers_zero_when_obs_matches_g() -> None:
    g = {"a": 12, "b": 4}
    e = {"a": 10.0, "b": 1.0}  # would skew strongly for large lambda
    p_obs = {"a": 0.75, "b": 0.25}  # exactly g-normalized
    fit = fit_lambda(p_obs, g, e, T=1.0, lam_grid=[0.0, 1.0, 2.0, 3.0])
    assert fit.lam_star == 0.0
    p0 = compute_p_pred(g, e, T=1.0, lam=0.0)
    assert kl_divergence(p_obs, p0) <= fit.kl_min + 1e-12

