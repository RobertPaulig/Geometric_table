import numpy as np

from hetero2.integration.adaptive import adaptive_approximate_on_grid
from hetero2.integration.types import AdaptiveIntegrationConfig


def test_adaptive_integrator_is_not_more_expensive_than_baseline_on_smooth_function() -> None:
    energy_grid = np.linspace(-1.0, 1.0, 128, dtype=float)

    def f(x: np.ndarray) -> np.ndarray:
        vals = np.asarray(x, dtype=float)
        return np.sin(vals)

    cfg = AdaptiveIntegrationConfig(
        eps_abs=1e-6,
        eps_rel=1e-4,
        subdomains_max=64,
        poly_degree_max=8,
        quad_order_max=16,
        eval_budget_max=4096,
        split_criterion="max_abs_error",
    )

    # Use the baseline-aware fast mode (midpoint refinement) to validate the speed rails.
    res = adaptive_approximate_on_grid(
        f=f,
        energy_grid=energy_grid,
        cfg=cfg,
        tol_scale=1.0,
        baseline_noise_abs=0.0,
        baseline_noise_rel=0.0,
    )
    evals_total = int(res.summary.get("evals_total", 10**9))

    # Baseline grid uses energy_points evaluations by definition. Adaptive must not exceed it on smooth cases.
    assert evals_total <= int(energy_grid.size * 1.1)


def test_adaptive_integrator_reports_nonzero_cache_hit_rate() -> None:
    energy_grid = np.linspace(-2.0, 2.0, 128, dtype=float)

    def f(x: np.ndarray) -> np.ndarray:
        vals = np.asarray(x, dtype=float)
        return np.sin(30.0 * vals) + 0.2 * np.cos(5.0 * vals)

    cfg = AdaptiveIntegrationConfig(
        eps_abs=1e-6,
        eps_rel=1e-4,
        subdomains_max=64,
        poly_degree_max=8,
        quad_order_max=16,
        eval_budget_max=4096,
        split_criterion="max_abs_error",
    )

    baseline = f(energy_grid)
    tol_scale = float(np.max(np.abs(baseline)))

    # Provide baseline values to enable baseline-aware error estimation and deterministic p-refinement,
    # which must reuse previously evaluated nodes (cache hits).
    res = adaptive_approximate_on_grid(
        f=f,
        energy_grid=energy_grid,
        cfg=cfg,
        tol_scale=tol_scale,
        baseline_values=baseline,
    )
    hit_rate = float(res.summary.get("cache_hit_rate", float("nan")))

    # p-refinement must reuse previously evaluated nodes, therefore cache hits must be visible.
    assert np.isfinite(hit_rate)
    assert hit_rate > 0.2
