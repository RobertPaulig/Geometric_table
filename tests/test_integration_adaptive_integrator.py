import math

import numpy as np

from hetero2.integration.adaptive import adaptive_approximate_on_grid
from hetero2.integration.metrics import curve_checksum_sha256
from hetero2.integration.types import AdaptiveIntegrationConfig


def test_adaptive_integrator_smooth_matches_baseline() -> None:
    grid = np.linspace(-1.0, 1.0, 128, dtype=float)

    def f(x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        return np.sin(x) + 0.1 * x * x

    baseline = f(grid)
    tol_scale = float(np.max(np.abs(baseline)))
    cfg = AdaptiveIntegrationConfig(
        eps_abs=1e-4,
        eps_rel=1e-4,
        subdomains_max=8,
        poly_degree_max=12,
        quad_order_max=32,
        eval_budget_max=2048,
        split_criterion="max_abs_error",
    )

    res = adaptive_approximate_on_grid(f=f, energy_grid=grid, cfg=cfg, tol_scale=tol_scale)
    assert str(res.summary.get("verdict", "")) in {"SUCCESS", "INCONCLUSIVE_LIMIT_HIT"}

    values = np.asarray(res.values, dtype=float)
    max_err = float(np.max(np.abs(values - baseline))) if values.size else float("nan")
    tol = float(cfg.eps_abs) + float(cfg.eps_rel) * abs(float(tol_scale))
    assert math.isfinite(max_err)
    assert max_err <= tol * 5.0


def test_adaptive_integrator_gaussian_mixture_splits_and_matches_baseline() -> None:
    grid = np.linspace(-1.0, 1.0, 128, dtype=float)
    centers = np.array([-0.7, -0.1, 0.3, 0.9], dtype=float)
    eta = 0.05

    def f(x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        out = np.zeros_like(x)
        for c in centers.tolist():
            out += np.exp(-0.5 * ((x - float(c)) / float(eta)) ** 2)
        return out

    baseline = f(grid)
    tol_scale = float(np.max(np.abs(baseline)))
    cfg = AdaptiveIntegrationConfig(
        eps_abs=1e-3,
        eps_rel=1e-3,
        subdomains_max=64,
        poly_degree_max=8,
        quad_order_max=16,
        eval_budget_max=4096,
        split_criterion="max_abs_error",
    )

    res = adaptive_approximate_on_grid(f=f, energy_grid=grid, cfg=cfg, tol_scale=tol_scale)
    assert str(res.summary.get("verdict", "")) in {"SUCCESS", "INCONCLUSIVE_LIMIT_HIT"}
    assert int(res.summary.get("segments_used", 0) or 0) > 1

    values = np.asarray(res.values, dtype=float)
    max_err = float(np.max(np.abs(values - baseline))) if values.size else float("nan")
    tol = float(cfg.eps_abs) + float(cfg.eps_rel) * abs(float(tol_scale))
    assert math.isfinite(max_err)
    assert max_err <= tol * 10.0


def test_adaptive_integrator_oscillatory_splits_and_is_deterministic() -> None:
    grid = np.linspace(-1.0, 1.0, 128, dtype=float)

    def f(x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        return np.sin(30.0 * x) + 0.2 * np.cos(5.0 * x)

    baseline = f(grid)
    tol_scale = float(np.max(np.abs(baseline)))
    cfg = AdaptiveIntegrationConfig(
        eps_abs=5e-2,
        eps_rel=1e-4,
        subdomains_max=32,
        poly_degree_max=8,
        quad_order_max=16,
        eval_budget_max=4096,
        split_criterion="max_abs_error",
    )

    res1 = adaptive_approximate_on_grid(f=f, energy_grid=grid, cfg=cfg, tol_scale=tol_scale)
    res2 = adaptive_approximate_on_grid(f=f, energy_grid=grid, cfg=cfg, tol_scale=tol_scale)

    assert int(res1.summary.get("segments_used", 0) or 0) > 1
    assert int(res2.summary.get("segments_used", 0) or 0) > 1
    assert int(res1.summary.get("segments_used", 0) or 0) == int(res2.summary.get("segments_used", 0) or 0)
    assert int(res1.summary.get("evals_total", 0) or 0) == int(res2.summary.get("evals_total", 0) or 0)

    checksum_1 = curve_checksum_sha256(energy_grid=grid.tolist(), values=np.asarray(res1.values, dtype=float).tolist())
    checksum_2 = curve_checksum_sha256(energy_grid=grid.tolist(), values=np.asarray(res2.values, dtype=float).tolist())
    assert checksum_1 == checksum_2

