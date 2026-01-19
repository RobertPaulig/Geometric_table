from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Callable, Mapping

import numpy as np
from numpy.polynomial import chebyshev, legendre

from hetero2.integration.types import AdaptiveIntegrationConfig


@dataclass(frozen=True, slots=True)
class AdaptiveSegmentTrace:
    segment_id: int
    e_left: float
    e_right: float
    n_probe_points: int
    poly_degree: int
    quad_order: int
    n_function_evals: int
    error_est: float
    walltime_ms_segment: float
    split_reason: str


@dataclass(frozen=True, slots=True)
class AdaptiveCurveResult:
    values: np.ndarray
    trace: list[AdaptiveSegmentTrace]
    summary: Mapping[str, object]


class _EvalCache:
    def __init__(self, f: Callable[[np.ndarray], np.ndarray]) -> None:
        self._f = f
        self._cache: dict[str, float] = {}
        self.evals_total = 0

    @staticmethod
    def _key(e: float) -> str:
        v = float(e)
        if not math.isfinite(v):
            return "nan"
        # Stable token for deterministic caching.
        return f"{v:.12g}"

    def eval(self, energies: np.ndarray) -> np.ndarray:
        e = np.asarray(energies, dtype=float)
        if e.size == 0:
            return np.array([], dtype=float)

        keys = [self._key(float(x)) for x in e.tolist()]
        missing: dict[str, float] = {k: float(v) for k, v in zip(keys, e.tolist(), strict=False) if k not in self._cache}
        if missing:
            # Deterministic ordering for evaluation and cache fill.
            ordered = sorted(missing.items(), key=lambda kv: kv[0])
            vals = np.array([v for _, v in ordered], dtype=float)
            out = np.asarray(self._f(vals), dtype=float)
            if out.size != vals.size:
                raise ValueError("adaptive integrator: evaluator returned wrong shape")
            for (k, _), y in zip(ordered, out.tolist(), strict=True):
                self._cache[k] = float(y)
            self.evals_total += int(vals.size)

        return np.array([float(self._cache[k]) for k in keys], dtype=float)


def _map_x_to_energy(x: np.ndarray, *, e_left: float, e_right: float) -> np.ndarray:
    left = float(e_left)
    right = float(e_right)
    return 0.5 * (right - left) * np.asarray(x, dtype=float) + 0.5 * (right + left)


def _map_energy_to_x(e: np.ndarray, *, e_left: float, e_right: float) -> np.ndarray:
    left = float(e_left)
    right = float(e_right)
    denom = right - left
    if denom == 0.0:
        return np.zeros_like(np.asarray(e, dtype=float))
    return (2.0 * np.asarray(e, dtype=float) - (right + left)) / denom


def _chebyshev_nodes(n: int) -> np.ndarray:
    k = np.arange(int(n), dtype=float)
    # First-kind Chebyshev nodes on [-1, 1].
    return np.cos(math.pi * (2.0 * k + 1.0) / (2.0 * float(n)))


def adaptive_approximate_on_grid(
    *,
    f: Callable[[np.ndarray], np.ndarray],
    energy_grid: np.ndarray,
    cfg: AdaptiveIntegrationConfig,
    tol_scale: float,
    eps_floor: float = 1e-12,
) -> AdaptiveCurveResult:
    grid = np.asarray(energy_grid, dtype=float)
    if grid.size == 0:
        return AdaptiveCurveResult(
            values=np.array([], dtype=float),
            trace=[],
            summary={
                "verdict": "SUCCESS",
                "limit_hit_reason": "",
                "eps_abs": float(cfg.eps_abs),
                "eps_rel": float(cfg.eps_rel),
                "error_est_total": float("nan"),
                "segments_used": 0,
                "evals_total": 0,
                "walltime_ms_total": 0.0,
            },
        )

    e_min = float(np.min(grid))
    e_max = float(np.max(grid))
    if not (math.isfinite(e_min) and math.isfinite(e_max) and e_max > e_min):
        return AdaptiveCurveResult(
            values=np.zeros_like(grid),
            trace=[],
            summary={
                "verdict": "INCONCLUSIVE_LIMIT_HIT",
                "limit_hit_reason": "invalid_energy_range",
                "eps_abs": float(cfg.eps_abs),
                "eps_rel": float(cfg.eps_rel),
                "error_est_total": float("nan"),
                "segments_used": 0,
                "evals_total": 0,
                "walltime_ms_total": 0.0,
            },
        )

    tol = float(cfg.eps_abs) + float(cfg.eps_rel) * float(abs(float(tol_scale)))
    tol = max(float(tol), float(eps_floor))

    cache = _EvalCache(f)
    trace: list[AdaptiveSegmentTrace] = []

    # Stack of segments to process: (e_left, e_right).
    pending: list[tuple[float, float]] = [(e_min, e_max)]
    accepted: list[tuple[float, float, np.ndarray]] = []

    limit_hit_reason = ""
    verdict = "SUCCESS"
    t_total = time.perf_counter()

    subdomains_max = int(cfg.subdomains_max)
    eval_budget_max = int(cfg.eval_budget_max)
    poly_degree = max(1, min(int(cfg.poly_degree_max), 64))
    quad_order = max(2, min(int(cfg.quad_order_max), 128))

    while pending:
        left, right = pending.pop()
        seg_t0 = time.perf_counter()
        evals_before = int(cache.evals_total)

        split_reason = "accepted"
        error_est = float("nan")

        # Probe points (Chebyshev nodes).
        n_probe = int(poly_degree + 1)
        x_probe = _chebyshev_nodes(n_probe)
        e_probe = _map_x_to_energy(x_probe, e_left=left, e_right=right)
        y_probe = cache.eval(e_probe)
        coeffs = chebyshev.chebfit(x_probe, y_probe, deg=int(poly_degree))

        # Error estimate on Gauss nodes (max abs diff).
        x_quad, _ = legendre.leggauss(int(quad_order))
        e_quad = _map_x_to_energy(x_quad, e_left=left, e_right=right)
        y_true = cache.eval(e_quad)
        y_hat = chebyshev.chebval(x_quad, coeffs)
        if y_true.size:
            error_est = float(np.max(np.abs(y_true - y_hat)))
        else:
            error_est = 0.0

        # Decide split / accept.
        accept_segment = True
        if error_est > tol:
            can_split = len(accepted) + len(pending) + 1 < subdomains_max
            can_eval_more = cache.evals_total < eval_budget_max
            if can_split and can_eval_more:
                mid = 0.5 * (left + right)
                # Deterministic depth-first: push right then left.
                pending.append((mid, right))
                pending.append((left, mid))
                split_reason = "split_error_gt_tol"
                accept_segment = False
            else:
                split_reason = "limit_hit"
                verdict = "INCONCLUSIVE_LIMIT_HIT"
                if not limit_hit_reason:
                    if not can_split:
                        limit_hit_reason = "subdomains_max"
                    elif not can_eval_more:
                        limit_hit_reason = "eval_budget_max"
                    else:
                        limit_hit_reason = "unknown"
        if accept_segment:
            accepted.append((left, right, coeffs))

        evals_after = int(cache.evals_total)
        trace.append(
            AdaptiveSegmentTrace(
                segment_id=len(trace),
                e_left=float(left),
                e_right=float(right),
                n_probe_points=int(n_probe),
                poly_degree=int(poly_degree),
                quad_order=int(quad_order),
                n_function_evals=int(evals_after - evals_before),
                error_est=float(error_est),
                walltime_ms_segment=float((time.perf_counter() - seg_t0) * 1000.0),
                split_reason=str(split_reason),
            )
        )

    # Evaluate piecewise polynomial on the requested grid.
    accepted_sorted = sorted(accepted, key=lambda s: (float(s[0]), float(s[1])))
    values_out = np.zeros_like(grid)
    for idx, (left, right, coeffs) in enumerate(accepted_sorted):
        # Include right endpoint only for the last segment to avoid double coverage.
        if idx == len(accepted_sorted) - 1:
            mask = (grid >= float(left)) & (grid <= float(right))
        else:
            mask = (grid >= float(left)) & (grid < float(right))
        if not np.any(mask):
            continue
        x = _map_energy_to_x(grid[mask], e_left=float(left), e_right=float(right))
        values_out[mask] = chebyshev.chebval(x, coeffs)

    error_est_total = float(
        max(
            (
                float(t.error_est)
                for t in trace
                if str(t.split_reason) != "split_error_gt_tol" and math.isfinite(float(t.error_est))
            ),
            default=float("nan"),
        )
    )
    walltime_ms_total = float((time.perf_counter() - t_total) * 1000.0)

    return AdaptiveCurveResult(
        values=values_out,
        trace=trace,
        summary={
            "verdict": str(verdict),
            "limit_hit_reason": str(limit_hit_reason),
            "eps_abs": float(cfg.eps_abs),
            "eps_rel": float(cfg.eps_rel),
            "error_est_total": float(error_est_total),
            "segments_used": int(len(accepted_sorted)),
            "evals_total": int(cache.evals_total),
            "walltime_ms_total": float(walltime_ms_total),
        },
    )

