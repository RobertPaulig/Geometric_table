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
        self.requests_total = 0
        self.hits_total = 0

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
        self.requests_total += int(len(keys))
        for k in keys:
            if k in self._cache:
                self.hits_total += 1

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

    @property
    def hit_rate(self) -> float:
        total = int(self.requests_total)
        if total <= 0:
            return float("nan")
        return float(self.hits_total) / float(total)


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
    n = int(n)
    if n <= 1:
        return np.array([0.0], dtype=float)
    k = np.arange(n, dtype=float)
    # Chebyshev–Lobatto nodes on [-1, 1]. These are nested for n -> (2n-1),
    # enabling deterministic p-refinement with cache reuse.
    return np.cos(math.pi * k / float(n - 1))


def adaptive_approximate_on_grid(
    *,
    f: Callable[[np.ndarray], np.ndarray],
    energy_grid: np.ndarray,
    cfg: AdaptiveIntegrationConfig,
    tol_scale: float,
    baseline_values: np.ndarray | None = None,
    baseline_noise_abs: float | None = None,
    baseline_noise_rel: float | None = None,
    baseline_noise_k_abs: float = 1.0,
    baseline_noise_k_rel: float = 1.0,
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
                "cache_hit_rate": float("nan"),
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
                "cache_hit_rate": float("nan"),
            },
        )

    baseline_arr = None
    if baseline_values is not None:
        arr = np.asarray(baseline_values, dtype=float)
        if arr.shape == grid.shape:
            baseline_arr = arr

    tol = float(cfg.eps_abs) + float(cfg.eps_rel) * float(abs(float(tol_scale)))
    if baseline_noise_abs is not None and math.isfinite(float(baseline_noise_abs)):
        tol = max(float(tol), float(baseline_noise_k_abs) * float(abs(float(baseline_noise_abs))))
    if baseline_noise_rel is not None and math.isfinite(float(baseline_noise_rel)):
        tol = max(float(tol), float(baseline_noise_k_rel) * float(abs(float(baseline_noise_rel))) * float(abs(float(tol_scale))))
    tol = max(float(tol), float(eps_floor))

    cache = _EvalCache(f)
    trace: list[AdaptiveSegmentTrace] = []

    # Chebyshev polynomial per segment. If baseline grid values are provided (integrator_mode=both),
    # use them for error estimation to avoid over-refining beyond the baseline resolution.
    max_poly_degree = max(1, min(int(cfg.poly_degree_max), 64))
    start_poly_degree = min(max_poly_degree, 4)
    max_n_probe = int(max_poly_degree + 1)
    start_n_probe = int(start_poly_degree + 1)

    pending: list[tuple[float, float, int]] = [(e_min, e_max, start_n_probe)]
    accepted: list[tuple[float, float, np.ndarray]] = []

    limit_hit_reason = ""
    verdict = "SUCCESS"
    t_total = time.perf_counter()

    subdomains_max = int(cfg.subdomains_max)
    eval_budget_max = int(cfg.eval_budget_max)
    quad_order = max(2, min(int(cfg.quad_order_max), 128))

    while pending:
        left, right, n_probe = pending.pop()
        seg_t0 = time.perf_counter()
        evals_before = int(cache.evals_total)

        split_reason = "accepted"
        error_est = float("nan")

        x_probe = _chebyshev_nodes(n_probe)
        e_probe = _map_x_to_energy(x_probe, e_left=left, e_right=right)
        y_probe = np.asarray(cache.eval(e_probe), dtype=float)

        poly_degree = int(max(1, n_probe - 1))
        coeffs = chebyshev.chebfit(x_probe, y_probe, deg=int(poly_degree))

        if baseline_arr is not None:
            mask = (grid >= float(left)) & (grid <= float(right))
            if np.any(mask):
                x_grid = _map_energy_to_x(grid[mask], e_left=float(left), e_right=float(right))
                y_hat = chebyshev.chebval(x_grid, coeffs)
                error_est = float(np.max(np.abs(np.asarray(y_hat, dtype=float) - np.asarray(baseline_arr[mask], dtype=float))))
            else:
                error_est = 0.0
        else:
            # Fallback: Gauss-point error estimate (adaptive-only mode / unit tests).
            x_quad, _ = legendre.leggauss(int(quad_order))
            e_quad = _map_x_to_energy(x_quad, e_left=left, e_right=right)
            y_true = np.asarray(cache.eval(e_quad), dtype=float)
            y_hat = chebyshev.chebval(x_quad, coeffs)
            error_est = float(np.max(np.abs(y_true - y_hat))) if y_true.size else 0.0

        accept_segment = True
        if error_est > tol:
            can_split = len(accepted) + len(pending) + 1 < int(subdomains_max)
            can_eval_more = int(cache.evals_total) < int(eval_budget_max)
            can_refine = int(n_probe) < int(max_n_probe)
            scheduled = False

            # Prefer splitting over aggressive p-refinement for very wide segments in baseline-aware mode.
            # This avoids wasting many evaluations on parent segments that will be split anyway.
            prefer_split = False
            if baseline_arr is not None:
                mask = (grid >= float(left)) & (grid <= float(right))
                grid_points = int(np.count_nonzero(mask))
                if grid_points > int(max_n_probe) * 2:
                    prefer_split = True

            if prefer_split and can_split and can_eval_more:
                mid = 0.5 * (left + right)
                # Deterministic depth-first: push right then left.
                pending.append((mid, right, int(start_n_probe)))
                pending.append((left, mid, int(start_n_probe)))
                split_reason = "split_error_gt_tol"
                accept_segment = False
                scheduled = True

            if not scheduled and can_eval_more and can_refine:
                # Deterministic p-refinement using nested n -> (2n-1) nodes (clipped to max).
                n_probe_next = min(int(max_n_probe), int(2 * int(n_probe) - 1))
                if n_probe_next > int(n_probe):
                    pending.append((left, right, int(n_probe_next)))
                    split_reason = "p_refine_error_gt_tol"
                    accept_segment = False
                    scheduled = True

            if not scheduled and can_split and can_eval_more:
                mid = 0.5 * (left + right)
                # Deterministic depth-first: push right then left.
                pending.append((mid, right, int(start_n_probe)))
                pending.append((left, mid, int(start_n_probe)))
                split_reason = "split_error_gt_tol"
                accept_segment = False
                scheduled = True

            if not scheduled:
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

    accepted_sorted = sorted(accepted, key=lambda s: (float(s[0]), float(s[1])))
    values_out = np.zeros_like(grid)
    for idx, (left, right, coeffs) in enumerate(accepted_sorted):
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
                if str(t.split_reason) == "accepted" and math.isfinite(float(t.error_est))
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
            "cache_hit_rate": float(cache.hit_rate),
        },
    )

