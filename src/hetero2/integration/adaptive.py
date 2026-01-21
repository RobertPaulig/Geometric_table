from __future__ import annotations

import functools
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
        self._cache: dict[int, float] = {}
        self.evals_total = 0
        self.requests_total = 0
        self.hits_total = 0
        self.eval_walltime_ms_total = 0.0

    def eval(self, energies: np.ndarray) -> np.ndarray:
        e = np.asarray(energies, dtype=np.float64)
        if e.size == 0:
            return np.array([], dtype=float)

        e_flat = e.reshape(-1)
        keys = e_flat.view(np.int64)
        keys_unique, first_idx, inv = np.unique(keys, return_index=True, return_inverse=True)
        counts = np.bincount(inv, minlength=int(keys_unique.size))
        self.requests_total += int(e_flat.size)

        cache = self._cache
        out_unique = np.empty((int(keys_unique.size),), dtype=float)

        missing_pos: list[int] = []
        missing_keys: list[int] = []
        missing_energies: list[float] = []

        keys_unique_list = keys_unique.tolist()
        first_idx_list = first_idx.tolist()
        for i, key in enumerate(keys_unique_list):
            cached = cache.get(int(key))
            if cached is not None:
                self.hits_total += int(counts[int(i)])
                out_unique[int(i)] = float(cached)
                continue
            missing_pos.append(int(i))
            missing_keys.append(int(key))
            missing_energies.append(float(e_flat[int(first_idx_list[int(i)])]))

        if missing_keys:
            vals = np.asarray(missing_energies, dtype=float)
            t0 = time.perf_counter()
            y_new = np.asarray(self._f(vals), dtype=float)
            self.eval_walltime_ms_total += float((time.perf_counter() - t0) * 1000.0)
            if y_new.size != vals.size:
                raise ValueError("adaptive integrator: evaluator returned wrong shape")
            y_list = y_new.tolist()
            for k, y in zip(missing_keys, y_list, strict=True):
                cache[int(k)] = float(y)
            self.evals_total += int(len(missing_keys))
            for pos, y in zip(missing_pos, y_list, strict=True):
                out_unique[int(pos)] = float(y)

        out_flat = out_unique[inv]
        return np.asarray(out_flat.reshape(e.shape), dtype=float)

    @property
    def hit_rate(self) -> float:
        total = int(self.requests_total)
        if total <= 0:
            return float("nan")
        return float(self.hits_total) / float(total)


@functools.lru_cache(maxsize=64)
def _leggauss_cached(n: int) -> tuple[np.ndarray, np.ndarray]:
    x, w = legendre.leggauss(int(n))
    return np.asarray(x, dtype=float), np.asarray(w, dtype=float)


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


@functools.lru_cache(maxsize=256)
def _chebyshev_nodes(n: int) -> np.ndarray:
    n = int(n)
    if n <= 1:
        return np.array([0.0], dtype=float)
    k = np.arange(n, dtype=float)
    # Chebyshev–Lobatto nodes on [-1, 1]. These are nested for n -> (2n-1),
    # enabling deterministic p-refinement with cache reuse.
    return np.cos(math.pi * k / float(n - 1))


@functools.lru_cache(maxsize=256)
def _chebyshev_vander_inv(n: int) -> np.ndarray:
    n = int(n)
    if n <= 0:
        raise ValueError("adaptive integrator: invalid chebyshev size")
    x = _chebyshev_nodes(int(n))
    deg = int(n - 1)
    vander = np.asarray(chebyshev.chebvander(x, deg), dtype=float)
    return np.asarray(np.linalg.inv(vander), dtype=float)


def _chebyshev_interpolate_coeffs(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y, dtype=float).reshape(-1)
    n = int(y.size)
    if n == 0:
        return np.array([], dtype=float)
    return np.asarray(_chebyshev_vander_inv(n) @ y, dtype=float)


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

    grid_sorted = True
    if int(grid.size) >= 2:
        grid_sorted = bool(np.all(np.diff(grid) >= 0.0))

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
        coeffs = _chebyshev_interpolate_coeffs(y_probe)

        if baseline_arr is not None:
            if grid_sorted:
                lo = int(np.searchsorted(grid, float(left), side="left"))
                hi = int(np.searchsorted(grid, float(right), side="right"))
                if hi > lo:
                    x_grid = _map_energy_to_x(grid[lo:hi], e_left=float(left), e_right=float(right))
                    y_hat = np.asarray(chebyshev.chebval(x_grid, coeffs), dtype=float)
                    error_est = float(np.max(np.abs(y_hat - np.asarray(baseline_arr[lo:hi], dtype=float))))
                else:
                    error_est = 0.0
            else:
                mask = (grid >= float(left)) & (grid <= float(right))
                if np.any(mask):
                    x_grid = _map_energy_to_x(grid[mask], e_left=float(left), e_right=float(right))
                    y_hat = chebyshev.chebval(x_grid, coeffs)
                    error_est = float(np.max(np.abs(np.asarray(y_hat, dtype=float) - np.asarray(baseline_arr[mask], dtype=float))))
                else:
                    error_est = 0.0
        else:
            # Fallback: Gauss-point error estimate (adaptive-only mode / unit tests).
            x_quad, _ = _leggauss_cached(int(quad_order))
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

            # Upgrade-before-split: try deterministic p-refinement first.
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
        if grid_sorted:
            lo = int(np.searchsorted(grid, float(left), side="left"))
            side = "right" if idx == len(accepted_sorted) - 1 else "left"
            hi = int(np.searchsorted(grid, float(right), side=side))
            if hi <= lo:
                continue
            x = _map_energy_to_x(grid[lo:hi], e_left=float(left), e_right=float(right))
            values_out[lo:hi] = chebyshev.chebval(x, coeffs)
        else:
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
    eval_walltime_ms_total = float(cache.eval_walltime_ms_total)
    overhead_walltime_ms_total = float(walltime_ms_total - eval_walltime_ms_total)
    if not math.isfinite(overhead_walltime_ms_total) or overhead_walltime_ms_total < 0.0:
        overhead_walltime_ms_total = 0.0

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
            "eval_walltime_ms_total": float(eval_walltime_ms_total),
            "overhead_walltime_ms_total": float(overhead_walltime_ms_total),
            "cache_hit_rate": float(cache.hit_rate),
        },
    )

