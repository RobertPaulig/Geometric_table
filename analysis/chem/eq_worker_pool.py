from __future__ import annotations

import os
import pickle
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Mapping, MutableMapping, Optional, Tuple

from analysis.chem.topology_mcmc import Edge, run_fixed_n_tree_mcmc, tree_topology_edge_key_from_edges


CacheKey = Tuple[str, object]

_WORKER_BASE_CACHE: Dict[CacheKey, float] = {}
_WORKER_ENERGY_CACHE_PATH: Optional[Path] = None


class TrackingCache(MutableMapping):
    """
    Wraps an existing cache dict and tracks keys that were newly added.
    """

    def __init__(self, base: Dict[CacheKey, float]) -> None:
        self._base = base
        self.added: Dict[CacheKey, float] = {}

    def __getitem__(self, key: CacheKey) -> float:
        return self._base[key]

    def __setitem__(self, key: CacheKey, value: float) -> None:
        if key not in self._base:
            self.added[key] = value
        self._base[key] = value

    def __delitem__(self, key: CacheKey) -> None:  # pragma: no cover - not expected
        del self._base[key]

    def __iter__(self):
        return iter(self._base)

    def __len__(self) -> int:
        return len(self._base)

    def __contains__(self, key: object) -> bool:
        return key in self._base

    def get(self, key: CacheKey, default: Optional[float] = None) -> Optional[float]:
        return self._base.get(key, default)


def _safe_load_cache(path: Optional[Path]) -> Dict[CacheKey, float]:
    if path is None or not path.exists():
        return {}
    with path.open("rb") as f:
        obj = pickle.load(f)
    if isinstance(obj, dict):
        return dict(obj)
    return {}


def init_worker(energy_cache_path: Optional[str]) -> None:
    """
    Initializer for ProcessPoolExecutor workers (spawn-safe).
    Loads cache snapshot into the worker and clamps BLAS threads.
    """
    global _WORKER_BASE_CACHE, _WORKER_ENERGY_CACHE_PATH
    _WORKER_ENERGY_CACHE_PATH = Path(energy_cache_path) if energy_cache_path else None
    _WORKER_BASE_CACHE = _safe_load_cache(_WORKER_ENERGY_CACHE_PATH)
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")


def set_sequential_base_cache(cache: Dict[CacheKey, float]) -> None:
    """
    Configure worker globals for in-process execution (workers=1).
    """
    global _WORKER_BASE_CACHE, _WORKER_ENERGY_CACHE_PATH
    _WORKER_ENERGY_CACHE_PATH = None
    _WORKER_BASE_CACHE = cache


@dataclass(frozen=True)
class EqTask:
    n: int
    steps_per_chain: int
    burnin_steps: int
    thin: int
    start_spec: str
    chain_idx: int
    seed: int
    start_edges: Tuple[Edge, ...]
    backend: str = "fdm"
    lam: float = 1.0
    temperature: float = 1.0
    max_valence: int = 4
    n_samples_expected: int = 0


@dataclass
class EqTaskResult:
    start_spec: str
    chain_idx: int
    topo_counts: Counter[str]
    topo_counts_first: Counter[str]
    topo_counts_second: Counter[str]
    n_samples_recorded: int
    energy_mean: float
    energy_var: float
    energy_samples: int
    ess_energy_est: float
    accepted: int
    proposed: int
    cache_hits: int
    cache_misses: int
    steps_total: int
    cache_delta: Dict[CacheKey, float]


def _finalize_energy_stats(
    n: int,
    sum_x: float,
    sum_x2: float,
    sum_lag1: float,
) -> Tuple[float, float, float]:
    if n <= 0:
        return 0.0, 0.0, 0.0
    mean = sum_x / float(n)
    if n > 1:
        var = max(0.0, (sum_x2 - n * mean * mean) / float(n - 1))
        exy = sum_lag1 / float(n - 1)
        cov1 = exy - mean * mean
        rho1 = cov1 / var if var > 0 else 0.0
    else:
        var = 0.0
        rho1 = 0.0
    rho1 = max(min(rho1, 0.9999), -0.9999)
    return mean, var, rho1


def _ess_from_rho1(n: int, rho1: float) -> float:
    if n <= 1:
        return float(n)
    denom = 1.0 + rho1
    if denom <= 1e-12:
        return float(n)
    ess = n * (1.0 - rho1) / denom
    ess = max(min(ess, float(n)), 1.0)
    return float(ess)


def run_task(task: EqTask) -> EqTaskResult:
    global _WORKER_BASE_CACHE

    tracking_cache = TrackingCache(_WORKER_BASE_CACHE)

    topo_counts: Counter[str] = Counter()
    topo_counts_first: Counter[str] = Counter()
    topo_counts_second: Counter[str] = Counter()
    samples_recorded = 0
    half_point = max(0, int(task.n_samples_expected // 2))

    energy_n = 0
    energy_sum = 0.0
    energy_sum2 = 0.0
    energy_sum_lag1 = 0.0
    prev_energy: Optional[float] = None

    def on_sample(topo: str, energy: float) -> None:
        nonlocal samples_recorded, energy_n, energy_sum, energy_sum2, energy_sum_lag1, prev_energy
        topo_counts[topo] += 1
        if samples_recorded < half_point:
            topo_counts_first[topo] += 1
        else:
            topo_counts_second[topo] += 1
        samples_recorded += 1

        e = float(energy)
        energy_n += 1
        energy_sum += e
        energy_sum2 += e * e
        if prev_energy is not None:
            energy_sum_lag1 += e * prev_energy
        prev_energy = e

    _, summary = run_fixed_n_tree_mcmc(
        n=int(task.n),
        steps=int(task.steps_per_chain),
        burnin=int(task.burnin_steps),
        thin=int(task.thin),
        backend=str(task.backend),
        lam=float(task.lam),
        temperature_T=float(task.temperature),
        seed=int(task.seed),
        max_valence=int(task.max_valence),
        topology_key_fn_edges=tree_topology_edge_key_from_edges,
        start_edges=list(task.start_edges),
        energy_cache=tracking_cache,
        progress=None,
        profile_every=0,
        step_heartbeat_every=0,
        step_heartbeat=None,
        sample_callback=on_sample,
        collect_samples=False,
    )

    mean, var, rho1 = _finalize_energy_stats(energy_n, energy_sum, energy_sum2, energy_sum_lag1)
    ess_est = _ess_from_rho1(energy_n, rho1)

    return EqTaskResult(
        start_spec=task.start_spec,
        chain_idx=task.chain_idx,
        topo_counts=topo_counts,
        topo_counts_first=topo_counts_first,
        topo_counts_second=topo_counts_second,
        n_samples_recorded=samples_recorded,
        energy_mean=float(mean),
        energy_var=float(var),
        energy_samples=int(energy_n),
        ess_energy_est=float(ess_est),
        accepted=int(getattr(summary, "accepted", 0)),
        proposed=int(getattr(summary, "proposals", 0)),
        cache_hits=int(getattr(summary, "energy_cache_hits", 0)),
        cache_misses=int(getattr(summary, "energy_cache_misses", 0)),
        steps_total=int(getattr(summary, "steps", task.steps_per_chain)),
        cache_delta=dict(tracking_cache.added),
    )
