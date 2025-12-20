from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from analysis.chem.core2_fit import kl_divergence


def _normalize_counts(counts: Mapping[str, int]) -> Dict[str, float]:
    total = float(sum(int(v) for v in counts.values()))
    if total <= 0:
        return {}
    return {str(k): float(int(v)) / total for k, v in counts.items()}


def _hist_from_sequence(seq: Sequence[str]) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for x in seq:
        out[str(x)] = out.get(str(x), 0) + 1
    return out


def kl_pairwise_max_mean(p_by_chain: Sequence[Mapping[str, float]]) -> Tuple[float, float]:
    if len(p_by_chain) < 2:
        return float("nan"), float("nan")
    kls: List[float] = []
    for i in range(len(p_by_chain)):
        for j in range(len(p_by_chain)):
            if i == j:
                continue
            kls.append(float(kl_divergence(p_by_chain[i], p_by_chain[j])))
    return float(max(kls)), float(np.mean(np.asarray(kls, dtype=float)))


def split_chain_kl(seq: Sequence[str]) -> float:
    n = int(len(seq))
    if n < 4:
        return float("nan")
    mid = n // 2
    p1 = _normalize_counts(_hist_from_sequence(seq[:mid]))
    p2 = _normalize_counts(_hist_from_sequence(seq[mid:]))
    return float(kl_divergence(p1, p2))


def rhat(x_by_chain: Sequence[Sequence[float]]) -> float:
    chains = [np.asarray(c, dtype=float) for c in x_by_chain if len(c) > 1]
    m = int(len(chains))
    if m < 2:
        return float("nan")
    n = min(int(len(c)) for c in chains)
    if n < 2:
        return float("nan")
    xs = np.stack([c[:n] for c in chains], axis=0)
    chain_means = np.mean(xs, axis=1)
    chain_vars = np.var(xs, axis=1, ddof=1)
    W = float(np.mean(chain_vars))
    B = float(n * np.var(chain_means, ddof=1))
    var_hat = ((n - 1) / n) * W + (1 / n) * B
    if W <= 0:
        return float("nan")
    return float(math.sqrt(var_hat / W))


def ess(x: Sequence[float]) -> float:
    arr = np.asarray(x, dtype=float)
    n = int(arr.size)
    if n < 4:
        return float("nan")
    arr = arr - float(np.mean(arr))
    var = float(np.dot(arr, arr)) / float(n)
    if var <= 0:
        return float("nan")
    # autocorrelation via direct method (OK for thinned traces)
    rho_sum = 0.0
    for lag in range(1, n - 1):
        ac = float(np.dot(arr[:-lag], arr[lag:])) / float(n - lag)
        rho = ac / var
        if rho <= 0:
            break
        rho_sum += rho
    tau = 1.0 + 2.0 * rho_sum
    return float(n / tau) if tau > 0 else float("nan")


@dataclass(frozen=True)
class CoverageDynamics:
    bin_edges: List[int]
    n_unique: List[int]


def coverage_dynamics(seq: Sequence[str], *, n_bins: int = 20) -> CoverageDynamics:
    n = int(len(seq))
    if n <= 0:
        return CoverageDynamics(bin_edges=[0], n_unique=[0])
    n_bins = max(1, int(n_bins))
    edges: List[int] = [int(round(i * n / n_bins)) for i in range(n_bins + 1)]
    edges[0] = 0
    edges[-1] = n
    out: List[int] = []
    seen: set[str] = set()
    idx = 0
    for b in range(1, len(edges)):
        end = int(edges[b])
        while idx < end:
            seen.add(str(seq[idx]))
            idx += 1
        out.append(int(len(seen)))
    return CoverageDynamics(bin_edges=edges, n_unique=out)


@dataclass
class MixingDiagnosticsSummary:
    n: int
    steps: int
    burnin: int
    thin: int
    start_spec: str
    n_chains: int

    kl_pairwise_max: float
    kl_pairwise_mean: float
    kl_split_max: float
    kl_split_mean: float

    rhat_energy: float
    ess_energy_min: float
    ess_energy_mean: float

    coverage_bins: int
    coverage_n_unique_last: int


def compute_mixing_diagnostics(
    *,
    n: int,
    steps: int,
    burnin: int,
    thin: int,
    start_spec: str,
    topology_sequences_by_chain: Sequence[Sequence[str]],
    energy_traces_by_chain: Optional[Sequence[Sequence[float]]] = None,
    coverage_bins: int = 20,
) -> MixingDiagnosticsSummary:
    p_by_chain: List[Dict[str, float]] = []
    kl_splits: List[float] = []
    coverage_last: List[int] = []
    for seq in topology_sequences_by_chain:
        p_by_chain.append(_normalize_counts(_hist_from_sequence(seq)))
        kl_splits.append(float(split_chain_kl(seq)))
        dyn = coverage_dynamics(seq, n_bins=int(coverage_bins))
        coverage_last.append(int(dyn.n_unique[-1]) if dyn.n_unique else 0)

    kl_pairwise_max, kl_pairwise_mean = kl_pairwise_max_mean(p_by_chain)
    kl_split_max = float(np.nanmax(np.asarray(kl_splits, dtype=float))) if kl_splits else float("nan")
    kl_split_mean = float(np.nanmean(np.asarray(kl_splits, dtype=float))) if kl_splits else float("nan")

    rhat_e = float("nan")
    ess_min = float("nan")
    ess_mean = float("nan")
    if energy_traces_by_chain is not None and len(energy_traces_by_chain) >= 2:
        rhat_e = float(rhat(energy_traces_by_chain))
        esses = [float(ess(c)) for c in energy_traces_by_chain if len(c) > 1]
        if esses:
            ess_min = float(np.nanmin(np.asarray(esses, dtype=float)))
            ess_mean = float(np.nanmean(np.asarray(esses, dtype=float)))

    return MixingDiagnosticsSummary(
        n=int(n),
        steps=int(steps),
        burnin=int(burnin),
        thin=int(thin),
        start_spec=str(start_spec),
        n_chains=int(len(topology_sequences_by_chain)),
        kl_pairwise_max=float(kl_pairwise_max),
        kl_pairwise_mean=float(kl_pairwise_mean),
        kl_split_max=float(kl_split_max),
        kl_split_mean=float(kl_split_mean),
        rhat_energy=float(rhat_e),
        ess_energy_min=float(ess_min),
        ess_energy_mean=float(ess_mean),
        coverage_bins=int(coverage_bins),
        coverage_n_unique_last=int(max(coverage_last) if coverage_last else 0),
    )

