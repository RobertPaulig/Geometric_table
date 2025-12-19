from __future__ import annotations

import csv
import math
import time
from collections import Counter
from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from analysis.chem.alkane_exact_1 import tree_automorphism_size
from analysis.chem.core2_fit import kl_divergence
from analysis.chem.topology_mcmc import Edge, run_fixed_n_tree_mcmc
from analysis.io_utils import results_path
from analysis.utils.progress import progress_iter
from analysis.utils.timing import now_iso
from analysis.growth.reporting import write_growth_txt
from core.complexity import compute_complexity_features_v2
from core.tree_canonical import canonical_relabel_tree


def make_path_edges(n: int) -> List[Edge]:
    return [(int(i), int(i + 1)) for i in range(int(n) - 1)]


def make_max_branch_edges(n: int) -> List[Edge]:
    """
    Deterministic "highly branched" alkane-valid tree (deg<=4).

    - Create degree-4 center at node 0
    - Attach remaining nodes as a short chain from one leaf
    """
    n = int(n)
    if n < 2:
        return []
    if n <= 5:
        return [(0, i) for i in range(1, n)]
    edges: List[Edge] = [(0, 1), (0, 2), (0, 3), (0, 4)]
    last = 4
    for v in range(5, n):
        edges.append((last, v))
        last = v
    return edges


def topology_id_tree(adj: np.ndarray) -> str:
    can = canonical_relabel_tree(np.asarray(adj, dtype=float))
    n = int(can.shape[0])
    edges: List[Tuple[int, int]] = []
    for i in range(n):
        for j in range(i + 1, n):
            if can[i, j] > 0:
                edges.append((i, j))
    return "tree:" + ",".join(f"{a}-{b}" for a, b in edges)


def parse_tree_topology_edges(topo: str) -> List[Tuple[int, int]]:
    if not topo.startswith("tree:"):
        raise ValueError(f"Expected topology key 'tree:<...>', got: {topo!r}")
    s = topo[len("tree:") :]
    if not s:
        return []
    edges: List[Tuple[int, int]] = []
    for part in s.split(","):
        a_str, b_str = part.split("-", 1)
        a = int(a_str)
        b = int(b_str)
        edges.append((a, b) if a < b else (b, a))
    edges.sort()
    return edges


def edges_to_adj(n: int, edges: Sequence[Tuple[int, int]]) -> np.ndarray:
    adj = np.zeros((int(n), int(n)), dtype=float)
    for a, b in edges:
        i = int(a)
        j = int(b)
        adj[i, j] = 1.0
        adj[j, i] = 1.0
    return adj


def _kl(P: Mapping[str, float], Q: Mapping[str, float], eps: float = 1e-12) -> float:
    keys = set(P.keys()) | set(Q.keys())
    p = np.asarray([float(P.get(k, 0.0)) + eps for k in keys], dtype=float)
    q = np.asarray([float(Q.get(k, 0.0)) + eps for k in keys], dtype=float)
    p = p / float(p.sum())
    q = q / float(q.sum())
    return float(np.sum(p * np.log(p / q)))


def p_pred_from_energy(
    topo_keys: Sequence[str],
    *,
    n: int,
    backend: str,
    lam: float,
    temperature_T: float,
) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, int], Dict[str, int]]:
    n = int(n)
    n_fact = math.factorial(n)

    energies: Dict[str, float] = {}
    aut_sizes: Dict[str, int] = {}
    g_vals: Dict[str, int] = {}
    weights: Dict[str, float] = {}

    beta = float(lam) / float(temperature_T) if temperature_T > 0 else float("inf")
    for topo in topo_keys:
        edges = parse_tree_topology_edges(str(topo))
        adj = edges_to_adj(n, edges)
        feats = compute_complexity_features_v2(adj, backend=str(backend))
        e = float(feats.total)
        aut = int(tree_automorphism_size(adj))
        g = int(round(float(n_fact) / float(aut)))
        w = float(g) * math.exp(-beta * float(e))
        energies[str(topo)] = float(e)
        aut_sizes[str(topo)] = int(aut)
        g_vals[str(topo)] = int(g)
        weights[str(topo)] = float(w)

    z = float(sum(weights.values()))
    p_pred = {k: (float(v) / z if z > 0 else 0.0) for k, v in weights.items()}
    return p_pred, energies, aut_sizes, g_vals


@dataclass
class EqRunConfig:
    n: int
    expected_n_topologies: int
    mode: str = "A"
    backend: str = "fdm"
    lam: float = 1.0
    temperature_T: float = 1.0
    steps: int = 20_000
    burnin: int = 2_000
    thin: int = 10
    chains: int = 5
    start_specs: Tuple[str, ...] = ("path", "max_branch")
    seed: int = 0
    progress: bool = True
    top_k: int = 20
    max_attempts: int = 3
    guardrail_kl_max_target: Optional[float] = None


def _start_edges_for_spec(n: int, spec: str) -> List[Edge]:
    spec = str(spec)
    if spec == "path":
        return make_path_edges(n)
    if spec == "max_branch":
        return make_max_branch_edges(n)
    raise ValueError(f"Unknown start_spec: {spec!r}")


def _aggregate_chain_probs(chain_probs: Sequence[Dict[str, float]]) -> Dict[str, float]:
    keys = sorted({k for p in chain_probs for k in p.keys()})
    out = {k: float(np.mean(np.asarray([float(p.get(k, 0.0)) for p in chain_probs], dtype=float))) for k in keys}
    s = float(sum(out.values()))
    return {k: (float(v) / s if s > 0 else 0.0) for k, v in out.items()}


def _kl_pairwise_max_mean(p_by_start: Sequence[Dict[str, float]]) -> Tuple[float, float]:
    if len(p_by_start) < 2:
        return float("nan"), float("nan")
    kls: List[float] = []
    for i in range(len(p_by_start)):
        for j in range(len(p_by_start)):
            if i == j:
                continue
            kls.append(float(_kl(p_by_start[i], p_by_start[j])))
    return float(max(kls)), float(np.mean(np.asarray(kls, dtype=float)))


def run_equilibrium_with_guardrail(cfg: EqRunConfig) -> Tuple[Dict[str, float], Dict[str, object]]:
    """
    Runs per-start multi-chain fixed-N MCMC, computes guardrail KL between starts, and returns P_eq.

    Returns:
      p_eq: mean across starts of mean-across-chains (normalized)
      meta: dict with per-start summaries, cache metrics, timing, coverage
    """
    start_ts = now_iso()
    t0_total = time.perf_counter()

    steps = int(cfg.steps)
    burnin = int(cfg.burnin)

    for attempt in range(int(cfg.max_attempts)):
        per_start_chain_probs: List[Tuple[str, List[Dict[str, float]]]] = []
        per_start_chain_counts: List[Tuple[str, List[Counter[str]]]] = []
        cache_hits_total = 0
        cache_misses_total = 0
        proposals_total = 0
        accepted_total = 0
        steps_total = 0
        chain_steps_sec: List[float] = []

        for start_spec in cfg.start_specs:
            chain_probs: List[Dict[str, float]] = []
            chain_counts: List[Counter[str]] = []
            for chain_id in progress_iter(
                range(int(cfg.chains)),
                total=int(cfg.chains),
                desc=f"EQ N{cfg.n} mode{cfg.mode} steps={steps} start={start_spec}",
                enabled=bool(cfg.progress),
            ):
                edges0 = _start_edges_for_spec(cfg.n, start_spec)
                samples, summary = run_fixed_n_tree_mcmc(
                    n=int(cfg.n),
                    steps=int(steps),
                    burnin=int(burnin),
                    thin=int(cfg.thin),
                    backend=str(cfg.backend),
                    lam=float(cfg.lam),
                    temperature_T=float(cfg.temperature_T),
                    seed=int(cfg.seed) + 101 * int(chain_id) + 10_000 * int(attempt),
                    max_valence=4,
                    topology_classifier=topology_id_tree,
                    start_edges=edges0,
                    progress=None,
                )
                chain_probs.append(dict(summary.p_topology))
                c: Counter[str] = Counter()
                for s in samples:
                    c[str(s["topology"])] += 1
                chain_counts.append(c)

                cache_hits_total += int(summary.energy_cache_hits)
                cache_misses_total += int(summary.energy_cache_misses)
                proposals_total += int(summary.proposals)
                accepted_total += int(summary.accepted)
                steps_total += int(summary.steps)
                chain_steps_sec.append(float(summary.steps_per_sec))

            per_start_chain_probs.append((str(start_spec), chain_probs))
            per_start_chain_counts.append((str(start_spec), chain_counts))

        # Mean P per start (across chains), and overall mean across starts.
        p_start_means: List[Tuple[str, Dict[str, float]]] = []
        for st, cps in per_start_chain_probs:
            p_start_means.append((st, _aggregate_chain_probs(cps)))

        keys_all = sorted({k for _, p in p_start_means for k in p.keys()})
        p_eq = {k: float(np.mean(np.asarray([float(p.get(k, 0.0)) for _, p in p_start_means], dtype=float))) for k in keys_all}
        s = float(sum(p_eq.values()))
        p_eq = {k: (float(v) / s if s > 0 else 0.0) for k, v in p_eq.items()}

        n_unique_eq = int(len(p_eq))
        if n_unique_eq >= int(cfg.expected_n_topologies):
            elapsed_total = time.perf_counter() - t0_total
            end_ts = now_iso()
            cache_total = cache_hits_total + cache_misses_total
            cache_hit_rate = float(cache_hits_total) / float(cache_total) if cache_total > 0 else 0.0

            kl_max, kl_mean = _kl_pairwise_max_mean([p for _, p in p_start_means])

            if cfg.guardrail_kl_max_target is not None and float(kl_max) > float(cfg.guardrail_kl_max_target):
                # Coverage ok but guardrail too loose: increase steps and retry.
                steps = int(steps) * 2
                burnin = int(max(burnin, int(round(0.1 * float(steps)))))
                continue

            meta: Dict[str, object] = {
                "start_ts": start_ts,
                "end_ts": end_ts,
                "elapsed_total_sec": float(elapsed_total),
                "eq_elapsed_sec": float(elapsed_total),
                "steps_total": int(steps_total),
                "steps_per_sec_total": float(steps_total) / float(elapsed_total) if elapsed_total > 0 else 0.0,
                "cache_hits": int(cache_hits_total),
                "cache_misses": int(cache_misses_total),
                "cache_hit_rate": float(cache_hit_rate),
                "accept_rate": (float(accepted_total) / float(proposals_total)) if proposals_total > 0 else 0.0,
                "kl_max_pairwise": float(kl_max),
                "kl_mean_pairwise": float(kl_mean),
                "n_unique_eq": int(n_unique_eq),
                "per_start_p": {st: p for st, p in p_start_means},
                "per_start_chain_counts": per_start_chain_counts,
                "chain_steps_per_sec": float(np.mean(np.asarray(chain_steps_sec, dtype=float))) if chain_steps_sec else 0.0,
                "attempt": int(attempt),
                "steps_used": int(steps),
            }
            return p_eq, meta

        # Not enough coverage: increase steps and try again.
        steps = int(steps) * 2
        burnin = int(max(burnin, int(round(0.1 * float(steps)))))

    raise RuntimeError(
        f"Coverage failed after {cfg.max_attempts} attempts: n_unique_eq={n_unique_eq} < expected {cfg.expected_n_topologies}"
    )


def write_report_and_csv(
    *,
    out_stub: str,
    cfg: EqRunConfig,
    p_eq: Dict[str, float],
    meta: Dict[str, object],
    p_pred: Dict[str, float],
    energies: Dict[str, float],
    aut_sizes: Dict[str, int],
    g_vals: Dict[str, int],
) -> Tuple[str, str]:
    out_txt = results_path(f"{out_stub}.txt")
    out_csv = results_path(f"{out_stub}.csv")

    topo_keys = sorted(p_eq.keys())
    top_k = max(1, int(cfg.top_k))

    # CSV: per chain counts + derived columns
    fieldnames = [
        "mode",
        "N",
        "start_spec",
        "chain_id",
        "steps",
        "thin",
        "burnin",
        "topology_id",
        "count",
        "prob_eq",
        "energy",
        "aut_size",
        "g",
    ]
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        per_start_chain_counts = meta["per_start_chain_counts"]
        for start_spec, chain_counts in per_start_chain_counts:
            for chain_id, ctr in enumerate(chain_counts):
                total = float(sum(ctr.values())) if ctr else 0.0
                for topo in topo_keys:
                    cnt = int(ctr.get(topo, 0))
                    w.writerow(
                        {
                            "mode": str(cfg.mode),
                            "N": int(cfg.n),
                            "start_spec": str(start_spec),
                            "chain_id": int(chain_id),
                            "steps": int(meta["steps_used"]),
                            "thin": int(cfg.thin),
                            "burnin": int(cfg.burnin),
                            "topology_id": str(topo),
                            "count": int(cnt),
                            "prob_eq": float(cnt) / total if total > 0 else 0.0,
                            "energy": float(energies.get(topo, 0.0)),
                            "aut_size": int(aut_sizes.get(topo, 0)),
                            "g": int(g_vals.get(topo, 0)),
                        }
                    )

    # TXT report
    lines: List[str] = []
    lines.append(f"{out_stub}: equilibrium-first fixed-N MCMC (tree-only alkane skeleton)")
    lines.append(
        f"N={cfg.n}, mode={cfg.mode}, backend={cfg.backend}, lambda={cfg.lam}, T={cfg.temperature_T}, "
        f"steps={meta['steps_used']}, burnin={cfg.burnin}, thin={cfg.thin}, chains={cfg.chains}, starts={list(cfg.start_specs)}"
    )
    lines.append("")
    lines.append(f"Coverage: n_unique_eq={meta['n_unique_eq']} (expected {cfg.expected_n_topologies})")
    lines.append(
        f"Guardrail: KL_max_pairwise={meta['kl_max_pairwise']:.6f}, KL_mean_pairwise={meta['kl_mean_pairwise']:.6f}"
    )
    lines.append(
        f"Self-consistency: KL(P_eq||P_pred)={float(kl_divergence(p_eq, p_pred)):.6f}"
    )
    lines.append("")
    lines.append("Top-K by P_eq:")
    for topo, p in sorted(p_eq.items(), key=lambda kv: (-kv[1], kv[0]))[:top_k]:
        lines.append(f"  {topo} = {p:.6f}  (P_pred={p_pred.get(topo, 0.0):.6f})")
    lines.append("")
    lines.append("P_eq vs P_pred table:")
    for topo in topo_keys:
        peq = float(p_eq.get(topo, 0.0))
        pp = float(p_pred.get(topo, 0.0))
        lines.append(
            f"  {topo}: P_eq={peq:.6f} P_pred={pp:.6f} log(P_eq/P_pred)={math.log((peq+1e-12)/(pp+1e-12)):+.6f} "
            f"E={energies.get(topo, 0.0):.6f} |Aut|={aut_sizes.get(topo, 0)} g={g_vals.get(topo, 0)}"
        )
    lines.append("")
    lines.append("TIMING")
    lines.append(f"START_TS={meta['start_ts']}")
    lines.append(f"END_TS={meta['end_ts']}")
    lines.append(f"ELAPSED_TOTAL_SEC={meta['elapsed_total_sec']:.6f}")
    lines.append(
        f"STEPS_TOTAL={meta['steps_total']}, STEPS_PER_SEC_TOTAL={meta['steps_per_sec_total']:.1f}"
    )
    lines.append(
        f"ENERGY_CACHE: hit_rate={meta['cache_hit_rate']:.3f}, hits={meta['cache_hits']}, misses={meta['cache_misses']}"
    )

    write_growth_txt(out_stub, lines)
    print(f"[CHEM-VALIDATION-2] wrote {out_csv}")
    print(f"[CHEM-VALIDATION-2] wrote {out_txt}")
    print(
        f"Wall-clock: start={meta['start_ts']} end={meta['end_ts']} elapsed_total_sec={float(meta['elapsed_total_sec']):.3f} "
        f"steps_total={int(meta['steps_total'])} steps_per_sec_total={float(meta['steps_per_sec_total']):.1f}"
    )
    return str(out_csv), str(out_txt)
