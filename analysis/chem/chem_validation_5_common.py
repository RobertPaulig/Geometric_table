from __future__ import annotations

import csv
import math
import time
from collections import Counter
from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from analysis.chem.alkane_expected_counts import expected_unique_alkane_tree_topologies
from analysis.chem.alkane_exact_1 import tree_automorphism_size
from analysis.chem.core2_fit import kl_divergence
from analysis.chem.mixing_diagnostics_1 import compute_mixing_diagnostics
from analysis.chem.topology_mcmc import Edge, run_fixed_n_tree_mcmc, tree_topology_edge_key_from_edges
from analysis.io_utils import results_path
from analysis.utils.progress import progress_iter
from analysis.utils.timing import now_iso
from core.complexity import compute_complexity_features_v2
from core.grower import GrowthParams, grow_molecule_christmas_tree
from core.tree_canonical import canonical_relabel_tree


def _canonical_tree_id_from_adj(adj: np.ndarray) -> str:
    can = canonical_relabel_tree(np.asarray(adj, dtype=float))
    n = int(can.shape[0])
    edges: List[Tuple[int, int]] = []
    for i in range(n):
        for j in range(i + 1, n):
            if float(can[i, j]) > 0:
                edges.append((i, j))
    edges.sort()
    return "tree:" + ",".join(f"{a}-{b}" for a, b in edges)


def _parse_tree_edges(topo: str) -> List[Tuple[int, int]]:
    if not str(topo).startswith("tree:"):
        raise ValueError(f"Expected 'tree:' topology_id, got: {topo!r}")
    s = str(topo)[len("tree:") :]
    if not s:
        return []
    out: List[Tuple[int, int]] = []
    for part in s.split(","):
        a_str, b_str = part.split("-", 1)
        a = int(a_str)
        b = int(b_str)
        out.append((a, b) if a < b else (b, a))
    out.sort()
    return out


def _edges_to_adj(n: int, edges: Sequence[Tuple[int, int]]) -> np.ndarray:
    adj = np.zeros((int(n), int(n)), dtype=float)
    for a, b in edges:
        i = int(a)
        j = int(b)
        adj[i, j] = 1.0
        adj[j, i] = 1.0
    return adj


def _wiener_index_tree(adj: np.ndarray) -> int:
    n = int(adj.shape[0])
    adj_list: List[List[int]] = [[] for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            if float(adj[i, j]) > 0:
                adj_list[i].append(j)
                adj_list[j].append(i)
    total = 0
    for src in range(n):
        dist = [-1] * n
        dist[src] = 0
        q = [src]
        qi = 0
        while qi < len(q):
            u = q[qi]
            qi += 1
            for v in adj_list[u]:
                if dist[v] < 0:
                    dist[v] = dist[u] + 1
                    q.append(v)
        for dst in range(src + 1, n):
            total += int(dist[dst])
    return int(total)


def _diameter_tree(adj: np.ndarray) -> int:
    n = int(adj.shape[0])
    adj_list: List[List[int]] = [[] for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            if float(adj[i, j]) > 0:
                adj_list[i].append(j)
                adj_list[j].append(i)

    def bfs(start: int) -> Tuple[int, List[int]]:
        dist = [-1] * n
        dist[start] = 0
        q = [start]
        qi = 0
        while qi < len(q):
            u = q[qi]
            qi += 1
            for v in adj_list[u]:
                if dist[v] < 0:
                    dist[v] = dist[u] + 1
                    q.append(v)
        far = max(range(n), key=lambda i: dist[i])
        return int(far), dist

    a, _ = bfs(0)
    b, dist = bfs(a)
    return int(max(dist))


def _tree_metrics(n: int, topo_id: str) -> Dict[str, float]:
    edges = _parse_tree_edges(topo_id)
    adj = _edges_to_adj(int(n), edges)
    deg = [int(x) for x in np.sum(adj, axis=0).tolist()]
    return {
        "diameter": float(_diameter_tree(adj)),
        "max_degree": float(max(deg) if deg else 0),
        "n_leaves": float(sum(1 for d in deg if d == 1)),
        "wiener": float(_wiener_index_tree(adj)),
    }


def _spearman_rho(x: Sequence[float], y: Sequence[float]) -> float:
    if len(x) != len(y) or not x:
        return float("nan")
    rx = np.asarray(x, dtype=float).argsort().argsort().astype(float) + 1.0
    ry = np.asarray(y, dtype=float).argsort().argsort().astype(float) + 1.0
    rx = rx - float(np.mean(rx))
    ry = ry - float(np.mean(ry))
    num = float(np.dot(rx, ry))
    den = float(math.sqrt(float(np.dot(rx, rx)) * float(np.dot(ry, ry))))
    return float(num / den) if den > 0 else float("nan")


@dataclass(frozen=True)
class EqCfg:
    n: int
    steps_init: int
    max_steps: int
    chains: int
    thin: int
    seed: int
    start_specs: Tuple[str, ...]
    guardrail_target: float
    profile_every: int
    progress: bool


@dataclass(frozen=True)
class GrowthCfg:
    n_runs: int
    seeds: Tuple[int, ...]
    progress: bool


def run_growth_distribution(*, n: int, cfg: GrowthCfg) -> Tuple[Dict[str, float], Dict[str, int], float]:
    t0 = time.perf_counter()
    counts: Counter[str] = Counter()
    total = int(cfg.n_runs) * int(len(cfg.seeds))
    for seed in cfg.seeds:
        for run_idx in progress_iter(
            range(int(cfg.n_runs)),
            total=int(cfg.n_runs),
            desc=f"GROWTH N{n} seed={seed}",
            enabled=bool(cfg.progress),
        ):
            gp = GrowthParams(
                stop_at_n_atoms=int(n),
                allowed_symbols=["C"],
                enforce_tree_alkane=True,
            )
            rng = np.random.default_rng(int(seed) + 1_000_000 * int(run_idx))
            mol = grow_molecule_christmas_tree("C", gp, rng=rng)
            topo = _canonical_tree_id_from_adj(np.asarray(mol.adjacency_matrix(), dtype=float))
            counts[topo] += 1
    p = {k: float(v) / float(total) for k, v in counts.items()} if total > 0 else {}
    return p, dict(counts), float(time.perf_counter() - t0)


def _start_edges_for_spec(n: int, spec: str) -> List[Edge]:
    spec = str(spec)
    if spec == "path":
        return [(int(i), int(i + 1)) for i in range(int(n) - 1)]
    if spec == "max_branch":
        if int(n) <= 5:
            return [(0, i) for i in range(1, int(n))]
        edges: List[Edge] = [(0, 1), (0, 2), (0, 3), (0, 4)]
        last = 4
        for v in range(5, int(n)):
            edges.append((last, v))
            last = v
        return edges
    raise ValueError(f"Unknown start_spec: {spec!r}")


def run_equilibrium_distribution_mode_a(
    *,
    n: int,
    cfg: EqCfg,
) -> Tuple[Dict[str, float], Dict[str, object], float, float]:
    """
    Returns p_eq (mean across starts), meta dict, eq_elapsed, io_elapsed is computed by caller.
    """
    t0 = time.perf_counter()
    start_ts = now_iso()

    # `steps` in CHEM-VALIDATION-5 is interpreted as TOTAL steps across all starts×chains.
    steps_total_budget = int(cfg.steps_init)
    n_chains_total = int(cfg.chains) * int(len(cfg.start_specs))
    steps_per_chain = max(1, int(round(float(steps_total_budget) / float(n_chains_total))))
    burnin = int(max(0, round(0.1 * float(steps_per_chain))))
    attempts = 0
    last_meta: Dict[str, object] = {}

    while True:
        attempts += 1
        per_start_chain_seqs: Dict[str, List[List[str]]] = {}
        per_start_chain_energy: Dict[str, List[List[float]]] = {}
        per_start_chain_probs: List[Tuple[str, List[Dict[str, float]]]] = []
        cache_hits_total = 0
        cache_misses_total = 0
        proposals_total = 0
        accepted_total = 0
        steps_total = 0
        chain_steps_sec: List[float] = []
        chain_t_move: List[float] = []
        chain_t_energy: List[float] = []
        chain_t_canon: List[float] = []

        for start_spec in cfg.start_specs:
            chain_probs: List[Dict[str, float]] = []
            chain_seqs: List[List[str]] = []
            chain_energies: List[List[float]] = []
            for chain_idx in progress_iter(
                range(int(cfg.chains)),
                total=int(cfg.chains),
                desc=f"EQ N{n} modeA steps_per_chain={steps_per_chain} start={start_spec}",
                enabled=bool(cfg.progress),
            ):
                edges0 = _start_edges_for_spec(n, start_spec)
                def _heartbeat(info: Mapping[str, float]) -> None:
                    print(
                        f"[EQ-HB N{n} start={start_spec} chain={int(chain_idx)}] "
                        f"step={int(info['step'])}/{int(info['steps_total'])} "
                        f"steps_per_sec={info['heartbeat_steps_per_sec']:.0f} "
                        f"acc={info['accept_rate']:.3f} "
                        f"hit={info['energy_cache_hit_rate']:.3f} "
                        f"misses_seen={int(info['energy_cache_misses_seen'])}"
                    )

                samples, summary = run_fixed_n_tree_mcmc(
                    n=int(n),
                    steps=int(steps_per_chain),
                    burnin=int(burnin),
                    thin=int(cfg.thin),
                    backend="fdm",
                    lam=1.0,
                    temperature_T=1.0,
                    seed=int(cfg.seed) + 101 * int(chain_idx) + 10_000 * int(attempts),
                    max_valence=4,
                    topology_key_fn_edges=tree_topology_edge_key_from_edges,
                    start_edges=edges0,
                    progress=None,
                    profile_every=int(cfg.profile_every),
                    step_heartbeat_every=1_000_000,
                    step_heartbeat=_heartbeat if bool(cfg.progress) else None,
                )
                chain_probs.append(dict(summary.p_topology))
                chain_seqs.append([str(s["topology"]) for s in samples])
                chain_energies.append([float(s["energy"]) for s in samples])
                cache_hits_total += int(summary.energy_cache_hits)
                cache_misses_total += int(summary.energy_cache_misses)
                proposals_total += int(summary.proposals)
                accepted_total += int(summary.accepted)
                steps_total += int(summary.steps)
                chain_steps_sec.append(float(summary.steps_per_sec))
                chain_t_move.append(float(getattr(summary, "t_move_avg", 0.0)))
                chain_t_energy.append(float(getattr(summary, "t_energy_avg", 0.0)))
                chain_t_canon.append(float(getattr(summary, "t_canon_avg", 0.0)))

            per_start_chain_seqs[str(start_spec)] = chain_seqs
            per_start_chain_energy[str(start_spec)] = chain_energies
            per_start_chain_probs.append((str(start_spec), chain_probs))

        # aggregate per-start mean across chains
        p_start_means: List[Tuple[str, Dict[str, float]]] = []
        for st, cps in per_start_chain_probs:
            keys = sorted({k for p in cps for k in p.keys()})
            p_mean = {k: float(np.mean(np.asarray([float(p.get(k, 0.0)) for p in cps], dtype=float))) for k in keys}
            s = float(sum(p_mean.values()))
            if s > 0:
                p_mean = {k: float(v) / s for k, v in p_mean.items()}
            p_start_means.append((st, p_mean))

        keys_all = sorted({k for _, p in p_start_means for k in p.keys()})
        p_eq = {k: float(np.mean(np.asarray([float(p.get(k, 0.0)) for _, p in p_start_means], dtype=float))) for k in keys_all}
        s = float(sum(p_eq.values()))
        if s > 0:
            p_eq = {k: float(v) / s for k, v in p_eq.items()}

        # mixing diagnostics per start
        md_by_start = {}
        for st in cfg.start_specs:
            md = compute_mixing_diagnostics(
                n=int(n),
                steps=int(steps_per_chain),
                burnin=int(burnin),
                thin=int(cfg.thin),
                start_spec=str(st),
                topology_sequences_by_chain=per_start_chain_seqs[str(st)],
                energy_traces_by_chain=per_start_chain_energy[str(st)],
            )
            md_by_start[str(st)] = md

        kl_max = float(np.nanmax(np.asarray([md.kl_pairwise_max for md in md_by_start.values()], dtype=float)))
        kl_mean = float(np.nanmean(np.asarray([md.kl_pairwise_mean for md in md_by_start.values()], dtype=float)))
        kl_split_max = float(np.nanmax(np.asarray([md.kl_split_max for md in md_by_start.values()], dtype=float)))
        rhat_max = float(np.nanmax(np.asarray([md.rhat_energy for md in md_by_start.values()], dtype=float)))
        ess_min = float(np.nanmin(np.asarray([md.ess_energy_min for md in md_by_start.values()], dtype=float)))

        expected = expected_unique_alkane_tree_topologies(int(n))
        coverage = float(len(p_eq)) / float(expected) if expected > 0 else 0.0

        elapsed = time.perf_counter() - t0
        end_ts = now_iso()
        cache_total = cache_hits_total + cache_misses_total
        cache_hit_rate = float(cache_hits_total) / float(cache_total) if cache_total > 0 else 0.0
        last_meta = {
            "start_ts": start_ts,
            "end_ts": end_ts,
            "elapsed_total_sec": float(elapsed),
            "eq_elapsed_sec": float(elapsed),
            "steps_total_budget": int(steps_total_budget),
            "steps_per_chain": int(steps_per_chain),
            "steps_total": int(steps_total),
            "steps_per_sec_total": float(steps_total) / float(elapsed) if elapsed > 0 else 0.0,
            "cache_hits": int(cache_hits_total),
            "cache_misses": int(cache_misses_total),
            "cache_hit_rate": float(cache_hit_rate),
            "accept_rate": float(accepted_total) / float(proposals_total) if proposals_total > 0 else 0.0,
            "kl_max_pairwise": float(kl_max),
            "kl_mean_pairwise": float(kl_mean),
            "kl_split_max": float(kl_split_max),
            "rhat_energy_max": float(rhat_max),
            "ess_energy_min": float(ess_min),
            "n_unique_eq": int(len(p_eq)),
            "expected_unique_eq": int(expected),
            "coverage_unique_eq": float(coverage),
            "t_move_avg": float(np.mean(np.asarray(chain_t_move, dtype=float))) if chain_t_move else 0.0,
            "t_energy_avg": float(np.mean(np.asarray(chain_t_energy, dtype=float))) if chain_t_energy else 0.0,
            "t_canon_avg": float(np.mean(np.asarray(chain_t_canon, dtype=float))) if chain_t_canon else 0.0,
            "mixing_by_start": md_by_start,
        }

        # DoD criteria for continuing escalation (guardrail + mixing)
        guard_ok = (kl_max <= float(cfg.guardrail_target))
        split_ok = (kl_split_max <= 0.01)
        rhat_ok = (rhat_max <= 1.05)
        ess_ok = (ess_min >= 500.0)
        coverage_ok = (coverage >= (0.90 if n == 15 else 0.85))

        if guard_ok and split_ok and rhat_ok and ess_ok and coverage_ok:
            return p_eq, last_meta, float(elapsed), float(0.0)
        if steps_total_budget >= int(cfg.max_steps):
            last_meta["FAIL"] = True
            return p_eq, last_meta, float(elapsed), float(0.0)
        steps_total_budget = min(int(cfg.max_steps), int(steps_total_budget) * 2)
        steps_per_chain = max(1, int(round(float(steps_total_budget) / float(n_chains_total))))
        burnin = int(max(burnin, round(0.1 * float(steps_per_chain))))


def write_report(
    *,
    out_stub: str,
    n: int,
    p_growth: Mapping[str, float],
    growth_counts: Mapping[str, int],
    p_eq: Mapping[str, float],
    meta: Mapping[str, object],
) -> Tuple[str, str]:
    out_txt = results_path(f"{out_stub}.txt")
    out_csv = results_path(f"{out_stub}.csv")

    # Build P_pred on P_eq support.
    topo_keys = sorted(p_eq.keys())
    n_fact = math.factorial(int(n))
    energies: Dict[str, float] = {}
    aut_sizes: Dict[str, int] = {}
    g_vals: Dict[str, int] = {}
    weights: Dict[str, float] = {}
    beta = float(meta.get("lambda_scale", 1.0)) / float(meta.get("temperature_T", 1.0))
    for topo in topo_keys:
        edges = _parse_tree_edges(str(topo))
        adj = _edges_to_adj(int(n), edges)
        feats = compute_complexity_features_v2(adj, backend="fdm")
        e = float(feats.total)
        aut = int(tree_automorphism_size(adj))
        g = int(round(float(n_fact) / float(aut)))
        energies[str(topo)] = e
        aut_sizes[str(topo)] = aut
        g_vals[str(topo)] = g
        weights[str(topo)] = float(g) * math.exp(-beta * float(e))
    z = float(sum(weights.values()))
    p_pred = {k: (float(v) / z if z > 0 else 0.0) for k, v in weights.items()}

    kl_growth_eq = float(kl_divergence(p_growth, p_eq))
    kl_eq_pred = float(kl_divergence(p_eq, p_pred))

    growth_total_sec = float(meta.get("growth_total_sec", 0.0))
    eq_total_sec = float(meta.get("eq_elapsed_sec", 0.0))
    t0_io = time.perf_counter()

    # CSV (topology-level)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "N",
            "mode",
            "topology_id",
            "P_growth",
            "P_eq",
            "P_pred",
            "count_growth",
            "energy",
            "aut_size",
            "g",
        ]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        keys = sorted(set(p_eq.keys()) | set(p_growth.keys()))
        for topo in keys:
            w.writerow(
                {
                    "N": int(n),
                    "mode": "A",
                    "topology_id": str(topo),
                    "P_growth": float(p_growth.get(topo, 0.0)),
                    "P_eq": float(p_eq.get(topo, 0.0)),
                    "P_pred": float(p_pred.get(topo, 0.0)),
                    "count_growth": int(growth_counts.get(topo, 0)),
                    "energy": float(energies.get(topo, 0.0)),
                    "aut_size": int(aut_sizes.get(topo, 0)),
                    "g": int(g_vals.get(topo, 0)),
                }
            )

    # TXT report
    lines: List[str] = []
    lines.append(f"{out_stub}: CHEM-VALIDATION-5 (growth vs equilibrium, Mode A)")
    lines.append("TOPOLOGY_KEY_INVARIANT=1 (tree:<edge-list> on canonical relabeling)")
    lines.append("")
    lines.append(f"N={n}, mode=A")
    lines.append(f"lambda_scale={float(meta.get('lambda_scale', 1.0))}  T_eff={float(meta.get('temperature_T', 1.0))}")
    lines.append(f"coverage_unique_eq={meta.get('coverage_unique_eq')}  n_unique_eq={meta.get('n_unique_eq')} expected={meta.get('expected_unique_eq')}")
    lines.append(f"Guardrail: KL_max_pairwise={meta.get('kl_max_pairwise'):.6f} KL_mean_pairwise={meta.get('kl_mean_pairwise'):.6f}")
    lines.append(f"Split: KL_split_max={meta.get('kl_split_max'):.6f}")
    lines.append(f"Rhat_energy_max={meta.get('rhat_energy_max'):.6f}  ESS_energy_min={meta.get('ess_energy_min'):.1f}")
    lines.append(f"KL(P_growth||P_eq)={kl_growth_eq:.6f}")
    lines.append(f"Self-consistency: KL(P_eq||P_pred)={kl_eq_pred:.6f}")
    if bool(meta.get('FAIL', False)):
        lines.append("FAIL: thresholds not met at max_steps")
    lines.append("")
    lines.append("TIMING")
    lines.append(f"START_TS={meta.get('start_ts')}")
    lines.append(f"END_TS={meta.get('end_ts')}")
    elapsed_total = float(meta.get("elapsed_total_sec", 0.0))
    io_total_sec = float(time.perf_counter() - t0_io)
    lines.append(f"ELAPSED_TOTAL_SEC={elapsed_total}")
    lines.append(f"ELAPSED_GROWTH_SEC={growth_total_sec}")
    lines.append(f"ELAPSED_EQ_SEC={eq_total_sec}")
    lines.append(f"ELAPSED_IO_SEC={io_total_sec}")
    lines.append(f"STEPS_TOTAL={meta.get('steps_total')} STEPS_PER_SEC_TOTAL={meta.get('steps_per_sec_total')}")
    lines.append(
        f"ENERGY_CACHE hit_rate={meta.get('cache_hit_rate'):.3f} hits={meta.get('cache_hits')} misses={meta.get('cache_misses')}"
    )
    lines.append(f"ENERGY_CACHE_MISSES_PER_CHAIN_MEAN={float(meta.get('cache_misses_per_chain_mean', 0.0)):.3f}")
    lines.append(f"Profiler: t_move_avg={meta.get('t_move_avg'):.6g}s t_energy_avg={meta.get('t_energy_avg'):.6g}s t_canon_avg={meta.get('t_canon_avg'):.6g}s")
    lines.append("")
    lines.append("Top-20 by P_eq (with metrics):")
    for topo, p in sorted(p_eq.items(), key=lambda kv: (-kv[1], kv[0]))[:20]:
        m = _tree_metrics(int(n), str(topo))
        lines.append(
            f"  {topo} P_eq={p:.6f} E={energies.get(topo, 0.0):.6f} W={m['wiener']:.0f} diam={m['diameter']:.0f} "
            f"leaves={m['n_leaves']:.0f} maxdeg={m['max_degree']:.0f}"
        )
    lines.append("")
    lines.append("Top-20 by lowest energy (with metrics):")
    for topo, e in sorted(energies.items(), key=lambda kv: (kv[1], kv[0]))[:20]:
        m = _tree_metrics(int(n), str(topo))
        lines.append(
            f"  {topo} E={e:.6f} P_eq={p_eq.get(topo, 0.0):.6f} W={m['wiener']:.0f} diam={m['diameter']:.0f} "
            f"leaves={m['n_leaves']:.0f} maxdeg={m['max_degree']:.0f}"
        )

    out_txt.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[CHEM-VALIDATION-5] wrote {out_csv}")
    print(f"[CHEM-VALIDATION-5] wrote {out_txt}")
    return str(out_csv), str(out_txt)
