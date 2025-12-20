from __future__ import annotations

import argparse
import math
import time
from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from analysis.chem.topology_mcmc import Edge, run_fixed_n_tree_mcmc, tree_topology_edge_key_from_edges
from analysis.io_utils import results_path
from analysis.utils.progress import progress_iter
from analysis.utils.timing import now_iso, timed
from analysis.growth.reporting import write_growth_txt
from core.grower import GrowthParams, grow_molecule_christmas_tree
from core.thermo_config import ThermoConfig, set_current_thermo_config
from core.tree_canonical import canonical_relabel_tree


def _backend_for_mode(mode: str) -> str:
    mode = mode.upper()
    if mode == "A":
        return "fdm"
    if mode in {"B", "C"}:
        return "fdm_entanglement"
    raise ValueError(f"Unknown mode: {mode!r}")


def _thermo_for_mode(mode: str) -> ThermoConfig:
    mode = mode.upper()
    if mode == "R":
        return ThermoConfig(
            experiment_name="CHEM-VALIDATION-1C-R",
            grower_use_mh=False,
            coupling_delta_G=0.0,
            coupling_complexity=0.0,
            coupling_topo_3d=0.0,
            coupling_shape_softness=0.0,
            coupling_shape_chi=0.0,
        )
    if mode == "A":
        return ThermoConfig(
            experiment_name="CHEM-VALIDATION-1C-A",
            grower_use_mh=True,
            coupling_delta_G=1.0,
            coupling_complexity=1.0,
            coupling_topo_3d=0.0,
            coupling_shape_softness=0.0,
            coupling_shape_chi=0.0,
            topo3d_prefilter_tree=True,
            topo3d_prefilter_min_n=10,
        )
    if mode == "B":
        return ThermoConfig(
            experiment_name="CHEM-VALIDATION-1C-B",
            grower_use_mh=True,
            coupling_delta_G=1.0,
            coupling_complexity=1.0,
            coupling_topo_3d=1.0,
            coupling_shape_softness=0.0,
            coupling_shape_chi=0.0,
            topo3d_prefilter_tree=True,
            topo3d_prefilter_min_n=10,
        )
    if mode == "C":
        return ThermoConfig(
            experiment_name="CHEM-VALIDATION-1C-C",
            grower_use_mh=True,
            coupling_delta_G=1.0,
            coupling_complexity=1.0,
            coupling_topo_3d=1.0,
            coupling_shape_softness=1.0,
            coupling_shape_chi=1.0,
            topo3d_prefilter_tree=True,
            topo3d_prefilter_min_n=10,
        )
    raise ValueError(f"Unknown mode: {mode!r}")


def _mol_to_adj(mol: object) -> np.ndarray:
    bonds = list(getattr(mol, "bonds", []))
    n = len(getattr(mol, "atoms", []))
    adj = np.zeros((n, n), dtype=float)
    for i, j in bonds:
        a = int(i)
        b = int(j)
        adj[a, b] = 1.0
        adj[b, a] = 1.0
    return adj


def _topology_id_tree(adj: np.ndarray) -> str:
    can = canonical_relabel_tree(np.asarray(adj, dtype=float))
    n = int(can.shape[0])
    edges: List[Tuple[int, int]] = []
    for i in range(n):
        for j in range(i + 1, n):
            if can[i, j] > 0:
                edges.append((i, j))
    return "tree:" + ",".join(f"{a}-{b}" for a, b in edges)


def _make_path_edges(n: int) -> List[Edge]:
    return [(int(i), int(i + 1)) for i in range(int(n) - 1)]


def _make_max_branch_edges(n: int) -> List[Edge]:
    n = int(n)
    if n <= 5:
        return [(0, i) for i in range(1, n)]
    edges: List[Edge] = [(0, 1), (0, 2), (0, 3), (0, 4)]
    last = 4
    for v in range(5, n):
        edges.append((last, v))
        last = v
    return edges


def _kl(P: Mapping[str, float], Q: Mapping[str, float], eps: float = 1e-12) -> float:
    keys = set(P.keys()) | set(Q.keys())
    p = np.asarray([float(P.get(k, 0.0)) + eps for k in keys], dtype=float)
    q = np.asarray([float(Q.get(k, 0.0)) + eps for k in keys], dtype=float)
    p = p / float(p.sum())
    q = q / float(q.sum())
    return float(np.sum(p * np.log(p / q)))


def _load_p_exact_from_csv(n: int, mode: str) -> Dict[str, float]:
    import csv

    path = results_path(f"alkane_exact_1_N{int(n)}.csv")
    if not path.exists():
        return {}
    out: Dict[str, float] = {}
    with path.open("r", newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            if int(row.get("N", 0)) != int(n):
                continue
            if str(row.get("mode", "")).upper() != str(mode).upper():
                continue
            topo = str(row.get("topology", ""))
            try:
                out[topo] = float(row.get("p_exact", 0.0))
            except Exception:
                continue
    return out


@dataclass
class Config:
    n_runs: int = 1000
    seeds: Tuple[int, ...] = (0, 1, 2, 3, 4)
    modes: Tuple[str, ...] = ("R", "A", "B", "C")
    progress: bool = True
    eq_chains: int = 3
    eq_steps: int = 5000
    eq_burnin_frac: float = 0.1
    eq_thin: int = 10
    eq_start_specs: Tuple[str, ...] = ("path", "max_branch")
    top_k: int = 20


def _parse_args(argv: Optional[Sequence[str]] = None) -> Config:
    ap = argparse.ArgumentParser(description="CHEM-VALIDATION-1C: C7 heptane skeleton (proposal vs equilibrium).")
    ap.add_argument("--n_runs", type=int, default=1000)
    ap.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2, 3, 4])
    ap.add_argument("--modes", type=str, nargs="*", default=["R", "A", "B", "C"])
    ap.add_argument("--progress", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--eq_chains", type=int, default=3)
    ap.add_argument("--eq_steps", type=int, default=5000)
    ap.add_argument("--eq_burnin_frac", type=float, default=0.1)
    ap.add_argument("--eq_thin", type=int, default=10)
    ap.add_argument("--eq_start_specs", type=str, nargs="*", default=["path", "max_branch"])
    ap.add_argument("--top_k", type=int, default=20)
    args = ap.parse_args(argv)
    return Config(
        n_runs=int(args.n_runs),
        seeds=tuple(int(x) for x in args.seeds),
        modes=tuple(str(x).upper() for x in args.modes),
        progress=bool(args.progress),
        eq_chains=int(args.eq_chains),
        eq_steps=int(args.eq_steps),
        eq_burnin_frac=float(args.eq_burnin_frac),
        eq_thin=int(args.eq_thin),
        eq_start_specs=tuple(str(x) for x in args.eq_start_specs),
        top_k=int(args.top_k),
    )


def main(argv: Optional[Sequence[str]] = None) -> None:
    cfg = _parse_args(argv)
    start_ts = now_iso()
    t0_total = time.perf_counter()

    out_stub = "chem_validation_1c_heptane"
    out_csv = results_path(f"{out_stub}.csv")
    out_txt = results_path(f"{out_stub}.txt")

    acc: Dict[str, float] = {}

    # --- Growth / proposal runs (final trees) ---
    total_runs = int(cfg.n_runs) * len(cfg.seeds) * len(cfg.modes)
    growth_counts: Dict[Tuple[str, str], int] = {}
    n_atoms = 7

    with timed("growth_total", acc):
        run_iter = progress_iter(
            range(total_runs),
            total=total_runs,
            desc="CHEM-VALIDATION-1C growth runs",
            enabled=bool(cfg.progress),
        )
        it = iter(run_iter)
        for mode in cfg.modes:
            thermo = _thermo_for_mode(mode)
            set_current_thermo_config(thermo)
            for seed in cfg.seeds:
                rng = np.random.default_rng(int(seed))
                for _ in range(int(cfg.n_runs)):
                    next(it)
                    params = GrowthParams(
                        stop_at_n_atoms=n_atoms,
                        allowed_symbols=["C"],
                        max_depth=4,
                    )
                    params.enforce_tree_alkane = True  # type: ignore[attr-defined]
                    mol = grow_molecule_christmas_tree("C", params, rng=rng)
                    adj = _mol_to_adj(mol)
                    topo = _topology_id_tree(adj)
                    growth_counts[(str(mode), topo)] = int(growth_counts.get((str(mode), topo), 0)) + 1

    # --- Equilibrium runs (fixed-N MCMC) ---
    eq_by_mode: Dict[str, Dict[str, float]] = {}
    eq_guardrail_by_mode: Dict[str, Dict[str, float]] = {}
    eq_metrics_by_mode: Dict[str, Dict[str, float]] = {}

    def _edges_for_start(spec: str) -> List[Edge]:
        if spec == "path":
            return _make_path_edges(7)
        if spec == "max_branch":
            return _make_max_branch_edges(7)
        raise ValueError(f"Unknown eq_start_spec: {spec!r}")

    with timed("eq_total", acc):
        for mode in [m for m in cfg.modes if m != "R"]:
            mode_t0 = time.perf_counter()
            steps_total = int(cfg.eq_steps) * int(cfg.eq_chains) * max(1, len(cfg.eq_start_specs))
            cache_hits_total = 0
            cache_misses_total = 0
            backend = _backend_for_mode(mode)
            burnin = int(max(0, round(float(cfg.eq_burnin_frac) * float(cfg.eq_steps))))
            per_start: List[Tuple[str, Dict[str, float], float, float]] = []
            for start_spec in cfg.eq_start_specs:
                chain_probs: List[Dict[str, float]] = []
                chain_hit_rates: List[float] = []
                chain_steps_sec: List[float] = []
                for chain_idx in progress_iter(
                    range(int(cfg.eq_chains)),
                    total=int(cfg.eq_chains),
                    desc=f"CHEM-VALIDATION-1C EQ mode{mode} start={start_spec}",
                    enabled=bool(cfg.progress),
                ):
                    edges0 = _edges_for_start(start_spec)
                    _, summary = run_fixed_n_tree_mcmc(
                        n=7,
                        steps=int(cfg.eq_steps),
                        burnin=int(burnin),
                        thin=int(cfg.eq_thin),
                        backend=str(backend),
                        lam=1.0,
                        temperature_T=1.0,
                        seed=12345 + 101 * int(chain_idx),
                        max_valence=4,
                        topology_key_fn_edges=tree_topology_edge_key_from_edges,
                        start_edges=edges0,
                        progress=None,
                    )
                    chain_probs.append(dict(summary.p_topology))
                    chain_hit_rates.append(float(summary.energy_cache_hit_rate))
                    chain_steps_sec.append(float(summary.steps_per_sec))
                    cache_hits_total += int(summary.energy_cache_hits)
                    cache_misses_total += int(summary.energy_cache_misses)

                keys: List[str] = []
                for p in chain_probs:
                    for k in p.keys():
                        if k not in keys:
                            keys.append(k)
                p_mean: Dict[str, float] = {}
                for k in keys:
                    p_mean[k] = float(np.mean(np.asarray([float(p.get(k, 0.0)) for p in chain_probs], dtype=float)))
                s = float(sum(p_mean.values()))
                if s > 0:
                    p_mean = {k: float(v) / s for k, v in p_mean.items()}
                per_start.append(
                    (
                        str(start_spec),
                        p_mean,
                        float(np.mean(np.asarray(chain_hit_rates, dtype=float))),
                        float(np.mean(np.asarray(chain_steps_sec, dtype=float))),
                    )
                )

            # Aggregate P_eq as mean of start-specific means (guardrail remains explicit).
            keys_all = sorted({k for _, p, _, _ in per_start for k in p.keys()})
            p_eq: Dict[str, float] = {k: float(np.mean(np.asarray([p.get(k, 0.0) for _, p, _, _ in per_start], dtype=float))) for k in keys_all}
            s = float(sum(p_eq.values()))
            if s > 0:
                p_eq = {k: float(v) / s for k, v in p_eq.items()}

            # Guardrail: pairwise KL between starts.
            kls: List[float] = []
            for i in range(len(per_start)):
                for j in range(len(per_start)):
                    if i == j:
                        continue
                    kls.append(_kl(per_start[i][1], per_start[j][1]))
            eq_by_mode[str(mode)] = p_eq
            eq_guardrail_by_mode[str(mode)] = {
                "kl_max_pairwise": float(max(kls)) if kls else float("nan"),
                "kl_mean_pairwise": float(np.mean(np.asarray(kls, dtype=float))) if kls else float("nan"),
                "cache_hit_rate_mean": float(np.mean(np.asarray([x[2] for x in per_start], dtype=float))) if per_start else 0.0,
                "steps_per_sec_mean": float(np.mean(np.asarray([x[3] for x in per_start], dtype=float))) if per_start else 0.0,
            }
            mode_elapsed = float(time.perf_counter() - mode_t0)
            cache_total = cache_hits_total + cache_misses_total
            eq_metrics_by_mode[str(mode)] = {
                "eq_elapsed_sec": float(mode_elapsed),
                "steps_total": float(steps_total),
                "steps_per_sec_total": float(steps_total) / float(mode_elapsed) if mode_elapsed > 0 else 0.0,
                "energy_cache_hits": float(cache_hits_total),
                "energy_cache_misses": float(cache_misses_total),
                "energy_cache_hit_rate": (float(cache_hits_total) / float(cache_total)) if cache_total > 0 else 0.0,
            }

    # --- Scoring (optional, for sample) ---
    with timed("scoring_total", acc):
        # No per-run scoring here; EQ/growth already evaluate energy during MH.
        pass

    # --- IO ---
    rows: List[Dict[str, object]] = []
    with timed("io_total", acc):
        for (mode, topo), cnt in sorted(growth_counts.items()):
            rows.append({"mode": mode, "topology": topo, "count": int(cnt)})
        import csv

        out_csv.parent.mkdir(parents=True, exist_ok=True)
        with out_csv.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["mode", "topology", "count"])
            w.writeheader()
            for r in rows:
                w.writerow(r)

        lines: List[str] = []
        lines.append("CHEM-VALIDATION-1C: C7 heptane skeleton (proposal vs equilibrium)")
        lines.append(f"Config: n_runs={cfg.n_runs}, seeds={list(cfg.seeds)}, modes={list(cfg.modes)}")
        lines.append(f"EQ budget: eq_steps={cfg.eq_steps}, eq_chains={cfg.eq_chains}, starts={list(cfg.eq_start_specs)}")
        lines.append(f"Reporting: top_k={cfg.top_k}")
        lines.append("")

        # Mode-consistent reporting: per-mode growth + eq + KL + guardrail + unique + mass_topK.
        for mode in [m for m in cfg.modes if m != "R"]:
            mode_counts = {topo: int(cnt) for (m, topo), cnt in growth_counts.items() if m == mode}
            total = float(sum(mode_counts.values())) if mode_counts else 0.0
            p_growth = {k: (float(v) / total if total > 0 else 0.0) for k, v in mode_counts.items()}
            n_unique_growth = int(len(p_growth))

            p_eq = eq_by_mode.get(mode, {})
            n_unique_eq = int(len(p_eq))
            p_exact = _load_p_exact_from_csv(n_atoms, mode)
            n_unique_exact = int(len(p_exact)) if p_exact else 0

            lines.append(f"[Mode {mode}]")
            if p_exact:
                lines.append(
                    f"  n_unique_growth={n_unique_growth}, n_unique_eq={n_unique_eq}, n_unique_exact={n_unique_exact}"
                )
            else:
                lines.append(f"  n_unique_growth={n_unique_growth}, n_unique_eq={n_unique_eq}")

            lines.append("  P_growth(topology):")
            growth_sorted = sorted(mode_counts.items(), key=lambda kv: (-kv[1], kv[0]))
            top_k = max(1, int(cfg.top_k))
            top_growth = growth_sorted[:top_k]
            mass_topk_growth = 0.0
            for topo, cnt in top_growth:
                p = float(cnt) / total if total > 0 else 0.0
                mass_topk_growth += float(p)
                lines.append(f"    {topo} = {p:.6f} (count={cnt})")
            lines.append(f"  mass_topK_growth={mass_topk_growth:.6f}")

            lines.append("  P_eq(topology):")
            eq_sorted = sorted(p_eq.items(), key=lambda kv: (-float(kv[1]), kv[0]))
            top_eq = eq_sorted[:top_k]
            mass_topk_eq = float(sum(float(p) for _, p in top_eq))
            for topo, p in top_eq:
                lines.append(f"    {topo} = {float(p):.6f}")
            lines.append(f"  mass_topK_eq={mass_topk_eq:.6f}")

            lines.append(f"  KL(P_growth||P_eq) = {_kl(p_growth, p_eq):.6f}")
            if p_exact:
                lines.append(f"  KL(P_eq||P_exact) = {_kl(p_eq, p_exact):.6f}")
                lines.append(f"  KL(P_growth||P_exact) = {_kl(p_growth, p_exact):.6f}")
            guard = eq_guardrail_by_mode.get(mode, {})
            lines.append(
                f"  Guardrail: KL_max_pairwise={guard.get('kl_max_pairwise', float('nan')):.6f}, "
                f"KL_mean_pairwise={guard.get('kl_mean_pairwise', float('nan')):.6f}"
            )
            eqm = eq_metrics_by_mode.get(mode, {})
            lines.append(
                f"  EQ_METRICS: ELAPSED_EQ_SEC={eqm.get('eq_elapsed_sec', 0.0):.6f}, "
                f"STEPS_TOTAL={int(eqm.get('steps_total', 0.0))}, "
                f"STEPS_PER_SEC_TOTAL={eqm.get('steps_per_sec_total', 0.0):.1f}"
            )
            lines.append(
                f"  ENERGY-CACHE: hit_rate={eqm.get('energy_cache_hit_rate', 0.0):.3f}, "
                f"hits={int(eqm.get('energy_cache_hits', 0.0))}, misses={int(eqm.get('energy_cache_misses', 0.0))}"
            )
            if p_exact:
                eps = 1e-12
                bias_rows: List[Tuple[str, float, float, float]] = []
                for topo in sorted(set(p_growth.keys()) | set(p_exact.keys())):
                    pg = float(p_growth.get(topo, 0.0))
                    pe = float(p_exact.get(topo, 0.0))
                    bias = float(math.log((pg + eps) / (pe + eps)))
                    bias_rows.append((topo, bias, pg, pe))
                bias_rows.sort(key=lambda x: (-abs(x[1]), x[0]))
                lines.append("  Bias log(P_growth/P_exact):")
                for topo, b, pg, pe in bias_rows[:top_k]:
                    lines.append(f"    {topo}: bias={b:+.6f} P_growth={pg:.6f} P_exact={pe:.6f}")
            lines.append("")

        elapsed_total = time.perf_counter() - t0_total
        end_ts = now_iso()
        lines.append("TIMING")
        lines.append(f"START_TS={start_ts}")
        lines.append(f"END_TS={end_ts}")
        lines.append(f"ELAPSED_TOTAL_SEC={elapsed_total:.6f}")
        lines.append(
            "BREAKDOWN_SEC="
            + ",".join(f"{k}={float(v):.6f}" for k, v in sorted(acc.items(), key=lambda kv: kv[0]))
        )
        write_growth_txt(out_stub, lines)

    elapsed_total = time.perf_counter() - t0_total
    end_ts = now_iso()
    print(f"[CHEM-VALIDATION-1C] wrote {out_csv}")
    print(f"[CHEM-VALIDATION-1C] wrote {out_txt}")
    print(f"Wall-clock: start={start_ts} end={end_ts} elapsed_total_sec={elapsed_total:.3f}")


if __name__ == "__main__":
    main()
