from __future__ import annotations

import argparse
import csv
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from analysis.chem.core2_fit import compute_p_pred, fit_lambda, kl_divergence
from analysis.io_utils import results_path
from analysis.utils.progress import progress_iter
from analysis.growth.rng import make_rng
from analysis.growth.reporting import write_growth_txt
from core.complexity import compute_complexity_features_v2
from core.grower import GrowthParams, grow_molecule_christmas_tree
from core.thermo_config import ThermoConfig, override_thermo_config
from analysis.chem.core2_fit import kl_divergence


HEXANE_DEGENERACY: Dict[str, int] = {
    "n_hexane": 360,
    "2_methylpentane": 360,
    "3_methylpentane": 360,
    "2,2_dimethylbutane": 120,
    "2,3_dimethylbutane": 90,
}


def _adj_to_shortest_path_lengths(adj: np.ndarray) -> np.ndarray:
    n = int(adj.shape[0])
    dist = np.full((n, n), -1, dtype=int)
    for s in range(n):
        dist[s, s] = 0
        q: List[int] = [s]
        while q:
            u = q.pop(0)
            nbrs = np.where(adj[u] > 0)[0]
            for v in nbrs:
                v = int(v)
                if dist[s, v] < 0:
                    dist[s, v] = dist[s, u] + 1
                    q.append(v)
    return dist


def classify_hexane_topology(adj: np.ndarray) -> str:
    """
    Classify a C6 tree topology by degree multiset and simple invariants.

    Returns:
      - n_hexane
      - 2_methylpentane
      - 3_methylpentane
      - 2,2_dimethylbutane
      - 2,3_dimethylbutane
      - other
    """
    if adj.ndim != 2 or adj.shape[0] != adj.shape[1]:
        return "other"
    if adj.shape[0] != 6:
        return "other"

    deg = np.asarray(adj.sum(axis=1), dtype=int)
    deg_sorted = tuple(sorted(int(x) for x in deg))

    if deg_sorted == (1, 1, 2, 2, 2, 2):
        return "n_hexane"
    if deg_sorted == (1, 1, 1, 1, 2, 4):
        return "2,2_dimethylbutane"
    if deg_sorted == (1, 1, 1, 1, 3, 3):
        return "2,3_dimethylbutane"
    if deg_sorted == (1, 1, 1, 2, 2, 3):
        deg2_nodes = np.where(deg == 2)[0].tolist()
        if len(deg2_nodes) != 2:
            return "other"
        dist = _adj_to_shortest_path_lengths(adj)
        d = int(dist[int(deg2_nodes[0]), int(deg2_nodes[1])])
        if d == 1:
            return "2_methylpentane"
        if d == 2:
            return "3_methylpentane"
        return "other"
    return "other"


def _make_thermo_for_mode(mode: str) -> ThermoConfig:
    base = ThermoConfig()
    mode = mode.upper()

    if mode == "R":
        return ThermoConfig(
            temperature=base.temperature,
            coupling_delta_F=0.0,
            coupling_complexity=0.0,
            coupling_softness=0.0,
            coupling_density=0.0,
            coupling_density_shape=0.0,
            coupling_port_geometry=0.0,
            coupling_ws_Z=0.0,
            coupling_shape_softness=0.0,
            coupling_shape_chi=0.0,
            coupling_topo_3d=0.0,
            coupling_delta_G=0.0,
            density_model=base.density_model,
            density_blend=base.density_blend,
            density_Z_ref=base.density_Z_ref,
            density_source=base.density_source,
            ws_R_max=base.ws_R_max,
            ws_R_well=base.ws_R_well,
            ws_V0=base.ws_V0,
            ws_N_grid=base.ws_N_grid,
            ws_ell=base.ws_ell,
            ws_state_index=base.ws_state_index,
            port_geometry_source=base.port_geometry_source,
            port_geometry_blend=base.port_geometry_blend,
            ws_geom_R_max=base.ws_geom_R_max,
            ws_geom_R_well=base.ws_geom_R_well,
            ws_geom_V0=base.ws_geom_V0,
            ws_geom_N_grid=base.ws_geom_N_grid,
            ws_geom_gap_scale=base.ws_geom_gap_scale,
            ws_geom_gap_ref=base.ws_geom_gap_ref,
            ws_Z_ref=base.ws_Z_ref,
            ws_Z_alpha=base.ws_Z_alpha,
            shape_kurt_scale=base.shape_kurt_scale,
            shape_rrms_scale=base.shape_rrms_scale,
            shape_softness_gain=base.shape_softness_gain,
            shape_chi_gain=base.shape_chi_gain,
            topo_3d_beta=base.topo_3d_beta,
            ws_integrator="fdm",
            ws_fdm_depth=base.ws_fdm_depth,
            ws_fdm_base=base.ws_fdm_base,
            topo3d_prefilter_tree=True,
            topo3d_prefilter_min_n=10,
            temperature_T=1.0,
            grower_use_mh=False,
            deltaG_backend="fdm_entanglement",
            consume_port_on_reject=True,
            max_attempts_per_port=1,
            grower_proposal_policy="uniform",
            proposal_beta=0.0,
            proposal_ports_gamma=0.0,
            experiment_name="chem_validation_1b_mode_R",
        )

    if mode == "A":
        return ThermoConfig(
            temperature=base.temperature,
            coupling_delta_F=0.0,
            coupling_complexity=1.0,
            coupling_softness=0.0,
            coupling_density=0.0,
            coupling_density_shape=0.0,
            coupling_port_geometry=0.0,
            coupling_ws_Z=0.0,
            coupling_shape_softness=0.0,
            coupling_shape_chi=0.0,
            coupling_topo_3d=0.0,
            coupling_delta_G=1.0,
            density_model=base.density_model,
            density_blend=base.density_blend,
            density_Z_ref=base.density_Z_ref,
            density_source=base.density_source,
            ws_R_max=base.ws_R_max,
            ws_R_well=base.ws_R_well,
            ws_V0=base.ws_V0,
            ws_N_grid=base.ws_N_grid,
            ws_ell=base.ws_ell,
            ws_state_index=base.ws_state_index,
            port_geometry_source=base.port_geometry_source,
            port_geometry_blend=base.port_geometry_blend,
            ws_geom_R_max=base.ws_geom_R_max,
            ws_geom_R_well=base.ws_geom_R_well,
            ws_geom_V0=base.ws_geom_V0,
            ws_geom_N_grid=base.ws_geom_N_grid,
            ws_geom_gap_scale=base.ws_geom_gap_scale,
            ws_geom_gap_ref=base.ws_geom_gap_ref,
            ws_Z_ref=base.ws_Z_ref,
            ws_Z_alpha=base.ws_Z_alpha,
            shape_kurt_scale=base.shape_kurt_scale,
            shape_rrms_scale=base.shape_rrms_scale,
            shape_softness_gain=base.shape_softness_gain,
            shape_chi_gain=base.shape_chi_gain,
            topo_3d_beta=base.topo_3d_beta,
            ws_integrator="fdm",
            ws_fdm_depth=base.ws_fdm_depth,
            ws_fdm_base=base.ws_fdm_base,
            topo3d_prefilter_tree=True,
            topo3d_prefilter_min_n=10,
            temperature_T=1.0,
            grower_use_mh=True,
            deltaG_backend="fdm_entanglement",
            consume_port_on_reject=False,
            max_attempts_per_port=50,
            grower_proposal_policy="uniform",
            proposal_beta=0.0,
            proposal_ports_gamma=0.0,
            experiment_name="chem_validation_1b_mode_A",
        )

    if mode == "B":
        return ThermoConfig(
            temperature=base.temperature,
            coupling_delta_F=0.0,
            coupling_complexity=1.0,
            coupling_softness=0.0,
            coupling_density=0.0,
            coupling_density_shape=0.0,
            coupling_port_geometry=0.0,
            coupling_ws_Z=0.0,
            coupling_shape_softness=0.0,
            coupling_shape_chi=0.0,
            coupling_topo_3d=1.0,
            coupling_delta_G=1.0,
            density_model=base.density_model,
            density_blend=base.density_blend,
            density_Z_ref=base.density_Z_ref,
            density_source=base.density_source,
            ws_R_max=base.ws_R_max,
            ws_R_well=base.ws_R_well,
            ws_V0=base.ws_V0,
            ws_N_grid=base.ws_N_grid,
            ws_ell=base.ws_ell,
            ws_state_index=base.ws_state_index,
            port_geometry_source=base.port_geometry_source,
            port_geometry_blend=base.port_geometry_blend,
            ws_geom_R_max=base.ws_geom_R_max,
            ws_geom_R_well=base.ws_geom_R_well,
            ws_geom_V0=base.ws_geom_V0,
            ws_geom_N_grid=base.ws_geom_N_grid,
            ws_geom_gap_scale=base.ws_geom_gap_scale,
            ws_geom_gap_ref=base.ws_geom_gap_ref,
            ws_Z_ref=base.ws_Z_ref,
            ws_Z_alpha=base.ws_Z_alpha,
            shape_kurt_scale=base.shape_kurt_scale,
            shape_rrms_scale=base.shape_rrms_scale,
            shape_softness_gain=base.shape_softness_gain,
            shape_chi_gain=base.shape_chi_gain,
            topo_3d_beta=base.topo_3d_beta,
            ws_integrator="fdm",
            ws_fdm_depth=base.ws_fdm_depth,
            ws_fdm_base=base.ws_fdm_base,
            topo3d_prefilter_tree=True,
            topo3d_prefilter_min_n=10,
            temperature_T=1.0,
            grower_use_mh=True,
            deltaG_backend="fdm_entanglement",
            consume_port_on_reject=False,
            max_attempts_per_port=50,
            grower_proposal_policy="uniform",
            proposal_beta=0.0,
            proposal_ports_gamma=0.0,
            experiment_name="chem_validation_1b_mode_B",
        )

    if mode == "C":
        return ThermoConfig(
            temperature=base.temperature,
            coupling_delta_F=0.0,
            coupling_complexity=1.0,
            coupling_softness=0.0,
            coupling_density=0.0,
            coupling_density_shape=0.0,
            coupling_port_geometry=0.0,
            coupling_ws_Z=0.0,
            coupling_shape_softness=1.0,
            coupling_shape_chi=1.0,
            coupling_topo_3d=1.0,
            coupling_delta_G=1.0,
            density_model=base.density_model,
            density_blend=base.density_blend,
            density_Z_ref=base.density_Z_ref,
            density_source=base.density_source,
            ws_R_max=base.ws_R_max,
            ws_R_well=base.ws_R_well,
            ws_V0=base.ws_V0,
            ws_N_grid=base.ws_N_grid,
            ws_ell=base.ws_ell,
            ws_state_index=base.ws_state_index,
            port_geometry_source=base.port_geometry_source,
            port_geometry_blend=base.port_geometry_blend,
            ws_geom_R_max=base.ws_geom_R_max,
            ws_geom_R_well=base.ws_geom_R_well,
            ws_geom_V0=base.ws_geom_V0,
            ws_geom_N_grid=base.ws_geom_N_grid,
            ws_geom_gap_scale=base.ws_geom_gap_scale,
            ws_geom_gap_ref=base.ws_geom_gap_ref,
            ws_Z_ref=base.ws_Z_ref,
            ws_Z_alpha=base.ws_Z_alpha,
            shape_kurt_scale=base.shape_kurt_scale,
            shape_rrms_scale=base.shape_rrms_scale,
            shape_softness_gain=base.shape_softness_gain,
            shape_chi_gain=base.shape_chi_gain,
            topo_3d_beta=base.topo_3d_beta,
            ws_integrator="fdm",
            ws_fdm_depth=base.ws_fdm_depth,
            ws_fdm_base=base.ws_fdm_base,
            topo3d_prefilter_tree=True,
            topo3d_prefilter_min_n=10,
            temperature_T=1.0,
            grower_use_mh=True,
            deltaG_backend="fdm_entanglement",
            consume_port_on_reject=False,
            max_attempts_per_port=50,
            grower_proposal_policy="uniform",
            proposal_beta=0.0,
            proposal_ports_gamma=0.0,
            experiment_name="chem_validation_1b_mode_C",
        )

    raise ValueError(f"Unsupported CHEM-VALIDATION-1B mode: {mode!r}")


@dataclass
class ChemValidation1BConfig:
    n_runs: int = 1000
    seeds: Tuple[int, ...] = (0, 1, 2, 3, 4)
    modes: Tuple[str, ...] = ("R", "A", "B", "C")
    max_depth: int = 6
    equilibrate_steps: int = 0
    equilibrate_burnin: int = 0
    equilibrate_thin: int = 1
    progress: bool = True


def _make_reference_adjs() -> Dict[str, np.ndarray]:
    refs: Dict[str, np.ndarray] = {}

    # n-hexane: path 0-1-2-3-4-5
    adj_n = np.zeros((6, 6), dtype=float)
    for a, b in ((0, 1), (1, 2), (2, 3), (3, 4), (4, 5)):
        adj_n[a, b] = adj_n[b, a] = 1.0
    refs["n_hexane"] = adj_n

    # 2-methylpentane: main chain 0-1-2-3-4, methyl at 1 (node 5)
    adj_2m = np.zeros((6, 6), dtype=float)
    for a, b in ((0, 1), (1, 2), (2, 3), (3, 4), (1, 5)):
        adj_2m[a, b] = adj_2m[b, a] = 1.0
    refs["2_methylpentane"] = adj_2m

    # 3-methylpentane: main chain 0-1-2-3-4, methyl at 2 (node 5)
    adj_3m = np.zeros((6, 6), dtype=float)
    for a, b in ((0, 1), (1, 2), (2, 3), (3, 4), (2, 5)):
        adj_3m[a, b] = adj_3m[b, a] = 1.0
    refs["3_methylpentane"] = adj_3m

    # 2,2-dimethylbutane: chain 0-1-2-3, two methyl at 1 (4,5)
    adj_22 = np.zeros((6, 6), dtype=float)
    for a, b in ((0, 1), (1, 2), (2, 3), (1, 4), (1, 5)):
        adj_22[a, b] = adj_22[b, a] = 1.0
    refs["2,2_dimethylbutane"] = adj_22

    # 2,3-dimethylbutane: chain 0-1-2-3, methyl at 1 (4) and at 2 (5)
    adj_23 = np.zeros((6, 6), dtype=float)
    for a, b in ((0, 1), (1, 2), (2, 3), (1, 4), (2, 5)):
        adj_23[a, b] = adj_23[b, a] = 1.0
    refs["2,3_dimethylbutane"] = adj_23

    return refs


def _load_p_exact_hexane_mode_a() -> Optional[Dict[str, float]]:
    """
    Parse P_exact(topology) from MH-KERNEL-3 exact report (if present).
    """
    try:
        txt = results_path("mh_kernel_3_c6_exact_modeA.txt").read_text(encoding="utf-8", errors="replace")
    except Exception:
        return None
    if "P_exact(topology):" not in txt:
        return None
    tail = txt.split("P_exact(topology):", 1)[1]
    block = tail.split("\n\n", 1)[0]
    out: Dict[str, float] = {}
    for line in block.splitlines():
        line = line.strip()
        if not line or "=" not in line:
            continue
        k, v = [x.strip() for x in line.split("=", 1)]
        try:
            out[k] = float(v)
        except Exception:
            continue
    return out or None


def run_chem_validation_1b(cfg: ChemValidation1BConfig) -> Tuple[Path, Path]:
    t_start = time.perf_counter()
    all_rows: List[Dict[str, object]] = []
    by_mode: Dict[str, List[Dict[str, object]]] = {m.upper(): [] for m in cfg.modes}

    total_runs = int(cfg.n_runs) * len(cfg.seeds) * len(cfg.modes)
    run_plan: List[Tuple[str, int, int]] = []
    for mode in cfg.modes:
        for seed in cfg.seeds:
            for run_idx in range(cfg.n_runs):
                run_plan.append((str(mode).upper(), int(seed), int(run_idx)))

    rng_by_mode_seed: Dict[Tuple[str, int], np.random.Generator] = {}
    for mode in cfg.modes:
        m = str(mode).upper()
        for seed in cfg.seeds:
            base_rng = make_rng(f"chem_validation_1b_hexane_{m}_{seed}")
            rng_by_mode_seed[(m, int(seed))] = np.random.default_rng(
                base_rng.integers(0, 2**32 - 1)
            )

    current_mode: Optional[str] = None
    current_ctx = None

    for mode, seed, run_idx in progress_iter(
        run_plan, total=total_runs, desc="CHEM-VALIDATION-1B", enabled=bool(cfg.progress)
    ):
        if mode != current_mode:
            if current_ctx is not None:
                current_ctx.__exit__(None, None, None)
            current_mode = mode
            thermo = _make_thermo_for_mode(mode)
            current_ctx = override_thermo_config(thermo)
            current_ctx.__enter__()

        rng = rng_by_mode_seed[(mode, seed)]
        params = GrowthParams(
            max_depth=cfg.max_depth,
            max_atoms=20,
            stop_at_n_atoms=6,
            allowed_symbols=["C"],
            enforce_tree_alkane=True,
            equilibrate_fixed_n_steps=int(cfg.equilibrate_steps),
            equilibrate_burnin=int(cfg.equilibrate_burnin),
            equilibrate_thin=int(cfg.equilibrate_thin),
            equilibrate_max_degree=4,
            p_continue_base=1.0,
            chi_sensitivity=0.0,
            role_bonus_hub=0.0,
            role_penalty_terminator=0.0,
            temperature=1.0,
        )

        t0 = time.perf_counter()
        mol = grow_molecule_christmas_tree("C", params, rng=rng)
        bonds_before = getattr(mol, "bonds_before_eq", None)
        if isinstance(bonds_before, list) and bonds_before:
            n = len(mol.atoms)
            adj_before = np.zeros((n, n), dtype=float)
            for i, j in bonds_before:
                a = int(i)
                b = int(j)
                adj_before[a, b] = 1.0
                adj_before[b, a] = 1.0
        else:
            adj_before = mol.adjacency_matrix()

        adj = mol.adjacency_matrix()
        growth_sec = time.perf_counter() - t0

        topology_before = classify_hexane_topology(adj_before) if adj_before.shape == (6, 6) else "other"
        topology_after = classify_hexane_topology(adj) if adj.shape == (6, 6) else "other"

        eq_stats = getattr(mol, "eq_stats", None)
        if isinstance(eq_stats, dict):
            eq_steps = int(eq_stats.get("eq_steps", 0))
            eq_accept_rate = float(eq_stats.get("eq_accept_rate", 0.0))
            eq_moves_mean = float(eq_stats.get("eq_moves_mean", 0.0))
        else:
            eq_steps = 0
            eq_accept_rate = 0.0
            eq_moves_mean = 0.0

        if adj.shape != (6, 6):
            topology = "other"
            deg_sorted = str([])
            feats_fdm = compute_complexity_features_v2(adj, backend="fdm")
            feats_ent = compute_complexity_features_v2(adj, backend="fdm_entanglement")
            score_fdm = float(feats_fdm.total)
            score_topo3d = float(feats_ent.total) - float(feats_fdm.total)
            score_shape = 0.0
            score_total = score_fdm + score_topo3d + score_shape
            mh_proposals = 0
            mh_accepted = 0
            mh_accept_rate = 0.0
            scoring_sec = 0.0
        else:
            deg = np.asarray(adj.sum(axis=1), dtype=int)
            deg_sorted_tuple = tuple(sorted(int(x) for x in deg))
            deg_sorted = str(list(deg_sorted_tuple))
            topology = classify_hexane_topology(adj)

            t1 = time.perf_counter()
            feats_fdm = compute_complexity_features_v2(adj, backend="fdm")
            feats_ent = compute_complexity_features_v2(adj, backend="fdm_entanglement")
            scoring_sec = time.perf_counter() - t1

            score_fdm = float(feats_fdm.total)
            score_topo3d = float(feats_ent.total) - float(feats_fdm.total)
            score_shape = 0.0
            score_total = score_fdm if mode == "R" else (score_fdm + score_topo3d + score_shape)

            mh_stats = getattr(mol, "mh_stats", None)
            if isinstance(mh_stats, dict):
                mh_proposals = int(mh_stats.get("proposals", 0))
                mh_accepted = int(mh_stats.get("accepted", 0))
            else:
                mh_proposals = 0
                mh_accepted = 0
            mh_accept_rate = (mh_accepted / float(mh_proposals)) if mh_proposals > 0 else 0.0

        row = {
            "mode": mode,
            "seed": seed,
            "run_idx": run_idx,
            "topology_before_eq": str(topology_before),
            "topology_after_eq": str(topology_after),
            "topology": topology,
            "deg_sorted": deg_sorted,
            "score_fdm": float(score_fdm),
            "score_topo3d": float(score_topo3d),
            "score_shape": float(score_shape),
            "score_total": float(score_total),
            "mh_proposals": int(mh_proposals),
            "mh_accepted": int(mh_accepted),
            "mh_accept_rate": float(mh_accept_rate),
            "eq_steps": int(eq_steps),
            "eq_accept_rate": float(eq_accept_rate),
            "eq_moves_mean": float(eq_moves_mean),
            "runtime_growth_sec": float(growth_sec),
            "runtime_scoring_sec": float(scoring_sec),
            "runtime_total_sec": float(growth_sec + scoring_sec),
        }
        by_mode[mode].append(row)
        all_rows.append(row)

    if current_ctx is not None:
        current_ctx.__exit__(None, None, None)

    csv_path = results_path("chem_validation_1b_hexane.csv")
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "mode",
        "seed",
        "run_idx",
        "topology_before_eq",
        "topology_after_eq",
        "topology",
        "deg_sorted",
        "score_fdm",
        "score_topo3d",
        "score_shape",
        "score_total",
        "mh_proposals",
        "mh_accepted",
        "mh_accept_rate",
        "eq_steps",
        "eq_accept_rate",
        "eq_moves_mean",
        "runtime_growth_sec",
        "runtime_scoring_sec",
        "runtime_total_sec",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in all_rows:
            w.writerow({k: r.get(k, "") for k in fieldnames})

    elapsed = time.perf_counter() - t_start
    lines: List[str] = []
    lines.append("CHEM-VALIDATION-1B: C6 hexane skeleton (tree-only)")
    lines.append("")
    lines.append(f"Config: n_runs={cfg.n_runs}, seeds={list(cfg.seeds)}, modes={list(cfg.modes)}")
    lines.append(
        f"Equilibrate: steps={cfg.equilibrate_steps}, burnin={cfg.equilibrate_burnin}, thin={cfg.equilibrate_thin}"
    )
    lines.append(f"elapsed_sec={elapsed:.3f}, runs_done={len(all_rows)}")
    lines.append("")

    g_sum = sum(int(v) for v in HEXANE_DEGENERACY.values())
    lines.append("CORE-1: degeneracy table g(topology)=6!/|Aut|")
    for k in sorted(HEXANE_DEGENERACY.keys()):
        lines.append(f"  g({k})={HEXANE_DEGENERACY[k]}")
    lines.append(f"  sum_g={g_sum} (sanity: 1290 for all labeled non-star C6 trees)")
    lines.append("")

    # Aggregate per mode.
    for mode in cfg.modes:
        m = mode.upper()
        rows = by_mode.get(m, [])
        if not rows:
            continue
        lines.append(f"[Mode {m}]")
        topo_counts: Dict[str, int] = {}
        topo_counts_growth: Dict[str, int] = {}
        topo_counts_eq: Dict[str, int] = {}
        scores_by_topo: Dict[str, List[float]] = {}
        mh_by_topo: Dict[str, Tuple[int, int]] = {}
        eq_steps_vals: List[int] = []
        eq_moves_vals: List[float] = []
        eq_accept_vals: List[float] = []
        growth_times: List[float] = []
        scoring_times: List[float] = []
        for r in rows:
            topo = str(r["topology"])
            topo_counts[topo] = topo_counts.get(topo, 0) + 1
            topo_g = str(r.get("topology_before_eq", topo))
            topo_e = str(r.get("topology_after_eq", topo))
            topo_counts_growth[topo_g] = topo_counts_growth.get(topo_g, 0) + 1
            topo_counts_eq[topo_e] = topo_counts_eq.get(topo_e, 0) + 1
            scores_by_topo.setdefault(topo, []).append(float(r["score_total"]))
            p = int(r.get("mh_proposals", 0))
            a = int(r.get("mh_accepted", 0))
            old_p, old_a = mh_by_topo.get(topo, (0, 0))
            mh_by_topo[topo] = (old_p + p, old_a + a)
            eq_steps_vals.append(int(r.get("eq_steps", 0)))
            eq_moves_vals.append(float(r.get("eq_moves_mean", 0.0)))
            eq_accept_vals.append(float(r.get("eq_accept_rate", 0.0)))
            growth_times.append(float(r.get("runtime_growth_sec", 0.0)))
            scoring_times.append(float(r.get("runtime_scoring_sec", 0.0)))

        total = sum(topo_counts.values()) or 1
        for topo in sorted(topo_counts.keys()):
            lines.append(f"  P({topo}) = {topo_counts[topo] / float(total):.4f} (count={topo_counts[topo]})")

        if any(int(x) > 0 for x in eq_steps_vals):
            lines.append("  Proposal vs equilibrated:")
            for topo in sorted(set(list(topo_counts_growth.keys()) + list(topo_counts_eq.keys()))):
                pg = topo_counts_growth.get(topo, 0) / float(total)
                pe = topo_counts_eq.get(topo, 0) / float(total)
                lines.append(f"    P_growth({topo})={pg:.4f}, P_equilibrated({topo})={pe:.4f}")
            lines.append(
                f"  EQ stats: mean_accept_rate={float(np.mean(eq_accept_vals)):.4f}, mean_moves={float(np.mean(eq_moves_vals)):.3f}"
            )

        for topo, vals in scores_by_topo.items():
            arr = np.asarray(vals, dtype=float)
            lines.append(f"  score_total[{topo}]: median={float(np.median(arr)):.4f}, p90={float(np.percentile(arr, 90)):.4f}")

        for topo in sorted(mh_by_topo.keys()):
            p, a = mh_by_topo[topo]
            if p > 0:
                lines.append(f"  MH_acceptance[{topo}]: proposals={p}, accepted={a}, rate={a/float(p):.4f}")

        # Key log-ratios vs n_hexane.
        n0 = topo_counts.get("n_hexane", 0)
        for k in ("2_methylpentane", "3_methylpentane", "2,2_dimethylbutane", "2,3_dimethylbutane"):
            nk = topo_counts.get(k, 0)
            if nk > 0 and n0 > 0:
                lines.append(f"  log(P({k})/P(n_hexane)) = {float(np.log(nk / n0)):.4f}")
            else:
                lines.append(f"  log(P({k})/P(n_hexane)) undefined (zero counts)")

        if m == "A":
            p_exact = _load_p_exact_hexane_mode_a()
            if p_exact is not None:
                p_growth = {k: topo_counts_growth.get(k, 0) / float(total) for k in HEXANE_DEGENERACY.keys()}
                p_equil = {k: topo_counts_eq.get(k, 0) / float(total) for k in HEXANE_DEGENERACY.keys()}
                p_exact_vec = {k: float(p_exact.get(k, 0.0)) for k in HEXANE_DEGENERACY.keys()}
                lines.append(f"  KL(P_growth||P_exact) = {kl_divergence(p_growth, p_exact_vec):.6f}")
                lines.append(f"  KL(P_equilibrated||P_exact) = {kl_divergence(p_equil, p_exact_vec):.6f}")

        if growth_times:
            arr = np.asarray(growth_times, dtype=float)
            lines.append(f"  runtime_growth_sec: mean={float(np.mean(arr)):.4f}, p90={float(np.percentile(arr, 90)):.4f}")
        if scoring_times:
            arr = np.asarray(scoring_times, dtype=float)
            lines.append(f"  runtime_scoring_sec: mean={float(np.mean(arr)):.4f}, p90={float(np.percentile(arr, 90)):.4f}")

        lines.append("")

    refs = _make_reference_adjs()
    lines.append("Deterministic reference scores (C6: 5 hexane isomers)")

    def _score_parts_for_mode(mode: str, adj: np.ndarray) -> Tuple[float, float, float, float]:
        thermo = _make_thermo_for_mode(mode)
        with override_thermo_config(thermo):
            feats_fdm = compute_complexity_features_v2(adj, backend="fdm")
            feats_ent = compute_complexity_features_v2(adj, backend="fdm_entanglement")
        score_fdm = float(feats_fdm.total)
        score_topo3d = float(feats_ent.total) - float(feats_fdm.total)
        score_shape = 0.0
        return score_fdm, score_topo3d, score_shape, score_fdm + score_topo3d + score_shape

    for mode in cfg.modes:
        m = mode.upper()
        if m not in {"A", "B", "C"}:
            continue
        e_ref: Dict[str, float] = {}
        lines.append(f"  [Mode {m}]")
        for topo in ("n_hexane", "2_methylpentane", "3_methylpentane", "2,2_dimethylbutane", "2,3_dimethylbutane"):
            fdm, t3, sh, tot = _score_parts_for_mode(m, refs[topo])
            e_ref[topo] = float(tot)
            lines.append(f"    {topo}: fdm={fdm:.4f}, topo3d={t3:.4f}, shape={sh:.4f}, total={tot:.4f}")
        # Predicted order (min-score best)
        order = [k for k, _ in sorted(e_ref.items(), key=lambda kv: (kv[1], kv[0]))]
        lines.append("    Predicted order (min-score best): " + " > ".join(order))
        lines.append("")

    # CORE-2: fit lambda and compare predicted vs observed per mode A/B/C.
    lines.append("CORE-2: Fit lambda (degeneracy-aware)")
    for mode in cfg.modes:
        m = mode.upper()
        if m not in {"A", "B", "C"}:
            continue
        thermo = _make_thermo_for_mode(m)
        T = float(getattr(thermo, "temperature_T", 1.0))

        e_ref = {k: float(_score_parts_for_mode(m, refs[k])[3]) for k in HEXANE_DEGENERACY.keys()}

        topo_counts: Dict[str, int] = {}
        for r in by_mode.get(m, []):
            topo = str(r["topology"])
            topo_counts[topo] = topo_counts.get(topo, 0) + 1
        denom = sum(topo_counts.get(k, 0) for k in HEXANE_DEGENERACY.keys()) or 1
        p_obs = {k: topo_counts.get(k, 0) / float(denom) for k in HEXANE_DEGENERACY.keys()}

        fit = fit_lambda(p_obs, HEXANE_DEGENERACY, e_ref, T=T)
        p_pred_star = compute_p_pred(HEXANE_DEGENERACY, e_ref, T=T, lam=fit.lam_star)
        kl_star = kl_divergence(p_obs, p_pred_star)

        lines.append(f"  [Mode {m}] temperature_T={T:.3f}")
        lines.append(f"    lambda*={fit.lam_star:.4f}, KL_min={fit.kl_min:.6f}, KL@lambda*={kl_star:.6f}")
        lines.append(f"    P_obs ={ {k: float(p_obs[k]) for k in sorted(p_obs.keys())} }")
        lines.append(f"    P_pred={ {k: float(p_pred_star.get(k, 0.0)) for k in sorted(HEXANE_DEGENERACY.keys())} }")
        # log-ratios vs n_hexane
        pn = max(float(p_obs.get("n_hexane", 0.0)), 1e-300)
        qn = max(float(p_pred_star.get("n_hexane", 0.0)), 1e-300)
        for k in ("2_methylpentane", "3_methylpentane", "2,2_dimethylbutane", "2,3_dimethylbutane"):
            pk = max(float(p_obs.get(k, 0.0)), 1e-300)
            qk = max(float(p_pred_star.get(k, 0.0)), 1e-300)
            lr_obs = float(math.log(pk / pn))
            lr_pred = float(math.log(qk / qn))
            lines.append(f"    log(P({k})/P(n_hexane)): obs={lr_obs:.4f}, pred={lr_pred:.4f}, Δ={float(lr_obs-lr_pred):.4f}")
        lines.append("")

    summary_path = write_growth_txt("chem_validation_1b_hexane", lines)
    return csv_path, summary_path


def _parse_args(argv: Optional[Sequence[str]] = None) -> ChemValidation1BConfig:
    parser = argparse.ArgumentParser(description="CHEM-VALIDATION-1B: C6 hexane skeleton validation.")
    parser.add_argument("--n_runs", type=int, default=1000)
    parser.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2, 3, 4])
    parser.add_argument("--modes", type=str, nargs="*", default=["R", "A", "B", "C"])
    parser.add_argument("--max_depth", type=int, default=6)
    parser.add_argument("--equilibrate_steps", type=int, default=0)
    parser.add_argument("--equilibrate_burnin", type=int, default=0)
    parser.add_argument("--equilibrate_thin", type=int, default=1)
    parser.add_argument(
        "--progress",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Show progress bar / periodic progress output.",
    )
    args = parser.parse_args(argv)
    return ChemValidation1BConfig(
        n_runs=int(args.n_runs),
        seeds=tuple(int(x) for x in args.seeds),
        modes=tuple(str(m).upper() for m in args.modes),
        max_depth=int(args.max_depth),
        equilibrate_steps=int(args.equilibrate_steps),
        equilibrate_burnin=int(args.equilibrate_burnin),
        equilibrate_thin=int(args.equilibrate_thin),
        progress=bool(args.progress),
    )


def main(argv: Optional[Sequence[str]] = None) -> None:
    cfg = _parse_args(argv)
    csv_path, txt_path = run_chem_validation_1b(cfg)
    print("[CHEM-VALIDATION-1B] done.")
    print(f"CSV: {csv_path}")
    print(f"Summary: {txt_path}")


if __name__ == "__main__":
    main()
