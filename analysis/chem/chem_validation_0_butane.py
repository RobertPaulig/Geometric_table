from __future__ import annotations

import argparse
import csv
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from analysis.io_utils import results_path
from analysis.utils.progress import progress_iter
from analysis.growth.rng import make_rng
from analysis.growth.reporting import write_growth_txt
from analysis.chem.core2_fit import compute_p_pred, fit_lambda, kl_divergence
from analysis.utils.timing import format_sec, now_iso, timed
from core.complexity import compute_complexity_features_v2
from core.grower import GrowthParams, grow_molecule_christmas_tree
from core.thermo_config import ThermoConfig, override_thermo_config


BUTANE_N_DEG: Tuple[int, ...] = (1, 1, 2, 2)
BUTANE_ISO_DEG: Tuple[int, ...] = (1, 1, 1, 3)
BUTANE_DEGENERACY: Dict[str, int] = {"n_butane": 12, "isobutane": 4}


def classify_butane_topology(sorted_degrees: Sequence[int]) -> str:
    """
    Classify a C4 tree topology by its sorted degree sequence.

    Returns:
        "n_butane" for [1,1,2,2],
        "isobutane" for [1,1,1,3],
        "other" otherwise.
    """
    seq = tuple(sorted_degrees)
    if seq == BUTANE_N_DEG:
        return "n_butane"
    if seq == BUTANE_ISO_DEG:
        return "isobutane"
    return "other"


@dataclass
class ChemValidation0Config:
    n_runs: int = 1000
    seeds: Tuple[int, ...] = (0, 1, 2, 3, 4)
    modes: Tuple[str, ...] = ("R", "A", "B", "C")
    max_depth: int = 4
    progress: bool = True


def _make_thermo_for_mode(mode: str) -> ThermoConfig:
    """
    Construct a minimal ThermoConfig for a given ablation mode.

    Modes:
        R: proposal-only (no MH, no topo3d, no shape).
        A: only FDM complexity (no topo3d/layout couplings, MH enabled).
        B: FDM complexity + topo3d entanglement/layout (MH enabled).
        C: like B, but with shape couplings in ThermoConfig (for completeness).
    """
    base = ThermoConfig()

    mode = mode.upper()

    if mode == "R":
        # Proposal-only: MH отключён, все энергетические связи 0.
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
            topo3d_prefilter_tree=False,
            topo3d_prefilter_min_n=0,
            temperature_T=1.0,
            grower_use_mh=False,
            deltaG_backend="fdm_entanglement",
            consume_port_on_reject=True,
            max_attempts_per_port=1,
            grower_proposal_policy="uniform",
            proposal_beta=0.0,
            proposal_ports_gamma=0.0,
            experiment_name="chem_validation_0_mode_R",
        )

    if mode == "A":
        # Only FDM complexity; topo3d disabled, MH включён как в основном пайплайне.
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
            consume_port_on_reject=True,
            max_attempts_per_port=1,
            grower_proposal_policy="uniform",
            proposal_beta=0.0,
            proposal_ports_gamma=0.0,
            experiment_name="chem_validation_0_mode_A",
        )

    if mode == "B":
        # FDM + topo3d entanglement/layout via backend="fdm_entanglement".
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
            consume_port_on_reject=True,
            max_attempts_per_port=1,
            grower_proposal_policy="uniform",
            proposal_beta=0.0,
            proposal_ports_gamma=0.0,
            experiment_name="chem_validation_0_mode_B",
        )

    if mode == "C":
        # FDM + topo3d + shape couplings в ThermoConfig (для полноты картины).
        # В C4-тесте shape-термы по сути одинаковы для всех C-скелетов и
        # не вносят дифференциацию, но мы фиксируем режим для дальнейших тестов.
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
            topo3d_prefilter_tree=False,
            topo3d_prefilter_min_n=0,
            temperature_T=1.0,
            grower_use_mh=True,
            deltaG_backend="fdm_entanglement",
            consume_port_on_reject=True,
            max_attempts_per_port=1,
            grower_proposal_policy="uniform",
            proposal_beta=0.0,
            proposal_ports_gamma=0.0,
            experiment_name="chem_validation_0_mode_C",
        )

    raise ValueError(f"Unsupported CHEM-VALIDATION-0 mode: {mode!r}")


def _run_single_mode(cfg: ChemValidation0Config, mode: str) -> List[Dict[str, object]]:
    thermo = _make_thermo_for_mode(mode)
    rows: List[Dict[str, object]] = []

    with override_thermo_config(thermo):
        for seed in cfg.seeds:
            base_rng = make_rng(f"chem_validation_0_butane_{mode}_{seed}")
            rng = np.random.default_rng(base_rng.integers(0, 2**32 - 1))

            params = GrowthParams(
                max_depth=cfg.max_depth,
                max_atoms=10,
                stop_at_n_atoms=4,
                allowed_symbols=["C"],
            )

            for run_idx in range(cfg.n_runs):
                t0 = time.perf_counter()
                mol = grow_molecule_christmas_tree("C", params, rng=rng)
                dt = time.perf_counter() - t0

                adj = mol.adjacency_matrix()
                degrees = np.asarray(adj.sum(axis=1), dtype=int)
                sorted_degrees = tuple(sorted(int(x) for x in degrees))
                topology = classify_butane_topology(sorted_degrees)

                feats_fdm = compute_complexity_features_v2(adj, backend="fdm")
                feats_ent = compute_complexity_features_v2(
                    adj, backend="fdm_entanglement"
                )

                # Декомпозиция score по термам (минимальная):
                score_fdm = float(feats_fdm.total)
                # topo3d-компонента как приращение к FDM
                score_topo3d = float(feats_ent.total) - float(feats_fdm.total)
                # В текущем C4-тесте shape не различает топологии, оставляем 0.
                score_shape = 0.0

                if mode == "R":
                    # Proposal-only: используем чистый FDM как proxy score, но он никак не
                    # влияет на рост (MH выключен).
                    total_score = score_fdm
                elif mode in ("A", "B", "C"):
                    # Для A/B/C total_score соответствует FDM+topo3d (+shape для C,
                    # когда он будет задействован).
                    total_score = score_fdm + score_topo3d + score_shape
                else:
                    total_score = score_fdm

                mh_stats = getattr(mol, "mh_stats", None)
                if isinstance(mh_stats, dict):
                    mh_proposals = int(mh_stats.get("proposals", 0))
                    mh_accepted = int(mh_stats.get("accepted", 0))
                else:
                    mh_proposals = 0
                    mh_accepted = 0
                if mh_proposals > 0:
                    mh_accept_rate = mh_accepted / float(mh_proposals)
                else:
                    mh_accept_rate = 0.0

                rows.append(
                    {
                        "mode": mode,
                        "seed": int(seed),
                        "run_idx": int(run_idx),
                        "n_atoms": int(len(mol.atoms)),
                        "topology": topology,
                        "deg_sorted": ";".join(str(x) for x in sorted_degrees),
                        "complexity_fdm": float(feats_fdm.total),
                        "complexity_fdm_entanglement": float(feats_ent.total),
                        "score_fdm": score_fdm,
                        "score_topo3d": score_topo3d,
                        "score_shape": score_shape,
                        "score_total": total_score,
                        "mh_proposals": mh_proposals,
                        "mh_accepted": mh_accepted,
                        "mh_accept_rate": mh_accept_rate,
                        "runtime_sec": float(dt),
                    }
                )

    return rows


def run_chem_validation_0(cfg: ChemValidation0Config) -> Tuple[Path, Path]:
    start_ts = now_iso()
    t0_total = time.perf_counter()
    acc: Dict[str, float] = {}
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
            base_rng = make_rng(f"chem_validation_0_butane_{m}_{seed}")
            rng_by_mode_seed[(m, int(seed))] = np.random.default_rng(
                base_rng.integers(0, 2**32 - 1)
            )

    current_mode: Optional[str] = None
    current_ctx = None

    for mode, seed, run_idx in progress_iter(
        run_plan, total=total_runs, desc="CHEM-VALIDATION-0", enabled=bool(cfg.progress)
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
            max_atoms=12,
            stop_at_n_atoms=4,
            allowed_symbols=["C"],
            enforce_tree_alkane=True,
            p_continue_base=1.0,
            chi_sensitivity=0.0,
            role_bonus_hub=0.0,
            role_penalty_terminator=0.0,
            temperature=1.0,
        )

        t0 = time.perf_counter()
        with timed("growth_total", acc):
            mol = grow_molecule_christmas_tree("C", params, rng=rng)
        adj = mol.adjacency_matrix()
        growth_sec = time.perf_counter() - t0

        degrees = np.asarray(adj.sum(axis=1), dtype=int)
        sorted_degrees = tuple(sorted(int(x) for x in degrees))
        topology = classify_butane_topology(sorted_degrees)

        t1 = time.perf_counter()
        with timed("scoring_total", acc):
            feats_fdm = compute_complexity_features_v2(adj, backend="fdm")
            feats_ent = compute_complexity_features_v2(adj, backend="fdm_entanglement")
        scoring_sec = time.perf_counter() - t1

        score_fdm = float(feats_fdm.total)
        score_topo3d = float(feats_ent.total) - float(feats_fdm.total)
        score_shape = 0.0
        total_score = score_fdm if mode == "R" else (score_fdm + score_topo3d + score_shape)

        mh_stats = getattr(mol, "mh_stats", None)
        if isinstance(mh_stats, dict):
            mh_proposals = int(mh_stats.get("proposals", 0))
            mh_accepted = int(mh_stats.get("accepted", 0))
        else:
            mh_proposals = 0
            mh_accepted = 0
        mh_accept_rate = (mh_accepted / float(mh_proposals)) if mh_proposals > 0 else 0.0

        r = {
            "mode": mode,
            "seed": seed,
            "run_idx": run_idx,
            "n_atoms": int(adj.shape[0]),
            "topology": topology,
            "deg_sorted": ";".join(str(x) for x in sorted_degrees),
            "complexity_fdm": float(feats_fdm.total),
            "complexity_fdm_entanglement": float(feats_ent.total),
            "score_fdm": score_fdm,
            "score_topo3d": score_topo3d,
            "score_shape": score_shape,
            "score_total": total_score,
            "mh_proposals": mh_proposals,
            "mh_accepted": mh_accepted,
            "mh_accept_rate": mh_accept_rate,
            "resample_attempts_used": 1,
            "t_growth_sec": float(growth_sec),
            "t_scoring_sec": float(scoring_sec),
            "t_total_sec": float(growth_sec + scoring_sec),
        }
        by_mode[mode].append(r)
        all_rows.append(r)

    if current_ctx is not None:
        current_ctx.__exit__(None, None, None)

    csv_path = results_path("chem_validation_0_butane.csv")
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "mode",
        "seed",
        "run_idx",
        "n_atoms",
        "topology",
        "deg_sorted",
        "complexity_fdm",
        "complexity_fdm_entanglement",
        "score_fdm",
        "score_topo3d",
        "score_shape",
        "score_total",
        "mh_proposals",
        "mh_accepted",
        "mh_accept_rate",
        "resample_attempts_used",
        "t_growth_sec",
        "t_scoring_sec",
        "t_total_sec",
    ]

    with timed("io_total", acc):
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in all_rows:
                writer.writerow(r)

    # Aggregate statistics for summary TXT.
    lines: List[str] = []
    lines.append("CHEM-VALIDATION-0 (C4 butane skeleton)")
    lines.append(f"n_runs_per_seed={cfg.n_runs}")
    lines.append(f"seeds={list(cfg.seeds)}")
    lines.append(f"modes={list(cfg.modes)}")
    elapsed_total = time.perf_counter() - t0_total
    end_ts = now_iso()
    lines.append(f"elapsed_sec={elapsed_total:.3f}, runs_done={len(all_rows)}")
    lines.append("")

    # group by mode
    for mode, rows in by_mode.items():
        lines.append(f"[Mode {mode}]")
        topo_counts: Dict[str, int] = {}
        scores_by_topo: Dict[str, List[float]] = {}
        mh_by_topo: Dict[str, Tuple[int, int]] = {}

        for r in rows:
            topo = str(r["topology"])
            topo_counts[topo] = topo_counts.get(topo, 0) + 1
            scores_by_topo.setdefault(topo, []).append(float(r["score_total"]))
            p = int(r.get("mh_proposals", 0))
            a = int(r.get("mh_accepted", 0))
            old_p, old_a = mh_by_topo.get(topo, (0, 0))
            mh_by_topo[topo] = (old_p + p, old_a + a)

        total = sum(topo_counts.values()) or 1
        for topo in sorted(topo_counts.keys()):
            p = topo_counts[topo] / float(total)
            lines.append(f"  P({topo}) = {p:.4f} (count={topo_counts[topo]})")

        for topo, vals in scores_by_topo.items():
            arr = np.asarray(vals, dtype=float)
            median = float(np.median(arr))
            p90 = float(np.percentile(arr, 90))
            lines.append(
                f"  score_total[{topo}]: median={median:.4f}, p90={p90:.4f}"
            )

        # MH-acceptance по топологиям (для A/B/C; для R значения будут 0).
        for topo in sorted(mh_by_topo.keys()):
            p, a = mh_by_topo[topo]
            if p > 0:
                acc = a / float(p)
                lines.append(
                    f"  MH_acceptance[{topo}]: proposals={p}, accepted={a}, rate={acc:.4f}"
                )

        n_iso = topo_counts.get("isobutane", 0)
        n_n = topo_counts.get("n_butane", 0)
        if n_iso > 0 and n_n > 0:
            log_ratio = np.log(n_iso / n_n)
            lines.append(f"  log(P(iso)/P(n)) = {float(log_ratio):.4f}")
        else:
            lines.append("  log(P(iso)/P(n)) undefined (zero counts)")

        lines.append("")

    # Детерминированные эталоны: path (n_butane) vs star (isobutane).
    lines.append("Deterministic reference scores (path vs star)")

    def _make_adj_path_star() -> Tuple[np.ndarray, np.ndarray]:
        # path: 0-1-2-3
        adj_path = np.zeros((4, 4), dtype=float)
        adj_path[0, 1] = adj_path[1, 0] = 1.0
        adj_path[1, 2] = adj_path[2, 1] = 1.0
        adj_path[2, 3] = adj_path[3, 2] = 1.0
        # star: 0 connected to 1,2,3
        adj_star = np.zeros((4, 4), dtype=float)
        for j in (1, 2, 3):
            adj_star[0, j] = adj_star[j, 0] = 1.0
        return adj_path, adj_star

    adj_path, adj_star = _make_adj_path_star()

    def _score_for_mode(mode: str, adj: np.ndarray) -> float:
        thermo = _make_thermo_for_mode(mode)
        with override_thermo_config(thermo):
            feats_fdm = compute_complexity_features_v2(adj, backend="fdm")
            feats_ent = compute_complexity_features_v2(adj, backend="fdm_entanglement")
        score_fdm = float(feats_fdm.total)
        score_topo3d = float(feats_ent.total) - float(feats_fdm.total)
        score_shape = 0.0
        if mode == "R":
            return score_fdm
        return score_fdm + score_topo3d + score_shape

    for mode in sorted(by_mode.keys()):
        m = mode.upper()
        if m not in {"R", "A", "B", "C"}:
            continue
        s_path = _score_for_mode(m, adj_path)
        s_star = _score_for_mode(m, adj_star)
        delta = s_path - s_star
        lines.append(
            f"  Mode {m}: score(path)={s_path:.4f}, score(star)={s_star:.4f}, "
            f"Δscore(n-iso)={delta:.4f}"
        )

        lines.append("")

    # CORE-1: degeneracy-aware predicted frequencies for tree-only unlabeled topologies.
    lines.append("CORE-1: Degeneracy-aware predicted vs observed (tree-only)")
    lines.append(
        "  Model: P_pred(topo) ∝ g(topo) * exp(-coupling_delta_G * E_ref(topo) / temperature_T)"
    )
    lines.append(f"  g(topology)={BUTANE_DEGENERACY}")

    def _safe_log(x: float) -> float:
        if x <= 0.0:
            return float("nan")
        return float(math.log(x))

    for mode in sorted(by_mode.keys()):
        m = mode.upper()
        if m not in {"A", "B", "C"}:
            continue
        thermo = _make_thermo_for_mode(m)
        coupling = float(getattr(thermo, "coupling_delta_G", 1.0))
        T = float(getattr(thermo, "temperature_T", 1.0))
        beta = (coupling / T) if T > 0 else float("inf")

        e_n = _score_for_mode(m, adj_path)
        e_iso = _score_for_mode(m, adj_star)
        w_n = float(BUTANE_DEGENERACY["n_butane"]) * math.exp(-beta * float(e_n))
        w_iso = float(BUTANE_DEGENERACY["isobutane"]) * math.exp(-beta * float(e_iso))
        z = w_n + w_iso
        p_pred_n = w_n / z if z > 0 else 0.0
        p_pred_iso = w_iso / z if z > 0 else 0.0

        topo_counts: Dict[str, int] = {}
        for r in by_mode.get(m, []):
            topo = str(r["topology"])
            topo_counts[topo] = topo_counts.get(topo, 0) + 1
        n_n = topo_counts.get("n_butane", 0)
        n_iso = topo_counts.get("isobutane", 0)
        denom = float(n_n + n_iso) or 1.0
        p_obs_n = n_n / denom
        p_obs_iso = n_iso / denom

        lines.append(f"  [Mode {m}] coupling={coupling:.3f}, temperature_T={T:.3f}, beta={beta:.3f}")
        lines.append(f"    P_obs : n={p_obs_n:.4f}, iso={p_obs_iso:.4f}")
        lines.append(f"    P_pred: n={p_pred_n:.4f}, iso={p_pred_iso:.4f}")
        lr_obs = _safe_log(p_obs_iso / p_obs_n) if p_obs_n > 0 else float("nan")
        lr_pred = _safe_log(p_pred_iso / p_pred_n) if p_pred_n > 0 else float("nan")
        lines.append(
            f"    log(P(iso)/P(n)): obs={lr_obs:.4f}, pred={lr_pred:.4f}, Δ={float(lr_obs - lr_pred):.4f}"
        )

    lines.append("")

    # CORE-2: fit lambda scale for degeneracy-aware Boltzmann model.
    lines.append("CORE-2: Fit lambda (degeneracy-aware)")
    for mode in sorted(by_mode.keys()):
        m = mode.upper()
        if m not in {"A", "B", "C"}:
            continue
        thermo = _make_thermo_for_mode(m)
        T = float(getattr(thermo, "temperature_T", 1.0))

        e_ref = {"n_butane": float(_score_for_mode(m, adj_path)), "isobutane": float(_score_for_mode(m, adj_star))}

        topo_counts: Dict[str, int] = {}
        for r in by_mode.get(m, []):
            topo = str(r["topology"])
            topo_counts[topo] = topo_counts.get(topo, 0) + 1
        n_n = topo_counts.get("n_butane", 0)
        n_iso = topo_counts.get("isobutane", 0)
        denom = float(n_n + n_iso) or 1.0
        p_obs = {"n_butane": n_n / denom, "isobutane": n_iso / denom}

        fit = fit_lambda(p_obs, BUTANE_DEGENERACY, e_ref, T=T)
        p_pred_star = compute_p_pred(BUTANE_DEGENERACY, e_ref, T=T, lam=fit.lam_star)
        kl_star = kl_divergence(p_obs, p_pred_star)

        lines.append(f"  [Mode {m}] temperature_T={T:.3f}")
        lines.append(f"    E_ref={e_ref}")
        lines.append(f"    lambda*={fit.lam_star:.4f}, KL_min={fit.kl_min:.6f}, KL@lambda*={kl_star:.6f}")
        lines.append(f"    P_obs ={ {k: float(v) for k, v in p_obs.items()} }")
        lines.append(f"    P_pred={ {k: float(v) for k, v in p_pred_star.items()} }")
        if p_obs["n_butane"] > 0 and p_pred_star.get("n_butane", 0.0) > 0:
            lr_obs = math.log(p_obs["isobutane"] / p_obs["n_butane"])
            lr_pred = math.log(p_pred_star["isobutane"] / p_pred_star["n_butane"])
            lines.append(
                f"    log(P(iso)/P(n)): obs={lr_obs:.4f}, pred={lr_pred:.4f}, Δ={float(lr_obs-lr_pred):.4f}"
            )
        lines.append("")

    lines.append("")
    lines.append("TIMING")
    lines.append(f"START_TS={start_ts}")
    lines.append(f"END_TS={end_ts}")
    lines.append(f"ELAPSED_TOTAL_SEC={elapsed_total:.6f}")
    lines.append(f"ELAPSED_GROWTH_SEC={float(acc.get('growth_total', 0.0)):.6f}")
    lines.append(f"ELAPSED_SCORING_SEC={float(acc.get('scoring_total', 0.0)):.6f}")
    lines.append(f"ELAPSED_IO_SEC={float(acc.get('io_total', 0.0)):.6f}")

    with timed("io_total", acc):
        summary_path = write_growth_txt("chem_validation_0_butane", lines)

    print(f"Wall-clock: start={start_ts} end={end_ts} elapsed_total_sec={elapsed_total:.3f}")
    print(
        "Breakdown: "
        f"growth={format_sec(acc.get('growth_total', 0.0))} "
        f"scoring={format_sec(acc.get('scoring_total', 0.0))} "
        f"io={format_sec(acc.get('io_total', 0.0))}"
    )
    return csv_path, summary_path


def _parse_args(argv: Optional[Sequence[str]] = None) -> ChemValidation0Config:
    parser = argparse.ArgumentParser(
        description="CHEM-VALIDATION-0: C4 butane skeleton validation."
    )
    parser.add_argument(
        "--n_runs",
        type=int,
        default=1000,
        help="Number of growth runs per seed.",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="*",
        default=[0, 1, 2, 3, 4],
        help="List of RNG seeds.",
    )
    parser.add_argument(
        "--modes",
        type=str,
        nargs="*",
        default=["R", "A", "B", "C"],
        help="Ablation modes: R (proposal-only), A (FDM), B (FDM+topo3d), C (FDM+topo3d+shape).",
    )
    parser.add_argument(
        "--progress",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Show progress bar / periodic progress output.",
    )
    args = parser.parse_args(argv)
    return ChemValidation0Config(
        n_runs=int(args.n_runs),
        seeds=tuple(int(x) for x in args.seeds),
        modes=tuple(str(m).upper() for m in args.modes),
        progress=bool(args.progress),
    )


def main(argv: Optional[Sequence[str]] = None) -> None:
    cfg = _parse_args(argv)
    csv_path, summary_path = run_chem_validation_0(cfg)
    print("[CHEM-VALIDATION-0] done.")
    print(f"CSV: {csv_path}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
