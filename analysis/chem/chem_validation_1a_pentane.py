from __future__ import annotations

import argparse
import csv
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from analysis.io_utils import results_path
from analysis.growth.rng import make_rng
from analysis.growth.reporting import write_growth_txt
from core.complexity import compute_complexity_features_v2
from core.grower import GrowthParams, grow_molecule_christmas_tree
from core.thermo_config import ThermoConfig, override_thermo_config


PENTANE_N_DEG: Tuple[int, ...] = (1, 1, 2, 2, 2)
PENTANE_ISO_DEG: Tuple[int, ...] = (1, 1, 1, 2, 3)
PENTANE_NEO_DEG: Tuple[int, ...] = (1, 1, 1, 1, 4)


def classify_pentane_topology(sorted_degrees: Sequence[int]) -> str:
    """
    Classify a C5 tree topology by its sorted degree sequence.

    Returns:
        "n_pentane" for [1,1,2,2,2],
        "isopentane" for [1,1,1,2,3],
        "neopentane" for [1,1,1,1,4],
        "other" otherwise.
    """
    seq = tuple(sorted_degrees)
    if seq == PENTANE_N_DEG:
        return "n_pentane"
    if seq == PENTANE_ISO_DEG:
        return "isopentane"
    if seq == PENTANE_NEO_DEG:
        return "neopentane"
    return "other"


@dataclass
class ChemValidation1AConfig:
    n_runs: int = 1000
    seeds: Tuple[int, ...] = (0, 1, 2, 3, 4)
    modes: Tuple[str, ...] = ("R", "A", "B", "C")
    max_depth: int = 5


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
            experiment_name="chem_validation_1a_mode_R",
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
            experiment_name="chem_validation_1a_mode_A",
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
            experiment_name="chem_validation_1a_mode_B",
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
            experiment_name="chem_validation_1a_mode_C",
        )

    raise ValueError(f"Unsupported CHEM-VALIDATION-1A mode: {mode!r}")


def _run_single_mode(cfg: ChemValidation1AConfig, mode: str) -> List[Dict[str, object]]:
    thermo = _make_thermo_for_mode(mode)
    rows: List[Dict[str, object]] = []

    with override_thermo_config(thermo):
        for seed in cfg.seeds:
            base_rng = make_rng(f"chem_validation_1a_pentane_{mode}_{seed}")
            rng = np.random.default_rng(base_rng.integers(0, 2**32 - 1))

            params = GrowthParams(
                max_depth=cfg.max_depth,
                max_atoms=12,
                stop_at_n_atoms=5,
                allowed_symbols=["C"],
            )

            for run_idx in range(cfg.n_runs):
                t0 = time.perf_counter()
                mol = grow_molecule_christmas_tree("C", params, rng=rng)
                dt = time.perf_counter() - t0

                adj = mol.adjacency_matrix()
                degrees = np.asarray(adj.sum(axis=1), dtype=int)
                sorted_degrees = tuple(sorted(int(x) for x in degrees))
                topology = classify_pentane_topology(sorted_degrees)

                feats_fdm = compute_complexity_features_v2(adj, backend="fdm")
                feats_ent = compute_complexity_features_v2(
                    adj, backend="fdm_entanglement"
                )

                score_fdm = float(feats_fdm.total)
                score_topo3d = float(feats_ent.total) - float(feats_fdm.total)
                score_shape = 0.0

                if mode == "R":
                    total_score = score_fdm
                else:
                    total_score = score_fdm + score_topo3d + score_shape

                mh_stats = getattr(mol, "mh_stats", None)
                if isinstance(mh_stats, dict):
                    mh_proposals = int(mh_stats.get("proposals", 0))
                    mh_accepted = int(mh_stats.get("accepted", 0))
                else:
                    mh_proposals = 0
                    mh_accepted = 0
                mh_accept_rate = (
                    (mh_accepted / float(mh_proposals)) if mh_proposals > 0 else 0.0
                )

                rows.append(
                    {
                        "mode": mode,
                        "seed": seed,
                        "run_idx": run_idx,
                        "n_atoms": int(adj.shape[0]),
                        "topology": topology,
                        "deg_sorted": str(list(sorted_degrees)),
                        "score_fdm": score_fdm,
                        "score_topo3d": score_topo3d,
                        "score_shape": score_shape,
                        "score_total": total_score,
                        "mh_proposals": mh_proposals,
                        "mh_accepted": mh_accepted,
                        "mh_accept_rate": mh_accept_rate,
                        "runtime_sec": dt,
                    }
                )
    return rows


def run_chem_validation_1a(cfg: ChemValidation1AConfig) -> Tuple[Path, Path]:
    all_rows: List[Dict[str, object]] = []
    by_mode: Dict[str, List[Dict[str, object]]] = {}
    for mode in cfg.modes:
        mode = mode.upper()
        rows = _run_single_mode(cfg, mode)
        by_mode[mode] = rows
        all_rows.extend(rows)

    csv_path = results_path("chem_validation_1a_pentane.csv")
    fieldnames = [
        "mode",
        "seed",
        "run_idx",
        "n_atoms",
        "topology",
        "deg_sorted",
        "score_fdm",
        "score_topo3d",
        "score_shape",
        "score_total",
        "mh_proposals",
        "mh_accepted",
        "mh_accept_rate",
        "runtime_sec",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in all_rows:
            w.writerow({k: r.get(k, "") for k in fieldnames})

    lines: List[str] = []
    lines.append("CHEM-VALIDATION-1A: C5 pentane skeleton (n/iso/neo)")
    lines.append("")
    lines.append(f"Config: n_runs={cfg.n_runs}, seeds={list(cfg.seeds)}, modes={list(cfg.modes)}")
    lines.append("")

    for mode in cfg.modes:
        rows = by_mode.get(mode.upper(), [])
        if not rows:
            continue
        lines.append(f"[Mode {mode.upper()}]")
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
            lines.append(f"  score_total[{topo}]: median={median:.4f}, p90={p90:.4f}")

        for topo in sorted(mh_by_topo.keys()):
            p, a = mh_by_topo[topo]
            if p > 0:
                acc = a / float(p)
                lines.append(
                    f"  MH_acceptance[{topo}]: proposals={p}, accepted={a}, rate={acc:.4f}"
                )

        n_iso = topo_counts.get("isopentane", 0)
        n_n = topo_counts.get("n_pentane", 0)
        n_neo = topo_counts.get("neopentane", 0)
        if n_iso > 0 and n_n > 0:
            lines.append(f"  log(P(iso)/P(n)) = {float(np.log(n_iso / n_n)):.4f}")
        else:
            lines.append("  log(P(iso)/P(n)) undefined (zero counts)")
        if n_neo > 0 and n_n > 0:
            lines.append(f"  log(P(neo)/P(n)) = {float(np.log(n_neo / n_n)):.4f}")
        else:
            lines.append("  log(P(neo)/P(n)) undefined (zero counts)")
        lines.append("")

    lines.append("Deterministic reference scores (C5: path / iso / star)")

    def _make_adj_refs() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        adj_n = np.zeros((5, 5), dtype=float)
        for a, b in ((0, 1), (1, 2), (2, 3), (3, 4)):
            adj_n[a, b] = adj_n[b, a] = 1.0

        adj_iso = np.zeros((5, 5), dtype=float)
        for a, b in ((1, 0), (1, 2), (1, 4), (2, 3)):
            adj_iso[a, b] = adj_iso[b, a] = 1.0

        adj_neo = np.zeros((5, 5), dtype=float)
        for j in (1, 2, 3, 4):
            adj_neo[0, j] = adj_neo[j, 0] = 1.0
        return adj_n, adj_iso, adj_neo

    adj_n, adj_iso, adj_neo = _make_adj_refs()

    def _score_for_mode(mode: str, adj: np.ndarray) -> float:
        thermo = _make_thermo_for_mode(mode)
        with override_thermo_config(thermo):
            feats_fdm = compute_complexity_features_v2(adj, backend="fdm")
            feats_ent = compute_complexity_features_v2(adj, backend="fdm_entanglement")
        score_fdm = float(feats_fdm.total)
        score_topo3d = float(feats_ent.total) - float(feats_fdm.total)
        score_shape = 0.0
        return score_fdm + score_topo3d + score_shape

    for mode in cfg.modes:
        m = mode.upper()
        if m not in {"A", "B", "C"}:
            continue
        s_n = _score_for_mode(m, adj_n)
        s_i = _score_for_mode(m, adj_iso)
        s_neo = _score_for_mode(m, adj_neo)
        lines.append(f"  Mode {m}: score(n)={s_n:.4f}, score(iso)={s_i:.4f}, score(neo)={s_neo:.4f}")
        lines.append(
            f"    Δ(n-iso)={float(s_n - s_i):.4f}, Δ(n-neo)={float(s_n - s_neo):.4f}, "
            f"Δ(iso-neo)={float(s_i - s_neo):.4f}"
        )

    lines.append("")
    summary_path = write_growth_txt("chem_validation_1a_pentane", lines)
    return csv_path, summary_path


def _parse_args(argv: Optional[Sequence[str]] = None) -> ChemValidation1AConfig:
    parser = argparse.ArgumentParser(
        description="CHEM-VALIDATION-1A: C5 pentane skeleton validation."
    )
    parser.add_argument("--n_runs", type=int, default=1000, help="Number of growth runs per seed.")
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
    parser.add_argument("--max_depth", type=int, default=5, help="Max depth for grower.")
    args = parser.parse_args(argv)
    return ChemValidation1AConfig(
        n_runs=int(args.n_runs),
        seeds=tuple(int(x) for x in args.seeds),
        modes=tuple(str(m).upper() for m in args.modes),
        max_depth=int(args.max_depth),
    )


def main(argv: Optional[Sequence[str]] = None) -> None:
    cfg = _parse_args(argv)
    csv_path, summary_path = run_chem_validation_1a(cfg)
    print("[CHEM-VALIDATION-1A] done.")
    print(f"CSV: {csv_path}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()

