from __future__ import annotations

import csv
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Tuple

import numpy as np

from core.complexity import compute_complexity_features_v2
from core.entanglement_3d import entanglement_score
from core.grower import grow_molecule_loopy
from core.layout_3d import force_directed_layout_3d
from core.thermo_config import ThermoConfig, override_thermo_config


@dataclass
class ProposalSweepConfig:
    n_seeds: int = 50
    seed_offset: int = 0
    temperature_T: float = 1.0
    deltaG_backend: str = "fdm_entanglement"


def _molecule_to_edges(mol) -> List[Tuple[int, int]]:
    edges: List[Tuple[int, int]] = []
    for i, j in mol.bonds:
        if i == j:
            continue
        a, b = (i, j) if i < j else (j, i)
        edges.append((a, b))
    return edges


def _quantiles(values: List[float], probs: Iterable[float]) -> List[float]:
    if not values:
        return [0.0 for _ in probs]
    arr = np.asarray(values, dtype=float)
    return [float(np.quantile(arr, p)) for p in probs]


def _run_single(seed: int, policy: str) -> dict:
    rng = np.random.default_rng(seed)

    thermo = ThermoConfig(
        grower_use_mh=True,
        coupling_delta_G=1.0,
        temperature_T=1.0,
        deltaG_backend="fdm_entanglement",
        grower_proposal_policy=policy,
        proposal_beta=0.5 if policy == "ctt_biased" else 0.0,
        proposal_ports_gamma=1.0 if policy == "ctt_biased" else 0.0,
        coupling_topo_3d=1.0,
        topo_3d_beta=1.0,
    )

    with override_thermo_config(thermo):
        start = time.time()
        mol = grow_molecule_loopy("C", rng=rng)
        elapsed = time.time() - start

        stats = getattr(mol, "mh_stats", {})
        mh_proposals = int(stats.get("proposals", 0))
        mh_accepted = int(stats.get("accepted", 0))
        mh_rejected = int(stats.get("rejected", 0))
        acceptance_rate = (
            mh_accepted / mh_proposals if mh_proposals > 0 else 1.0
        )

        n = len(mol.atoms)
        edges = _molecule_to_edges(mol)
        m = len(edges)

        if n > 0 and m > 0:
            layout = force_directed_layout_3d(n, edges, seed=42)
            E_3d = float(entanglement_score(layout, edges))

            adj = mol.adjacency_matrix()
            feats_fdm = compute_complexity_features_v2(adj, backend="fdm")
            feats_ent = compute_complexity_features_v2(
                adj, backend="fdm_entanglement"
            )
            total_fdm = float(feats_fdm.total)
            total_ent = float(feats_ent.total)
            cyclomatic = int(getattr(feats_fdm, "cyclomatic", 0))
        else:
            E_3d = 0.0
            total_fdm = 0.0
            total_ent = 0.0
            cyclomatic = 0

    penalty_factor_3d = (
        total_ent / total_fdm if total_fdm > 0 else 0.0
    )

    return {
        "seed": seed,
        "proposal_policy": policy,
        "mh_proposals": mh_proposals,
        "mh_accepted": mh_accepted,
        "mh_rejected": mh_rejected,
        "acceptance_rate": acceptance_rate,
        "n": n,
        "m": m,
        "cyclomatic": cyclomatic,
        "E_3d": E_3d,
        "total_fdm": total_fdm,
        "total_entangled": total_ent,
        "penalty_factor_3d": penalty_factor_3d,
        "elapsed_sec": elapsed,
    }


def run_proposal_sweep(cfg: ProposalSweepConfig):
    results_dir = Path("results")
    results_dir.mkdir(parents=True, exist_ok=True)
    csv_path = results_dir / "proposal_policy_sweep.csv"
    summary_path = results_dir / "proposal_policy_sweep_summary.txt"

    rows: List[dict] = []

    t0 = time.time()

    for policy in ["uniform", "ctt_biased"]:
        for k in range(cfg.n_seeds):
            seed = cfg.seed_offset + k
            rows.append(_run_single(seed, policy))

    total_time = time.time() - t0

    fieldnames = [
        "seed",
        "proposal_policy",
        "mh_proposals",
        "mh_accepted",
        "mh_rejected",
        "acceptance_rate",
        "n",
        "m",
        "cyclomatic",
        "E_3d",
        "total_fdm",
        "total_entangled",
        "penalty_factor_3d",
        "elapsed_sec",
    ]

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    with summary_path.open("w", encoding="utf-8") as f:
        f.write(
            "SMART-GROWTH-0 proposal policy sweep\n"
            f"N_seeds={cfg.n_seeds}, policies=2, total_time_sec={total_time:.2f}\n\n"
        )

        for policy in ["uniform", "ctt_biased"]:
            subset = [r for r in rows if r["proposal_policy"] == policy]
            if not subset:
                continue

            acc = [r["acceptance_rate"] for r in subset]
            cyc = [r["cyclomatic"] for r in subset]
            e3d = [r["E_3d"] for r in subset]
            pen = [r["penalty_factor_3d"] for r in subset]

            acc_q10, acc_med, acc_q90 = _quantiles(acc, [0.1, 0.5, 0.9])
            cyc_med, cyc_q90 = _quantiles(cyc, [0.5, 0.9])
            e3d_med, e3d_q90 = _quantiles(e3d, [0.5, 0.9])
            pen_med, pen_q90 = _quantiles(pen, [0.5, 0.9])

            f.write(f"policy={policy}\n")
            f.write(
                f"  acceptance_rate: p10={acc_q10:.3f}, median={acc_med:.3f}, p90={acc_q90:.3f}\n"
            )
            f.write(
                f"  cyclomatic: median={cyc_med:.3f}, p90={cyc_q90:.3f}\n"
            )
            f.write(f"  E_3d: median={e3d_med:.3f}, p90={e3d_q90:.3f}\n")
            f.write(
                f"  penalty_factor_3d: median={pen_med:.3f}, p90={pen_q90:.3f}\n"
            )

            sorted_subset = sorted(
                subset, key=lambda r: r["E_3d"], reverse=True
            )
            f.write("  top-5 entangled:\n")
            for r in sorted_subset[:5]:
                f.write(
                    "    seed={seed}, E_3d={E_3d:.3f}, penalty={penalty_factor_3d:.3f}, "
                    "n={n}, m={m}, cyclomatic={cyclomatic}\n".format(**r)
                )
            f.write("\n")

    return csv_path, summary_path


def main() -> None:
    cfg = ProposalSweepConfig()
    csv_path, summary_path = run_proposal_sweep(cfg)
    print("[SMART-GROWTH-0] proposal policy sweep done.")
    print(f"CSV: {csv_path}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()

