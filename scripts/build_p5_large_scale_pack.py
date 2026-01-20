from __future__ import annotations

import argparse
from pathlib import Path

from hetero2.scale_proof import P5Config, write_p5_evidence_pack


def _parse_bins(text: str) -> tuple[int, ...]:
    items = [p.strip() for p in str(text).split(",") if p.strip()]
    if not items:
        raise ValueError("n_atoms_bins must be a comma-separated list of ints (e.g. 20,50,100)")
    bins: list[int] = []
    for it in items:
        bins.append(int(it))
    return tuple(bins)


def main() -> int:
    p = argparse.ArgumentParser(description="Build PHYSICS-P5 large-scale evidence pack (speedup vs N_atoms).")
    p.add_argument("--out_dir", required=True, help="Output directory for evidence pack files.")
    p.add_argument("--n_atoms_bins", default="20,50,100,200,400,800", help="Comma-separated N_atoms bins.")
    p.add_argument("--samples_per_bin", type=int, default=10, help="Number of samples per N_atoms bin.")
    p.add_argument("--seed", type=int, default=0, help="Global seed for fixture generation.")
    p.add_argument("--curve_id", default="dos_H", help="Curve id label written to metadata (informational).")
    p.add_argument("--energy_points", type=int, default=128, help="Number of energy grid points for baseline.")
    p.add_argument("--dos_eta", type=float, default=0.05, help="Kernel width eta for DOS.")
    p.add_argument("--potential_scale_gamma", type=float, default=1.0, help="Global scale gamma for V (dimensionless).")
    p.add_argument(
        "--edge_weight_mode",
        default="bond_order_delta_chi",
        choices=["unweighted", "bond_order", "bond_order_delta_chi"],
        help="Edge weighting mode for the chain operator.",
    )
    p.add_argument("--integrator_eps_abs", type=float, default=1e-6, help="Adaptive abs tolerance.")
    p.add_argument("--integrator_eps_rel", type=float, default=1e-4, help="Adaptive rel tolerance.")
    p.add_argument("--integrator_subdomains_max", type=int, default=32, help="Adaptive max subdomains.")
    p.add_argument("--integrator_poly_degree_max", type=int, default=16, help="Adaptive max polynomial degree.")
    p.add_argument("--integrator_quad_order_max", type=int, default=32, help="Adaptive max Gauss quad order.")
    p.add_argument("--integrator_eval_budget_max", type=int, default=256, help="Adaptive eval budget cap.")
    p.add_argument("--integrator_split_criterion", default="curvature", help="Adaptive split criterion label.")
    p.add_argument("--overhead_region_n_max", type=int, default=100, help="Overhead region threshold for verdict.")
    p.add_argument("--gate_n_min", type=int, default=200, help="Minimum N_atoms bin for speedup gating.")
    p.add_argument("--correctness_gate_rate", type=float, default=1.0, help="Correctness pass-rate gate at scale.")
    p.add_argument("--min_scale_samples", type=int, default=5, help="Minimum scale samples for speedup verdict.")
    p.add_argument("--speedup_gate_break_even", type=float, default=1.0, help="Speedup gate (break-even).")
    p.add_argument("--speedup_gate_strong", type=float, default=2.0, help="Speedup gate (strong).")
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = P5Config(
        n_atoms_bins=_parse_bins(args.n_atoms_bins),
        samples_per_bin=int(args.samples_per_bin),
        seed=int(args.seed),
        curve_id=str(args.curve_id),
        energy_points=int(args.energy_points),
        dos_eta=float(args.dos_eta),
        potential_scale_gamma=float(args.potential_scale_gamma),
        edge_weight_mode=str(args.edge_weight_mode),
        integrator_eps_abs=float(args.integrator_eps_abs),
        integrator_eps_rel=float(args.integrator_eps_rel),
        integrator_subdomains_max=int(args.integrator_subdomains_max),
        integrator_poly_degree_max=int(args.integrator_poly_degree_max),
        integrator_quad_order_max=int(args.integrator_quad_order_max),
        integrator_eval_budget_max=int(args.integrator_eval_budget_max),
        integrator_split_criterion=str(args.integrator_split_criterion),
        overhead_region_n_max=int(args.overhead_region_n_max),
        gate_n_min=int(args.gate_n_min),
        correctness_gate_rate=float(args.correctness_gate_rate),
        min_scale_samples=int(args.min_scale_samples),
        speedup_gate_break_even=float(args.speedup_gate_break_even),
        speedup_gate_strong=float(args.speedup_gate_strong),
    )
    zip_path = write_p5_evidence_pack(out_dir=out_dir, cfg=cfg)
    print(str(zip_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

