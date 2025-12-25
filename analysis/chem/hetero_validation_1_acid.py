from __future__ import annotations

import argparse
import csv
import math
import time
from collections import Counter
from typing import Dict, List, Tuple

from analysis.chem.core2_fit import kl_divergence
from analysis.chem.hetero_exact_small import (
    DEFAULT_RHO,
    DEFAULT_VALENCE,
    exact_distribution_for_formula_C3H8O,
)
from analysis.chem.hetero_mcmc import HeteroState, run_hetero_mcmc
from analysis.chem.hetero_operator import hetero_energy_from_state
from analysis.io_utils import results_path


def _default_state() -> HeteroState:
    n = 4
    edges = ((0, 1), (1, 2), (2, 3))
    # Oxygen at node 3.
    types = (0, 0, 0, 2)
    return HeteroState(n=n, edges=tuple(sorted(edges)), types=types)


def _energy_fn_builder(
    *,
    rho_by_type: Dict[int, float],
    alpha_H: float,
    valence_by_type: Dict[int, int],
) -> callable:
    def _energy(state: HeteroState) -> float:
        return hetero_energy_from_state(
            state.n,
            state.edges,
            state.types,
            rho_by_type=rho_by_type,
            alpha_H=alpha_H,
            valence_by_type=valence_by_type,
        )

    return _energy


def _empirical_distribution(samples: List[Dict[str, object]]) -> Dict[str, float]:
    counts = Counter()
    for s in samples:
        counts[str(s["state_id"])] += 1
    total = sum(counts.values()) or 1
    return {k: v / float(total) for k, v in counts.items()}


def run_validation(args: argparse.Namespace) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, float]]:
    exact = exact_distribution_for_formula_C3H8O(
        beta=float(args.beta),
        rho_by_type=dict(DEFAULT_RHO),
        alpha_H=float(args.alpha_H),
        valence_by_type=dict(DEFAULT_VALENCE),
    )

    steps_total = int(args.steps_init)
    rng_seed = int(args.seed)
    valence = dict(DEFAULT_VALENCE)
    rho = dict(DEFAULT_RHO)
    energy_fn = _energy_fn_builder(rho_by_type=rho, alpha_H=float(args.alpha_H), valence_by_type=valence)
    summary_meta: Dict[str, float] = {}

    while True:
        steps_per_chain = max(1, int(round(float(steps_total) / float(args.chains))))
        burnin = int(max(0, round(float(args.burnin_frac) * float(steps_per_chain))))
        samples: List[Dict[str, object]] = []
        accepted = 0
        proposals = 0
        t0 = time.perf_counter()
        for chain_idx in range(int(args.chains)):
            chain_seed = rng_seed + 101 * chain_idx
            chain_samples, chain_summary = run_hetero_mcmc(
                init=_default_state(),
                steps=steps_per_chain,
                burnin=burnin,
                thin=int(args.thin),
                beta=float(args.beta),
                rng_seed=chain_seed,
                energy_fn=energy_fn,
                p_rewire=float(args.p_rewire),
                valence_by_type=valence,
            )
            samples.extend(chain_samples)
            accepted += chain_summary.accepted
            proposals += chain_summary.proposals
        elapsed = time.perf_counter() - t0
        p_emp = _empirical_distribution(samples)
        coverage = float(len(p_emp)) / float(len(exact)) if exact else 0.0
        kl_exact_emp = kl_divergence(exact, p_emp)
        kl_emp_exact = kl_divergence(p_emp, exact)
        summary_meta = {
            "steps_total": steps_total,
            "steps_per_chain": steps_per_chain,
            "burnin": burnin,
            "chains": int(args.chains),
            "samples_collected": len(samples),
            "accept_rate": (accepted / proposals) if proposals > 0 else 0.0,
            "elapsed_sec": elapsed,
            "coverage_unique_eq": coverage,
            "kl_exact_emp": kl_exact_emp,
            "kl_emp_exact": kl_emp_exact,
        }
        if coverage >= float(args.coverage_target):
            break
        if steps_total >= int(args.max_steps):
            break
        steps_total = min(int(args.max_steps), int(steps_total) * 2)

    return exact, p_emp, summary_meta


def write_report(exact: Dict[str, float], emp: Dict[str, float], meta: Dict[str, float]) -> None:
    out_txt = results_path("hetero_validation_1_acid.txt")
    lines = []
    lines.append("hetero_validation_1_acid: HETERO-1A acid test (C3H8O)")
    lines.append(f"steps_total={int(meta['steps_total'])} steps_per_chain={int(meta['steps_per_chain'])} burnin={int(meta['burnin'])}")
    lines.append(f"chains={int(meta['chains'])} samples={int(meta['samples_collected'])}")
    lines.append(f"accept_rate={meta['accept_rate']:.4f} elapsed_sec={meta['elapsed_sec']:.2f}")
    lines.append(f"coverage_unique_eq={meta['coverage_unique_eq']:.4f} support_exact={len(exact)} support_emp={len(emp)}")
    lines.append(f"KL(exact||emp)={meta['kl_exact_emp']:.6f}")
    lines.append(f"KL(emp||exact)={meta['kl_emp_exact']:.6f}")
    lines.append("")
    lines.append("Top states by P_emp:")
    for state_id, p in sorted(emp.items(), key=lambda kv: -kv[1])[:10]:
        lines.append(f"  {state_id} P_emp={p:.6f} P_exact={exact.get(state_id, 0.0):.6f}")
    out_txt.write_text("\n".join(lines), encoding="utf-8")

    out_csv = results_path("hetero_validation_1_acid.csv")
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["state_id", "P_exact", "P_emp"])
        writer.writeheader()
        for state_id in sorted(set(exact.keys()) | set(emp.keys())):
            writer.writerow(
                {
                    "state_id": state_id,
                    "P_exact": exact.get(state_id, 0.0),
                    "P_emp": emp.get(state_id, 0.0),
                }
            )
    print(f"[HETERO-ACID] wrote {out_txt}")
    print(f"[HETERO-ACID] wrote {out_csv}")


def main(argv: List[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="HETERO-1A acid validation (C3H8O).")
    ap.add_argument("--steps_init", type=int, default=2000)
    ap.add_argument("--max_steps", type=int, default=10000)
    ap.add_argument("--chains", type=int, default=4)
    ap.add_argument("--thin", type=int, default=5)
    ap.add_argument("--burnin_frac", type=float, default=0.30)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--beta", type=float, default=1.0)
    ap.add_argument("--alpha_H", type=float, default=0.5)
    ap.add_argument("--p_rewire", type=float, default=0.7)
    ap.add_argument("--coverage_target", type=float, default=1.0)
    args = ap.parse_args(argv)

    exact, emp, meta = run_validation(args)
    write_report(exact, emp, meta)


if __name__ == "__main__":
    main()
