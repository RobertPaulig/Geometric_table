from __future__ import annotations

import argparse
import csv
import math
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Tuple

from analysis.chem.core2_fit import kl_divergence
from analysis.chem.hetero_exact_small import (
    DEFAULT_RHO,
    DEFAULT_VALENCE,
    exact_distribution_for_type_counts,
)
from analysis.chem.hetero_mcmc import HeteroState, run_hetero_mcmc
from analysis.chem.hetero_operator import hetero_energy_from_state
from analysis.io_utils import ensure_results_dir


@dataclass(frozen=True)
class FormulaSpec:
    name: str
    type_counts: Dict[int, int]
    steps_init: int
    max_steps: int
    chains: int
    thin: int
    burnin_frac: float
    p_rewire: float

    @property
    def n_vertices(self) -> int:
        return int(sum(int(v) for v in self.type_counts.values()))


FORMULA_SPECS: Dict[str, FormulaSpec] = {
    "C2H6O": FormulaSpec("C2H6O", {0: 2, 2: 1}, steps_init=2000, max_steps=8000, chains=4, thin=4, burnin_frac=0.3, p_rewire=0.7),
    "C2H7N": FormulaSpec("C2H7N", {0: 2, 1: 1}, steps_init=2000, max_steps=8000, chains=4, thin=4, burnin_frac=0.3, p_rewire=0.7),
    "C3H8O": FormulaSpec("C3H8O", {0: 3, 2: 1}, steps_init=4000, max_steps=16000, chains=6, thin=4, burnin_frac=0.3, p_rewire=0.7),
    "C4H10O": FormulaSpec("C4H10O", {0: 4, 2: 1}, steps_init=8000, max_steps=32000, chains=6, thin=4, burnin_frac=0.3, p_rewire=0.7),
    "C4H11N": FormulaSpec("C4H11N", {0: 4, 1: 1}, steps_init=8000, max_steps=32000, chains=6, thin=4, burnin_frac=0.3, p_rewire=0.7),
    "C5H12O": FormulaSpec("C5H12O", {0: 5, 2: 1}, steps_init=12000, max_steps=48000, chains=6, thin=4, burnin_frac=0.3, p_rewire=0.7),
    "C5H13N": FormulaSpec("C5H13N", {0: 5, 1: 1}, steps_init=12000, max_steps=48000, chains=6, thin=4, burnin_frac=0.3, p_rewire=0.7),
}

DEFAULT_FORMULA = "C3H8O"


def _default_state_for_spec(spec: FormulaSpec) -> HeteroState:
    n = spec.n_vertices
    edges = tuple((i, i + 1) for i in range(n - 1))
    types: List[int] = []
    for t in sorted(spec.type_counts.keys()):
        types.extend([int(t)] * int(spec.type_counts[t]))
    return HeteroState(n=n, edges=edges, types=tuple(types))


def _energy_fn_builder(
    *,
    rho_by_type: Mapping[int, float],
    alpha_H: float,
    valence_by_type: Mapping[int, int],
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


def run_formula_validation(
    spec: FormulaSpec,
    *,
    steps_init: Optional[int] = None,
    max_steps: Optional[int] = None,
    chains: Optional[int] = None,
    thin: Optional[int] = None,
    burnin_frac: Optional[float] = None,
    beta: float,
    alpha_H: float,
    seed: int,
    coverage_target: float,
    p_rewire: Optional[float] = None,
) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, float]]:
    steps_total = int(steps_init or spec.steps_init)
    max_steps_total = int(max_steps or spec.max_steps)
    n_chains = int(chains or spec.chains)
    thin_val = int(thin or spec.thin)
    burnin_f = float(burnin_frac if burnin_frac is not None else spec.burnin_frac)
    p_rewire_val = float(p_rewire if p_rewire is not None else spec.p_rewire)

    exact = exact_distribution_for_type_counts(
        spec.type_counts,
        beta=float(beta),
        rho_by_type=dict(DEFAULT_RHO),
        alpha_H=float(alpha_H),
        valence_by_type=dict(DEFAULT_VALENCE),
    )

    rng_seed = int(seed)
    valence = dict(DEFAULT_VALENCE)
    rho = dict(DEFAULT_RHO)
    energy_fn = _energy_fn_builder(rho_by_type=rho, alpha_H=float(alpha_H), valence_by_type=valence)
    summary_meta: Dict[str, float] = {"formula": spec.name}

    while True:
        steps_per_chain = max(1, int(round(float(steps_total) / float(n_chains))))
        burnin = int(max(0, round(float(burnin_f) * float(steps_per_chain))))
        samples: List[Dict[str, object]] = []
        accepted = 0
        proposals = 0
        t0 = time.perf_counter()
        for chain_idx in range(n_chains):
            chain_seed = rng_seed + 101 * chain_idx
            chain_samples, chain_summary = run_hetero_mcmc(
                init=_default_state_for_spec(spec),
                steps=steps_per_chain,
                burnin=burnin,
                thin=thin_val,
                beta=float(beta),
                rng_seed=chain_seed,
                energy_fn=energy_fn,
                p_rewire=p_rewire_val,
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
            "formula": spec.name,
            "steps_total": steps_total,
            "steps_per_chain": steps_per_chain,
            "burnin": burnin,
            "chains": int(n_chains),
            "samples_collected": len(samples),
            "accept_rate": (accepted / proposals) if proposals > 0 else 0.0,
            "elapsed_sec": elapsed,
            "coverage_unique_eq": coverage,
            "kl_exact_emp": kl_exact_emp,
            "kl_emp_exact": kl_emp_exact,
        }
        if coverage >= float(coverage_target):
            break
        if steps_total >= max_steps_total:
            break
        steps_total = min(int(max_steps_total), int(steps_total) * 2)

    return exact, p_emp, summary_meta


def _resolve_out_dir(out_dir: Optional[str]) -> Path:
    if out_dir is None:
        base = ensure_results_dir()
        return base / "hetero_validation"
    return Path(out_dir)


def write_report(
    formula_name: str,
    exact: Dict[str, float],
    emp: Dict[str, float],
    meta: Dict[str, float],
    *,
    out_dir: Optional[str],
    out_stub: Optional[str] = None,
) -> Tuple[Path, Path]:
    dir_path = _resolve_out_dir(out_dir)
    dir_path.mkdir(parents=True, exist_ok=True)
    stub = out_stub or f"acid_{formula_name}"
    out_txt = dir_path / f"{stub}.txt"
    out_csv = dir_path / f"{stub}.csv"
    lines = []
    lines.append(f"{stub}: HETERO-1A acid test ({formula_name})")
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
    return out_txt, out_csv


def _parse_args(argv: Optional[List[str]]) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="HETERO-1A acid validation (single formula).")
    ap.add_argument("--formula", type=str, choices=sorted(FORMULA_SPECS.keys()), default=DEFAULT_FORMULA)
    ap.add_argument("--steps_init", type=int, default=None)
    ap.add_argument("--max_steps", type=int, default=None)
    ap.add_argument("--chains", type=int, default=None)
    ap.add_argument("--thin", type=int, default=None)
    ap.add_argument("--burnin_frac", type=float, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--beta", type=float, default=1.0)
    ap.add_argument("--alpha_H", type=float, default=0.5)
    ap.add_argument("--p_rewire", type=float, default=None)
    ap.add_argument("--coverage_target", type=float, default=1.0)
    ap.add_argument("--out_dir", type=str, default=None)
    ap.add_argument("--out_stub", type=str, default=None)
    return ap.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    args = _parse_args(argv)
    spec = FORMULA_SPECS[args.formula]
    exact, emp, meta = run_formula_validation(
        spec,
        steps_init=args.steps_init,
        max_steps=args.max_steps,
        chains=args.chains,
        thin=args.thin,
        burnin_frac=args.burnin_frac,
        beta=float(args.beta),
        alpha_H=float(args.alpha_H),
        seed=int(args.seed),
        coverage_target=float(args.coverage_target),
        p_rewire=args.p_rewire,
    )
    write_report(
        spec.name,
        exact,
        emp,
        meta,
        out_dir=args.out_dir,
        out_stub=args.out_stub,
    )


if __name__ == "__main__":
    main()
