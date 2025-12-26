from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import List

from analysis.chem.hetero_validation_1_acid import (
    FORMULA_SPECS,
    run_formula_validation,
    write_report,
)


def _parse_args(argv: List[str] | None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="HETERO-1A validation suite for multiple formulas.")
    ap.add_argument("--formulas", type=str, nargs="*", default=list(FORMULA_SPECS.keys()), choices=sorted(FORMULA_SPECS.keys()))
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
    ap.add_argument("--out_dir", type=str, default="results/golden/hetero")
    ap.add_argument("--stub_prefix", type=str, default="acid")
    return ap.parse_args(argv)


def main(argv: List[str] | None = None) -> None:
    args = _parse_args(argv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_rows = []
    for formula_name in args.formulas:
        spec = FORMULA_SPECS[formula_name]
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
        stub = f"{args.stub_prefix}_{formula_name}"
        write_report(formula_name, exact, emp, meta, out_dir=str(out_dir), out_stub=stub)
        row = {
            "formula": formula_name,
            "steps_total": int(meta["steps_total"]),
            "steps_per_chain": int(meta["steps_per_chain"]),
            "coverage_unique_eq": float(meta["coverage_unique_eq"]),
            "kl_exact_emp": float(meta["kl_exact_emp"]),
            "kl_emp_exact": float(meta["kl_emp_exact"]),
            "accept_rate": float(meta["accept_rate"]),
            "samples": int(meta["samples_collected"]),
        }
        summary_rows.append(row)
    summary_path = out_dir / "hetero_validation_suite.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "formula",
                "steps_total",
                "steps_per_chain",
                "coverage_unique_eq",
                "kl_exact_emp",
                "kl_emp_exact",
                "accept_rate",
                "samples",
            ],
        )
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"[HETERO-SUITE] wrote {summary_path}")


if __name__ == "__main__":
    main()
