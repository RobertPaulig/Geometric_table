from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List

from analysis.chem.hetero_validation_suite import main as suite_main


FORMULAS_DEFAULT = ["C4H10O", "C5H12O", "C4H11N", "C5H13N", "C6H14O", "C7H16O", "C6H15N", "C7H17N"]


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="HETERO-1A P0.8: seed robustness scan at fixed theta.")
    ap.add_argument("--theta_json", type=str, default="analysis/chem/configs/hetero_theta_best.json")
    ap.add_argument("--formulas", nargs="+", default=FORMULAS_DEFAULT)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--out_csv", type=str, default="results/hetero_p08_seed_scan.csv")
    ap.add_argument("--out_root", type=str, default="results/hetero_p08_seed_runs")
    return ap.parse_args()


def main() -> None:
    args = _parse_args()
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    out_csv = Path(args.out_csv)
    rows_out: List[Dict[str, object]] = []
    for seed in args.seeds:
        out_dir = out_root / f"seed_{seed}"
        suite_main(
            [
                "--theta_json",
                args.theta_json,
                "--fp_exclude_energy_like",
                "--seed",
                str(seed),
                "--formulas",
                *args.formulas,
                "--out_dir",
                str(out_dir),
                "--stub_prefix",
                f"seed{seed}",
            ]
        )
        suite_csv = out_dir / "hetero_validation_suite.csv"
        import csv as _csv

        with suite_csv.open("r", newline="", encoding="utf-8") as f:
            reader = _csv.DictReader(f)
            for row in reader:
                if row.get("weight_source") != "P_exact":
                    continue
                rows_out.append(
                    {
                        "seed": seed,
                        "formula": row.get("formula"),
                        "coverage_unique_eq": float(row.get("coverage_unique_eq") or 0.0),
                        "kl_exact_emp": float(row.get("kl_exact_emp") or 0.0),
                        "coll_cross_pairs_strict": float(row.get("coll_cross_pairs_strict") or 0.0),
                        "fp_best_auc_best": float(row.get("fp_best_auc_best") or 0.0),
                        "fp_best_auc_gap": float(row.get("fp_best_auc_gap") or 0.0),
                    }
                )
    fieldnames = ["seed", "formula", "coverage_unique_eq", "kl_exact_emp", "coll_cross_pairs_strict", "fp_best_auc_best", "fp_best_auc_gap"]
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows_out:
            writer.writerow(row)
    print(f"[P0.8] wrote {out_csv}")


if __name__ == "__main__":
    main()

