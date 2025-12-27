from __future__ import annotations

import argparse
import csv
import itertools
import json
from pathlib import Path
from typing import Dict, List

from analysis.chem.hetero_validation_suite import main as suite_main


FORMULAS_DEFAULT = ["C4H10O", "C5H12O", "C4H11N", "C5H13N", "C6H14O", "C7H16O", "C6H15N", "C7H17N"]


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="HETERO-1A P0.8: local theta grid around theta_best.")
    ap.add_argument("--theta_json", type=str, default="analysis/chem/configs/hetero_theta_best.json")
    ap.add_argument("--formulas", nargs="+", default=FORMULAS_DEFAULT)
    ap.add_argument("--delta_beta", type=float, default=0.05)
    ap.add_argument("--delta_alpha_H", type=float, default=0.02)
    ap.add_argument("--out_root", type=str, default="results/hetero_p08_theta_grid")
    ap.add_argument("--out_csv", type=str, default="results/hetero_p08_theta_grid.csv")
    return ap.parse_args()


def main() -> None:
    args = _parse_args()
    base = json.loads(Path(args.theta_json).read_text(encoding="utf-8"))
    beta0 = float(base["beta"])
    alpha0 = float(base["alpha_H"])
    beta_grid = [beta0 - args.delta_beta, beta0, beta0 + args.delta_beta]
    alpha_grid = [alpha0 - args.delta_alpha_H, alpha0, alpha0 + args.delta_alpha_H]

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    out_rows: List[Dict[str, object]] = []

    for beta, alpha in itertools.product(beta_grid, alpha_grid):
        theta_tag = f"beta={beta:.4f}_alphaH={alpha:.4f}"
        theta_file = out_root / f"theta_{theta_tag}.json"
        theta_file.write_text(json.dumps({"beta": beta, "alpha_H": alpha}, indent=2), encoding="utf-8")
        out_dir = out_root / theta_tag
        suite_main(
            [
                "--theta_json",
                str(theta_file),
                "--fp_exclude_energy_like",
                "--formulas",
                *args.formulas,
                "--out_dir",
                str(out_dir),
                "--stub_prefix",
                "grid",
            ]
        )
        suite_csv = out_dir / "hetero_validation_suite.csv"
        with suite_csv.open("r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get("weight_source") != "P_exact":
                    continue
                out_rows.append(
                    {
                        "beta": beta,
                        "alpha_H": alpha,
                        "formula": row.get("formula"),
                        "coll_cross_pairs_strict": float(row.get("coll_cross_pairs_strict") or 0.0),
                        "coll_cross_pairs": float(row.get("coll_cross_pairs") or 0.0),
                        "max_abs_delta_cross": float(row.get("max_abs_delta_cross") or 0.0),
                        "coverage_unique_eq": float(row.get("coverage_unique_eq") or 0.0),
                        "kl_exact_emp": float(row.get("kl_exact_emp") or 0.0),
                    }
                )

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["beta", "alpha_H", "formula", "coverage_unique_eq", "kl_exact_emp", "coll_cross_pairs", "coll_cross_pairs_strict", "max_abs_delta_cross"]
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in out_rows:
            writer.writerow(r)
    print(f"[P0.8] wrote {out_csv}")


if __name__ == "__main__":
    main()

