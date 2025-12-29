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
    ap.add_argument("--seeds", type=int, nargs="+", default=list(range(20)))
    ap.add_argument("--neg_controls", action="store_true", help="Compute negative controls (perm labels + random fp).")
    ap.add_argument("--neg_control_seed", type=int, default=0)
    ap.add_argument("--out_csv", type=str, default="results/robustness/seed_robustness_raw.csv")
    ap.add_argument("--out_summary_csv", type=str, default="results/robustness/seed_robustness_summary.csv")
    ap.add_argument("--out_root", type=str, default="results/robustness/seed_runs")
    ap.add_argument("--coverage_req", type=float, default=1.0)
    ap.add_argument("--kl_max", type=float, default=0.05)
    ap.add_argument("--fp_auc_min", type=float, default=0.95)
    ap.add_argument("--fp_auc_gap_min", type=float, default=0.02)
    ap.add_argument(
        "--neg_auc_max",
        type=float,
        default=None,
        help="Optional absolute max AUC for negative controls (override). If omitted, gates use per-row fp_neg_auc_gate.",
    )
    ap.add_argument("--neg_control_reps", type=int, default=50)
    ap.add_argument("--neg_control_quantile", type=float, default=0.95)
    ap.add_argument("--neg_auc_margin", type=float, default=0.05)
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
                "--neg_control_seed",
                str(int(args.neg_control_seed)),
                "--neg_control_reps",
                str(int(args.neg_control_reps)),
                "--neg_control_quantile",
                str(float(args.neg_control_quantile)),
                "--neg_auc_margin",
                str(float(args.neg_auc_margin)),
                *(["--neg_control_permute_labels"] if args.neg_controls else []),
                *(["--neg_control_random_fp"] if args.neg_controls else []),
            ]
        )
        suite_csv = out_dir / "hetero_validation_suite.csv"
        import csv as _csv

        with suite_csv.open("r", newline="", encoding="utf-8") as f:
            reader = _csv.DictReader(f)
            for row in reader:
                if row.get("weight_source") != "P_exact":
                    continue
                neg_perm = row.get("fp_neg_auc_best_perm_labels", "")
                neg_rand = row.get("fp_neg_auc_best_rand_fp", "")
                neg_perm_f = float(neg_perm) if neg_perm not in {"", "nan", "NaN", None} else float("nan")
                neg_rand_f = float(neg_rand) if neg_rand not in {"", "nan", "NaN", None} else float("nan")
                max_neg = float("nan")
                gate_val = float("nan")
                if args.neg_controls:
                    vals = [v for v in [neg_perm_f, neg_rand_f] if v == v]
                    max_neg = max(vals) if vals else float("nan")
                    gate_str = row.get("fp_neg_auc_gate", "")
                    gate_val = float(gate_str) if gate_str not in {"", "nan", "NaN", None} else float("nan")
                rows_out.append(
                    {
                        "seed": seed,
                        "formula": row.get("formula"),
                        "coverage_unique_eq": float(row.get("coverage_unique_eq") or 0.0),
                        "kl_exact_emp": float(row.get("kl_exact_emp") or 0.0),
                        "coll_cross_pairs_strict": float(row.get("coll_cross_pairs_strict") or 0.0),
                        "fp_best_auc_best": float(row.get("fp_best_auc_best") or 0.0),
                        "fp_best_auc_gap": float(row.get("fp_best_auc_gap") or 0.0),
                        "fp_neg_auc_best_perm_labels": neg_perm_f,
                        "fp_neg_auc_best_rand_fp": neg_rand_f,
                        "max_neg_auc": max_neg,
                        "neg_auc_gate": gate_val,
                    }
                )
    fieldnames = [
        "seed",
        "formula",
        "coverage_unique_eq",
        "kl_exact_emp",
        "coll_cross_pairs_strict",
        "fp_best_auc_best",
        "fp_best_auc_gap",
        "fp_neg_auc_best_perm_labels",
        "fp_neg_auc_best_rand_fp",
        "max_neg_auc",
        "neg_auc_gate",
    ]
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows_out:
            writer.writerow(row)
    print(f"[P0.8] wrote {out_csv}")

    # Aggregate per formula across seeds into a PASS/FAIL gate table.
    by_formula: Dict[str, List[Dict[str, object]]] = {}
    for r in rows_out:
        by_formula.setdefault(str(r["formula"]), []).append(r)
    summary_rows: List[Dict[str, object]] = []
    for formula, rr in sorted(by_formula.items()):
        min_cov = min(float(x["coverage_unique_eq"]) for x in rr)
        max_kl = max(float(x["kl_exact_emp"]) for x in rr)
        max_cross_strict = max(float(x["coll_cross_pairs_strict"]) for x in rr)
        min_auc = min(float(x["fp_best_auc_best"]) for x in rr)
        min_gap = min(float(x["fp_best_auc_gap"]) for x in rr)
        max_neg_auc = max(
            (float(x["max_neg_auc"]) for x in rr if float(x["max_neg_auc"]) == float(x["max_neg_auc"])),
            default=float("nan"),
        )
        max_neg_gate = max(
            (float(x["neg_auc_gate"]) for x in rr if float(x["neg_auc_gate"]) == float(x["neg_auc_gate"])),
            default=float("nan"),
        )
        gate_ok = (
            min_cov >= float(args.coverage_req)
            and max_kl <= float(args.kl_max)
            and max_cross_strict <= 0.0
            and min_auc >= float(args.fp_auc_min)
            and min_gap >= float(args.fp_auc_gap_min)
            and (
                (not args.neg_controls)
                or (
                    (args.neg_auc_max is not None and max_neg_auc == max_neg_auc and max_neg_auc <= float(args.neg_auc_max))
                    or (args.neg_auc_max is None and max_neg_auc == max_neg_auc and max_neg_gate == max_neg_gate and max_neg_auc <= max_neg_gate)
                )
            )
        )
        summary_rows.append(
            {
                "formula": formula,
                "seeds_n": len(rr),
                "min_coverage": min_cov,
                "max_kl_exact_emp": max_kl,
                "max_coll_cross_pairs_strict": max_cross_strict,
                "min_fp_best_auc_best": min_auc,
                "min_fp_best_auc_gap": min_gap,
                "max_neg_auc_any": max_neg_auc,
                "neg_auc_gate_any": max_neg_gate,
                "neg_auc_margin": float(args.neg_auc_margin),
                "gate_pass": bool(gate_ok),
            }
        )
    out_summary = Path(args.out_summary_csv)
    out_summary.parent.mkdir(parents=True, exist_ok=True)
    summary_fields = [
        "formula",
        "seeds_n",
        "min_coverage",
        "max_kl_exact_emp",
        "max_coll_cross_pairs_strict",
        "min_fp_best_auc_best",
        "min_fp_best_auc_gap",
        "max_neg_auc_any",
        "neg_auc_gate_any",
        "neg_auc_margin",
        "gate_pass",
    ]
    with out_summary.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=summary_fields)
        writer.writeheader()
        for r in summary_rows:
            writer.writerow(r)
    overall = all(bool(r["gate_pass"]) for r in summary_rows) if summary_rows else False
    print(f"[P0.8] wrote {out_summary} overall_pass={overall}")


if __name__ == "__main__":
    main()
