from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, List


TRAIN_DEFAULT = ["C4H10O", "C5H12O", "C4H11N", "C5H13N"]
HOLDOUT_DEFAULT = ["C6H14O", "C7H16O", "C6H15N", "C7H17N"]


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="HETERO-1A P0.7: train/holdout calibration and validation.")
    ap.add_argument("--train", nargs="+", default=TRAIN_DEFAULT)
    ap.add_argument("--holdout", nargs="+", default=HOLDOUT_DEFAULT)
    ap.add_argument("--out_root", type=str, default="results/hetero_calibration_p07")
    ap.add_argument("--out_holdout", type=str, default="results/hetero_suite_p07_holdout")
    ap.add_argument("--beta_grid", type=float, nargs="+", default=[0.9, 1.0])
    ap.add_argument("--alpha_grid", type=float, nargs="+", default=[0.45, 0.5])
    ap.add_argument("--coverage_min", type=float, default=1.0)
    ap.add_argument("--kl_max", type=float, default=0.05)
    ap.add_argument("--fp_auc_min", type=float, default=0.85)
    ap.add_argument("--fp_auc_gap_min", type=float, default=0.02)
    ap.add_argument("--coll_cross_pairs_strict_max", type=int, default=0)
    ap.add_argument("--clean", action="store_true", help="Delete output dirs before running.")
    return ap.parse_args()


def _run(cmd: List[str]) -> None:
    print(f"[P0.7] running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def _read_suite_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return [dict(r) for r in reader]


def _print_table(rows: List[Dict[str, str]], formulas: List[str], *, title: str) -> None:
    wanted = [
        "formula",
        "coverage_unique_eq",
        "kl_exact_emp",
        "coll_cross_pairs",
        "coll_cross_pairs_strict",
        "fp_policy_used",
        "fp_best_idx",
        "fp_best_auc_best",
        "fp_best_auc_gap",
        "fp_energy_spearman_abs",
        "n_other",
        "other_frac",
    ]
    by_formula = {r.get("formula", ""): r for r in rows if r.get("weight_source") == "P_exact"}
    print(title)
    for formula in formulas:
        r = by_formula.get(formula, {})
        print(", ".join(f"{k}={r.get(k, '')}" for k in wanted))


def main() -> None:
    args = _parse_args()
    out_root = Path(args.out_root)
    out_holdout = Path(args.out_holdout)
    if args.clean:
        if out_root.exists():
            shutil.rmtree(out_root)
        if out_holdout.exists():
            shutil.rmtree(out_holdout)

    _run(
        [
            sys.executable,
            "-m",
            "analysis.chem.hetero_calibration_loop",
            "--out_root",
            str(out_root),
            "--formulas",
            *args.train,
            "--beta_grid",
            *[str(x) for x in args.beta_grid],
            "--alpha_grid",
            *[str(x) for x in args.alpha_grid],
            "--coverage_min",
            str(args.coverage_min),
            "--kl_max",
            str(args.kl_max),
            "--fp_auc_min",
            str(args.fp_auc_min),
            "--fp_auc_gap_min",
            str(args.fp_auc_gap_min),
            "--coll_cross_pairs_strict_max",
            str(args.coll_cross_pairs_strict_max),
        ]
    )

    best_json = out_root / "calib_best.json"
    if not best_json.exists():
        raise FileNotFoundError(f"Expected {best_json} after calibration")
    best = json.loads(best_json.read_text(encoding="utf-8"))
    print(f"[P0.7] best: {best}")

    # Train table (copied by calibration loop).
    train_rows = _read_suite_csv(out_root / "calib_best_suite.csv")
    _print_table(train_rows, args.train, title="[P0.7] TRAIN (calib_best_suite.csv)")

    # Holdout run with the chosen theta.
    _run(
        [
            sys.executable,
            "-m",
            "analysis.chem.hetero_validation_suite",
            "--theta_json",
            str(best_json),
            "--fp_exclude_energy_like",
            "--formulas",
            *args.holdout,
            "--out_dir",
            str(out_holdout),
            "--stub_prefix",
            "holdout",
        ]
    )
    holdout_rows = _read_suite_csv(out_holdout / "hetero_validation_suite.csv")
    _print_table(holdout_rows, args.holdout, title="[P0.7] HOLDOUT (hetero_validation_suite.csv)")


if __name__ == "__main__":
    main()

