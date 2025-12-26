from __future__ import annotations

import argparse
import csv
import itertools
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

from analysis.chem.hetero_validation_1_acid import FORMULA_SPECS


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="HETERO-1A calibration loop over validation suite.")
    ap.add_argument("--out_root", type=str, default="results/hetero_calibration", help="Directory to keep calibration trials.")
    ap.add_argument("--formulas", type=str, nargs="*", choices=sorted(FORMULA_SPECS.keys()), help="Subset of formulas (default: all).")
    ap.add_argument("--beta_grid", type=float, nargs="+", default=[1.0], help="Grid of beta values.")
    ap.add_argument("--alpha_grid", type=float, nargs="+", default=[0.5], help="Grid of alpha_H values.")
    ap.add_argument("--coverage_min", type=float, default=1.0)
    ap.add_argument("--kl_max", type=float, default=0.02)
    ap.add_argument("--lambda_other", type=float, default=0.3)
    ap.add_argument("--lambda_collision", type=float, default=0.1)
    ap.add_argument("--min_nontrivial_rows", type=int, default=2)
    ap.add_argument("--suite_kwargs", type=str, nargs="*", help="Extra passthrough pairs key=value for suite CLI.")
    return ap.parse_args(argv)


def _boolify(value: str | int | float | bool) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _run_suite(out_dir: Path, *, beta: float, alpha_H: float, formulas: List[str] | None, extra_args: Dict[str, str]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        "-m",
        "analysis.chem.hetero_validation_suite",
        "--out_dir",
        str(out_dir),
        "--beta",
        str(beta),
        "--alpha_H",
        str(alpha_H),
    ]
    if formulas:
        cmd.extend(["--formulas", *formulas])
    for key, val in extra_args.items():
        cmd.append(f"--{key}")
        if val:
            cmd.append(val)
    subprocess.run(cmd, check=True)


def _read_summary(path: Path) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(dict(row))
    return rows


def _evaluate_trial(rows: List[Dict[str, object]], *, args: argparse.Namespace) -> tuple[str, str, float]:
    if not rows:
        return "fail", "empty_summary", float("-inf")
    for row in rows:
        coverage = float(row.get("coverage_unique_eq", 0.0))
        kl_val = float(row.get("kl_exact_emp", 0.0))
        if coverage < args.coverage_min:
            return "fail", "coverage_below_min", float("-inf")
        if kl_val > args.kl_max:
            return "fail", "kl_above_threshold", float("-inf")
    non_trivial = [row for row in rows if not _boolify(row.get("fp_best_is_trivial", False))]
    if len(non_trivial) < args.min_nontrivial_rows:
        return "fail", "nontrivial_rows_lt_min", float("-inf")
    total_score = 0.0
    for row in non_trivial:
        n_a = float(row.get("n_a", 0))
        n_b = float(row.get("n_b", 0))
        n_other = float(row.get("n_other", 0))
        denom = n_a + n_b + n_other
        other_frac = (n_other / denom) if denom > 0 else 0.0
        collision = float(row.get("energy_collision_rate", 0.0))
        auc_best = float(row.get("fp_best_auc_best", 0.0))
        row_score = auc_best - args.lambda_other * other_frac - args.lambda_collision * collision
        total_score += row_score
    return "ok", "pass", total_score


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    formulas = list(args.formulas) if args.formulas else sorted(FORMULA_SPECS.keys())
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    trials_path = out_root / "calib_trials.csv"
    best_json_path = out_root / "calib_best.json"
    best_suite_path = out_root / "calib_best_suite.csv"
    extra_kwargs: Dict[str, str] = {}
    if args.suite_kwargs:
        for item in args.suite_kwargs:
            if "=" in item:
                key, val = item.split("=", 1)
            else:
                key, val = item, ""
            extra_kwargs[key.strip().replace("--", "")] = val.strip()

    trial_records: List[Dict[str, object]] = []
    best_trial: Dict[str, object] | None = None
    trial_id = 0
    for beta, alpha in itertools.product(args.beta_grid, args.alpha_grid):
        trial_id += 1
        trial_dir = out_root / f"trial_{trial_id:03d}"
        if trial_dir.exists():
            shutil.rmtree(trial_dir)
        _run_suite(trial_dir, beta=beta, alpha_H=alpha, formulas=formulas, extra_args=extra_kwargs)
        summary_path = trial_dir / "hetero_validation_suite.csv"
        rows = _read_summary(summary_path)
        status, reason, score = _evaluate_trial(rows, args=args)
        record = {
            "trial_id": trial_id,
            "beta": beta,
            "alpha_H": alpha,
            "status": status,
            "reason": reason,
            "score": score if status == "ok" else "",
            "nontrivial_rows": sum(0 if _boolify(r.get("fp_best_is_trivial", False)) else 1 for r in rows),
            "output_dir": str(trial_dir),
        }
        trial_records.append(record)
        if status == "ok" and (best_trial is None or score > float(best_trial["score"])):
            best_trial = {"score": score, "beta": beta, "alpha_H": alpha, "output_dir": str(trial_dir)}

    fieldnames = ["trial_id", "beta", "alpha_H", "status", "reason", "score", "nontrivial_rows", "output_dir"]
    with trials_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for record in trial_records:
            writer.writerow(record)

    if best_trial is None:
        best_json_path.write_text(json.dumps({"status": "fail", "reason": "no_successful_trials"}, indent=2), encoding="utf-8")
        if best_suite_path.exists():
            best_suite_path.unlink()
        print("[HETERO-CALIB] no successful trials; see calib_trials.csv for details")
        return

    best_json = {
        "status": "ok",
        "score": best_trial["score"],
        "beta": best_trial["beta"],
        "alpha_H": best_trial["alpha_H"],
        "output_dir": best_trial["output_dir"],
    }
    best_json_path.write_text(json.dumps(best_json, indent=2), encoding="utf-8")
    summary_src = Path(best_trial["output_dir"]) / "hetero_validation_suite.csv"
    shutil.copyfile(summary_src, best_suite_path)
    print(f"[HETERO-CALIB] best score={best_trial['score']:.4f} beta={best_trial['beta']} alpha_H={best_trial['alpha_H']}")


if __name__ == "__main__":
    main()
