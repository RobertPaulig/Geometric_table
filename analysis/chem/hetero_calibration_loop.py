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
    ap.add_argument("--fp_allow_energy_like", action="store_true", help="Allow FP candidates highly correlated with energy.")
    return ap.parse_args(argv)


def _boolify(value: str | int | float | bool) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _run_suite(
    out_dir: Path,
    *,
    beta: float,
    alpha_H: float,
    formulas: List[str] | None,
    extra_args: Dict[str, str],
    fp_allow_energy_like: bool,
) -> None:
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
    if fp_allow_energy_like:
        cmd.append("--fp_allow_energy_like")
    else:
        cmd.append("--fp_exclude_energy_like")
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


def _evaluate_trial(rows: List[Dict[str, object]], *, args: argparse.Namespace) -> tuple[str, str, float, Dict[str, float], Dict[str, object]]:
    if not rows:
        return "fail", "empty_summary", float("-inf"), {}, {}
    coverage_vals = [float(row.get("coverage_unique_eq", 0.0)) for row in rows]
    kl_vals = [float(row.get("kl_exact_emp", 0.0)) for row in rows]
    cross_pairs_vals = [float(row.get("coll_cross_pairs", 0.0) or 0.0) for row in rows]
    within_pairs_vals = [float(row.get("coll_within_pairs", 0.0) or 0.0) for row in rows]
    total_pairs_vals = [float(row.get("total_pairs", 0.0) or 0.0) for row in rows]
    coll_total_vals = [float(row.get("coll_total", row.get("energy_collision_rate", 0.0))) for row in rows]
    cross_rate_vals = [float(row.get("coll_cross", 0.0)) for row in rows]
    min_cov = min(coverage_vals) if coverage_vals else 0.0
    max_kl = max(kl_vals) if kl_vals else 0.0
    max_cross_pairs = max(cross_pairs_vals) if cross_pairs_vals else 0.0
    max_cross_rate = max(cross_rate_vals) if cross_rate_vals else 0.0
    fail_meta_template = {
        "min_coverage": min_cov,
        "max_kl_exact_emp": max_kl,
        "max_coll_cross_pairs": max_cross_pairs,
        "max_coll_cross_rate": max_cross_rate,
        "max_coll_within_pairs": max(within_pairs_vals) if within_pairs_vals else 0.0,
        "max_coll_total_pairs": max(
            (c + w) for c, w in zip(cross_pairs_vals, within_pairs_vals)
        )
        if within_pairs_vals
        else max_cross_pairs,
        "max_total_pairs": max(total_pairs_vals) if total_pairs_vals else 0.0,
        "max_coll_total": max(coll_total_vals) if coll_total_vals else 0.0,
    }
    formula_gate: Dict[str, Dict[str, float]] = {}
    for row in rows:
        formula = str(row.get("formula", ""))
        stats = formula_gate.setdefault(
            formula,
            {
                "coverage": float("inf"),
                "kl": 0.0,
                "coll_cross_pairs": 0.0,
                "coll_cross_rate": 0.0,
            },
        )
        coverage = float(row.get("coverage_unique_eq", 0.0))
        kl = float(row.get("kl_exact_emp", 0.0))
        cross_pairs = float(row.get("coll_cross_pairs", 0.0) or 0.0)
        cross_rate = float(row.get("coll_cross", 0.0) or 0.0)
        stats["coverage"] = min(stats["coverage"], coverage)
        stats["kl"] = max(stats["kl"], kl)
        stats["coll_cross_pairs"] = max(stats["coll_cross_pairs"], cross_pairs)
        stats["coll_cross_rate"] = max(stats["coll_cross_rate"], cross_rate)
    gate_reason_any = ""
    gate_formula_any = ""
    for formula, stats in formula_gate.items():
        if stats["coverage"] < args.coverage_min:
            gate_reason_any = "coverage"
            gate_formula_any = formula
            break
        if stats["kl"] > args.kl_max:
            gate_reason_any = "kl"
            gate_formula_any = formula
            break
        if stats["coll_cross_pairs"] > 0:
            gate_reason_any = "coll_cross"
            gate_formula_any = formula
            break
    gate_failed_any = bool(gate_reason_any)
    if gate_failed_any:
        meta = dict(fail_meta_template)
        meta.update(
            {
                "gate_failed_any": True,
                "gate_reason_any": gate_reason_any,
                "gate_formula_any": gate_formula_any,
            }
        )
        return "fail", gate_reason_any, float("-inf"), meta, {
            "gate_failed_any": True,
            "gate_reason_any": gate_reason_any,
            "gate_formula_any": gate_formula_any,
        }
    non_trivial = [row for row in rows if not _boolify(row.get("fp_best_is_trivial", False))]
    if len(non_trivial) < args.min_nontrivial_rows:
        return "fail", "nontrivial_rows_lt_min", float("-inf"), {
            "min_coverage": min_cov,
            "max_kl_exact_emp": max_kl,
            "max_coll_cross_pairs": max_cross_pairs,
            "max_coll_cross_rate": max_cross_rate,
        }, {
            "gate_failed_any": False,
            "gate_reason_any": "",
            "gate_formula_any": "",
        }
    total_score = 0.0
    avg_other_frac = 0.0
    avg_coll_total = 0.0
    details_list = []
    for row in non_trivial:
        n_a = float(row.get("n_a", 0))
        n_b = float(row.get("n_b", 0))
        n_other = float(row.get("n_other", 0))
        denom = n_a + n_b + n_other
        other_frac = (n_other / denom) if denom > 0 else 0.0
        collision = float(row.get("coll_total", row.get("energy_collision_rate", 0.0)))
        auc_best = float(row.get("fp_best_auc_best", 0.0))
        row_score = auc_best - args.lambda_other * other_frac - args.lambda_collision * collision
        total_score += row_score
        avg_other_frac += other_frac
        avg_coll_total += collision
        details_list.append(
            (
                str(row.get("fp_best_idx", "")),
                f"{auc_best:.6f}",
            )
        )
    rows_count = len(non_trivial) if non_trivial else 1
    avg_other_frac /= rows_count
    avg_coll_total /= rows_count
    meta = {
        "min_coverage": min_cov,
        "max_kl_exact_emp": max_kl,
        "max_coll_cross_pairs": max_cross_pairs,
        "max_coll_cross_rate": max_cross_rate,
        "max_coll_within_pairs": fail_meta_template["max_coll_within_pairs"],
        "max_coll_total_pairs": fail_meta_template["max_coll_total_pairs"],
        "max_total_pairs": fail_meta_template["max_total_pairs"],
        "max_coll_total": fail_meta_template["max_coll_total"],
        "avg_other_frac": avg_other_frac,
        "avg_coll_total": avg_coll_total,
        "fp_best_idx_list": ";".join(idx for idx, _ in details_list),
        "fp_best_auc_best_list": ";".join(val for _, val in details_list),
    }
    gate_meta = {
        "gate_failed_any": False,
        "gate_reason_any": "",
        "gate_formula_any": "",
    }
    return "ok", "pass", total_score, meta, gate_meta


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
        _run_suite(
            trial_dir,
            beta=beta,
            alpha_H=alpha,
            formulas=formulas,
            extra_args=extra_kwargs,
            fp_allow_energy_like=bool(args.fp_allow_energy_like),
        )
        summary_path = trial_dir / "hetero_validation_suite.csv"
        rows = _read_summary(summary_path)
        status, reason, score, meta, gate_meta = _evaluate_trial(rows, args=args)
        record = {
            "trial_id": trial_id,
            "beta": beta,
            "alpha_H": alpha,
            "status": status,
            "reason": reason,
            "score": score if status == "ok" else "",
            "nontrivial_rows": sum(0 if _boolify(r.get("fp_best_is_trivial", False)) else 1 for r in rows),
            "output_dir": str(trial_dir),
            "min_coverage": meta.get("min_coverage"),
            "max_kl_exact_emp": meta.get("max_kl_exact_emp"),
            "max_coll_cross_pairs": meta.get("max_coll_cross_pairs"),
            "max_coll_cross_rate": meta.get("max_coll_cross_rate"),
            "max_coll_within_pairs": meta.get("max_coll_within_pairs"),
            "max_coll_total_pairs": meta.get("max_coll_total_pairs"),
            "max_total_pairs": meta.get("max_total_pairs"),
            "max_coll_total": meta.get("max_coll_total"),
            "avg_other_frac": meta.get("avg_other_frac"),
            "avg_coll_total": meta.get("avg_coll_total"),
            "fp_best_idx_list": meta.get("fp_best_idx_list"),
            "fp_best_auc_best_list": meta.get("fp_best_auc_best_list"),
            "gate_failed": status != "ok",
            "gate_reason": reason,
            "gate_failed_any": gate_meta.get("gate_failed_any"),
            "gate_reason_any": gate_meta.get("gate_reason_any"),
            "gate_formula_any": gate_meta.get("gate_formula_any"),
        }
        trial_records.append(record)
        if status == "ok" and (best_trial is None or score > float(best_trial["score"])):
            best_trial = {"score": score, "beta": beta, "alpha_H": alpha, "output_dir": str(trial_dir)}

    fieldnames = [
        "trial_id",
        "beta",
        "alpha_H",
        "status",
        "reason",
        "score",
        "nontrivial_rows",
        "gate_failed",
        "gate_reason",
        "gate_failed_any",
        "gate_reason_any",
        "gate_formula_any",
        "min_coverage",
        "max_kl_exact_emp",
        "max_coll_cross_pairs",
        "max_coll_cross_rate",
        "max_coll_within_pairs",
        "max_coll_total_pairs",
        "max_total_pairs",
        "max_coll_total",
        "avg_other_frac",
        "avg_coll_total",
        "fp_best_idx_list",
        "fp_best_auc_best_list",
        "output_dir",
    ]
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
