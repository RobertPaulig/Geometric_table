from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
from pathlib import Path
import subprocess
from typing import Dict, List

from analysis.chem.hetero_score_utils import (
    ALPHA_H,
    RHO_BY_TYPE,
    compute_formula_scores,
    compute_state_table,
)
from analysis.chem.hetero_validation_1_acid import (
    FORMULA_SPECS,
    run_formula_validation,
    write_report,
)


def _git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "UNKNOWN"


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
    ap.add_argument("--theta_json", type=str, help="Path to JSON with theta overrides.")
    ap.add_argument(
        "--fp_exclude_energy_like",
        dest="fp_exclude_energy_like",
        action="store_true",
        help="Drop FP candidates highly correlated with energy (default).",
    )
    ap.add_argument(
        "--fp_allow_energy_like",
        dest="fp_exclude_energy_like",
        action="store_false",
        help="Allow FP candidates even if they are energy-like.",
    )
    ap.set_defaults(fp_exclude_energy_like=True)
    ap.add_argument("--fp_energy_like_threshold", type=float, default=0.999, help="Spearman|corr| threshold for energy-like FP.")
    ap.add_argument("--debug_fp", action="store_true")
    return ap.parse_args(argv)


def main(argv: List[str] | None = None) -> None:
    args = _parse_args(argv)
    theta_overrides: Dict[str, object] = {}
    theta_json_path: Path | None = None
    if args.theta_json:
        theta_json_path = Path(args.theta_json)
        if not theta_json_path.exists():
            raise FileNotFoundError(f"theta_json not found: {theta_json_path}")
        data = json.loads(theta_json_path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError("theta_json must contain an object with overrides")
        for key, value in data.items():
            if hasattr(args, key):
                setattr(args, key, value)
                theta_overrides[key] = value
    theta_overrides_meta = json.dumps(theta_overrides)
    theta_values = {
        "beta": float(args.beta),
        "alpha_H": float(args.alpha_H),
    }
    if args.p_rewire is not None:
        theta_values["p_rewire"] = float(args.p_rewire)
    theta_values_meta = json.dumps(theta_values)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_rows: List[dict] = []
    score_cols: set[str] = set()
    states_dir = out_dir / "states"
    states_dir.mkdir(parents=True, exist_ok=True)
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
        state_ids = sorted(set(exact.keys()) | set(emp.keys()))
        df_states = compute_state_table(
            formula=formula_name,
            state_ids=state_ids,
            p_exact=exact,
            p_emp=emp,
            rho_by_type=RHO_BY_TYPE,
            alpha_H_override=float(args.alpha_H),
        )
        df_states.to_csv(states_dir / f"states_{formula_name}.csv", index=False)
        run_meta = {
            "run_id": f"{formula_name}_{dt.datetime.utcnow().isoformat()}",
            "git_sha": _git_sha(),
            "timestamp_utc": dt.datetime.utcnow().isoformat(),
            "chains": int(args.chains or spec.chains),
            "steps_per_chain": int(meta["steps_per_chain"]),
            "burnin_steps": int(meta["burnin"]),
            "thin": int(args.thin or spec.thin),
            "seed_base": int(args.seed),
            "beta": float(args.beta),
            "rho_C": float(RHO_BY_TYPE[0]),
            "rho_N": float(RHO_BY_TYPE[1]),
            "rho_O": float(RHO_BY_TYPE[2]),
            "alpha_H": float(args.alpha_H),
            "tau_s": 0.0,
            "steps_total": int(meta["steps_total"]),
            "kl_exact_emp": float(meta["kl_exact_emp"]),
            "kl_emp_exact": float(meta["kl_emp_exact"]),
            "accept_rate": float(meta["accept_rate"]),
            "samples": int(meta["samples_collected"]),
            "coverage_unique_eq": float(meta["coverage_unique_eq"]),
            "theta_source": "theta_json" if theta_json_path else "cli",
            "theta_json_path": str(theta_json_path) if theta_json_path else "",
            "theta_overrides": theta_overrides_meta,
            "theta_values": theta_values_meta,
            "fp_policy_used": "exclude_energy_like" if bool(args.fp_exclude_energy_like) else "allow_energy_like",
        }
        score_rows = compute_formula_scores(
            df_states,
            formula=formula_name,
            weights_col="P_exact",
            run_meta=run_meta,
            debug_fp=bool(args.debug_fp),
            fp_exclude_energy_like=bool(args.fp_exclude_energy_like),
            fp_energy_like_threshold=float(args.fp_energy_like_threshold),
            collision_log_dir=out_dir / "collisions",
        )
        summary_rows.extend(score_rows)
        for row in score_rows:
            score_cols.update(row.keys())

    base_fields = [
        "run_id",
        "git_sha",
        "timestamp_utc",
        "formula",
        "weight_source",
        "chains",
        "steps_total",
        "steps_per_chain",
        "burnin_steps",
        "thin",
        "seed_base",
        "beta",
        "rho_C",
        "rho_N",
        "rho_O",
        "alpha_H",
        "tau_s",
        "support_exact",
        "support_emp",
        "coverage_unique_eq",
        "kl_exact_emp",
        "kl_emp_exact",
        "accept_rate",
        "samples",
        "theta_source",
        "theta_json_path",
        "theta_overrides",
        "theta_values",
        "fp_policy_used",
        "energy_collision_rate",
        "energy_collision_eps",
        "collision_eps",
        "energy_key_scheme",
        "coll_total",
        "coll_within",
        "coll_cross",
        "coll_total_pairs",
        "coll_within_pairs",
        "coll_cross_pairs",
        "total_pairs",
        "max_abs_delta_cross",
        "fp_collision_rate",
        "fp_energy_spearman_abs",
        "class_a",
        "class_b",
        "n_a",
        "n_b",
        "n_other",
        "other_frac",
        "pair_is_exhaustive",
        "E_is_trivial",
        "E_auc_raw",
        "E_auc_best",
        "E_delta_abs",
        "E_effect_size",
        "fp_dim",
        "fp_best_idx_default",
        "fp_best_auc_default",
        "fp_best_auc_gap_default",
        "fp_best_idx_excl_energy_like",
        "fp_best_auc_excl_energy_like",
        "fp_best_auc_gap_excl_energy_like",
        "fp_best_idx",
        "fp_best_is_trivial",
        "fp_best_auc_raw",
        "fp_best_auc_best",
        "fp_best_auc_gap",
        "fp_best_delta_abs",
        "fp_best_effect_size",
    ]
    fieldnames = list(dict.fromkeys(base_fields))
    extra_cols = sorted(c for c in score_cols if c not in fieldnames)
    fieldnames.extend(extra_cols)
    summary_path = out_dir / "hetero_validation_suite.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in summary_rows:
            writer.writerow(row)
    print(f"[HETERO-SUITE] wrote {summary_path}")


if __name__ == "__main__":
    main()
