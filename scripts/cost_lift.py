from __future__ import annotations

import argparse
import csv
import json
import math
import random
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping


class CostLiftError(ValueError):
    pass


@dataclass(frozen=True)
class EligibleRow:
    molecule_id: str
    expensive_label: str  # PASS|FAIL
    verdict: str  # PASS|FAIL|SKIP
    gate: float
    slack: float
    auc_tie_aware: float

    @property
    def is_hit(self) -> bool:
        return self.expensive_label == "PASS"


def _read_csv_dicts(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise CostLiftError(f"missing file: {path}")
    text = path.read_text(encoding="utf-8")
    reader = csv.DictReader(text.splitlines())
    if not reader.fieldnames:
        raise CostLiftError(f"empty csv or missing header: {path}")
    return [dict(row) for row in reader]


def _require_columns(rows: list[dict[str, str]], *, path: Path, required: Iterable[str]) -> None:
    fieldnames = set(rows[0].keys()) if rows else set()
    missing = [col for col in required if col not in fieldnames]
    if missing:
        raise CostLiftError(f"{path} missing required columns: {missing}")


def _stable_method_seed(seed: int, method: str) -> int:
    return int(seed) ^ int(zlib.adler32(method.encode("utf-8")) & 0xFFFFFFFF)


def _parse_float(value: str, *, where: str) -> float:
    try:
        return float(value)
    except Exception as exc:
        raise CostLiftError(f"failed to parse float at {where}: {value!r}") from exc


def _bootstrap_ci(hit_flags: list[int], *, seed: int, n_bootstrap: int) -> tuple[float, float]:
    if n_bootstrap <= 0:
        raise CostLiftError(f"n_bootstrap must be > 0, got: {n_bootstrap}")
    k = len(hit_flags)
    if k <= 0:
        raise CostLiftError("cannot bootstrap CI for empty selection (k=0)")
    rng = random.Random(int(seed))
    rates: list[float] = []
    for _ in range(int(n_bootstrap)):
        hits = 0
        for _ in range(k):
            hits += hit_flags[rng.randrange(k)]
        rates.append(hits / k)
    rates.sort()
    lo = rates[int(0.025 * (n_bootstrap - 1))]
    hi = rates[int(0.975 * (n_bootstrap - 1))]
    return lo, hi


def generate_cost_lift_report(
    *,
    summary_csv: Path,
    truth_csv: Path,
    k: int,
    seed: int,
    skip_policy: str,
    n_bootstrap: int = 500,
) -> dict[str, Any]:
    if k <= 0:
        raise CostLiftError(f"k must be > 0, got: {k}")
    if skip_policy != "unknown_bucket":
        raise CostLiftError(f"unsupported skip_policy: {skip_policy!r} (expected 'unknown_bucket')")

    summary_rows = _read_csv_dicts(summary_csv)
    _require_columns(summary_rows, path=summary_csv, required=["id", "status", "verdict", "gate", "slack"])

    truth_rows = _read_csv_dicts(truth_csv)
    _require_columns(
        truth_rows,
        path=truth_csv,
        required=["molecule_id", "expensive_label", "truth_source", "truth_version"],
    )

    truth_by_id: dict[str, dict[str, str]] = {}
    truth_versions: set[str] = set()
    for idx, row in enumerate(truth_rows):
        molecule_id = (row.get("molecule_id") or "").strip()
        if not molecule_id:
            raise CostLiftError(f"{truth_csv}: empty molecule_id at row {idx+2}")
        if molecule_id in truth_by_id:
            raise CostLiftError(f"{truth_csv}: duplicate molecule_id: {molecule_id}")
        expensive_label = (row.get("expensive_label") or "").strip().upper()
        if expensive_label not in {"PASS", "FAIL"}:
            raise CostLiftError(f"{truth_csv}: invalid expensive_label for {molecule_id}: {expensive_label!r}")
        truth_source = (row.get("truth_source") or "").strip()
        if not truth_source:
            raise CostLiftError(f"{truth_csv}: empty truth_source for {molecule_id}")
        truth_version = (row.get("truth_version") or "").strip()
        if not truth_version:
            raise CostLiftError(f"{truth_csv}: empty truth_version for {molecule_id}")
        truth_versions.add(truth_version)
        truth_by_id[molecule_id] = {
            "expensive_label": expensive_label,
            "truth_source": truth_source,
            "truth_version": truth_version,
        }

    if truth_versions != {"customer_truth.v1"}:
        raise CostLiftError(f"{truth_csv}: expected truth_version={{'customer_truth.v1'}}, got: {sorted(truth_versions)}")

    n_total = len(summary_rows)
    n_ok = sum(1 for r in summary_rows if (r.get("status") or "").strip().upper() == "OK")
    n_skip = sum(1 for r in summary_rows if (r.get("status") or "").strip().upper() == "SKIP")
    n_error = sum(1 for r in summary_rows if (r.get("status") or "").strip().upper() == "ERROR")

    ok_with_truth = []
    eligible: list[EligibleRow] = []
    for r in summary_rows:
        status = (r.get("status") or "").strip().upper()
        if status != "OK":
            continue
        mol_id = (r.get("id") or "").strip()
        if not mol_id:
            continue
        truth = truth_by_id.get(mol_id)
        if not truth:
            continue
        ok_with_truth.append(mol_id)

        verdict = (r.get("verdict") or "").strip().upper()
        if not verdict:
            raise CostLiftError(f"{summary_csv}: OK row {mol_id} has empty verdict")

        if verdict not in {"PASS", "FAIL"}:
            # OK rows may still have audit verdicts like SKIP (e.g., missing_scores_for_all_decoys),
            # where gate/slack are intentionally empty. Such rows are not eligible for selection.
            continue

        gate = _parse_float(str(r.get("gate", "")), where=f"{summary_csv}:id={mol_id}:gate")
        slack = _parse_float(str(r.get("slack", "")), where=f"{summary_csv}:id={mol_id}:slack")

        auc_raw = str(r.get("auc_tie_aware", "") or "").strip()
        auc_tie_aware = _parse_float(auc_raw, where=f"{summary_csv}:id={mol_id}:auc_tie_aware") if auc_raw else (gate + slack)

        eligible.append(
            EligibleRow(
                molecule_id=mol_id,
                expensive_label=str(truth["expensive_label"]),
                verdict=verdict,
                gate=gate,
                slack=slack,
                auc_tie_aware=auc_tie_aware,
            )
        )

    n_with_truth = len(ok_with_truth)
    if n_ok <= 0:
        raise CostLiftError(f"{summary_csv}: N_ok=0; cannot compute truth coverage")
    truth_coverage_rate = n_with_truth / n_ok
    unknown_bucket_rate = (n_ok - n_with_truth) / n_ok

    if not eligible:
        raise CostLiftError("no eligible rows: need status=OK + truth label + numeric gate/slack")

    k_effective_common = min(int(k), len(eligible))

    def _method_result(*, name: str, selected: list[EligibleRow]) -> dict[str, Any]:
        if not selected:
            raise CostLiftError(f"method {name}: empty selection; cannot compute hit_rate/uplift")
        hit_flags = [1 if r.is_hit else 0 for r in selected]
        hits = int(sum(hit_flags))
        k_eff = len(selected)
        hit_rate = hits / k_eff
        ci_low, ci_high = _bootstrap_ci(hit_flags, seed=_stable_method_seed(seed + 1, name), n_bootstrap=n_bootstrap)
        return {
            "k_effective": k_eff,
            "hits": hits,
            "hit_rate": hit_rate,
            "ci_low": ci_low,
            "ci_high": ci_high,
        }

    # 1) baseline_random: uniform among eligible
    rng_random = random.Random(_stable_method_seed(seed, "baseline_random"))
    selected_random = rng_random.sample(eligible, k_effective_common)

    # 2) baseline_score_only_topk: top-K by auc_tie_aware
    eligible_by_auc = sorted(eligible, key=lambda r: (-r.auc_tie_aware, r.molecule_id))
    selected_score_only = eligible_by_auc[:k_effective_common]

    # 3) filtered_score_plus_audit_topk: top-K by slack among verdict==PASS
    eligible_pass = [r for r in eligible if r.verdict == "PASS"]
    if not eligible_pass:
        raise CostLiftError("no eligible rows with verdict=PASS; cannot compute filtered_score_plus_audit_topk")
    eligible_pass_by_slack = sorted(eligible_pass, key=lambda r: (-r.slack, r.molecule_id))
    selected_score_plus_audit = eligible_pass_by_slack[:k_effective_common]

    methods: dict[str, dict[str, Any]] = {
        "baseline_random": _method_result(name="baseline_random", selected=selected_random),
        "baseline_score_only_topk": _method_result(name="baseline_score_only_topk", selected=selected_score_only),
        "filtered_score_plus_audit_topk": _method_result(name="filtered_score_plus_audit_topk", selected=selected_score_plus_audit),
    }

    uplift_vs_score_only = methods["filtered_score_plus_audit_topk"]["hit_rate"] - methods["baseline_score_only_topk"]["hit_rate"]
    uplift_vs_random = methods["filtered_score_plus_audit_topk"]["hit_rate"] - methods["baseline_random"]["hit_rate"]

    if not math.isfinite(uplift_vs_score_only) or not math.isfinite(uplift_vs_random):
        raise CostLiftError("uplift is not finite; check inputs")

    return {
        "report_schema": "cost_lift.v1",
        "truth_schema": "customer_truth.v1",
        "summary_csv": str(summary_csv.as_posix()),
        "truth_csv": str(truth_csv.as_posix()),
        "skip_policy": skip_policy,
        "seed": int(seed),
        "K_requested": int(k),
        "K_effective": int(k_effective_common),
        "N_total": int(n_total),
        "N_ok": int(n_ok),
        "N_skip": int(n_skip),
        "N_error": int(n_error),
        "N_with_truth": int(n_with_truth),
        "truth_coverage_rate": truth_coverage_rate,
        "unknown_bucket_rate": unknown_bucket_rate,
        "methods": methods,
        "uplift_score_plus_audit_vs_score_only": uplift_vs_score_only,
        "uplift_score_plus_audit_vs_random": uplift_vs_random,
    }


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compute cost&lift utility report from summary.csv + customer truth CSV.")
    p.add_argument("--summary_csv", type=Path, required=True)
    p.add_argument("--truth_csv", type=Path, required=True)
    p.add_argument("--k", type=int, default=10000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--skip_policy", type=str, default="unknown_bucket")
    p.add_argument("--out", type=Path, default=Path("cost_lift_report.json"))
    p.add_argument("--bootstrap_n", type=int, default=500)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    report = generate_cost_lift_report(
        summary_csv=args.summary_csv,
        truth_csv=args.truth_csv,
        k=int(args.k),
        seed=int(args.seed),
        skip_policy=str(args.skip_policy),
        n_bootstrap=int(args.bootstrap_n),
    )
    args.out.write_text(json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

