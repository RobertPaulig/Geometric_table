from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import numpy as np


def kl(p: Dict[str, float], q: Dict[str, float], eps: float = 1e-12) -> float:
    keys = set(p.keys()) | set(q.keys())
    ps = sum(p.values())
    qs = sum(q.values())
    if ps <= 0 or qs <= 0:
        return 0.0
    out = 0.0
    for k in keys:
        pk = p.get(k, 0.0) / ps + eps
        qk = q.get(k, 0.0) / qs + eps
        out += pk * math.log(pk / qk)
    return float(out)


def sym_kl(p: Dict[str, float], q: Dict[str, float], eps: float = 1e-12) -> float:
    a = kl(p, q, eps)
    b = kl(q, p, eps)
    return 0.5 * (a + b)


def rhat_from_means_vars(means: List[float], vars_: List[float], n: int) -> float:
    m = len(means)
    if m < 2 or n < 2:
        return 1.0
    means_np = np.asarray(means, dtype=np.float64)
    vars_np = np.asarray(vars_, dtype=np.float64)
    W = float(vars_np.mean())
    B = float(n * means_np.var(ddof=1))
    if W <= 0:
        return 1.0
    var_hat = ((n - 1) / n) * W + (1 / n) * B
    return float(math.sqrt(max(var_hat / W, 0.0)))


def load_tasks(tasks_jsonl: Path) -> List[Dict]:
    tasks: List[Dict] = []
    for line in tasks_jsonl.read_text(encoding="utf-8").splitlines():
        if line.strip():
            tasks.append(json.loads(line))
    return tasks


def load_submissions(submissions_dir: Path) -> List[Dict]:
    out: List[Dict] = []
    for path in sorted(submissions_dir.glob("task_*.json")):
        out.append(json.loads(path.read_text(encoding="utf-8")))
    return out


def main(argv: List[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Aggregate distributed EQ-TARGET-3 submissions.")
    ap.add_argument("--tasks_jsonl", type=str, required=True)
    ap.add_argument("--submissions_dir", type=str, required=True)
    ap.add_argument("--expected_unique", type=int, required=True)
    ap.add_argument("--out_csv", type=str, required=True)
    ap.add_argument("--out_txt", type=str, required=True)
    ap.add_argument(
        "--status",
        action="store_true",
        help="Print readiness per steps_per_chain and list missing task_id; does not require out paths to exist.",
    )
    ap.add_argument(
        "--verify_steps",
        type=int,
        nargs="*",
        default=[8_000_000, 16_000_000],
        help="Steps that require double submissions when --require_double_for_verify is set.",
    )
    ap.add_argument(
        "--require_double_for_verify",
        action="store_true",
        help="Require two submissions with matching counter_hash for verify_steps.",
    )
    args = ap.parse_args(argv)

    tasks = load_tasks(Path(args.tasks_jsonl))
    submissions = load_submissions(Path(args.submissions_dir))

    subs_by_task = defaultdict(list)
    for sub in submissions:
        subs_by_task[sub["task_id"]].append(sub)

    tasks_by_steps = defaultdict(list)
    for task in tasks:
        tasks_by_steps[int(task["steps_per_chain"])].append(task)

    if args.status:
        for steps, tlist in sorted(tasks_by_steps.items()):
            present = 0
            missing: List[str] = []
            bad_verify: List[str] = []
            for task in tlist:
                tid = task["task_id"]
                cand = subs_by_task.get(tid, [])
                if cand:
                    present += 1
                else:
                    missing.append(tid)
            verify_set = set(args.verify_steps or [])
            if args.require_double_for_verify and steps in verify_set:
                for task in tlist:
                    tid = task["task_id"]
                    cand = subs_by_task.get(tid, [])
                    if len(cand) < 2:
                        bad_verify.append(tid)
                        continue
                    cand.sort(key=lambda x: x.get("date_utc", ""))
                    if cand[-1].get("counter_hash") != cand[-2].get("counter_hash"):
                        bad_verify.append(tid)
            print(f"STEPS={steps} READY={present}/{len(tlist)}")
            if missing:
                print("MISSING=" + ",".join(missing))
            if bad_verify:
                print("VERIFY_FAIL=" + ",".join(bad_verify))
        return

    lines: List[str] = []
    csv_rows: List[Dict[str, float]] = []
    verify_set = set(args.verify_steps or [])

    for steps, tlist in sorted(tasks_by_steps.items()):
        # collect chosen submissions
        chosen = []
        missing = []
        for task in tlist:
            tid = task["task_id"]
            cand = subs_by_task.get(tid, [])
            if not cand:
                missing.append(tid)
                continue
            cand.sort(key=lambda x: x.get("date_utc", ""))
            chosen.append(cand[-1])

        if missing:
            lines.append(f"[DIST] steps={steps}: MISSING {len(missing)}/{len(tlist)} tasks")
            continue

        if args.require_double_for_verify and steps in verify_set:
            bad = []
            for task in tlist:
                tid = task["task_id"]
                cand = subs_by_task.get(tid, [])
                if len(cand) < 2:
                    bad.append(tid)
                    continue
                cand.sort(key=lambda x: x.get("date_utc", ""))
                if cand[-1].get("counter_hash") != cand[-2].get("counter_hash"):
                    bad.append(tid)
            if bad:
                lines.append(f"[DIST] steps={steps}: VERIFY_FAIL {len(bad)} tasks")
                continue

        # Aggregate metrics
        dist_list: List[Dict[str, float]] = []
        split_pairs: List[tuple[Dict[str, float], Dict[str, float]]] = []
        means: List[float] = []
        vars_: List[float] = []
        ess_list: List[float] = []
        unique_keys = set()

        elapsed_sum = 0.0
        steps_sum = 0
        acc_rates = []
        hit_rates = []
        misses = 0

        for sub in chosen:
            dist = sub["topology_counter"]
            dist_list.append(dist)
            split_pairs.append((sub["topology_counter_first_half"], sub["topology_counter_second_half"]))
            unique_keys |= set(dist.keys())
            means.append(float(sub.get("energy_mean", 0.0)))
            vars_.append(float(sub.get("energy_var", 0.0)))
            ess_list.append(float(sub.get("ess_energy", 1.0)))
            elapsed_sum += float(sub.get("elapsed_sec", 0.0))
            steps_sum += int(sub.get("steps_total", steps))
            acc_rates.append(float(sub.get("accept_rate", 0.0)))
            hit_rates.append(float(sub.get("hit_rate", 0.0)))
            misses += int(sub.get("misses_seen", 0))

        # KL pairwise
        kl_max = 0.0
        for i, j in itertools.combinations(range(len(dist_list)), 2):
            kl_val = sym_kl(dist_list[i], dist_list[j])
            if kl_val > kl_max:
                kl_max = kl_val

        kl_split_max = 0.0
        for first, second in split_pairs:
            kl_split = sym_kl(first, second)
            if kl_split > kl_split_max:
                kl_split_max = kl_split

        n_samples = int(chosen[0].get("n_samples", 1))
        rhat = rhat_from_means_vars(means, vars_, n_samples)
        ess_min = float(min(ess_list)) if ess_list else 1.0
        coverage = (len(unique_keys) / float(args.expected_unique)) if args.expected_unique > 0 else 0.0
        steps_per_sec_total = (steps_sum / elapsed_sum) if elapsed_sum > 0 else 0.0

        lines.extend(
            [
                "POINT",
                f"STEPS_PER_CHAIN={steps}",
                f"KL_MAX_PAIRWISE={kl_max:.6g}",
                f"KL_SPLIT_MAX={kl_split_max:.6g}",
                f"RHAT_ENERGY_MAX={rhat:.6g}",
                f"ESS_ENERGY_MIN={ess_min:.6g}",
                f"N_UNIQUE_EQ={len(unique_keys)}",
                f"EXPECTED_UNIQUE_EQ={args.expected_unique}",
                f"COVERAGE_UNIQUE_EQ={coverage:.6g}",
                f"ACCEPT_RATE={float(np.mean(acc_rates)) if acc_rates else 0.0:.6g}",
                f"HIT_RATE={float(np.mean(hit_rates)) if hit_rates else 0.0:.6g}",
                f"MISSES_SEEN={misses}",
                f"STEPS_TOTAL={steps_sum}",
                f"STEPS_PER_SEC_TOTAL={steps_per_sec_total:.6g}",
                f"ELAPSED_SEC={elapsed_sum:.6g}",
                "END",
            ]
        )

        csv_rows.append(
            {
                "N": chosen[0]["N"],
                "mode": chosen[0]["mode"],
                "steps": steps,
                "KL_max_pairwise": kl_max,
                "KL_split_max": kl_split_max,
                "Rhat_energy_max": rhat,
                "ESS_energy_min": ess_min,
                "n_unique_eq": len(unique_keys),
                "expected_unique_eq": args.expected_unique,
                "coverage_unique_eq": coverage,
                "steps_total": steps_sum,
                "steps_per_sec_total": steps_per_sec_total,
                "elapsed_sec": elapsed_sum,
            }
        )

    out_txt = Path(args.out_txt)
    out_txt.parent.mkdir(parents=True, exist_ok=True)
    out_txt.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")

    if csv_rows:
        out_csv = Path(args.out_csv)
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        with out_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
            writer.writeheader()
            for row in sorted(csv_rows, key=lambda r: int(r["steps"])):
                writer.writerow(row)

    print(f"WROTE {args.out_txt}")
    if csv_rows:
        print(f"WROTE {args.out_csv}")


if __name__ == "__main__":
    main()
