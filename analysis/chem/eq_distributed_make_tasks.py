from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import List


def now_utc_iso() -> str:
    return (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def compute_seed(seed_base: int, steps: int, start_idx: int, chain_idx: int) -> int:
    """
    Deterministic seed formula (договорённость):
    seed = base + 10_000*steps + 1_000_000*start_idx + 101*chain_idx
    """
    return int(seed_base) + 10_000 * int(steps) + 1_000_000 * int(start_idx) + 101 * int(chain_idx)


@dataclass(frozen=True)
class EqTask:
    task_id: str
    N: int
    mode: str
    steps_per_chain: int
    thin: int
    burnin_frac: float
    start_spec: str
    start_spec_idx: int
    chain_idx: int
    seed_base: int
    seed: int
    created_utc: str
    git_sha_expected: str
    notes: str = "EQ-DIST-P0"


def main(argv: List[str] | None = None) -> None:
    ap = argparse.ArgumentParser(
        description="Generate distributed EQ-TARGET-3 tasks (independent work units)."
    )
    ap.add_argument("--N", type=int, required=True)
    ap.add_argument("--mode", type=str, default="A")
    ap.add_argument("--steps_grid", type=int, nargs="+", required=True)
    ap.add_argument("--chains", type=int, default=3)
    ap.add_argument("--start_specs", type=str, nargs="+", default=["path", "max_branch"])
    ap.add_argument("--thin", type=int, default=10)
    ap.add_argument("--burnin_frac", type=float, default=0.1)
    ap.add_argument("--seed_base", type=int, default=12345)
    ap.add_argument("--git_sha_expected", type=str, default="")
    ap.add_argument("--out_dir", type=str, required=True)
    args = ap.parse_args(argv)

    out_dir = Path(args.out_dir)
    tasks_dir = out_dir / "tasks"
    tasks_dir.mkdir(parents=True, exist_ok=True)

    tasks_jsonl = tasks_dir / "tasks.jsonl"
    created = now_utc_iso()
    lines: List[str] = []
    task_idx = 1

    for steps in args.steps_grid:
        for start_idx, start_spec in enumerate(args.start_specs):
            for chain_idx in range(args.chains):
                task_id = f"task_{task_idx:06d}"
                seed = compute_seed(args.seed_base, steps, start_idx, chain_idx)
                task = EqTask(
                    task_id=task_id,
                    N=int(args.N),
                    mode=str(args.mode),
                    steps_per_chain=int(steps),
                    thin=int(args.thin),
                    burnin_frac=float(args.burnin_frac),
                    start_spec=str(start_spec),
                    start_spec_idx=int(start_idx),
                    chain_idx=int(chain_idx),
                    seed_base=int(args.seed_base),
                    seed=int(seed),
                    created_utc=created,
                    git_sha_expected=str(args.git_sha_expected),
                )
                task_path = tasks_dir / f"{task_id}.json"
                task_json = json.dumps(asdict(task), ensure_ascii=False, indent=2)
                task_path.write_text(task_json, encoding="utf-8")
                lines.append(json.dumps(asdict(task), ensure_ascii=False))
                task_idx += 1

    tasks_jsonl.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"WROTE {tasks_jsonl} ({len(lines)} tasks)")


if __name__ == "__main__":
    main()
