import argparse
import datetime as _dt
import subprocess
import sys
from pathlib import Path


def _run(cmd: list[str]) -> None:
    p = subprocess.run(cmd)
    if p.returncode != 0:
        raise SystemExit(p.returncode)


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(
        description="Run eq_distributed_aggregate, then git add/commit/push updated artifacts.",
    )
    ap.add_argument("--tasks_jsonl", required=True)
    ap.add_argument("--submissions_dir", required=True)
    ap.add_argument("--expected_unique", type=int, required=True)
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--out_txt", required=True)
    ap.add_argument(
        "--include_submissions_dir",
        action="store_true",
        help="Also `git add` the submissions_dir (JSON submissions) after aggregation.",
    )
    ap.add_argument(
        "--include_cache_path",
        type=str,
        default="",
        help="Optional cache file to `git add` (e.g. results/dist/.../cache_HOST.pkl).",
    )
    ap.add_argument(
        "--add_paths",
        nargs="*",
        default=[],
        help="Extra paths to `git add` (in addition to out_csv/out_txt).",
    )
    ap.add_argument(
        "--commit_message",
        default="",
        help="If empty, an auto message with timestamp is used.",
    )
    ap.add_argument("--no_push", action="store_true")
    args = ap.parse_args(argv)

    _run(
        [
            sys.executable,
            "-m",
            "analysis.chem.eq_distributed_aggregate",
            "--tasks_jsonl",
            args.tasks_jsonl,
            "--submissions_dir",
            args.submissions_dir,
            "--expected_unique",
            str(args.expected_unique),
            "--out_csv",
            args.out_csv,
            "--out_txt",
            args.out_txt,
        ]
    )

    out_csv = Path(args.out_csv)
    out_txt = Path(args.out_txt)
    add_paths = [str(out_csv), str(out_txt)] + list(args.add_paths)
    if args.include_submissions_dir:
        add_paths.append(str(Path(args.submissions_dir)))
    if args.include_cache_path:
        add_paths.append(str(Path(args.include_cache_path)))

    _run(["git", "add", *add_paths])

    msg = args.commit_message.strip()
    if not msg:
        ts = _dt.datetime.now().astimezone().isoformat(timespec="seconds")
        msg = f"EQ-DIST aggregate {out_txt.name} ({ts})"

    _run(["git", "commit", "-m", msg])

    if not args.no_push:
        _run(["git", "push"])

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
