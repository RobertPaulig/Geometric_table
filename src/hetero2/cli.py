from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from hetero2.pipeline import run_pipeline_v2
from hetero2.report import render_report_v2
from hetero2.batch import run_batch


def _parse_pipeline_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="HETERO-2 pipeline: SMILES -> decoys -> audit -> report JSON.")
    ap.add_argument("--smiles", required=True, help="Input SMILES string.")
    ap.add_argument("--out", default="", help="Output pipeline JSON (default: stdout).")
    ap.add_argument("--k_decoys", type=int, default=20, help="Number of decoys.")
    ap.add_argument("--seed", type=int, default=0, help="Seed.")
    ap.add_argument("--timestamp", default="", help="Timestamp override (ISO).")
    ap.add_argument("--score_mode", choices=["external_scores", "mock"], default="external_scores")
    ap.add_argument("--scores_input", default="", help="Path to hetero_scores.v1 (for external_scores).")
    return ap.parse_args(argv)


def _parse_report_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="HETERO-2 report: pipeline.json -> markdown (+ assets).")
    ap.add_argument("--input", required=True, help="Path to hetero2_pipeline.v1 JSON.")
    ap.add_argument("--out", default="aspirin_report.md", help="Output report md path.")
    ap.add_argument("--assets_dir", default="", help="Assets directory for images (default: <stem>_assets).")
    return ap.parse_args(argv)


def _parse_demo_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="HETERO-2 aspirin demo (SMILES -> report).")
    ap.add_argument("--out_dir", default=".", help="Output directory (default: current dir).")
    ap.add_argument("--stem", default="aspirin", help="Filename stem for outputs.")
    return ap.parse_args(argv)


def _parse_batch_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="HETERO-2 batch runner from CSV (id,smiles[,scores_input]).")
    ap.add_argument("--input", required=True, help="Input CSV with columns: id, smiles [,scores_input].")
    ap.add_argument("--out_dir", default="out", help="Output directory for artifacts.")
    ap.add_argument("--k_decoys", type=int, default=20, help="Decoys per molecule.")
    ap.add_argument("--seed", type=int, default=0, help="Seed.")
    ap.add_argument("--timestamp", default="", help="Timestamp override.")
    ap.add_argument("--scores_input", default="", help="Global scores_input path (optional).")
    return ap.parse_args(argv)


def main_pipeline(argv: list[str] | None = None) -> int:
    args = _parse_pipeline_args(argv)
    score_mode = "external_scores" if args.score_mode == "external_scores" else "mock"
    out = run_pipeline_v2(
        args.smiles,
        k_decoys=int(args.k_decoys),
        seed=int(args.seed),
        timestamp=str(args.timestamp),
        score_mode=score_mode,
        scores_input=str(args.scores_input) if args.scores_input else None,
    )
    text = json.dumps(out, ensure_ascii=False, sort_keys=True, indent=2)
    if args.out:
        Path(args.out).write_text(text + "\n", encoding="utf-8")
    else:
        sys.stdout.write(text + "\n")
    return 0


def main_report(argv: list[str] | None = None) -> int:
    args = _parse_report_args(argv)
    payload = json.loads(Path(args.input).read_text(encoding="utf-8"))
    render_report_v2(payload, out_path=args.out, assets_dir=args.assets_dir or None)
    return 0


def main_demo_aspirin(argv: list[str] | None = None) -> int:
    args = _parse_demo_args(argv)
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = args.stem
    pipeline_out = out_dir / f"{stem}_pipeline.json"
    report_out = out_dir / f"{stem}_report.md"

    smiles = "CC(=O)OC1=CC=CC=C1C(=O)O"
    pipeline = run_pipeline_v2(
        smiles,
        k_decoys=20,
        seed=0,
        timestamp="2026-01-02T00:00:00+00:00",
        score_mode="mock",
        scores_input=None,
    )
    pipeline_out.write_text(json.dumps(pipeline, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    render_report_v2(pipeline, out_path=str(report_out), assets_dir=out_dir / f"{stem}_assets")
    sys.stdout.write(f"Pipeline JSON: {pipeline_out}\n")
    sys.stdout.write(f"Report: {report_out}\n")
    return 0


def main_batch(argv: list[str] | None = None) -> int:
    args = _parse_batch_args(argv)
    input_csv = Path(args.input).resolve()
    out_dir = Path(args.out_dir).resolve()
    run_batch(
        input_csv=input_csv,
        out_dir=out_dir,
        seed=int(args.seed),
        timestamp=str(args.timestamp),
        k_decoys=int(args.k_decoys),
        score_mode="external_scores" if args.scores_input else "mock",
        scores_input=str(args.scores_input) if args.scores_input else None,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main_pipeline())
