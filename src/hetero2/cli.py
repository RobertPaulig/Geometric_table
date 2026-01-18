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
    ap.add_argument(
        "--score_mode",
        choices=["external_scores", "mock"],
        default="mock",
        help="Score mode: mock ignores any scores_input; external_scores requires scores_input to be set.",
    )
    ap.add_argument("--scores_input", default="", help="Path to hetero_scores.v1 (for external_scores).")
    ap.add_argument("--decoy_hard_mode", action="store_true", help="Enable hard decoys accept/reject by Tanimoto.")
    ap.add_argument("--decoy_hard_tanimoto_min", type=float, default=0.65, help="Hard decoys min Tanimoto (inclusive).")
    ap.add_argument("--decoy_hard_tanimoto_max", type=float, default=0.95, help="Hard decoys max Tanimoto (inclusive).")
    ap.add_argument(
        "--physics_mode",
        choices=["topological", "hamiltonian", "both"],
        default="topological",
        help="Physics operator mode: topological (L), hamiltonian (H=L+V), or both.",
    )
    ap.add_argument(
        "--edge_weight_mode",
        choices=["unweighted", "bond_order", "bond_order_delta_chi"],
        default="unweighted",
        help="Edge weights for weighted operators: unweighted (binary), bond_order (1/2/3/1.5), or bond_order_delta_chi (bond_order*(1+alpha*|Δchi|)).",
    )
    ap.add_argument(
        "--operator_mode",
        choices=["laplacian", "h_operator"],
        default=None,
        help="DEPRECATED: use --physics_mode. laplacian -> topological, h_operator -> hamiltonian.",
    )
    ap.add_argument("--guardrails_max_atoms", type=int, default=200, help="Guardrail: max heavy atoms (skip if exceeded).")
    ap.add_argument(
        "--guardrails_require_connected",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Guardrail: require molecule to be connected (default: True).",
    )
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
    ap.add_argument(
        "--artifacts",
        choices=["full", "light"],
        default="full",
        help="Artifacts mode: full emits reports/assets; light emits only summary/metrics/index/manifest/checksums/zip.",
    )
    ap.add_argument("--k_decoys", type=int, default=20, help="Decoys per molecule.")
    ap.add_argument("--seed", type=int, default=0, help="Seed.")
    ap.add_argument("--timestamp", default="", help="Timestamp override.")
    ap.add_argument("--scores_input", default="", help="Global scores_input path (optional).")
    ap.add_argument(
        "--score_mode",
        choices=["external_scores", "mock"],
        default="mock",
        help="Score mode: mock ignores any scores_input; external_scores requires scores_input to be set.",
    )
    ap.add_argument("--guardrails_max_atoms", type=int, default=200, help="Guardrail: max heavy atoms (skip if exceeded).")
    ap.add_argument(
        "--guardrails_require_connected",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Guardrail: require molecule to be connected (default: True).",
    )
    ap.add_argument("--decoy_hard_mode", action="store_true", help="Enable hard decoys accept/reject by Tanimoto.")
    ap.add_argument("--decoy_hard_tanimoto_min", type=float, default=0.65, help="Hard decoys min Tanimoto (inclusive).")
    ap.add_argument("--decoy_hard_tanimoto_max", type=float, default=0.95, help="Hard decoys max Tanimoto (inclusive).")
    ap.add_argument(
        "--physics_mode",
        choices=["topological", "hamiltonian", "both"],
        default="topological",
        help="Physics operator mode: topological (L), hamiltonian (H=L+V), or both.",
    )
    ap.add_argument(
        "--edge_weight_mode",
        choices=["unweighted", "bond_order", "bond_order_delta_chi"],
        default="unweighted",
        help="Edge weights for weighted operators: unweighted (binary), bond_order (1/2/3/1.5), or bond_order_delta_chi (bond_order*(1+alpha*|Δchi|)).",
    )
    ap.add_argument(
        "--operator_mode",
        choices=["laplacian", "h_operator"],
        default=None,
        help="DEPRECATED: use --physics_mode. laplacian -> topological, h_operator -> hamiltonian.",
    )
    ap.add_argument(
        "--seed_strategy",
        choices=["global", "per_row"],
        default="global",
        help="Seed usage: global uses provided seed for all rows; per_row uses stable_hash(id) XOR seed.",
    )
    ap.add_argument("--no_index", action="store_true", help="Do not emit index.md (evidence pack).")
    ap.add_argument("--no_manifest", action="store_true", help="Do not emit manifest.json (provenance).")
    ap.add_argument("--zip_pack", action="store_true", help="Create evidence_pack.zip with batch outputs.")
    ap.add_argument("--workers", type=int, default=1, help="Number of worker processes (default: 1).")
    ap.add_argument("--timeout_s", type=float, default=None, help="Per-molecule timeout in seconds (ERROR on hit).")
    ap.add_argument("--resume", action="store_true", help="Resume from existing summary.csv (skip processed ids).")
    ap.add_argument("--overwrite", action="store_true", help="Recompute even if id already processed.")
    ap.add_argument("--maxtasksperchild", type=int, default=100, help="Pool maxtasksperchild to mitigate RDKit leaks.")
    return ap.parse_args(argv)


def main_pipeline(argv: list[str] | None = None) -> int:
    args = _parse_pipeline_args(argv)
    if args.operator_mode is not None and args.physics_mode != "topological":
        raise SystemExit("Do not mix --operator_mode and --physics_mode; use --physics_mode only.")
    physics_mode = str(args.physics_mode)
    if args.operator_mode is not None:
        sys.stderr.write("WARNING: --operator_mode is deprecated; use --physics_mode instead.\n")
        physics_mode = "hamiltonian" if str(args.operator_mode) == "h_operator" else "topological"
    score_mode = "external_scores" if args.score_mode == "external_scores" else "mock"
    out = run_pipeline_v2(
        args.smiles,
        k_decoys=int(args.k_decoys),
        seed=int(args.seed),
        timestamp=str(args.timestamp),
        score_mode=score_mode,
        scores_input=str(args.scores_input) if args.scores_input else None,
        guardrails_max_atoms=int(args.guardrails_max_atoms),
        guardrails_require_connected=bool(args.guardrails_require_connected),
        decoy_hard_mode=bool(args.decoy_hard_mode),
        decoy_hard_tanimoto_min=float(args.decoy_hard_tanimoto_min),
        decoy_hard_tanimoto_max=float(args.decoy_hard_tanimoto_max),
        physics_mode=physics_mode,
        edge_weight_mode=str(args.edge_weight_mode),
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
        physics_mode="topological",
    )
    pipeline_out.write_text(json.dumps(pipeline, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    render_report_v2(pipeline, out_path=str(report_out), assets_dir=out_dir / f"{stem}_assets")
    sys.stdout.write(f"Pipeline JSON: {pipeline_out}\n")
    sys.stdout.write(f"Report: {report_out}\n")
    return 0


def main_batch(argv: list[str] | None = None) -> int:
    args = _parse_batch_args(argv)
    if args.operator_mode is not None and args.physics_mode != "topological":
        raise SystemExit("Do not mix --operator_mode and --physics_mode; use --physics_mode only.")
    physics_mode = str(args.physics_mode)
    if args.operator_mode is not None:
        sys.stderr.write("WARNING: --operator_mode is deprecated; use --physics_mode instead.\n")
        physics_mode = "hamiltonian" if str(args.operator_mode) == "h_operator" else "topological"
    input_csv = Path(args.input).resolve()
    out_dir = Path(args.out_dir).resolve()
    run_batch(
        input_csv=input_csv,
        out_dir=out_dir,
        artifacts=str(args.artifacts),
        seed=int(args.seed),
        timestamp=str(args.timestamp),
        k_decoys=int(args.k_decoys),
        score_mode=str(args.score_mode),
        scores_input=str(args.scores_input) if args.scores_input else None,
        guardrails_max_atoms=int(args.guardrails_max_atoms),
        guardrails_require_connected=bool(args.guardrails_require_connected),
        decoy_hard_mode=bool(args.decoy_hard_mode),
        decoy_hard_tanimoto_min=float(args.decoy_hard_tanimoto_min),
        decoy_hard_tanimoto_max=float(args.decoy_hard_tanimoto_max),
        physics_mode=physics_mode,
        edge_weight_mode=str(args.edge_weight_mode),
        seed_strategy=str(args.seed_strategy),
        no_index=bool(args.no_index),
        no_manifest=bool(args.no_manifest),
        zip_pack=bool(args.zip_pack),
        workers=int(args.workers),
        timeout_s=float(args.timeout_s) if args.timeout_s is not None else None,
        resume=bool(args.resume),
        overwrite=bool(args.overwrite),
        maxtasksperchild=int(args.maxtasksperchild),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main_pipeline())
