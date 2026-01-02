from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

from analysis.chem.decoys import edge_dist


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="HETERO-1A report: pipeline.json -> report.md + decoys.csv.")
    ap.add_argument("--input", required=True, help="Path to hetero_pipeline.v1 JSON.")
    ap.add_argument("--out_dir", default=".", help="Output directory (default: current dir).")
    ap.add_argument("--stem", default="", help="Output filename stem (default: tree_input_id).")
    return ap.parse_args(list(argv) if argv is not None else None)


def _edges_norm(edges: Sequence[Sequence[int]]) -> List[Tuple[int, int]]:
    return sorted((min(int(a), int(b)), max(int(a), int(b))) for a, b in edges)


def _edge_overlap_mean(decoys: Sequence[Dict[str, Any]], orig_edges: Sequence[Tuple[int, int]], n: int) -> Dict[str, float]:
    orig_set = set(orig_edges)
    values: List[float] = []
    for d in decoys:
        edges = _edges_norm(d["edges"])
        overlap = len(orig_set.intersection(set(edges)))
        values.append(float(overlap) / float(n - 1))
    return {"mean": float(sum(values) / len(values)) if values else float("nan")}


def render_report(payload: Dict[str, Any], *, out_dir: str = ".", stem: str = "") -> tuple[str, str]:
    out_dir_path = Path(out_dir).resolve()
    out_dir_path.mkdir(parents=True, exist_ok=True)
    stem_name = stem or str(payload.get("tree_input_id", "report"))

    schema_version = str(payload.get("schema_version", ""))
    run = payload.get("run", {})
    audit = payload.get("audit", {})
    selection = payload.get("selection", {})
    decoys = payload.get("decoys", {})
    warnings = payload.get("warnings", [])
    score_mode = str(payload.get("score_mode", ""))
    score_definition = str(payload.get("score_definition", ""))

    tree_input = payload.get("tree_input", {})
    orig_edges = _edges_norm(tree_input.get("edges", []))
    n = int(decoys.get("n", 0) or 0)

    decoy_list = list(decoys.get("decoys", []))
    decoys_sorted = sorted(decoy_list, key=lambda d: d.get("hash", ""))

    dist_to_original: List[float] = []
    for d in decoys_sorted:
        d_edges = _edges_norm(d["edges"])
        dist_to_original.append(edge_dist(d_edges, orig_edges, n))

    selected_hashes = set(selection.get("selected_hashes", []))

    report_md = out_dir_path / f"{stem_name}.report.md"
    report_csv = out_dir_path / f"{stem_name}.decoys.csv"

    md_lines = [
        "HETERO-1A Report (hetero_pipeline.v1)",
        "",
        "Summary",
        f"- score_mode: {score_mode}",
        f"- score_definition: {score_definition}",
        f"- audit.verdict: {audit.get('neg_controls', {}).get('verdict', '')}",
        f"- slack: {audit.get('neg_controls', {}).get('slack', '')}",
        f"- gate: {audit.get('neg_controls', {}).get('gate', '')}",
        f"- margin: {audit.get('neg_controls', {}).get('margin', '')}",
        f"- k_generated/k_selected: {decoys.get('k_generated', '')}/{selection.get('k_selected', '')}",
        f"- selection.method: {selection.get('method', '')}",
        f"- selection.metrics.min_pairwise_dist: {selection.get('metrics', {}).get('min_pairwise_dist', '')}",
        f"- decoys.metrics.dist_to_original.min: {decoys.get('metrics', {}).get('dist_to_original', {}).get('min', '')}",
        f"- decoys.metrics.dist_to_original.mean: {decoys.get('metrics', {}).get('dist_to_original', {}).get('mean', '')}",
        "",
        "Warnings",
    ]
    if warnings:
        for w in warnings:
            md_lines.append(f"- {w}")
    else:
        md_lines.append("- none")

    md_lines.extend(
        [
            "",
            "Repro",
            f"- schema_version: {schema_version}",
            f"- run.cmd: {' '.join(run.get('cmd', []))}",
            f"- seed: {run.get('seed', '')}",
            f"- timestamp: {run.get('timestamp', '')}",
        ]
    )

    report_md.write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    with report_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["index", "hash", "selected", "dist_to_original", "edge_overlap_to_original"])
        for idx, d in enumerate(decoys_sorted):
            h = str(d["hash"])
            selected = 1 if h in selected_hashes else 0
            d_edges = _edges_norm(d["edges"])
            dist = edge_dist(d_edges, orig_edges, n)
            overlap = float(len(set(orig_edges).intersection(set(d_edges)))) / float(n - 1) if n > 1 else 0.0
            writer.writerow([idx, h, selected, f"{dist:.6f}", f"{overlap:.6f}"])

    return str(report_md), str(report_csv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    payload = json.loads(Path(args.input).read_text(encoding="utf-8"))
    render_report(payload, out_dir=args.out_dir, stem=args.stem)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
