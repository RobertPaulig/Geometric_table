from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

from hetero2.pipeline import run_pipeline_v2
from hetero2.report import render_report_v2


def _read_rows(path: Path) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({k.strip(): v.strip() for k, v in row.items()})
    return rows


def run_batch(
    *,
    input_csv: Path,
    out_dir: Path,
    seed: int = 0,
    timestamp: str = "",
    k_decoys: int = 20,
    score_mode: str = "external_scores",
    scores_input: str | None = None,
) -> Path:
    rows = _read_rows(input_csv)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_rows: List[Dict[str, object]] = []
    for idx, row in enumerate(rows):
        mol_id = row.get("id") or f"mol_{idx}"
        smiles = row.get("smiles", "")
        if not smiles:
            continue
        scores_path = row.get("scores_input") or scores_input or ""
        pipeline = run_pipeline_v2(
            smiles,
            k_decoys=int(k_decoys),
            seed=int(seed),
            timestamp=timestamp,
            score_mode="external_scores" if scores_path else "mock",
            scores_input=scores_path or None,
        )
        stem = mol_id
        pipe_path = out_dir / f"{stem}.pipeline.json"
        rep_path = out_dir / f"{stem}.report.md"
        assets_dir = out_dir / f"{stem}_assets"
        pipe_path.write_text(json.dumps(pipeline, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
        render_report_v2(pipeline, out_path=str(rep_path), assets_dir=assets_dir)
        neg = pipeline.get("audit", {}).get("neg_controls", {}) if isinstance(pipeline, dict) else {}
        warnings = pipeline.get("warnings", []) if isinstance(pipeline, dict) else []
        summary_rows.append(
            {
                "id": mol_id,
                "verdict": neg.get("verdict", ""),
                "gate": neg.get("gate", ""),
                "slack": neg.get("slack", ""),
                "margin": neg.get("margin", ""),
                "n_decoys": len(pipeline.get("decoys", [])) if isinstance(pipeline, dict) else "",
                "warnings_count": len(set(warnings)) if isinstance(warnings, list) else 0,
                "report_path": str(rep_path),
            }
        )
    summary_path = out_dir / "summary.csv"
    if summary_rows:
        with summary_path.open("w", encoding="utf-8", newline="") as f:
            fieldnames: Sequence[str] = ["id", "verdict", "gate", "slack", "margin", "n_decoys", "warnings_count", "report_path"]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(summary_rows)
    else:
        summary_path.write_text("id,verdict,gate,slack,margin,n_decoys,warnings_count,report_path\n", encoding="utf-8")
    return summary_path
