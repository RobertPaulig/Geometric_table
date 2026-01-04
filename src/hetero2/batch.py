from __future__ import annotations

import csv
import json
import zlib
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


def _stable_hash_id(text: str) -> int:
    """Deterministic 32-bit hash for seeds (Python hash() is randomized)."""
    return int(zlib.crc32(text.encode("utf-8")) & 0xFFFFFFFF)


def run_batch(
    *,
    input_csv: Path,
    out_dir: Path,
    seed: int = 0,
    timestamp: str = "",
    k_decoys: int = 20,
    score_mode: str = "mock",
    scores_input: str | None = None,
    guardrails_max_atoms: int = 200,
    guardrails_require_connected: bool = True,
    seed_strategy: str = "global",
) -> Path:
    rows = _read_rows(input_csv)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_rows: List[Dict[str, object]] = []
    for idx, row in enumerate(rows):
        mol_id = row.get("id") or f"mol_{idx}"
        smiles = row.get("smiles", "")
        scores_path = row.get("scores_input") or scores_input or ""
        derived_seed = int(seed)
        if seed_strategy == "per_row":
            derived_seed = int(seed) ^ _stable_hash_id(str(mol_id))
        status = "OK"
        reason = ""
        warnings: List[str] = []
        pipeline: Dict[str, object] | None = None
        rep_path: Path | None = None
        try:
            if not smiles:
                status = "SKIP"
                reason = "missing_smiles"
            else:
                effective_score_mode = "mock" if score_mode == "mock" else "external_scores"
                pipeline = run_pipeline_v2(
                    smiles,
                    k_decoys=int(k_decoys),
                    seed=derived_seed,
                    timestamp=timestamp,
                    score_mode=effective_score_mode,
                    scores_input=scores_path or None,
                    guardrails_max_atoms=int(guardrails_max_atoms),
                    guardrails_require_connected=bool(guardrails_require_connected),
                )
                warnings = pipeline.get("warnings", []) if isinstance(pipeline, dict) else []
                if score_mode == "mock" and scores_path:
                    warnings.append("scores_input_ignored_in_mock_mode")
                skip = pipeline.get("skip") if isinstance(pipeline, dict) else None
                if isinstance(skip, dict):
                    status = "SKIP"
                    reason = str(skip.get("reason", "skip"))
        except Exception as exc:
            status = "ERROR"
            reason = repr(exc)
            pipeline = pipeline or {}
            warnings = warnings or []

        neg = pipeline.get("audit", {}).get("neg_controls", {}) if isinstance(pipeline, dict) else {}
        warnings_unique = sorted(set(warnings))
        if isinstance(pipeline, dict):
            pipeline["warnings"] = warnings_unique
            pipe_path = out_dir / f"{mol_id}.pipeline.json"
            pipe_path.write_text(json.dumps(pipeline, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
            rep_path = out_dir / f"{mol_id}.report.md"
            render_report_v2(pipeline, out_path=str(rep_path), assets_dir=out_dir / f"{mol_id}_assets")
        summary_rows.append(
            {
                "id": mol_id,
                "status": status,
                "reason": reason,
                "verdict": neg.get("verdict", ""),
                "gate": neg.get("gate", ""),
                "slack": neg.get("slack", ""),
                "margin": neg.get("margin", ""),
                "n_decoys": len(pipeline.get("decoys", [])) if isinstance(pipeline, dict) else "",
                "warnings_count": len(warnings_unique),
                "report_path": str(rep_path) if status == "OK" and rep_path is not None else "",
                "seed_used": derived_seed if isinstance(derived_seed, int) else "",
            }
        )
    summary_path = out_dir / "summary.csv"
    fieldnames: Sequence[str] = [
        "id",
        "status",
        "reason",
        "verdict",
        "gate",
        "slack",
        "margin",
        "n_decoys",
        "warnings_count",
        "report_path",
        "seed_used",
    ]
    with summary_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)
    return summary_path
