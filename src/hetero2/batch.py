from __future__ import annotations

import csv
import json
import os
import platform
import subprocess
import zlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import hetero1a
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


def _write_index_md(out_dir: Path, summary_rows: List[Dict[str, object]]) -> None:
    lines = [
        "# Evidence Index",
        "",
        "| id | status | verdict | gate | slack | warnings | seed_used | report | assets | pipeline |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in summary_rows:
        mol_id = str(row.get("id", ""))
        report_path = Path(str(row.get("report_path", ""))) if row.get("report_path") else None
        report_link = f"./{report_path.name}" if report_path and report_path.exists() else ""
        assets_dir = out_dir / f"{mol_id}_assets"
        assets_link = f"./{assets_dir.name}/" if assets_dir.exists() else ""
        pipeline_path = out_dir / f"{mol_id}.pipeline.json"
        pipeline_link = f"./{pipeline_path.name}" if pipeline_path.exists() else ""
        lines.append(
            f"| {mol_id} | {row.get('status','')} | {row.get('verdict','')} | {row.get('gate','')} | "
            f"{row.get('slack','')} | {row.get('warnings_count','')} | {row.get('seed_used','')} | "
            f"{report_link} | {assets_link} | {pipeline_link} |"
        )
    (out_dir / "index.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _git_sha_fallback() -> str | None:
    if os.environ.get("GITHUB_SHA"):
        return os.environ["GITHUB_SHA"]
    try:
        res = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, check=True, text=True)
        return res.stdout.strip()
    except Exception:
        return None


def _rdkit_version() -> str | None:
    try:
        import rdkit  # type: ignore

        return getattr(rdkit, "__version__", None) or None
    except Exception:
        return None


def _write_manifest(
    out_dir: Path,
    *,
    seed: int,
    seed_strategy: str,
    score_mode: str,
    guardrails_max_atoms: int,
    guardrails_require_connected: bool,
) -> None:
    payload: Dict[str, object] = {
        "tool_version": getattr(hetero1a, "__version__", None),
        "git_sha": _git_sha_fallback(),
        "python_version": platform.python_version(),
        "rdkit_version": _rdkit_version(),
        "timestamp_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "config": {
            "seed_strategy": seed_strategy,
            "seed": int(seed),
            "score_mode": score_mode,
            "guardrails_max_atoms": int(guardrails_max_atoms),
            "guardrails_require_connected": bool(guardrails_require_connected),
        },
    }
    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")


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
    no_index: bool = False,
    no_manifest: bool = False,
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

    if not no_index:
        _write_index_md(out_dir, summary_rows)
    if not no_manifest:
        _write_manifest(
            out_dir,
            seed=seed,
            seed_strategy=seed_strategy,
            score_mode=score_mode,
            guardrails_max_atoms=guardrails_max_atoms,
            guardrails_require_connected=guardrails_require_connected,
        )
    return summary_path
