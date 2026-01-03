from __future__ import annotations

from pathlib import Path
from typing import Dict


def render_report_v2(payload: Dict[str, object], *, out_path: str = "aspirin_report.md") -> str:
    out_file = Path(out_path).resolve()
    decoys = payload.get("decoys", []) or []
    ring_info = payload.get("ring_info", {}) or {}
    physchem = payload.get("physchem", {}) or {}
    hardness = payload.get("hardness", {}) or {}
    warnings = payload.get("warnings", []) or []
    audit = payload.get("audit", {}) or {}
    neg = audit.get("neg_controls", {}) if isinstance(audit, dict) else {}
    run = payload.get("run", {}) or {}

    lines = [
        "HETERO-2 Report (hetero2_pipeline.v1)",
        "",
        "Summary",
        f"- Verdict: {neg.get('verdict', '')}",
        f"- Slack: {neg.get('slack', '')}",
        f"- Gate: {neg.get('gate', '')}",
        f"- Margin: {neg.get('margin', '')}",
        f"- Decoys generated: {len(decoys)}",
        "",
        "Rings",
        f"- n_rings: {ring_info.get('n_rings', '')}",
        f"- n_aromatic_rings: {ring_info.get('n_aromatic_rings', '')}",
        "",
        "PhysChem",
        f"- MW: {physchem.get('mw', '')}",
        f"- LogP: {physchem.get('logp', '')}",
        f"- TPSA: {physchem.get('tpsa', '')}",
        f"- HBD: {physchem.get('hbd', '')}",
        f"- HBA: {physchem.get('hba', '')}",
        f"- QED: {physchem.get('qed', '') if 'qed' in physchem else 'n/a'}",
        "",
        "Hardness",
        f"- Tanimoto Morgan min: {hardness.get('min', '')}",
        f"- Tanimoto Morgan median: {hardness.get('median', '')}",
        "",
        "Warnings",
    ]
    if warnings:
        for w in warnings:
            lines.append(f"- {w}")
    else:
        lines.append("- none")

    lines.extend(
        [
            "",
            "Repro",
            f"- schema_version: {payload.get('schema_version', '')}",
            f"- seed: {run.get('seed', '')}",
            f"- timestamp: {run.get('timestamp', '')}",
            f"- cmd: {run.get('cmd', '')}",
        ]
    )
    out_file.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return str(out_file)
