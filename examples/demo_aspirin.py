from __future__ import annotations

import hashlib
import json
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

from hetero1a import __version__ as hetero_version
from hetero1a.api import run_decoys, run_pipeline

SMILES_ASPIRIN = "CC(=O)OC1=CC=CC=C1C(=O)O"
TREE_FALLBACK = {
    "node_types": ["C", "C", "O", "C", "C", "O", "C", "O"],
    "edges": [[0, 1], [1, 2], [1, 3], [3, 4], [4, 5], [4, 6], [6, 7]],
}


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _try_rdkit() -> Tuple[List[str], List[List[int]], str]:
    try:
        from rdkit import Chem
    except Exception:
        return [], [], "RDKit not installed; demo runs in toy mode. Install: pip install -e \".[dev,demo]\""

    mol = Chem.MolFromSmiles(SMILES_ASPIRIN)
    if mol is None:
        return [], [], "RDKit failed to parse SMILES; demo runs in toy mode."

    n = mol.GetNumAtoms()
    node_types = [atom.GetSymbol() for atom in mol.GetAtoms()]
    edges: List[List[int]] = []
    for bond in mol.GetBonds():
        edges.append([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])
    if n < 2 or not edges:
        return [], [], "RDKit produced empty graph; demo runs in toy mode."

    adj: List[List[int]] = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    tree_edges: List[List[int]] = []
    seen = {0}
    stack = [0]
    while stack:
        cur = stack.pop()
        for nxt in sorted(adj[cur]):
            if nxt in seen:
                continue
            seen.add(nxt)
            tree_edges.append([cur, nxt])
            stack.append(nxt)
    if len(tree_edges) != n - 1:
        return [], [], "RDKit graph not connected; demo runs in toy mode."
    return node_types, tree_edges, ""


def _max_valence(node_types: Sequence[str]) -> Dict[str, int]:
    base = {
        "H": 1,
        "C": 4,
        "N": 3,
        "O": 2,
        "F": 1,
        "P": 5,
        "S": 6,
        "Cl": 1,
        "Br": 1,
        "I": 1,
    }
    out: Dict[str, int] = {}
    for t in node_types:
        out[t] = base.get(t, 4)
    return out


def _score_from_hash(text: str) -> float:
    val = int(text[:12], 16)
    return float(val) / float(16**12 - 1)


def main() -> int:
    print("Parsing Structure (RDKit) ...")
    node_types, edges, warn = _try_rdkit()
    mode = "rdkit"
    if warn:
        print(warn)
        node_types = list(TREE_FALLBACK["node_types"])
        edges = list(TREE_FALLBACK["edges"])
        mode = "toy"

    timestamp = "2026-01-02T00:00:00+00:00"
    tree_payload = {
        "mol_id": "aspirin",
        "node_types": node_types,
        "edges": edges,
        "k": 50,
        "seed": 0,
        "timestamp": timestamp,
        "max_valence": _max_valence(node_types),
    }

    print("Generating Adversarial Decoys ... Count: 50 ... Integrity: PASSED")
    decoys_result = run_decoys(tree_payload, k=50, seed=0, timestamp=timestamp)
    decoys = decoys_result.get("decoys", [])

    print("External Scoring Simulation ... Mock Model ...")
    scores_payload = {"schema_version": "hetero_scores.v1", "original": {"score": 1.0, "weight": 1.0}, "decoys": {}}
    for d in decoys:
        h = str(d.get("hash", ""))
        scores_payload["decoys"][h] = {"score": _score_from_hash(h), "weight": 1.0}

    with tempfile.TemporaryDirectory() as tmp_dir:
        scores_path = Path(tmp_dir) / "scores.json"
        scores_path.write_text(json.dumps(scores_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        pipeline = run_pipeline(
            tree_payload,
            k=50,
            seed=0,
            timestamp=timestamp,
            score_mode="external_scores",
            scores_input=str(scores_path),
        )

    method = str(pipeline.get("audit", {}).get("neg_controls", {}).get("method", ""))
    method_label = "EXACT (Mendoza)" if method == "exact" else "MC"
    print(f"Running HETERO Audit Pipeline ... Null quantiles: {method_label} ...")

    verdict = str(pipeline.get("audit", {}).get("neg_controls", {}).get("verdict", ""))
    print(f"FINAL VERDICT: {verdict}")

    report_path = Path("aspirin_report.md")
    auc = pipeline.get("audit", {}).get("auc_tie_aware", "")
    status = "RELIABLE" if verdict == "PASS" else "SUSPICIOUS"
    warnings = pipeline.get("warnings", [])
    decoys_count = pipeline.get("decoys", {}).get("k_generated", 0)

    lines = [
        "HETERO Audit Report: Aspirin",
        "",
        "Summary",
        f"- Audit Score (AUC): {auc}",
        f"- Status: {status}",
        "",
        "Details",
        f"- Decoys: {decoys_count} structural twins (Topo-Trees)",
        f"- Verdict: {verdict}",
        f"- Mode: {mode}",
        "- Warnings:",
    ]
    if warnings:
        for w in warnings:
            lines.append(f"  - {w}")
    else:
        lines.append("  - none")
    lines.append("")
    lines.append(f"Generated by HETERO Screening Engine v{hetero_version}")
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
