from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

from analysis.chem.audit import run_audit
from analysis.chem.decoys import _edge_dist, run_decoys


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _normalized_cmd(argv: Sequence[str]) -> List[str]:
    out: List[str] = []
    skip_next = False
    for token in argv:
        if skip_next:
            skip_next = False
            continue
        if token in {"--out", "--tree_input"}:
            skip_next = True
            continue
        out.append(token)
    return out


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="HETERO-1A pipeline: decoys -> selection -> audit (single JSON report).")
    ap.add_argument("--tree_input", required=True, help="Path to tree JSON input.")
    ap.add_argument("--out", default="", help="Output JSON path (default: stdout).")
    ap.add_argument("--k", type=int, default=50, help="Number of decoys to request.")
    ap.add_argument("--seed", type=int, default=0, help="Seed.")
    ap.add_argument("--timestamp", default="", help="UTC timestamp override (ISO). Default: now().")

    ap.add_argument("--min_dist_to_original", type=float, default=0.0)
    ap.add_argument("--min_pair_dist", type=float, default=0.0)
    ap.add_argument("--max_attempts", type=int, default=None)

    ap.add_argument("--neg_control_reps", type=int, default=200)
    ap.add_argument("--margin", type=float, default=0.05)

    ap.add_argument("--select_k", type=int, default=20)
    ap.add_argument("--selection", choices=["maxmin", "firstk"], default="maxmin")
    return ap.parse_args(list(argv) if argv is not None else None)


def _edges_norm(edges: Sequence[Sequence[int]]) -> List[Tuple[int, int]]:
    return sorted((min(int(a), int(b)), max(int(a), int(b))) for a, b in edges)


def _selection_firstk(decoys_sorted: Sequence[Dict[str, Any]], k: int) -> List[int]:
    return list(range(min(k, len(decoys_sorted))))


def _selection_maxmin(decoys_sorted: Sequence[Dict[str, Any]], n: int, k: int) -> List[int]:
    if not decoys_sorted or k <= 0:
        return []
    selected = [0]
    while len(selected) < min(k, len(decoys_sorted)):
        best_idx = None
        best_score = -1.0
        for idx in range(len(decoys_sorted)):
            if idx in selected:
                continue
            cand_edges = _edges_norm(decoys_sorted[idx]["edges"])
            min_d = float("inf")
            for sidx in selected:
                s_edges = _edges_norm(decoys_sorted[sidx]["edges"])
                d = _edge_dist(cand_edges, s_edges, n)
                if d < min_d:
                    min_d = d
            cand_hash = str(decoys_sorted[idx]["hash"])
            if (min_d > best_score) or (min_d == best_score and best_idx is not None and cand_hash < str(decoys_sorted[best_idx]["hash"])) or (
                min_d == best_score and best_idx is None
            ):
                best_score = min_d
                best_idx = idx
        if best_idx is None:
            break
        selected.append(best_idx)
    return selected


def _pairwise_stats(decoys_sorted: Sequence[Dict[str, Any]], n: int, indices: Sequence[int]) -> Dict[str, float]:
    if len(indices) < 2:
        return {"min_pairwise_dist": float("nan"), "mean_pairwise_dist": float("nan")}
    dists: List[float] = []
    for i in range(len(indices)):
        for j in range(i + 1, len(indices)):
            a = _edges_norm(decoys_sorted[indices[i]]["edges"])
            b = _edges_norm(decoys_sorted[indices[j]]["edges"])
            dists.append(_edge_dist(a, b, n))
    return {
        "min_pairwise_dist": float(min(dists)) if dists else float("nan"),
        "mean_pairwise_dist": float(sum(dists) / len(dists)) if dists else float("nan"),
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    timestamp = str(args.timestamp).strip() or _utc_now_iso()

    tree_payload = json.loads(Path(args.tree_input).read_text(encoding="utf-8"))
    tree_payload["k"] = int(args.k)
    tree_payload["seed"] = int(args.seed)
    tree_payload["timestamp"] = timestamp

    decoys_result = run_decoys(
        tree_payload,
        k=int(args.k),
        seed=int(args.seed),
        timestamp=timestamp,
        min_dist_to_original=float(args.min_dist_to_original),
        min_pair_dist=float(args.min_pair_dist),
        max_attempts=int(args.max_attempts) if args.max_attempts is not None else None,
        cmd_argv=sys.argv,
    )

    decoys_list = list(decoys_result.get("decoys", []))
    decoys_sorted = sorted(decoys_list, key=lambda d: str(d.get("hash", "")))
    n = int(decoys_result.get("n", 0))
    orig_edges = _edges_norm(tree_payload["edges"])

    items = [{"label": 1, "score": 1.0, "weight": 1.0}]
    for d in decoys_sorted:
        dist_to_orig = _edge_dist(_edges_norm(d["edges"]), orig_edges, n)
        items.append({"label": 0, "score": float(1.0 - dist_to_orig), "weight": 1.0})

    audit_payload = {"dataset_id": f"pipeline:{tree_payload.get('mol_id','')}", "items": items}
    audit_result = run_audit(
        audit_payload,
        seed=int(args.seed),
        timestamp=timestamp,
        cmd_argv=sys.argv,
        neg_control_reps=int(args.neg_control_reps),
        neg_auc_margin=float(args.margin),
    )

    select_k = int(args.select_k)
    if args.selection == "firstk":
        selected_indices = _selection_firstk(decoys_sorted, select_k)
    else:
        selected_indices = _selection_maxmin(decoys_sorted, n, select_k)

    selection_metrics = _pairwise_stats(decoys_sorted, n, selected_indices)

    out: Dict[str, Any] = {
        "schema_version": "hetero_pipeline.v1",
        "tree_input_id": str(tree_payload.get("mol_id", "")),
        "decoys": decoys_result,
        "selection": {
            "method": str(args.selection),
            "k_requested": select_k,
            "k_selected": int(len(selected_indices)),
            "selected_indices": [int(i) for i in selected_indices],
            "metrics": selection_metrics,
        },
        "audit": audit_result,
        "warnings": [],
        "run": {"seed": int(args.seed), "timestamp": timestamp, "cmd": _normalized_cmd(sys.argv)},
    }

    text = json.dumps(out, ensure_ascii=False, sort_keys=True, indent=2) + os.linesep
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
    else:
        sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

