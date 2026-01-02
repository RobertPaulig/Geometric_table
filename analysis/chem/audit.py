from __future__ import annotations

import argparse
import json
import os
import random
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

from analysis.chem.hetero_score_utils import _roc_auc
from analysis.chem.neg_control_null_auc import null_auc_quantile


@dataclass(frozen=True)
class _Item:
    score: float
    label: int
    weight: float


def _read_version() -> str:
    try:
        return Path("VERSION").read_text(encoding="utf-8").strip()
    except Exception:
        return "unknown"


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _load_items(payload: Dict[str, Any]) -> Tuple[str, List[_Item]]:
    dataset_id = str(payload.get("dataset_id", "")).strip()
    if not dataset_id:
        raise ValueError("input JSON must contain non-empty 'dataset_id'")

    items_raw = payload.get("items", None)
    if not isinstance(items_raw, list) or not items_raw:
        raise ValueError("input JSON must contain non-empty list 'items'")

    items: List[_Item] = []
    for row in items_raw:
        if not isinstance(row, dict):
            raise ValueError("each item must be an object")
        score = float(row.get("score", 0.0))
        label = int(row.get("label", 0))
        if label not in (0, 1):
            raise ValueError("item.label must be 0/1")
        weight = float(row.get("weight", 1.0))
        items.append(_Item(score=score, label=label, weight=weight))

    return dataset_id, items


def _auc_from_items(items: Sequence[_Item]) -> float:
    scores = [it.score for it in items]
    labels = [it.label for it in items]
    weights = [it.weight for it in items]
    return float(_roc_auc(scores, labels, weights))


def _neg_control_quantiles(
    *,
    items: Sequence[_Item],
    seed: int,
    reps: int,
    q: float,
) -> Tuple[float, float]:
    if reps <= 0:
        raise ValueError("reps must be positive")
    if not (0.0 < q <= 1.0):
        raise ValueError("q must be in (0,1]")

    rng = random.Random(seed)
    scores = [it.score for it in items]
    weights = [it.weight for it in items]
    labels_orig = [it.label for it in items]

    aucs_perm: List[float] = []
    aucs_rand_fp: List[float] = []
    for _ in range(reps):
        labels = labels_orig[:]
        rng.shuffle(labels)
        aucs_perm.append(float(_roc_auc(scores, labels, weights)))

        rand_scores = [rng.random() for _ in scores]
        aucs_rand_fp.append(float(_roc_auc(rand_scores, labels_orig, weights)))

    def _quantile(values: Sequence[float], q_val: float) -> float:
        vals = sorted(values)
        if not vals:
            return float("nan")
        # nearest-rank quantile (deterministic)
        k = max(1, int(round(q_val * len(vals))))
        return float(vals[min(k - 1, len(vals) - 1)])

    return _quantile(aucs_perm, q), _quantile(aucs_rand_fp, q)


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="HETERO-1A audit: stable JSON summary for a simple scored dataset.")
    ap.add_argument("--input", required=True, help="Path to JSON with {dataset_id, items:[{score,label,weight?}]} .")
    ap.add_argument("--out", default="", help="Output JSON path (default: stdout).")
    ap.add_argument("--seed", type=int, default=0, help="Base seed for neg-controls sampling.")
    ap.add_argument("--timestamp", default="", help="UTC timestamp override (ISO). Default: now().")
    ap.add_argument("--neg_control_reps", type=int, default=200, help="Repetitions for neg-controls sampling.")
    ap.add_argument("--neg_control_quantile", type=float, default=0.95, help="Quantile for neg-controls and null-q.")
    ap.add_argument("--neg_auc_margin", type=float, default=0.05, help="Margin added to null-q to form gate.")
    return ap.parse_args(list(argv) if argv is not None else None)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)

    payload = json.loads(Path(args.input).read_text(encoding="utf-8"))
    dataset_id, items = _load_items(payload)

    n_pos = sum(1 for it in items if it.label == 1)
    n_neg = sum(1 for it in items if it.label == 0)

    auc = _auc_from_items(items)

    perm_q, rand_q = _neg_control_quantiles(
        items=items,
        seed=int(args.seed),
        reps=int(args.neg_control_reps),
        q=float(args.neg_control_quantile),
    )
    neg_auc_max = max(perm_q, rand_q)

    null_q = float(null_auc_quantile(n_pos, n_neg, float(args.neg_control_quantile)))
    margin = float(args.neg_auc_margin)
    gate = float(null_q + margin)
    slack = float(gate - neg_auc_max)
    verdict = "PASS" if slack >= 0.0 else "FAIL"

    out: Dict[str, Any] = {
        "version": _read_version(),
        "dataset_id": dataset_id,
        "n_pos": int(n_pos),
        "n_neg": int(n_neg),
        "auc_tie_aware": auc,
        "neg_controls": {
            "null_q": null_q,
            "gate": gate,
            "margin": margin,
            "slack": slack,
            "verdict": verdict,
        },
        "run": {
            "seed": int(args.seed),
            "timestamp": str(args.timestamp).strip() or _utc_now_iso(),
            "cmd": " ".join(_normalized_cmd(sys.argv)),
        },
    }

    text = json.dumps(out, ensure_ascii=False, sort_keys=True, indent=2) + os.linesep
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
    else:
        sys.stdout.write(text)
    return 0


def _normalized_cmd(argv: Sequence[str]) -> List[str]:
    # Keep a stable command string for audits/tests: drop `--out <path>` because it varies by environment.
    out: List[str] = []
    skip_next = False
    for token in argv:
        if skip_next:
            skip_next = False
            continue
        if token == "--out":
            skip_next = True
            continue
        out.append(token)
    return out


if __name__ == "__main__":
    raise SystemExit(main())
