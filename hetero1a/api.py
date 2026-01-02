from __future__ import annotations

from typing import Any, Dict, Sequence

from analysis.chem.audit import run_audit as _run_audit
from analysis.chem.decoys import run_decoys as _run_decoys
from analysis.chem.pipeline import run_pipeline as _run_pipeline
from analysis.chem.report import render_report as _render_report


def run_audit(
    payload: Dict[str, Any],
    *,
    seed: int = 0,
    timestamp: str = "",
    cmd_argv: Sequence[str] | None = None,
    neg_control_reps: int = 200,
    neg_control_quantile: float = 0.95,
    neg_auc_margin: float = 0.05,
) -> Dict[str, Any]:
    return _run_audit(
        payload,
        seed=seed,
        timestamp=timestamp,
        cmd_argv=cmd_argv,
        neg_control_reps=neg_control_reps,
        neg_control_quantile=neg_control_quantile,
        neg_auc_margin=neg_auc_margin,
    )


def run_decoys(
    payload: Dict[str, Any],
    *,
    k: int | None = None,
    seed: int | None = None,
    timestamp: str | None = None,
    min_dist_to_original: float = 0.0,
    min_pair_dist: float = 0.0,
    max_attempts: int | None = None,
    cmd_argv: Sequence[str] | None = None,
) -> Dict[str, Any]:
    return _run_decoys(
        payload,
        k=k,
        seed=seed,
        timestamp=timestamp,
        min_dist_to_original=min_dist_to_original,
        min_pair_dist=min_pair_dist,
        max_attempts=max_attempts,
        cmd_argv=cmd_argv,
    )


def run_pipeline(
    tree_payload: Dict[str, Any],
    *,
    k: int = 50,
    seed: int = 0,
    timestamp: str = "",
    min_dist_to_original: float = 0.0,
    min_pair_dist: float = 0.0,
    max_attempts: int | None = None,
    neg_control_reps: int = 200,
    margin: float = 0.05,
    select_k: int = 20,
    selection: str = "maxmin",
    score_mode: str = "toy_edge_dist",
    scores_input: str = "",
    cmd_argv: Sequence[str] | None = None,
) -> Dict[str, Any]:
    return _run_pipeline(
        tree_payload,
        k=k,
        seed=seed,
        timestamp=timestamp,
        min_dist_to_original=min_dist_to_original,
        min_pair_dist=min_pair_dist,
        max_attempts=max_attempts,
        neg_control_reps=neg_control_reps,
        margin=margin,
        select_k=select_k,
        selection=selection,
        score_mode=score_mode,
        scores_input=scores_input,
        cmd_argv=cmd_argv,
    )


def render_report(payload: Dict[str, Any], *, out_dir: str = ".", stem: str = "") -> tuple[str, str]:
    return _render_report(payload, out_dir=out_dir, stem=stem)
