from __future__ import annotations

import csv
import datetime as dt
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd

from analysis.chem.hetero_operator import build_operator_H, hetero_energy_from_state, hetero_fingerprint

TYPE_LABELS = {0: "C", 1: "N", 2: "O"}
VALENCE_BY_TYPE = {0: 4, 1: 3, 2: 2}
RHO_BY_TYPE = {0: 0.0, 1: 0.2, 2: 0.5}
ALPHA_H = 0.5
EPS_STD = 1e-12
EPS_WEIGHT = 1e-12
EPS_SCORE = 1e-12
ENERGY_COLLISION_EPS = 1e-9
ENERGY_COLLISION_EPS_STRICT = 1e-12
ENERGY_KEY_SCHEME = "absdiff"
FP_COLLISION_EPS = 1e-9


@dataclass(frozen=True)
class PairSpec:
    class_a: str
    class_b: str
    other_labels: Tuple[str, ...] = ()


PAIR_SPECS: Dict[str, Tuple[PairSpec, ...]] = {
    "C2H6O": (PairSpec("alcohol", "ether"),),
    "C3H8O": (PairSpec("alcohol", "ether"),),
    "C4H10O": (PairSpec("alcohol", "ether"),),
    "C5H12O": (PairSpec("alcohol", "ether"),),
    "C6H14O": (PairSpec("alcohol", "ether"),),
    "C7H16O": (PairSpec("alcohol", "ether"),),
    "C2H7N": (PairSpec("primary_amine", "secondary_amine", ("tertiary_amine",)),),
    "C4H11N": (PairSpec("primary_amine", "secondary_amine", ("tertiary_amine",)),),
    "C5H13N": (PairSpec("primary_amine", "secondary_amine", ("tertiary_amine",)),),
    "C6H15N": (PairSpec("primary_amine", "secondary_amine", ("tertiary_amine",)),),
    "C7H17N": (PairSpec("primary_amine", "secondary_amine", ("tertiary_amine",)),),
}


def parse_state_id(state_id: str) -> Tuple[List[Tuple[int, int]], List[int]]:
    if not state_id.startswith("het:"):
        raise ValueError(f"Unexpected state_id: {state_id}")
    parts = state_id.split(";")
    if len(parts) != 2:
        raise ValueError(f"Unexpected state_id: {state_id}")
    edges_part = parts[0].split("=", 1)[1]
    types_part = parts[1].split("=", 1)[1]
    edges: List[Tuple[int, int]] = []
    if edges_part:
        for item in edges_part.split(","):
            if not item:
                continue
            a, b = item.split("-")
            a_i = int(a)
            b_i = int(b)
            if a_i > b_i:
                a_i, b_i = b_i, a_i
            edges.append((a_i, b_i))
    types = [int(t) for t in types_part.split(",")] if types_part else []
    return edges, types


def _degrees(n: int, edges: Sequence[Tuple[int, int]]) -> List[int]:
    deg = [0] * n
    for u, v in edges:
        deg[int(u)] += 1
        deg[int(v)] += 1
    return deg


def classify_functional_group(formula: str, edges: Sequence[Tuple[int, int]], types: Sequence[int]) -> str:
    n = len(types)
    deg = _degrees(n, edges)
    o_idx = [i for i, t in enumerate(types) if t == 2]
    if len(o_idx) == 1:
        od = deg[o_idx[0]]
        if od == 1:
            return "alcohol"
        if od == 2:
            return "ether"
        return "invalid_O_valence"
    n_idx = [i for i, t in enumerate(types) if t == 1]
    if len(n_idx) == 1:
        nd = deg[n_idx[0]]
        if nd == 1:
            return "primary_amine"
        if nd == 2:
            return "secondary_amine"
        if nd == 3:
            return "tertiary_amine"
        return "invalid_N_valence"
    return "unknown"


def compute_state_table(
    *,
    formula: str,
    state_ids: Sequence[str],
    p_exact: Mapping[str, float],
    p_emp: Mapping[str, float],
    rho_by_type: Mapping[int, float] | None = None,
    alpha_H_override: float | None = None,
) -> pd.DataFrame:
    records: List[Dict[str, object]] = []
    rho = dict(RHO_BY_TYPE if rho_by_type is None else rho_by_type)
    alpha = ALPHA_H if alpha_H_override is None else float(alpha_H_override)
    for sid in state_ids:
        edges, types = parse_state_id(sid)
        n = len(types)
        deg = _degrees(n, edges)
        implicit_h = [VALENCE_BY_TYPE.get(types[i], 0) - deg[i] for i in range(n)]
        energy = hetero_energy_from_state(
            n,
            edges,
            types,
            rho_by_type=rho,
            alpha_H=alpha,
            valence_by_type=VALENCE_BY_TYPE,
        )
        H = build_operator_H(
            n,
            edges,
            types,
            rho_by_type=rho,
            alpha_H=alpha,
            valence_by_type=VALENCE_BY_TYPE,
        )
        fp = hetero_fingerprint(H)
        class_label = classify_functional_group(formula, edges, types)
        rec: Dict[str, object] = {
            "formula": formula,
            "state_id": sid,
            "class_label": class_label,
            "N_heavy": n,
            "P_exact": float(p_exact.get(sid, 0.0)),
            "P_emp": float(p_emp.get(sid, 0.0)),
            "energy": float(energy),
            "is_valid_valence": int(all(h >= 0 for h in implicit_h)),
            "degree_seq": ",".join(str(d) for d in sorted(deg)),
        }
        o_idx = next((i for i, t in enumerate(types) if t == 2), None)
        if o_idx is not None:
            rec["o_deg"] = deg[o_idx]
            rec["o_implicit_h"] = implicit_h[o_idx]
        else:
            rec["o_deg"] = float("nan")
            rec["o_implicit_h"] = float("nan")
        n_idx = next((i for i, t in enumerate(types) if t == 1), None)
        if n_idx is not None:
            rec["n_deg"] = deg[n_idx]
            rec["n_implicit_h"] = implicit_h[n_idx]
        else:
            rec["n_deg"] = float("nan")
            rec["n_implicit_h"] = float("nan")
        for idx, val in enumerate(fp):
            rec[f"fp{idx}"] = float(val)
        records.append(rec)
    df = pd.DataFrame(records)
    eps = 1e-9
    df["log_ratio_emp_exact"] = np.log((df["P_emp"] + eps) / (df["P_exact"] + eps))
    return df


def _weighted_mean_std(vals: Sequence[float], weights: Sequence[float]) -> Tuple[float, float]:
    arr = np.asarray(vals, dtype=float)
    w = np.asarray(weights, dtype=float)
    total_w = float(np.sum(w))
    if total_w <= 0:
        return 0.0, 0.0
    mean = float(np.sum(arr * w) / total_w)
    var = float(np.sum(w * (arr - mean) ** 2) / total_w)
    return mean, math.sqrt(var)


def _collision_breakdown(
    values: Sequence[float],
    labels: Sequence[str] | None = None,
    *,
    tol: float = ENERGY_COLLISION_EPS,
    state_ids: Sequence[str] | None = None,
    record_pairs: bool = False,
    energy_key_scheme: str = ENERGY_KEY_SCHEME,
    pair_tol_fn: callable | None = None,
) -> Dict[str, object]:
    arr = np.asarray(values, dtype=float)
    n = arr.size
    total_pairs = n * (n - 1) // 2
    breakdown: Dict[str, object] = {
        "total_pairs": float(total_pairs),
        "coll_total_pairs": 0.0,
        "coll_within_pairs": 0.0,
        "coll_cross_pairs": 0.0,
        "coll_total": 0.0,
        "coll_within": 0.0,
        "coll_cross": 0.0,
        "collision_eps": float(tol),
        "energy_key_scheme": energy_key_scheme,
        "max_abs_delta_cross": 0.0,
    }
    details: List[Dict[str, object]] = []
    if total_pairs <= 0:
        breakdown["cross_pair_records"] = details
        if record_pairs:
            breakdown["cross_pair_records"] = details
        return breakdown
    if tol <= 0:
        raise ValueError("collision tolerance must be > 0")
    if not str(energy_key_scheme).startswith("absdiff"):
        raise ValueError(f"Unknown energy_key_scheme: {energy_key_scheme}")
    lab_arr = np.asarray(labels) if labels is not None else np.array([None] * n, dtype=object)
    sid_arr = np.asarray(state_ids) if state_ids is not None else np.array([None] * n, dtype=object)
    key_arr = np.round(arr / tol).astype(np.int64)
    within = 0
    cross = 0
    for i in range(n):
        for j in range(i + 1, n):
            abs_delta = abs(arr[i] - arr[j])
            tol_ij = tol if pair_tol_fn is None else float(pair_tol_fn(arr[i], arr[j]))
            if abs_delta <= tol_ij:
                if lab_arr[i] == lab_arr[j]:
                    within += 1
                else:
                    cross += 1
                    breakdown["max_abs_delta_cross"] = max(float(breakdown["max_abs_delta_cross"]), float(abs_delta))
                    if record_pairs:
                        details.append(
                            {
                                "state_id_i": sid_arr[i],
                                "class_i": lab_arr[i],
                                "E_i": float(arr[i]),
                                "energy_key_i": str(key_arr[i]),
                                "state_id_j": sid_arr[j],
                                "class_j": lab_arr[j],
                                "E_j": float(arr[j]),
                                "energy_key_j": str(key_arr[j]),
                                "abs_delta": float(abs_delta),
                            }
                        )
    total = within + cross
    breakdown["coll_total_pairs"] = float(total)
    breakdown["coll_within_pairs"] = float(within)
    breakdown["coll_cross_pairs"] = float(cross)
    if total_pairs > 0:
        breakdown["coll_within"] = within / total_pairs
        breakdown["coll_cross"] = cross / total_pairs
        breakdown["coll_total"] = breakdown["coll_within"] + breakdown["coll_cross"]
    else:
        breakdown["coll_total"] = 0.0
    breakdown["cross_pair_records"] = details if record_pairs else []
    return breakdown


def _collision_rate(values: Sequence[float], tol: float = ENERGY_COLLISION_EPS) -> float:
    return float(
        _collision_breakdown(
            values,
            labels=None,
            tol=tol,
            state_ids=None,
            record_pairs=False,
        )["coll_total"]
    )


def _fp_collision_rate(fp_vectors: Sequence[Sequence[float]], tol: float = FP_COLLISION_EPS) -> float:
    if not fp_vectors:
        return float("nan")
    arr = np.asarray(fp_vectors, dtype=float)
    n = arr.shape[0]
    if n <= 1:
        return 0.0
    collisions = 0
    total = 0
    for i in range(n):
        for j in range(i + 1, n):
            total += 1
            if np.allclose(arr[i], arr[j], atol=tol):
                collisions += 1
    return float(collisions) / float(total) if total > 0 else 0.0


def _roc_auc(scores: Sequence[float], labels: Sequence[int], weights: Sequence[float]) -> float:
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=int)
    weights = np.asarray(weights, dtype=float)
    pos = labels == 1
    neg = labels == 0
    weight_pos = float(np.sum(weights[pos]))
    weight_neg = float(np.sum(weights[neg]))
    if weight_pos == 0 or weight_neg == 0:
        return float("nan")
    rank = np.argsort(scores)
    scores = scores[rank]
    labels = labels[rank]
    weights = weights[rank]
    cum_neg = 0.0
    auc = 0.0
    for lbl, w in zip(labels, weights):
        if lbl == 1:
            auc += w * cum_neg
        else:
            cum_neg += w
    auc /= (weight_pos * weight_neg)
    return float(auc)


def _pair_metric_components(
    values_a: Sequence[float],
    values_b: Sequence[float],
    *,
    weights_a: Sequence[float],
    weights_b: Sequence[float],
) -> Dict[str, object]:
    len_a = len(values_a)
    len_b = len(values_b)
    sum_w_a = float(np.sum(weights_a))
    sum_w_b = float(np.sum(weights_b))
    mean_a, std_a = _weighted_mean_std(values_a, weights_a)
    mean_b, std_b = _weighted_mean_std(values_b, weights_b)
    delta = mean_a - mean_b
    pooled_std = math.sqrt(std_a ** 2 + std_b ** 2)
    is_trivial = (
        len_a < 2
        or len_b < 2
        or pooled_std < EPS_STD
        or sum_w_a <= EPS_WEIGHT
        or sum_w_b <= EPS_WEIGHT
    )
    if is_trivial:
        return {
            "mean_a": mean_a,
            "mean_b": mean_b,
            "delta": delta,
            "effect_size": float("nan"),
            "auc_raw": 0.5,
            "auc_best": 0.5,
            "is_trivial": True,
        }
    labels = np.concatenate([np.ones(len_a), np.zeros(len_b)])
    scores = np.concatenate([np.asarray(values_a, dtype=float), np.asarray(values_b, dtype=float)])
    weights = np.concatenate([np.asarray(weights_a, dtype=float), np.asarray(weights_b, dtype=float)])
    auc = _roc_auc(scores, labels.astype(int), weights)
    if math.isnan(auc):
        auc = 0.5
    return {
        "mean_a": mean_a,
        "mean_b": mean_b,
        "delta": delta,
        "effect_size": delta / (pooled_std + 1e-12),
        "auc_raw": auc,
        "auc_best": max(auc, 1.0 - auc),
        "is_trivial": False,
    }


def _spearman_abs(x: Sequence[float], y: Sequence[float]) -> float:
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    if x_arr.size <= 1 or y_arr.size <= 1:
        return float("nan")
    if x_arr.size != y_arr.size:
        raise ValueError("Spearman arrays must have same length")
    rx = pd.Series(x_arr).rank(method="average").to_numpy()
    ry = pd.Series(y_arr).rank(method="average").to_numpy()
    if np.allclose(rx, ry, atol=1e-12, rtol=1e-12):
        return 1.0
    denom = np.std(rx) * np.std(ry)
    if denom == 0:
        return float("nan")
    corr = np.corrcoef(rx, ry)[0, 1]
    return float(abs(corr))


def compute_formula_scores(
    df_states: pd.DataFrame,
    *,
    formula: str,
    weights_col: str,
    run_meta: Mapping[str, object],
    debug_fp: bool = False,
    fp_exclude_energy_like: bool = True,
    fp_energy_like_threshold: float = 0.999,
    collision_log_dir: Path | None = None,
) -> List[Dict[str, object]]:
    weights = df_states[weights_col].fillna(0.0).to_numpy(dtype=float)
    if np.all(weights == 0):
        weights = np.ones_like(weights)
    score_rows: List[Dict[str, object]] = []
    run_id = run_meta.get("run_id", f"run_{formula}_{dt.datetime.utcnow().isoformat()}")
    coverage_meta = float(run_meta.get("coverage_unique_eq", 0.0))
    if coverage_meta == 0.0 and len(df_states) > 0:
        coverage_meta = float((df_states["P_emp"] > 0).sum() / len(df_states))
    record_pairs = collision_log_dir is not None
    collision_info = _collision_breakdown(
        df_states["energy"].tolist(),
        df_states["class_label"].tolist(),
        state_ids=df_states["state_id"].tolist(),
        record_pairs=record_pairs,
    )
    collision_info_strict = _collision_breakdown(
        df_states["energy"].tolist(),
        df_states["class_label"].tolist(),
        state_ids=None,
        record_pairs=False,
        tol=ENERGY_COLLISION_EPS_STRICT,
        pair_tol_fn=lambda ei, ej: max(
            ENERGY_COLLISION_EPS_STRICT,
            ENERGY_COLLISION_EPS_STRICT * max(abs(float(ei)), abs(float(ej))),
        ),
        energy_key_scheme="absdiff_strict",
    )
    energy_coll = collision_info["coll_total"]
    fp_cols_all = [c for c in df_states.columns if c.startswith("fp")]
    if fp_cols_all:
        fps = df_states[fp_cols_all].fillna(0.0).to_numpy().tolist()
        fp_coll = _fp_collision_rate(fps)
    else:
        fp_coll = float("nan")
    base_row = dict(run_meta)
    base_row.update(
        {
            "run_id": run_id,
            "formula": formula,
            "weight_source": weights_col,
            "support_exact": int(df_states["P_exact"].gt(0).sum()),
            "support_emp": int(df_states["P_emp"].gt(0).sum()),
            "coverage_unique_eq": coverage_meta,
            "energy_collision_rate": energy_coll,
            "energy_collision_eps": ENERGY_COLLISION_EPS,
            "collision_eps": collision_info["collision_eps"],
            "energy_key_scheme": collision_info["energy_key_scheme"],
            "coll_total": collision_info["coll_total"],
            "coll_within": collision_info["coll_within"],
            "coll_cross": collision_info["coll_cross"],
            "coll_total_pairs": collision_info["coll_total_pairs"],
            "coll_within_pairs": collision_info["coll_within_pairs"],
            "coll_cross_pairs": collision_info["coll_cross_pairs"],
            "total_pairs": collision_info["total_pairs"],
            "max_abs_delta_cross": collision_info["max_abs_delta_cross"],
            "collision_eps_strict": collision_info_strict["collision_eps"],
            "coll_cross_pairs_strict": collision_info_strict["coll_cross_pairs"],
            "coll_total_pairs_strict": collision_info_strict["coll_total_pairs"],
            "coll_cross_strict": collision_info_strict["coll_cross"],
            "coll_total_strict": collision_info_strict["coll_total"],
            "fp_collision_rate": fp_coll,
            "fp_energy_spearman_abs": float("nan"),
            "fp_policy_used": run_meta.get("fp_policy_used", "unknown"),
        }
    )
    if collision_log_dir:
        collision_log_dir.mkdir(parents=True, exist_ok=True)
        log_path = collision_log_dir / f"{formula}_cross_collisions.csv"
        fields = [
            "state_id_i",
            "class_i",
            "E_i",
            "energy_key_i",
            "state_id_j",
            "class_j",
            "E_j",
            "energy_key_j",
            "abs_delta",
        ]
        records = collision_info.get("cross_pair_records", [])
        with log_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            for rec in records[:1000]:
                writer.writerow(rec)
    pair_specs = PAIR_SPECS.get(formula, tuple())
    for spec in pair_specs:
        ga = df_states[df_states["class_label"] == spec.class_a]
        gb = df_states[df_states["class_label"] == spec.class_b]
        n_other = int(
            df_states[df_states["class_label"].isin(spec.other_labels)].shape[0]
        ) if spec.other_labels else 0
        if ga.empty or gb.empty:
            continue
        row = dict(base_row)
        row["class_a"] = spec.class_a
        row["class_b"] = spec.class_b
        row["n_a"] = int(len(ga))
        row["n_b"] = int(len(gb))
        row["n_other"] = int(n_other)
        total_groups = row["n_a"] + row["n_b"] + row["n_other"]
        row["other_frac"] = float(row["n_other"] / total_groups) if total_groups > 0 else 0.0
        row["pair_is_exhaustive"] = bool(n_other == 0)
        w_a = ga[weights_col].to_numpy(dtype=float)
        w_b = gb[weights_col].to_numpy(dtype=float)
        energy_metrics = _pair_metric_components(
            ga["energy"].to_numpy(dtype=float),
            gb["energy"].to_numpy(dtype=float),
            weights_a=w_a,
            weights_b=w_b,
        )
        row["E_mean_a"] = energy_metrics["mean_a"]
        row["E_mean_b"] = energy_metrics["mean_b"]
        row["E_delta_mean"] = energy_metrics["delta"]
        row["E_delta_abs"] = abs(float(energy_metrics["delta"]))
        row["E_effect_size"] = energy_metrics["effect_size"]
        row["E_is_trivial"] = bool(energy_metrics["is_trivial"])
        row["E_auc_raw"] = _clamp01(float(energy_metrics["auc_raw"]))
        row["E_auc_best"] = _clamp01(float(energy_metrics["auc_best"]))
        if fp_cols_all:
            best_idx = -1
            best_delta_abs = 0.0
            best_metrics: Dict[str, object] | None = None
            best_entry: Dict[str, object] | None = None
            fp_candidates: List[Dict[str, object]] = []
            energy_concat = np.concatenate(
                [ga["energy"].to_numpy(dtype=float), gb["energy"].to_numpy(dtype=float)]
            )
            for idx, col in enumerate(fp_cols_all):
                vals_a = ga[col].to_numpy(dtype=float)
                vals_b = gb[col].to_numpy(dtype=float)
                metrics = _pair_metric_components(
                    vals_a,
                    vals_b,
                    weights_a=w_a,
                    weights_b=w_b,
                )
                delta_abs = abs(float(metrics["delta"]))
                values_concat = np.concatenate([vals_a, vals_b])
                spearman_abs = _spearman_abs(values_concat, energy_concat)
                fp_candidates.append(
                    {
                        "idx": idx,
                        "col": col,
                        "metrics": metrics,
                        "vals_a": vals_a,
                        "vals_b": vals_b,
                        "values_concat": values_concat,
                        "spearman_abs": spearman_abs,
                    }
                )
                if delta_abs > best_delta_abs + EPS_SCORE:
                    best_delta_abs = delta_abs
                    best_idx = idx
                    best_metrics = metrics
                    best_entry = fp_candidates[-1]
            if best_entry is None and fp_candidates:
                best_entry = fp_candidates[0]
                best_metrics = best_entry["metrics"]
                best_idx = best_entry["idx"]
                best_delta_abs = abs(float(best_entry["metrics"]["delta"]))
            if best_metrics is None or best_entry is None:
                row["fp_dim"] = len(fp_cols_all)
                row["fp_best_idx"] = -1
                row["fp_best_delta_abs"] = 0.0
                row["fp_best_effect_size"] = float("nan")
                row["fp_best_auc_raw"] = 0.5
                row["fp_best_auc_best"] = 0.5
                row["fp_best_auc_gap"] = row["fp_best_auc_best"] - row["E_auc_best"]
                row["fp_best_is_trivial"] = True
                row["fp_energy_spearman_abs"] = float("nan")
                row["fp_best_idx_default"] = -1
                row["fp_best_auc_default"] = 0.5
                row["fp_best_auc_gap_default"] = row["fp_best_auc_default"] - row["E_auc_best"]
                row["fp_best_idx_excl_energy_like"] = -1
                row["fp_best_auc_excl_energy_like"] = 0.5
                row["fp_best_auc_gap_excl_energy_like"] = row["fp_best_auc_excl_energy_like"] - row["E_auc_best"]
            else:
                row["fp_dim"] = len(fp_cols_all)
                default_cand = best_entry
                filtered_candidates = [
                    cand
                    for cand in fp_candidates
                    if not (
                        np.isfinite(cand["spearman_abs"])
                        and cand["spearman_abs"] >= fp_energy_like_threshold
                    )
                ]
                filtered_cand = max(
                    filtered_candidates,
                    key=lambda c: float(c["metrics"]["auc_best"]),
                    default=None,
                )
                row["fp_best_idx_default"] = int(default_cand["idx"])
                row["fp_best_auc_default"] = _clamp01(float(default_cand["metrics"]["auc_best"]))
                row["fp_best_auc_gap_default"] = row["fp_best_auc_default"] - row["E_auc_best"]
                if filtered_cand is not None:
                    row["fp_best_idx_excl_energy_like"] = int(filtered_cand["idx"])
                    row["fp_best_auc_excl_energy_like"] = _clamp01(float(filtered_cand["metrics"]["auc_best"]))
                    row["fp_best_auc_gap_excl_energy_like"] = (
                        row["fp_best_auc_excl_energy_like"] - row["E_auc_best"]
                    )
                else:
                    row["fp_best_idx_excl_energy_like"] = -1
                    row["fp_best_auc_excl_energy_like"] = float("nan")
                    row["fp_best_auc_gap_excl_energy_like"] = float("nan")
                active = (
                    filtered_cand
                    if fp_exclude_energy_like and filtered_cand is not None
                    else (default_cand if not fp_exclude_energy_like else None)
                )
                if active is None and fp_exclude_energy_like and filtered_cand is None:
                    row["fp_best_idx"] = -1
                    row["fp_best_delta_abs"] = 0.0
                    row["fp_best_effect_size"] = float("nan")
                    row["fp_best_auc_raw"] = 0.5
                    row["fp_best_auc_best"] = 0.5
                    row["fp_best_auc_gap"] = row["fp_best_auc_best"] - row["E_auc_best"]
                    row["fp_best_is_trivial"] = True
                    row["fp_energy_spearman_abs"] = float("nan")
                    spearman_abs = float("nan")
                    best_col = ""
                else:
                    if active is None:
                        active = default_cand
                    row["fp_best_idx"] = int(active["idx"])
                    row["fp_best_delta_abs"] = abs(float(active["metrics"]["delta"]))
                    row["fp_best_effect_size"] = active["metrics"]["effect_size"]
                    row["fp_best_auc_raw"] = _clamp01(float(active["metrics"]["auc_raw"]))
                    row["fp_best_auc_best"] = _clamp01(float(active["metrics"]["auc_best"]))
                    row["fp_best_auc_gap"] = row["fp_best_auc_best"] - row["E_auc_best"]
                    row["fp_best_is_trivial"] = bool(active["metrics"]["is_trivial"])
                    spearman_abs = (
                        _spearman_abs(active["values_concat"], energy_concat)
                        if active.get("values_concat") is not None
                        else float("nan")
                    )
                    row["fp_energy_spearman_abs"] = spearman_abs
                    best_col = (
                        fp_cols_all[row["fp_best_idx"]]
                        if 0 <= row["fp_best_idx"] < len(fp_cols_all)
                        else f"fp{row['fp_best_idx']}"
                    )
                if debug_fp:
                    print(
                        f"[DEBUG-FP] formula={formula} pair={spec.class_a}/{spec.class_b} "
                        f"fp_best_idx={row['fp_best_idx']} col={best_col} auc_best={row['fp_best_auc_best']:.6f} "
                        f"E_auc_best={row['E_auc_best']:.6f} spearman_abs={spearman_abs:.6f}"
                    )
                    top_candidates = sorted(
                        fp_candidates,
                        key=lambda c: float(c["metrics"]["auc_best"]),
                        reverse=True,
                    )[:5]
                    for cand in top_candidates:
                        print(
                            f"  [DEBUG-FP] candidate idx={cand['idx']} col={cand['col']} "
                            f"auc_best={cand['metrics']['auc_best']:.6f}"
                        )
                    if best_col and "energy" in best_col.lower():
                        print("[DEBUG-FP][ERROR] fp_best is energy column (fingerprint selection degenerated)")
                    elif active is not None and np.allclose(active["values_concat"], energy_concat, rtol=1e-9, atol=1e-12):
                        print("[DEBUG-FP][WARNING] fp_best values allclose to energy (possible degeneracy)")
                    elif np.isfinite(spearman_abs) and spearman_abs >= fp_energy_like_threshold:
                        print(
                            "[DEBUG-FP][WARNING] fp_best ranking ~ energy ranking "
                            "(fp may add little information)"
                        )
                if active is not None:
                    best_col_name = (
                        fp_cols_all[row["fp_best_idx"]] if 0 <= row["fp_best_idx"] < len(fp_cols_all) else f"fp{row['fp_best_idx']}"
                    )
                    if "energy" in best_col_name.lower():
                        raise RuntimeError(
                            f"FP best column appears to alias energy ({best_col_name}); aborting for formula {formula}"
                        )
                    if np.allclose(active["values_concat"], energy_concat, rtol=1e-9, atol=1e-12):
                        print(
                            f"[FP][WARNING] fp_best values nearly identical to energy for formula={formula} pair={spec.class_a}/{spec.class_b}"
                        )
                    if np.isfinite(spearman_abs) and spearman_abs >= fp_energy_like_threshold:
                        print(
                            f"[FP][WARNING] fp_best spearman_abs={spearman_abs:.6f} exceeds threshold "
                            f"for formula={formula} pair={spec.class_a}/{spec.class_b}"
                        )
        else:
            row["fp_dim"] = 0
            row["fp_best_idx"] = -1
            row["fp_best_delta_abs"] = 0.0
            row["fp_best_effect_size"] = float("nan")
            row["fp_best_auc_raw"] = 0.5
            row["fp_best_auc_best"] = 0.5
            row["fp_best_auc_gap"] = row["fp_best_auc_best"] - row["E_auc_best"]
            row["fp_best_is_trivial"] = True
            row["fp_energy_spearman_abs"] = float("nan")
            row["fp_best_idx_default"] = -1
            row["fp_best_auc_default"] = 0.5
            row["fp_best_auc_gap_default"] = row["fp_best_auc_default"] - row["E_auc_best"]
            row["fp_best_idx_excl_energy_like"] = -1
            row["fp_best_auc_excl_energy_like"] = float("nan")
            row["fp_best_auc_gap_excl_energy_like"] = float("nan")
        score_rows.append(row)
    return score_rows
def _clamp01(val: float) -> float:
    if math.isnan(val):
        return float("nan")
    if val < 0.0:
        return 0.0
    if val > 1.0:
        return 1.0
    return float(val)
