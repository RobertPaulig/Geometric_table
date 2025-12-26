from __future__ import annotations

import datetime as dt
import math
from dataclasses import dataclass
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
    "C2H7N": (PairSpec("primary_amine", "secondary_amine", ("tertiary_amine",)),),
    "C4H11N": (PairSpec("primary_amine", "secondary_amine", ("tertiary_amine",)),),
    "C5H13N": (PairSpec("primary_amine", "secondary_amine", ("tertiary_amine",)),),
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


def _collision_rate(values: Sequence[float], tol: float = ENERGY_COLLISION_EPS) -> float:
    arr = np.asarray(values, dtype=float)
    n = arr.size
    if n <= 1:
        return 0.0
    collisions = 0
    total = 0
    for i in range(n):
        for j in range(i + 1, n):
            total += 1
            if abs(arr[i] - arr[j]) <= tol:
                collisions += 1
    return float(collisions) / float(total) if total > 0 else 0.0


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


def compute_formula_scores(
    df_states: pd.DataFrame,
    *,
    formula: str,
    weights_col: str,
    run_meta: Mapping[str, object],
) -> List[Dict[str, object]]:
    weights = df_states[weights_col].fillna(0.0).to_numpy(dtype=float)
    if np.all(weights == 0):
        weights = np.ones_like(weights)
    score_rows: List[Dict[str, object]] = []
    run_id = run_meta.get("run_id", f"run_{formula}_{dt.datetime.utcnow().isoformat()}")
    coverage_meta = float(run_meta.get("coverage_unique_eq", 0.0))
    if coverage_meta == 0.0 and len(df_states) > 0:
        coverage_meta = float((df_states["P_emp"] > 0).sum() / len(df_states))
    energy_coll = _collision_rate(df_states["energy"].tolist())
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
            "fp_collision_rate": fp_coll,
        }
    )
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
        row["E_auc_raw"] = float(energy_metrics["auc_raw"])
        row["E_auc_best"] = float(energy_metrics["auc_best"])
        if fp_cols_all:
            best_idx = -1
            best_delta_abs = 0.0
            best_metrics: Dict[str, object] | None = None
            for idx, col in enumerate(fp_cols_all):
                metrics = _pair_metric_components(
                    ga[col].to_numpy(dtype=float),
                    gb[col].to_numpy(dtype=float),
                    weights_a=w_a,
                    weights_b=w_b,
                )
                delta_abs = abs(float(metrics["delta"]))
                if delta_abs > best_delta_abs + EPS_SCORE:
                    best_delta_abs = delta_abs
                    best_idx = idx
                    best_metrics = metrics
            if best_metrics is None:
                row["fp_dim"] = len(fp_cols_all)
                row["fp_best_idx"] = -1
                row["fp_best_delta_abs"] = 0.0
                row["fp_best_effect_size"] = float("nan")
                row["fp_best_auc_raw"] = 0.5
                row["fp_best_auc_best"] = 0.5
                row["fp_best_is_trivial"] = True
            else:
                row["fp_dim"] = len(fp_cols_all)
                row["fp_best_idx"] = best_idx
                row["fp_best_delta_abs"] = best_delta_abs
                row["fp_best_effect_size"] = best_metrics["effect_size"]
                row["fp_best_auc_raw"] = float(best_metrics["auc_raw"])
                row["fp_best_auc_best"] = float(best_metrics["auc_best"])
                row["fp_best_is_trivial"] = bool(best_metrics["is_trivial"])
        else:
            row["fp_dim"] = 0
            row["fp_best_idx"] = -1
            row["fp_best_delta_abs"] = 0.0
            row["fp_best_effect_size"] = float("nan")
            row["fp_best_auc_raw"] = 0.5
            row["fp_best_auc_best"] = 0.5
            row["fp_best_is_trivial"] = True
        score_rows.append(row)
    return score_rows
