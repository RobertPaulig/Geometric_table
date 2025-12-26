from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd

from analysis.chem.hetero_canonical import canonicalize_hetero_state
from analysis.chem.hetero_operator import build_operator_H, hetero_energy_from_state, hetero_fingerprint

EDGE_PREFIX = "het:edges="
TYPE_PREFIX = "types="

TYPE_LABELS = {0: "C", 1: "N", 2: "O"}
VALENCE_BY_TYPE = {0: 4, 1: 3, 2: 2}
RHO_BY_TYPE = {0: 0.0, 1: 0.2, 2: 0.5}
ALPHA_H = 0.5


def parse_state_id(state_id: str) -> Tuple[List[Tuple[int, int]], List[int]]:
    if not state_id.startswith("het:"):
        raise ValueError(f"Unexpected state_id format: {state_id}")
    parts = state_id.split(";")
    if len(parts) != 2:
        raise ValueError(f"Unexpected state_id format: {state_id}")
    edges_part = parts[0].split("=", 1)[1]
    types_part = parts[1].split("=", 1)[1]
    edges = []
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
    try:
        n = len(types)
        deg = _degrees(n, edges)
        valence = VALENCE_BY_TYPE
        if formula in ("C2H6O", "C3H8O"):
            idx = [i for i, t in enumerate(types) if t == 2]
            if len(idx) != 1:
                return "invalid_O_count"
            o_idx = idx[0]
            d = deg[o_idx]
            if d == 1:
                return "alcohol"
            if d == 2:
                return "ether"
            return "invalid_O_valence"
        if formula == "C2H7N":
            idx = [i for i, t in enumerate(types) if t == 1]
            if len(idx) != 1:
                return "invalid_N_count"
            n_idx = idx[0]
            d = deg[n_idx]
            if d == 1:
                return "primary_amine"
            if d == 2:
                return "secondary_amine"
            if d == 3:
                return "tertiary_amine"
            return "invalid_N_valence"
        return "unknown"
    except Exception:
        return "invalid_state"


def compute_state_table(
    *,
    formula: str,
    state_ids: Sequence[str],
    p_exact: Mapping[str, float],
    p_emp: Mapping[str, float],
) -> pd.DataFrame:
    records: List[Dict[str, object]] = []
    for sid in state_ids:
        edges, types = parse_state_id(sid)
        n = len(types)
        deg = _degrees(n, edges)
        valence = VALENCE_BY_TYPE
        implicit_h = [int(valence.get(types[i], 0) - deg[i]) for i in range(n)]
        class_label = classify_functional_group(formula, edges, types)
        energy_val = hetero_energy_from_state(
            n,
            edges,
            types,
            rho_by_type=RHO_BY_TYPE,
            alpha_H=ALPHA_H,
            valence_by_type=VALENCE_BY_TYPE,
        )
        H = build_operator_H(
            n,
            edges,
            types,
            rho_by_type=RHO_BY_TYPE,
            alpha_H=ALPHA_H,
            valence_by_type=VALENCE_BY_TYPE,
        )
        fp = hetero_fingerprint(H)
        rec: Dict[str, object] = {
            "formula": formula,
            "state_id": sid,
            "class_label": class_label,
            "N_heavy": n,
            "P_exact": float(p_exact.get(sid, 0.0)),
            "P_emp": float(p_emp.get(sid, 0.0)),
            "energy": float(energy_val),
            "is_valid_valence": int(all(h >= 0 for h in implicit_h)),
            "degree_seq": ",".join(str(d) for d in sorted(deg)),
        }
        # oxygen features
        o_idx = next((i for i, t in enumerate(types) if t == 2), None)
        if o_idx is not None:
            rec["o_deg"] = deg[o_idx]
            rec["o_implicit_h"] = implicit_h[o_idx]
        else:
            rec["o_deg"] = float("nan")
            rec["o_implicit_h"] = float("nan")
        # nitrogen features
        n_idx = next((i for i, t in enumerate(types) if t == 1), None)
        if n_idx is not None:
            rec["n_deg"] = deg[n_idx]
            rec["n_implicit_h"] = implicit_h[n_idx]
        else:
            rec["n_deg"] = float("nan")
            rec["n_implicit_h"] = float("nan")
        # fingerprint
        for idx, val in enumerate(fp):
            rec[f"fp{idx}"] = float(val)
        records.append(rec)
    df = pd.DataFrame(records)
    eps = 1e-9
    df["log_ratio_emp_exact"] = np.log((df["P_emp"] + eps) / (df["P_exact"] + eps))
    return df


def _class_stats(df: pd.DataFrame, column: str) -> Dict[str, Tuple[float, float]]:
    stats: Dict[str, Tuple[float, float]] = {}
    for cls, group in df.groupby("class_label"):
        vals = np.asarray(group[column], dtype=float)
        stats[str(cls)] = (float(np.mean(vals)), float(np.std(vals)))
    return stats


def _collision_rate(values: Sequence[float], tol: float = 1e-6) -> float:
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


def _fp_collision_rate(fps: Sequence[Sequence[float]], tol: float = 1e-6) -> float:
    arr = np.asarray(fps, dtype=float)
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


def compute_formula_score(
    df_states: pd.DataFrame,
    *,
    formula: str,
    weights_col: str = "P_exact",
) -> Dict[str, object]:
    df = df_states.copy()
    weights = np.asarray(df[weights_col].fillna(0.0), dtype=float)
    if np.all(weights == 0):
        weights = np.ones_like(weights)
    classes = sorted(df["class_label"].unique())
    score: Dict[str, object] = {
        "formula": formula,
        "support_exact": int(df["P_exact"].gt(0).sum()),
        "support_emp": int(df["P_emp"].gt(0).sum()),
        "coverage_unique_eq": float(len(df["P_emp"].gt(0)) / len(df)) if len(df) > 0 else 0.0,
        "energy_collision_rate": _collision_rate(df["energy"].fillna(0.0).tolist()),
    }
    fp_cols = [c for c in df.columns if c.startswith("fp")]
    if fp_cols:
        fps = df[fp_cols].fillna(0.0).to_numpy()
        score["fp_collision_rate"] = _fp_collision_rate(fps.tolist())
    else:
        score["fp_collision_rate"] = float("nan")

    # target classes
    class_pairs = []
    if formula in ("C2H6O", "C3H8O"):
        class_pairs.append(("alcohol", "ether"))
    if formula == "C2H7N":
        class_pairs.append(("primary_amine", "secondary_amine"))
    for idx, (a, b) in enumerate(class_pairs):
        ga = df[df["class_label"] == a]
        gb = df[df["class_label"] == b]
        if ga.empty or gb.empty:
            continue
        mean_a = float(np.average(ga["energy"], weights=ga[weights_col]))
        mean_b = float(np.average(gb["energy"], weights=gb[weights_col]))
        std_a = float(np.sqrt(np.average((ga["energy"] - mean_a) ** 2, weights=ga[weights_col])))
        std_b = float(np.sqrt(np.average((gb["energy"] - mean_b) ** 2, weights=gb[weights_col])))
        delta = mean_a - mean_b
        margin = abs(delta) / math.sqrt((std_a ** 2) + (std_b ** 2) + 1e-12)
        prefix = f"pair{idx}_{a}_vs_{b}"
        score[f"{prefix}_E_mean_a"] = mean_a
        score[f"{prefix}_E_mean_b"] = mean_b
        score[f"{prefix}_E_delta_mean"] = delta
        score[f"{prefix}_E_margin"] = margin
        score[f"{prefix}_n_a"] = int(len(ga))
        score[f"{prefix}_n_b"] = int(len(gb))
        if fp_cols:
            best_idx = -1
            best_margin = -float("inf")
            for fp_idx, col in enumerate(fp_cols):
                fpa = np.asarray(ga[col], dtype=float)
                fpb = np.asarray(gb[col], dtype=float)
                mean_fpa = float(np.mean(fpa))
                mean_fpb = float(np.mean(fpb))
                std_fpa = float(np.std(fpa))
                std_fpb = float(np.std(fpb))
                delta_fp = mean_fpa - mean_fpb
                margin_fp = abs(delta_fp) / math.sqrt((std_fpa ** 2) + (std_fpb ** 2) + 1e-12)
                score[f"{prefix}_fp{fp_idx}_delta_mean"] = delta_fp
                score[f"{prefix}_fp{fp_idx}_margin"] = margin_fp
                if margin_fp > best_margin:
                    best_margin = margin_fp
                    best_idx = fp_idx
            score[f"{prefix}_fp_best_idx"] = best_idx
            score[f"{prefix}_fp_best_margin"] = best_margin
    return score
