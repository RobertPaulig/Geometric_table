from __future__ import annotations

import argparse
import math
import re
import time
from collections import Counter
from typing import Dict, Optional, Tuple

import numpy as np

from analysis.chem.core2_fit import compute_p_pred, kl_divergence
from analysis.chem.exact_trees import enumerate_labeled_trees
from analysis.chem.chem_validation_1b_hexane import HEXANE_DEGENERACY, classify_hexane_topology, _make_thermo_for_mode
from analysis.io_utils import results_path
from analysis.utils.progress import progress_iter
from analysis.growth.reporting import write_growth_txt
from analysis.utils.timing import now_iso
from core.complexity import compute_complexity_features_v2
from core.thermo_config import override_thermo_config


def _load_lambda_star_from_hexane(mode: str) -> Optional[float]:
    mode = mode.upper()
    path = results_path("chem_validation_1b_hexane.txt")
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return None
    m_core = re.search(r"CORE-2: Fit lambda \(degeneracy-aware\)(.*)\Z", text, flags=re.S)
    if not m_core:
        return None
    tail = m_core.group(1)
    m_mode = re.search(rf"\[Mode {re.escape(mode)}\].*?lambda\*=([0-9.]+)", tail, flags=re.S)
    if not m_mode:
        return None
    try:
        return float(m_mode.group(1))
    except Exception:
        return None


def _load_growth_distribution(mode: str) -> Dict[str, float]:
    import csv

    path = results_path("chem_validation_1b_hexane.csv")
    counts: Counter[str] = Counter()
    total = 0
    with path.open("r", newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            if str(row.get("mode", "")).upper() != mode.upper():
                continue
            topo = str(row.get("topology", ""))
            counts[topo] += 1
            total += 1
    if total <= 0:
        return {}
    return {k: v / float(total) for k, v in sorted(counts.items(), key=lambda kv: kv[0])}


def _load_mcmc_distribution(mode: str) -> Dict[str, float]:
    path = results_path(f"mh_kernel_3_c6_mcmc_mode{mode.upper()}.txt")
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return {}
    probs: Dict[str, float] = {}
    for line in text.splitlines():
        m = re.match(r"\s*([A-Za-z0-9_,]+)\s*=\s*([0-9.]+)\s*$", line)
        if m:
            probs[m.group(1)] = float(m.group(2))
    return probs


def _ref_energies(mode: str) -> Dict[str, float]:
    mode = mode.upper()
    thermo = _make_thermo_for_mode(mode)

    def _adj_from_edges(edges: Tuple[Tuple[int, int], ...]) -> np.ndarray:
        adj = np.zeros((6, 6), dtype=float)
        for a, b in edges:
            adj[a, b] = 1.0
            adj[b, a] = 1.0
        return adj

    refs = {
        "n_hexane": _adj_from_edges(((0, 1), (1, 2), (2, 3), (3, 4), (4, 5))),
        # 2-methylpentane: branch at C2 (index 1), leaf at 5.
        "2_methylpentane": _adj_from_edges(((0, 1), (1, 2), (2, 3), (3, 4), (1, 5))),
        # 3-methylpentane: branch at C3 (index 2), leaf at 5.
        "3_methylpentane": _adj_from_edges(((0, 1), (1, 2), (2, 3), (3, 4), (2, 5))),
        # 2,2-dimethylbutane: degree-4 center at 0 with 1,2,3; and 1-4 chain.
        "2,2_dimethylbutane": _adj_from_edges(((0, 1), (0, 2), (0, 3), (0, 5), (1, 4))),
        # 2,3-dimethylbutane: edge between degree-3 nodes 1-2, each with two leaves.
        "2,3_dimethylbutane": _adj_from_edges(((1, 2), (1, 0), (1, 3), (2, 4), (2, 5))),
    }

    out: Dict[str, float] = {}
    for topo, adj in refs.items():
        with override_thermo_config(thermo):
            feats = compute_complexity_features_v2(adj, backend="fdm_entanglement")
        out[str(topo)] = float(feats.total)
    return out


def _energy_state(adj: np.ndarray, mode: str) -> float:
    thermo = _make_thermo_for_mode(mode.upper())
    with override_thermo_config(thermo):
        feats = compute_complexity_features_v2(adj, backend="fdm_entanglement")
    return float(feats.total)


def _normalize(weights: Dict[str, float]) -> Dict[str, float]:
    s = float(sum(weights.values()))
    if s <= 0:
        return {k: 0.0 for k in weights}
    return {k: float(v) / s for k, v in weights.items()}


def main() -> None:
    start_ts = now_iso()
    t0_total = time.perf_counter()
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="A", choices=["A", "B", "C"])
    ap.add_argument("--lambda", dest="lam", type=float, default=None)
    ap.add_argument("--progress", action=argparse.BooleanOptionalAction, default=True)
    args = ap.parse_args()

    mode = str(args.mode).upper()
    lam = float(args.lam) if args.lam is not None else None
    if lam is None:
        lam = _load_lambda_star_from_hexane(mode) or 1.0

    n = 6
    temperature_T = 1.0
    beta = float(lam) / float(temperature_T) if temperature_T > 0 else float("inf")

    trees = enumerate_labeled_trees(n)
    # Filter alkane: max degree <= 4 (removes the star K1,5).
    trees_alkane: list[np.ndarray] = []
    for adj in trees:
        deg = np.sum(adj > 0, axis=1).astype(int)
        if int(np.max(deg)) <= 4:
            trees_alkane.append(adj)

    t0 = time.perf_counter()
    weights_by_topo: Dict[str, float] = {}
    counts_by_topo: Dict[str, int] = {}
    for adj in progress_iter(
        trees_alkane, total=len(trees_alkane), desc=f"[MH-KERNEL-3:C6:EXACT:{mode}]", enabled=bool(args.progress)
    ):
        topo = str(classify_hexane_topology(adj))
        e = _energy_state(adj, mode=mode)
        w = math.exp(-beta * float(e))
        weights_by_topo[topo] = weights_by_topo.get(topo, 0.0) + float(w)
        counts_by_topo[topo] = counts_by_topo.get(topo, 0) + 1
    dt = time.perf_counter() - t0
    elapsed_total = time.perf_counter() - t0_total
    end_ts = now_iso()

    p_exact = _normalize(weights_by_topo)
    p_mcmc = _load_mcmc_distribution(mode)
    p_obs = _load_growth_distribution(mode)

    e_ref = _ref_energies(mode)
    p_pred = compute_p_pred(g=HEXANE_DEGENERACY, e_ref=e_ref, T=temperature_T, lam=float(lam))

    topologies = [k for k in HEXANE_DEGENERACY.keys()]
    p_exact_vec = {k: float(p_exact.get(k, 0.0)) for k in topologies}
    p_mcmc_vec = {k: float(p_mcmc.get(k, 0.0)) for k in topologies}
    p_obs_vec = {k: float(p_obs.get(k, 0.0)) for k in topologies}
    p_pred_vec = {k: float(p_pred.get(k, 0.0)) for k in topologies}

    kl_mcmc_exact = kl_divergence(p_mcmc_vec, p_exact_vec) if p_mcmc else float("nan")
    kl_obs_exact = kl_divergence(p_obs_vec, p_exact_vec) if p_obs else float("nan")
    kl_obs_pred = kl_divergence(p_obs_vec, p_pred_vec) if p_obs else float("nan")
    kl_pred_exact = kl_divergence(p_pred_vec, p_exact_vec)

    out_name = f"mh_kernel_3_c6_exact_mode{mode}"

    lines: list[str] = []
    lines.append("MH-KERNEL-3: C6 exact baseline (Pruefer enumeration) + comparisons")
    lines.append(f"mode={mode}, lambda={lam:.6g}, T={temperature_T:.3f}")
    lines.append(f"N=6 labeled trees total={len(trees)}, alkane(deg<=4)={len(trees_alkane)} (expected 1290)")
    lines.append(f"elapsed_sec={dt:.3f}")
    lines.append("")
    lines.append("TIMING")
    lines.append(f"START_TS={start_ts}")
    lines.append(f"END_TS={end_ts}")
    lines.append(f"ELAPSED_TOTAL_SEC={elapsed_total:.6f}")
    lines.append("")
    lines.append("Counts per topology (alkane labeled states):")
    for k in topologies:
        lines.append(f"  {k}: count={counts_by_topo.get(k, 0)}")
    lines.append("")
    lines.append("P_exact(topology):")
    for k in topologies:
        lines.append(f"  {k} = {p_exact_vec[k]:.6f}")
    lines.append("")
    lines.append("P_pred(topology) = g*exp(-lambda*E_ref/T) normalized:")
    for k in topologies:
        lines.append(f"  {k} = {p_pred_vec[k]:.6f}")
    lines.append("")
    lines.append("P_obs_growth(topology) from chem_validation_1b_hexane.csv:")
    for k in topologies:
        lines.append(f"  {k} = {p_obs_vec[k]:.6f}")
    lines.append("")
    lines.append("P_mcmc_fixedN(topology) from mh_kernel_3_c6_mcmc_mode*.txt:")
    for k in topologies:
        lines.append(f"  {k} = {p_mcmc_vec[k]:.6f}")
    lines.append("")
    lines.append("KL matrix:")
    lines.append(f"  KL(P_mcmc || P_exact) = {kl_mcmc_exact:.6g}")
    lines.append(f"  KL(P_obs  || P_exact) = {kl_obs_exact:.6g}")
    lines.append(f"  KL(P_obs  || P_pred)  = {kl_obs_pred:.6g}")
    lines.append(f"  KL(P_pred || P_exact) = {kl_pred_exact:.6g}")

    write_growth_txt(out_name, lines)
    print(f"[MH-KERNEL-3:C6:EXACT] wrote {results_path(out_name + '.txt')}")
    print(f"Wall-clock: start={start_ts} end={end_ts} elapsed_total_sec={elapsed_total:.3f}")


if __name__ == "__main__":
    main()
