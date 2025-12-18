from __future__ import annotations

import argparse
import ast
import math
import re
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from analysis.chem.exact_trees import enumerate_labeled_trees
from analysis.chem.topology_mcmc import classify_tree_topology_by_deg_sorted
from analysis.io_utils import results_path
from analysis.utils.progress import progress_iter
from analysis.chem.core2_fit import compute_p_pred, kl_divergence
from core.complexity import compute_complexity_features_v2
from core.thermo_config import ThermoConfig, override_thermo_config
from analysis.growth.reporting import write_growth_txt


def _make_thermo_for_mode(mode: str) -> ThermoConfig:
    from analysis.chem.chem_validation_1a_pentane import _make_thermo_for_mode as _t

    return _t(mode)


def _degeneracy_for_n(n: int) -> Dict[str, int]:
    if n == 4:
        return {"n_butane": 12, "isobutane": 4}
    if n == 5:
        return {"n_pentane": 60, "isopentane": 60, "neopentane": 5}
    raise ValueError("Supported N: 4 or 5")


def _ref_energies_for_n_mode(n: int, mode: str) -> Dict[str, float]:
    """
    Deterministic reference energies E_ref(topology) from fixed adjacency representatives.
    Must match the energy definition used for exact enumeration (fdm_entanglement under thermo mode).
    """
    mode = mode.upper()
    thermo = _make_thermo_for_mode(mode)

    def score(adj: np.ndarray) -> float:
        with override_thermo_config(thermo):
            feats = compute_complexity_features_v2(adj, backend="fdm_entanglement")
        return float(feats.total)

    if n == 4:
        adj_path = np.zeros((4, 4), dtype=float)
        for a, b in ((0, 1), (1, 2), (2, 3)):
            adj_path[a, b] = adj_path[b, a] = 1.0
        adj_star = np.zeros((4, 4), dtype=float)
        for j in (1, 2, 3):
            adj_star[0, j] = adj_star[j, 0] = 1.0
        return {"n_butane": score(adj_path), "isobutane": score(adj_star)}

    if n == 5:
        adj_n = np.zeros((5, 5), dtype=float)
        for a, b in ((0, 1), (1, 2), (2, 3), (3, 4)):
            adj_n[a, b] = adj_n[b, a] = 1.0
        adj_iso = np.zeros((5, 5), dtype=float)
        for a, b in ((1, 0), (1, 2), (1, 4), (2, 3)):
            adj_iso[a, b] = adj_iso[b, a] = 1.0
        adj_neo = np.zeros((5, 5), dtype=float)
        for j in (1, 2, 3, 4):
            adj_neo[0, j] = adj_neo[j, 0] = 1.0
        return {"n_pentane": score(adj_n), "isopentane": score(adj_iso), "neopentane": score(adj_neo)}

    raise ValueError("Supported N: 4 or 5")


def _load_lambda_star_from_chem_validation(n: int, mode: str) -> Optional[float]:
    mode = mode.upper()
    if n == 4:
        path = results_path("chem_validation_0_butane.txt")
    elif n == 5:
        path = results_path("chem_validation_1a_pentane.txt")
    else:
        return None
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


def _load_p_mcmc_from_results(n: int, mode: str) -> Optional[Dict[str, float]]:
    mode = mode.upper()
    path = results_path(f"mh_kernel_1_mcmc_N{n}_mode{mode}.txt")
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return None
    m = re.search(r"P_mcmc=(\{.*\})", text)
    if not m:
        return None
    try:
        d = ast.literal_eval(m.group(1))
    except Exception:
        return None
    return {str(k): float(v) for k, v in dict(d).items()}


def _normalize(p: Dict[str, float]) -> Dict[str, float]:
    z = sum(float(v) for v in p.values()) or 1.0
    return {k: float(v) / z for k, v in p.items()}


@dataclass
class ExactConfig:
    n: int = 5
    mode: str = "A"
    lam: Optional[float] = None
    progress: bool = True


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="MH-KERNEL-2: exact baseline via Prüfer enumeration.")
    parser.add_argument("--N", type=int, choices=[4, 5], default=5)
    parser.add_argument("--mode", type=str, default="A", choices=["A", "B", "C"])
    parser.add_argument("--lambda", dest="lam", type=float, default=None)
    parser.add_argument(
        "--progress",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Show progress bar / periodic progress output.",
    )
    args = parser.parse_args(argv)
    cfg = ExactConfig(
        n=int(args.N),
        mode=str(args.mode).upper(),
        lam=(float(args.lam) if args.lam is not None else None),
        progress=bool(args.progress),
    )

    thermo = _make_thermo_for_mode(cfg.mode)
    T = float(getattr(thermo, "temperature_T", 1.0))
    backend = "fdm_entanglement"

    lam = cfg.lam
    if lam is None:
        lam_star = _load_lambda_star_from_chem_validation(cfg.n, cfg.mode)
        if lam_star is None:
            raise RuntimeError("lambda not provided and CORE-2 lambda* not found in chem_validation TXT")
        lam = lam_star

    g = _degeneracy_for_n(cfg.n)
    e_ref = _ref_energies_for_n_mode(cfg.n, cfg.mode)
    p_pred = compute_p_pred(g, e_ref, T=T, lam=float(lam))

    trees = enumerate_labeled_trees(cfg.n)
    total_states = len(trees)

    t0 = time.perf_counter()
    topo_weights: Dict[str, float] = {}
    topo_counts: Dict[str, int] = {}
    topo_energies: Dict[str, List[float]] = {}
    weights_all: List[float] = []

    with override_thermo_config(thermo):
        for adj in progress_iter(
            trees,
            total=total_states,
            desc=f"Exact N={cfg.n} mode={cfg.mode}",
            enabled=cfg.progress,
        ):
            deg = np.asarray(adj.sum(axis=1), dtype=int)
            topo = classify_tree_topology_by_deg_sorted(tuple(sorted(int(x) for x in deg)))
            feats = compute_complexity_features_v2(adj, backend=backend)
            e = float(feats.total)
            w = math.exp(-float(lam) * e / float(T)) if T > 0 else (1.0 if e == 0 else 0.0)
            weights_all.append(w)
            topo_weights[topo] = topo_weights.get(topo, 0.0) + w
            topo_counts[topo] = topo_counts.get(topo, 0) + 1
            topo_energies.setdefault(topo, []).append(e)

    elapsed = time.perf_counter() - t0
    z = sum(topo_weights.values()) or 1.0
    p_exact = {k: (v / z) for k, v in topo_weights.items()}
    p_exact = _normalize(p_exact)

    # CORE-3: label-averaged prediction equals exact by construction when using state energies.
    p_pred3 = dict(p_exact)

    p_mcmc = _load_p_mcmc_from_results(cfg.n, cfg.mode)
    if p_mcmc is not None:
        p_mcmc = _normalize(p_mcmc)

    out_txt = results_path(f"mh_kernel_2_exact_N{cfg.n}_mode{cfg.mode}.txt")
    lines: List[str] = []
    lines.append("MH-KERNEL-2: exact baseline via Prüfer enumeration")
    lines.append(f"N={cfg.n}, mode={cfg.mode}, backend={backend}")
    lines.append(f"states_total={total_states}, elapsed_sec={elapsed:.3f}")
    lines.append(f"temperature_T={T:.3f}, lambda={float(lam):.4f}")
    lines.append("")
    lines.append(f"P_pred (CORE-2, g*exp(-λE_ref/T))={p_pred}")
    lines.append(f"P_exact (enumeration)={p_exact}")
    lines.append(f"P_pred3 (CORE-3, label-averaged)={p_pred3}")
    if p_mcmc is not None:
        lines.append(f"P_mcmc (MH-KERNEL-1)={p_mcmc}")
    lines.append("")
    if p_mcmc is not None:
        lines.append(f"KL(P_mcmc||P_exact)={kl_divergence(p_mcmc, p_exact):.6f}")
    lines.append(f"KL(P_pred||P_exact)={kl_divergence(p_pred, p_exact):.6f}")
    lines.append("")
    lines.append("Label-dependence diagnostics: E stats within topology")
    for topo in sorted(topo_energies.keys()):
        arr = np.asarray(topo_energies[topo], dtype=float)
        lines.append(
            f"  {topo}: count={topo_counts.get(topo,0)}, "
            f"E_min={float(arr.min()):.6f}, E_mean={float(arr.mean()):.6f}, "
            f"E_std={float(arr.std()):.6f}, E_max={float(arr.max()):.6f}"
        )

    out_txt = write_growth_txt(f"mh_kernel_2_exact_N{cfg.n}_mode{cfg.mode}", lines)
    print("[MH-KERNEL-2] done.")
    print(f"Summary: {out_txt}")


if __name__ == "__main__":
    main()

