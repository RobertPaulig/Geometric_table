from __future__ import annotations

import argparse
import csv
import math
import re
import time
from collections import Counter
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from analysis.chem.core2_fit import kl_divergence
from analysis.chem.exact_trees import enumerate_labeled_trees
from analysis.io_utils import results_path
from analysis.utils.progress import progress_iter
from analysis.utils.timing import now_iso
from analysis.growth.reporting import write_growth_txt
from core.complexity import compute_complexity_features_v2
from core.tree_canonical import _adj_list_from_adj, _tree_centers, _rooted_ahu_code
from core.thermo_config import override_thermo_config
from analysis.chem.chem_validation_1c_heptane import _thermo_for_mode as thermo_c7
from analysis.chem.chem_validation_1d_octane import _thermo_for_mode as thermo_c8


Edge = Tuple[int, int]


def _max_degree(adj: np.ndarray) -> int:
    deg = np.sum(adj > 0, axis=1).astype(int)
    return int(np.max(deg)) if deg.size else 0


def _rooted_ahu_code_local(root: int, parent: int, adj_list: Sequence[Sequence[int]]) -> Tuple:
    labels: Dict[int, Tuple] = {}
    return _rooted_ahu_code(int(root), int(parent), adj_list, labels)


def canonical_unrooted_code(adj: np.ndarray) -> Tuple:
    adj_list = _adj_list_from_adj(np.asarray(adj, dtype=float))
    centers = _tree_centers([list(nei) for nei in adj_list])
    codes = []
    for c in centers:
        codes.append(_rooted_ahu_code_local(int(c), -1, adj_list))
    return min(codes) if codes else tuple()


def rooted_aut_count(root: int, parent: int, adj_list: Sequence[Sequence[int]]) -> Tuple[Tuple, int]:
    """
    Returns (rooted_code, aut_count) for rooted tree.
    aut_count counts automorphisms that fix the root.
    """
    child_infos: List[Tuple[Tuple, int]] = []
    for v in adj_list[int(root)]:
        if int(v) == int(parent):
            continue
        code_v, aut_v = rooted_aut_count(int(v), int(root), adj_list)
        child_infos.append((code_v, int(aut_v)))
    child_infos.sort(key=lambda x: x[0])
    code = tuple(code for code, _ in child_infos)

    aut = 1
    i = 0
    while i < len(child_infos):
        j = i + 1
        while j < len(child_infos) and child_infos[j][0] == child_infos[i][0]:
            j += 1
        mult = j - i
        for _, a in child_infos[i:j]:
            aut *= int(a)
        aut *= math.factorial(int(mult))
        i = j
    return code, int(aut)


def tree_automorphism_size(adj: np.ndarray) -> int:
    """
    |Aut(T)| for an unrooted tree T using AHU centers.
    """
    adj_list = _adj_list_from_adj(np.asarray(adj, dtype=float))
    centers = _tree_centers([list(nei) for nei in adj_list])
    if not centers:
        return 1
    if len(centers) == 1:
        _, aut = rooted_aut_count(int(centers[0]), -1, adj_list)
        return int(aut)
    if len(centers) == 2:
        a, b = int(centers[0]), int(centers[1])
        code_a, aut_a = rooted_aut_count(a, b, adj_list)
        code_b, aut_b = rooted_aut_count(b, a, adj_list)
        swap = 2 if code_a == code_b else 1
        return int(aut_a) * int(aut_b) * int(swap)
    raise ValueError("Invalid number of centers for a tree")


def topology_key(adj: np.ndarray) -> str:
    can = canonical_unrooted_code(adj)
    # tuple->string key (stable under Python run for nested tuples of ints)
    return f"ahu:{repr(can)}"


def _load_growth_counts(path: str, mode: str) -> Counter[str]:
    c: Counter[str] = Counter()
    lines = results_path(path).read_text(encoding="utf-8", errors="replace").splitlines()
    in_mode = False
    in_block = False
    for line in lines:
        s = line.strip()
        if s == f"[Mode {mode}]":
            in_mode = True
            in_block = False
            continue
        if in_mode and s.startswith("[Mode ") and s.endswith("]") and s != f"[Mode {mode}]":
            break
        if in_mode and s == "P_growth(topology):":
            in_block = True
            continue
        if in_mode and in_block and s.startswith("mass_topK_growth="):
            break
        if in_mode and in_block:
            if not s.startswith("tree:") or "count=" not in s:
                continue
            topo = s.split("=", 1)[0].strip()
            try:
                cnt = int(s.split("count=")[1].split(")")[0])
            except Exception:
                continue
            c[topo] += cnt
    return c


def _load_eq_probs(path: str, mode: str) -> Dict[str, float]:
    out: Dict[str, float] = {}
    lines = results_path(path).read_text(encoding="utf-8", errors="replace").splitlines()
    in_mode = False
    in_block = False
    for line in lines:
        s = line.strip()
        if s == f"[Mode {mode}]":
            in_mode = True
            in_block = False
            continue
        if in_mode and s.startswith("[Mode ") and s.endswith("]") and s != f"[Mode {mode}]":
            break
        if in_mode and s == "P_eq(topology):":
            in_block = True
            continue
        if in_mode and in_block and s.startswith("mass_topK_eq="):
            break
        if in_mode and in_block:
            if not s.startswith("tree:") or "=" not in s:
                continue
            k, v = [x.strip() for x in s.split("=", 1)]
            try:
                out[k] = float(v)
            except Exception:
                continue
    return out


def _energy(adj: np.ndarray, thermo, backend: str) -> float:
    with override_thermo_config(thermo):
        feats = compute_complexity_features_v2(adj, backend=backend)
    return float(feats.total)


@dataclass
class Config:
    N: int = 7
    lam: float = 1.0
    T: float = 1.0
    modes: Tuple[str, ...] = ("A", "B", "C")
    progress: bool = True


def _parse_args(argv: Optional[Sequence[str]] = None) -> Config:
    ap = argparse.ArgumentParser(description="ALKANE-EXACT-1: exact equilibrium over tree-only alkane topologies.")
    ap.add_argument("--N", type=int, required=True, choices=[7, 8])
    ap.add_argument("--lambda", dest="lam", type=float, default=1.0)
    ap.add_argument("--T", type=float, default=1.0)
    ap.add_argument("--modes", type=str, nargs="*", default=["A", "B", "C"])
    ap.add_argument("--progress", action=argparse.BooleanOptionalAction, default=True)
    args = ap.parse_args(argv)
    return Config(
        N=int(args.N),
        lam=float(args.lam),
        T=float(args.T),
        modes=tuple(str(x).upper() for x in args.modes),
        progress=bool(args.progress),
    )


def main(argv: Optional[Sequence[str]] = None) -> None:
    cfg = _parse_args(argv)
    start_ts = now_iso()
    t0_total = time.perf_counter()

    N = int(cfg.N)
    beta = float(cfg.lam) / float(cfg.T) if cfg.T > 0 else float("inf")

    if N == 7:
        growth_txt = "chem_validation_1c_heptane.txt"
        thermo_fn = thermo_c7
    else:
        growth_txt = "chem_validation_1d_octane.txt"
        thermo_fn = thermo_c8

    # Enumerate all labeled trees and collapse to unique unlabeled topologies (deg<=4).
    trees = enumerate_labeled_trees(N)
    uniq: Dict[Tuple, np.ndarray] = {}
    counts_labeled: Counter[Tuple] = Counter()
    for adj in progress_iter(trees, total=len(trees), desc=f"[ALKANE-EXACT-1:N{N}:ENUM]", enabled=bool(cfg.progress)):
        if _max_degree(adj) > 4:
            continue
        code = canonical_unrooted_code(adj)
        counts_labeled[code] += 1
        if code not in uniq:
            uniq[code] = np.asarray(adj, dtype=float)

    # Compute |Aut| and g from counts: count = n!/|Aut| for trees with distinct labels.
    n_fact = math.factorial(N)
    topo_rows: List[Dict[str, object]] = []
    for code, adj in uniq.items():
        aut = tree_automorphism_size(adj)
        g = n_fact / float(aut)
        topo_rows.append(
            {
                "N": N,
                "code": repr(code),
                "aut_size": int(aut),
                "g": float(g),
                "labeled_count": int(counts_labeled[code]),
            }
        )

    # Sanity: g should match labeled_count.
    for r in topo_rows:
        if int(r["labeled_count"]) != int(round(float(r["g"]))):
            raise RuntimeError("Degeneracy sanity failed: labeled_count != g (check automorphism calc)")

    # Build canonical edge-list key used by chem_validation_1c/1d for joining.
    topo_key_by_code: Dict[Tuple, str] = {}
    for code, adj in uniq.items():
        from core.tree_canonical import canonical_relabel_tree

        can_adj = canonical_relabel_tree(np.asarray(adj, dtype=float))
        edges: List[Tuple[int, int]] = []
        for i in range(N):
            for j in range(i + 1, N):
                if can_adj[i, j] > 0:
                    edges.append((i, j))
        topo_key_by_code[code] = "tree:" + ",".join(f"{a}-{b}" for a, b in edges)

    # Load observed growth and MCMC eq distributions from existing artifacts.
    growth_counts_by_mode: Dict[str, Counter[str]] = {}
    eq_probs_by_mode: Dict[str, Dict[str, float]] = {}
    for mode in cfg.modes:
        growth_counts_by_mode[mode] = _load_growth_counts(growth_txt, mode)
        eq_probs_by_mode[mode] = _load_eq_probs(growth_txt, mode)

    # Exact equilibrium by topology (unlabeled) computed via g*exp(-beta*E_topo).
    exact_by_mode: Dict[str, Dict[str, float]] = {}
    energy_by_mode: Dict[str, Dict[str, float]] = {}
    for mode in cfg.modes:
        thermo = thermo_fn(mode)
        backend = "fdm" if mode == "A" else "fdm_entanglement"
        weights: Dict[str, float] = {}
        energies: Dict[str, float] = {}
        for code, adj in uniq.items():
            topo_key = topo_key_by_code[code]
            e = _energy(adj, thermo=thermo, backend=backend)
            g = float(n_fact) / float(tree_automorphism_size(adj))
            w = float(g) * math.exp(-beta * float(e))
            weights[topo_key] = float(w)
            energies[topo_key] = float(e)
        s = float(sum(weights.values()))
        exact_by_mode[mode] = {k: (float(v) / s if s > 0 else 0.0) for k, v in weights.items()}
        energy_by_mode[mode] = energies

    # Write CSV of topologies + energies for each mode.
    out_csv = results_path(f"alkane_exact_1_N{N}.csv")
    out_txt = results_path(f"alkane_exact_1_N{N}.txt")

    fieldnames = ["N", "topology", "aut_size", "g", "mode", "energy", "p_exact"]
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for code, adj in sorted(uniq.items(), key=lambda kv: repr(kv[0])):
            topo_key = topo_key_by_code[code]
            aut = tree_automorphism_size(adj)
            g = float(n_fact) / float(aut)
            for mode in cfg.modes:
                w.writerow(
                    {
                        "N": N,
                        "topology": topo_key,
                        "aut_size": int(aut),
                        "g": float(g),
                        "mode": mode,
                        "energy": float(energy_by_mode[mode][topo_key]),
                        "p_exact": float(exact_by_mode[mode][topo_key]),
                    }
                )

    # Summary TXT
    lines: List[str] = []
    lines.append(f"ALKANE-EXACT-1: exact equilibrium over tree-only alkane topologies (N={N})")
    lines.append(f"Enumerated labeled trees total={len(trees)}, alkane(deg<=4)={sum(counts_labeled.values())}")
    lines.append(f"Unique unlabeled alkane topologies={len(uniq)}")
    lines.append(f"lambda={cfg.lam}, T={cfg.T}")
    lines.append("")

    # List degeneracies
    lines.append("Topologies: aut_size, g (=n!/|Aut|)")
    for code, adj in sorted(uniq.items(), key=lambda kv: repr(kv[0])):
        topo_key = topo_key_by_code[code]
        aut = tree_automorphism_size(adj)
        g = float(n_fact) / float(aut)
        lines.append(f"  {topo_key}: |Aut|={aut}, g={int(g)}")
    lines.append("")

    for mode in cfg.modes:
        lines.append(f"[Mode {mode}]")
        p_exact = exact_by_mode[mode]
        # compare to eq and growth if available
        p_eq = eq_probs_by_mode.get(mode, {})
        growth_counts = growth_counts_by_mode.get(mode, Counter())
        total_growth = float(sum(growth_counts.values())) if growth_counts else 0.0
        p_growth = {k: (v / total_growth if total_growth > 0 else 0.0) for k, v in growth_counts.items()}

        top_sorted = sorted(p_exact.items(), key=lambda kv: (-kv[1], kv[0]))
        lines.append("  P_exact(topology):")
        for topo, p in top_sorted:
            lines.append(f"    {topo} = {p:.6f}")
        if p_eq:
            keys = sorted(set(p_exact.keys()) | set(p_eq.keys()))
            p_exact_vec = {k: float(p_exact.get(k, 0.0)) for k in keys}
            p_eq_vec = {k: float(p_eq.get(k, 0.0)) for k in keys}
            lines.append(f"  KL(P_eq||P_exact) = {float(kl_divergence(p_eq_vec, p_exact_vec)):.6f}")
        if p_growth:
            keys = sorted(set(p_exact.keys()) | set(p_growth.keys()))
            p_exact_vec = {k: float(p_exact.get(k, 0.0)) for k in keys}
            p_growth_vec = {k: float(p_growth.get(k, 0.0)) for k in keys}
            lines.append(f"  KL(P_growth||P_exact) = {float(kl_divergence(p_growth_vec, p_exact_vec)):.6f}")
        lines.append("")

    elapsed_total = time.perf_counter() - t0_total
    end_ts = now_iso()
    lines.append("TIMING")
    lines.append(f"START_TS={start_ts}")
    lines.append(f"END_TS={end_ts}")
    lines.append(f"ELAPSED_TOTAL_SEC={elapsed_total:.6f}")

    write_growth_txt(f"alkane_exact_1_N{N}", lines)
    print(f"[ALKANE-EXACT-1] wrote {out_csv}")
    print(f"[ALKANE-EXACT-1] wrote {out_txt}")
    print(f"Wall-clock: start={start_ts} end={end_ts} elapsed_total_sec={elapsed_total:.3f}")


if __name__ == "__main__":
    main()
