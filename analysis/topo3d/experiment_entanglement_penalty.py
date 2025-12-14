from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import argparse
from typing import Iterable, List, Tuple, Dict, Any

import numpy as np
import pandas as pd

from core.crossing import estimate_crossing_number_circle
from core.complexity import compute_complexity_features_v2
from core.layout_3d import force_directed_layout_3d
from core.entanglement_3d import entanglement_score
from core.thermo_config import ThermoConfig, override_thermo_config
from core.grower import GrowthParams, grow_molecule_christmas_tree, grow_molecule_loopy


RESULTS_DIR = Path("results")


def _adjacency(n: int, edges: List[Tuple[int, int]]) -> np.ndarray:
    A = np.zeros((n, n), dtype=float)
    for i, j in edges:
        if i == j:
            continue
        a, b = (i, j) if i < j else (j, i)
        if 0 <= a < n and 0 <= b < n:
            A[a, b] = 1.0
            A[b, a] = 1.0
    return A


def _normalize_edges(edges: Iterable[Tuple[int, int]]) -> List[Tuple[int, int]]:
    s = set()
    out: List[Tuple[int, int]] = []
    for i, j in edges:
        if i == j:
            continue
        a, b = (i, j) if i < j else (j, i)
        if (a, b) not in s:
            s.add((a, b))
            out.append((a, b))
    return out


def _crossing_density(
    adj: np.ndarray,
    m: int,
    *,
    max_exact_n: int,
    rng: np.random.Generator,
) -> float:
    if m <= 0:
        return 0.0
    crossing, _ = estimate_crossing_number_circle(
        adj,
        max_exact_n=max_exact_n,
        rng=rng,
    )
    return float(crossing) / float(m)


def _manual_graphs() -> List[Dict[str, Any]]:
    graphs: List[Dict[str, Any]] = []

    def add(name: str, atoms: List[str], edges: List[Tuple[int, int]], source: str = "manual") -> None:
        graphs.append(
            {
                "name": name,
                "atoms": atoms,
                "edges": _normalize_edges(edges),
                "source": source,
            }
        )

    # Учебные молекулы
    add("H2", ["H", "H"], [(0, 1)])
    add("H2O", ["O", "H", "H"], [(0, 1), (0, 2)])
    add("CH4", ["C", "H", "H", "H", "H"], [(0, 1), (0, 2), (0, 3), (0, 4)])
    add("NH3", ["N", "H", "H", "H"], [(0, 1), (0, 2), (0, 3)])
    add("CO2", ["O", "C", "O"], [(0, 1), (1, 2)])
    add("C2H4", ["C", "C", "H", "H", "H", "H"], [(0, 1), (0, 2), (0, 3), (1, 4), (1, 5)])

    ring = [(i, (i + 1) % 6) for i in range(6)]
    hyd = [(i, 6 + i) for i in range(6)]
    add("C6H6", ["C"] * 6 + ["H"] * 6, ring + hyd)

    add("NaCl", ["Na", "Cl"], [(0, 1)])
    add("SiO2", ["O", "Si", "O"], [(0, 1), (1, 2)])

    # Эталоны
    add("chain_C3", ["C", "C", "C"], [(0, 1), (1, 2)], source="manual_ref")
    k4_edges: List[Tuple[int, int]] = []
    for i in range(4):
        for j in range(i + 1, 4):
            k4_edges.append((i, j))
    add("K4_C", ["C"] * 4, k4_edges, source="manual_ref")

    return graphs


def _grower_graphs(
    *,
    n_tree: int,
    n_loopy: int,
    seed: int,
    max_depth: int,
    max_atoms: int,
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    seeds = ["C", "O", "Si", "N", "Cl"]

    params_tree = GrowthParams(max_depth=max_depth, max_atoms=max_atoms)
    for i in range(n_tree):
        rng = np.random.default_rng(seed + 1000 + i)
        root = seeds[i % len(seeds)]
        mol = grow_molecule_christmas_tree(root, params_tree, rng=rng)
        atoms = [str(getattr(a, "name", a)) for a in mol.atoms]
        edges = _normalize_edges(mol.bonds)
        out.append(
            {
                "name": f"grow_tree_{root}_{i}",
                "atoms": atoms,
                "edges": edges,
                "source": "grower_tree",
            }
        )

    for i in range(n_loopy):
        rng = np.random.default_rng(seed + 2000 + i)
        root = seeds[i % len(seeds)]
        mol = grow_molecule_loopy(root, rng=rng)
        atoms = [str(getattr(a, "name", a)) for a in mol.atoms]
        edges = _normalize_edges(mol.bonds)
        out.append(
            {
                "name": f"grow_loopy_{root}_{i}",
                "atoms": atoms,
                "edges": edges,
                "source": "grower_loopy",
            }
        )

    return out


def main(argv=None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--coupling-topo-3d", type=float, default=1.0)
    parser.add_argument("--topo-3d-beta", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--layout-seed", type=int, default=42)
    parser.add_argument("--max-exact-n", type=int, default=8)
    parser.add_argument("--n-grower-tree", type=int, default=10)
    parser.add_argument("--n-grower-loopy", type=int, default=10)
    parser.add_argument("--grower-max-depth", type=int, default=4)
    parser.add_argument("--grower-max-atoms", type=int, default=25)
    args = parser.parse_args(argv)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    graphs = _manual_graphs()
    graphs.extend(
        _grower_graphs(
            n_tree=args.n_grower_tree,
            n_loopy=args.n_grower_loopy,
            seed=args.seed,
            max_depth=args.grower_max_depth,
            max_atoms=args.grower_max_atoms,
        )
    )

    base_cfg = ThermoConfig()
    cfg = replace(
        base_cfg,
        coupling_topo_3d=float(args.coupling_topo_3d),
        topo_3d_beta=float(args.topo_3d_beta),
    )

    rows: List[Dict[str, Any]] = []

    with override_thermo_config(cfg):
        for g in graphs:
            name = g["name"]
            atoms = g["atoms"]
            edges = g["edges"]
            n = len(atoms)
            m = len(edges)
            if n <= 1 or m <= 0:
                continue

            adj = _adjacency(n, edges)

            rng_cross = np.random.default_rng(args.seed + 999)
            crossing_density = _crossing_density(
                adj,
                m,
                max_exact_n=args.max_exact_n,
                rng=rng_cross,
            )

            pos = force_directed_layout_3d(
                n,
                edges,
                seed=int(args.layout_seed),
            )
            e3d = float(entanglement_score(pos, edges))

            feats_fdm = compute_complexity_features_v2(adj, backend="fdm")
            feats_ent = compute_complexity_features_v2(
                adj,
                backend="fdm_entanglement",
            )

            total_fdm = float(feats_fdm.total)
            total_ent = float(feats_ent.total)
            penalty = total_ent / total_fdm if total_fdm > 0.0 else float("nan")

            target_penalty = 1.0 + float(args.coupling_topo_3d) * float(args.topo_3d_beta) * e3d
            residual = penalty - target_penalty

            rows.append(
                dict(
                    name=name,
                    source=g.get("source", ""),
                    n=n,
                    m=m,
                    cyclomatic=int(feats_fdm.cyclomatic),
                    crossing_density=float(crossing_density),
                    E_3d=e3d,
                    total_fdm=total_fdm,
                    total_entangled=total_ent,
                    penalty_factor_3d=float(penalty),
                    formula_target=float(target_penalty),
                    formula_residual=float(residual),
                    coupling_topo_3d=float(args.coupling_topo_3d),
                    topo_3d_beta=float(args.topo_3d_beta),
                )
            )

    df = pd.DataFrame(rows)
    csv_path = RESULTS_DIR / "topo3d_entanglement_penalty.csv"
    df.to_csv(csv_path, index=False)

    def _corr(a: pd.Series, b: pd.Series) -> float:
        sub = pd.concat([a, b], axis=1).dropna()
        if len(sub) < 3:
            return float("nan")
        return float(sub.corr().iloc[0, 1])

    corr_e3d_cross = _corr(df["E_3d"], df["crossing_density"])
    corr_pen_cross = _corr(df["penalty_factor_3d"], df["crossing_density"])
    corr_pen_e3d = _corr(df["penalty_factor_3d"], df["E_3d"])

    max_abs_res = float(np.nanmax(np.abs(df["formula_residual"].to_numpy()))) if len(df) else float("nan")
    med_abs_res = float(np.nanmedian(np.abs(df["formula_residual"].to_numpy()))) if len(df) else float("nan")

    top = df.sort_values("penalty_factor_3d", ascending=False).head(15)

    txt_path = RESULTS_DIR / "topo3d_entanglement_penalty_summary.txt"
    with txt_path.open("w", encoding="utf-8") as f:
        f.write("TOPO-3D-2 experiment: 2D crossing proxy vs 3D entanglement penalty\n")
        f.write("===============================================================\n\n")
        f.write(f"coupling_topo_3d={args.coupling_topo_3d}\n")
        f.write(f"topo_3d_beta={args.topo_3d_beta}\n")
        f.write(f"layout_seed={args.layout_seed}\n")
        f.write(f"max_exact_n={args.max_exact_n}\n\n")

        f.write(f"corr(E_3d, crossing_density) = {corr_e3d_cross:.6f}\n")
        f.write(f"corr(penalty_factor_3d, crossing_density) = {corr_pen_cross:.6f}\n")
        f.write(f"corr(penalty_factor_3d, E_3d) = {corr_pen_e3d:.6f}\n\n")

        f.write(f"max|formula_residual| = {max_abs_res:.6e}\n")
        f.write(f"median|formula_residual| = {med_abs_res:.6e}\n\n")

        f.write("Top graphs by penalty_factor_3d:\n")
        f.write(top.to_string(index=False))
        f.write("\n")

    print(f"Wrote: {csv_path}")
    print(f"Wrote: {txt_path}")
    print(f"corr(E_3d, crossing_density) = {corr_e3d_cross:.6f}")
    print(f"corr(penalty_factor_3d, crossing_density) = {corr_pen_cross:.6f}")
    print(f"corr(penalty_factor_3d, E_3d) = {corr_pen_e3d:.6f}")
    print(f"max|formula_residual| = {max_abs_res:.6e}")
    print(f"median|formula_residual| = {med_abs_res:.6e}")


if __name__ == "__main__":
    main()

