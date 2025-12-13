from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd

from core.geom_atoms import Molecule, AtomGraph, PERIODIC_TABLE
from core.crossing import estimate_crossing_number_circle
from core.entanglement_3d import entanglement_score
from core.layout_3d import layout_molecule_3d


def _make_example_molecules() -> List[Molecule]:
    """
    Небольший набор игрушечных молекул/графов для сравнения 2D vs 3D.
    """
    C = PERIODIC_TABLE["C"]
    O = PERIODIC_TABLE["O"]
    N = PERIODIC_TABLE["N"]

    # Простое дерево (цепочка C-C-C)
    mol_chain = Molecule(
        name="chain_C3",
        atoms=[C, C, C],
        bonds=[(0, 1), (1, 2)],
    )

    # Малый цикл (треугольник C-O-N)
    mol_cycle = Molecule(
        name="cycle_CON",
        atoms=[C, O, N],
        bonds=[(0, 1), (1, 2), (2, 0)],
    )

    # Плотный граф (K4) на C-атомах
    atoms_k4 = [C, C, C, C]
    bonds_k4: List[Tuple[int, int]] = []
    for i in range(4):
        for j in range(i + 1, 4):
            bonds_k4.append((i, j))
    mol_k4 = Molecule(name="K4_C", atoms=atoms_k4, bonds=bonds_k4)

    return [mol_chain, mol_cycle, mol_k4]


def _crossing_proxy_2d(mol: Molecule) -> float:
    adj = mol.adjacency_matrix()
    crossing, _ = estimate_crossing_number_circle(adj, max_exact_n=8)
    m = len(mol.bonds)
    if m <= 0:
        return 0.0
    return float(crossing) / float(m)


def main() -> None:
    rows = []
    for mol in _make_example_molecules():
        pos3d = layout_molecule_3d(mol, n_steps=400, step=0.02, seed=42)
        score3d = entanglement_score(pos3d, mol.bonds, sigma=0.25, ignore_adjacent=True)
        score2d = _crossing_proxy_2d(mol)
        rows.append(
            {
                "name": mol.name,
                "n_atoms": len(mol.atoms),
                "n_bonds": len(mol.bonds),
                "crossing_proxy_2d": score2d,
                "entanglement_3d": score3d,
            }
        )

    df = pd.DataFrame(rows)
    out_dir = Path("results")
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "topo3d_compare.csv"
    df.to_csv(csv_path, index=False)

    txt_path = out_dir / "topo3d_compare_summary.txt"
    with txt_path.open("w", encoding="utf-8") as f:
        f.write("TOPO-3D-1 comparison: 2D crossing proxy vs 3D entanglement\n\n")
        f.write(df.to_string(index=False))
        f.write("\n")

    print(f"Saved comparison CSV to {csv_path}")
    print(f"Saved summary TXT to {txt_path}")


if __name__ == "__main__":
    main()

