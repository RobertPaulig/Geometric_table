from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

from hetero2.chemgraph import ChemGraph
from hetero2.phase_channel import magnetic_laplacian, normalize_flux_phi, phase_matrix_flux_on_cycles, sssr_cycles_from_mol
from hetero2.physics_operator import AtomsDbV1, load_atoms_db_v1


@dataclass(frozen=True)
class _MolRow:
    mid: int
    gid: str
    smiles: str
    truth_rel: float


def _heavy_atom_mapping(mol) -> tuple[list[int], dict[int, int]]:
    heavy: list[int] = []
    mapping: dict[int, int] = {}
    for idx, atom in enumerate(mol.GetAtoms()):
        if int(atom.GetAtomicNum()) > 1:
            mapping[int(idx)] = len(heavy)
            heavy.append(int(idx))
    return heavy, mapping


def _heavy_bonds_with_attrs(mol, mapping: dict[int, int]) -> tuple[tuple[int, int, float, int], ...]:
    edges: set[tuple[int, int, float, int]] = set()
    for bond in mol.GetBonds():
        u = int(bond.GetBeginAtomIdx())
        v = int(bond.GetEndAtomIdx())
        if u in mapping and v in mapping:
            i = int(mapping[u])
            j = int(mapping[v])
            if i == j:
                continue
            if i > j:
                i, j = j, i
            bo = float(bond.GetBondTypeAsDouble())
            arom = 1 if bool(bond.GetIsAromatic()) else 0
            edges.add((i, j, bo, int(arom)))
    return tuple(sorted(edges))


def _cycles_heavy_from_mol_sssr(*, mol, heavy_mapping: dict[int, int]) -> tuple[tuple[int, ...], ...]:
    cycles_full = sssr_cycles_from_mol(mol)
    cycles_heavy: list[tuple[int, ...]] = []
    for cyc in cycles_full:
        mapped: list[int] = []
        for a in cyc:
            if int(a) not in heavy_mapping:
                raise ValueError("unexpected non-heavy atom in SSSR cycle")
            mapped.append(int(heavy_mapping[int(a)]))
        cycles_heavy.append(tuple(mapped))
    return tuple(cycles_heavy)


def _build_weight_adjacency(
    bonds: Iterable[tuple[int, int, float, int]],
    *,
    n: int,
    atoms_db: AtomsDbV1,
    types: Iterable[int],
    aromatic_multiplier: float,
    delta_chi_alpha: float,
) -> np.ndarray:
    chi_by_z = atoms_db.chi_by_atomic_num
    z = [int(t) for t in types]
    missing = sorted({int(v) for v in z if int(v) not in chi_by_z})
    if missing:
        raise ValueError(f"missing atoms_db chi for Z={missing}")
    chi = np.asarray([float(chi_by_z[int(v)]) for v in z], dtype=float)

    w_adj = np.zeros((int(n), int(n)), dtype=float)
    am = float(aromatic_multiplier)
    alpha = float(delta_chi_alpha)
    for i, j, bond_order, aromatic in bonds:
        a = int(i)
        b = int(j)
        if a == b:
            continue
        w = float(bond_order)
        if int(aromatic):
            w = w * (1.0 + am)
        w = w * (1.0 + alpha * float(abs(float(chi[a]) - float(chi[b]))))
        w_adj[a, b] = w
        w_adj[b, a] = w
    return w_adj


def _phase_laplacian(
    *,
    weights: np.ndarray,
    cycles_heavy: tuple[tuple[int, ...], ...],
    phi_fixed: float,
) -> np.ndarray:
    n = int(weights.shape[0])
    A = phase_matrix_flux_on_cycles(n=n, cycles=cycles_heavy, phi=float(phi_fixed))
    L_phase = magnetic_laplacian(weights=weights, A=A)
    return np.asarray(L_phase, dtype=np.complex128)


def _heavy_canonical_perm(mol, *, heavy_atom_indices: list[int], heavy_mapping: dict[int, int]) -> list[int]:
    from rdkit import Chem

    ranks = list(Chem.CanonicalRankAtoms(mol, breakTies=True))
    heavy_order = sorted((int(idx) for idx in heavy_atom_indices), key=lambda i: (int(ranks[int(i)]), int(i)))
    return [int(heavy_mapping[int(idx)]) for idx in heavy_order]


def _load_rows(input_csv: Path, groups: set[str]) -> list[_MolRow]:
    rows: list[_MolRow] = []
    with input_csv.open("r", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        if not r.fieldnames:
            raise ValueError("missing CSV header")
        required = {"id", "group_id", "smiles", "energy_rel_kcalmol"}
        missing = [c for c in required if c not in set(r.fieldnames)]
        if missing:
            raise ValueError(f"missing required columns: {missing}")
        for row in r:
            gid = str(row.get("group_id") or "").strip()
            if gid not in groups:
                continue
            rows.append(
                _MolRow(
                    mid=int(str(row.get("id") or "").strip()),
                    gid=gid,
                    smiles=str(row.get("smiles") or "").strip(),
                    truth_rel=float(str(row.get("energy_rel_kcalmol") or "").strip()),
                )
            )
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description="A4.x ceiling-test: isomorphic_H on fixed H used in K=f(H).")
    ap.add_argument("--input_csv", type=Path, default=Path("data/accuracy/isomer_truth.v1.csv"))
    ap.add_argument(
        "--groups",
        type=str,
        default="C11H21B1N2O4,C15H24O1,C20H22N2O2",
        help="Comma-separated list of group_id values to test.",
    )
    ap.add_argument("--phi_fixed", type=float, default=float(math.pi / 2.0), help="Phi fixed (radians).")
    ap.add_argument("--heat_tau", type=float, default=1.0, help="Not used here; recorded for provenance only.")
    ap.add_argument("--edge_aromatic_multiplier", type=float, default=0.0)
    ap.add_argument("--edge_delta_chi_alpha", type=float, default=1.0)
    ap.add_argument("--out_csv", type=Path, default=Path("out_isomorphism_ceiling/isomorphism_ceiling.csv"))
    ap.add_argument("--tol", type=float, default=1e-12)
    ap.add_argument("--delta_truth_min", type=float, default=1e-6)
    args = ap.parse_args()

    groups = {g.strip() for g in str(args.groups).split(",") if g.strip()}
    if not groups:
        raise SystemExit("no groups specified")

    rows = _load_rows(Path(args.input_csv), groups)
    if not rows:
        raise SystemExit(f"no rows found for groups={sorted(groups)}")

    atoms_db = load_atoms_db_v1()
    phi_fixed = float(normalize_flux_phi(float(args.phi_fixed)))

    # Compute canonicalized H per molecule (heavy-only Laplacian with phase channel).
    per_group: dict[str, list[tuple[int, float, np.ndarray]]] = {}
    for r in rows:
        g = ChemGraph(smiles=r.smiles)
        mol = g.mol
        heavy, mapping = _heavy_atom_mapping(mol)
        types = [int(mol.GetAtomWithIdx(int(idx)).GetAtomicNum()) for idx in heavy]
        bonds = _heavy_bonds_with_attrs(mol, mapping)
        cycles_heavy = _cycles_heavy_from_mol_sssr(mol=mol, heavy_mapping=mapping)

        w_adj = _build_weight_adjacency(
            bonds,
            n=len(heavy),
            atoms_db=atoms_db,
            types=types,
            aromatic_multiplier=float(args.edge_aromatic_multiplier),
            delta_chi_alpha=float(args.edge_delta_chi_alpha),
        )
        H = _phase_laplacian(weights=w_adj, cycles_heavy=cycles_heavy, phi_fixed=phi_fixed)

        perm = _heavy_canonical_perm(mol, heavy_atom_indices=heavy, heavy_mapping=mapping)
        Hc = np.asarray(H)[np.ix_(perm, perm)]
        per_group.setdefault(r.gid, []).append((int(r.mid), float(r.truth_rel), Hc))

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    tol = float(args.tol)
    delta_truth_min = float(args.delta_truth_min)

    any_ceiling = False
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["group_id", "a_id", "b_id", "isomorphic_H", "delta_truth"])
        w.writeheader()
        for gid, items in sorted(per_group.items()):
            for i in range(len(items)):
                a_id, a_truth, Ha = items[i]
                for j in range(i + 1, len(items)):
                    b_id, b_truth, Hb = items[j]
                    diff = np.max(np.abs(Ha - Hb)) if Ha.size and Hb.size else 0.0
                    iso = bool(diff <= tol)
                    delta_truth = float(abs(float(a_truth) - float(b_truth)))
                    w.writerow(
                        {
                            "group_id": str(gid),
                            "a_id": int(a_id),
                            "b_id": int(b_id),
                            "isomorphic_H": "true" if iso else "false",
                            "delta_truth": delta_truth,
                        }
                    )
                    if iso and delta_truth > delta_truth_min:
                        any_ceiling = True

    print(f"wrote_csv={out_csv.as_posix()}")
    print(f"phi_fixed_norm={phi_fixed}")
    print(f"isomorphism_ceiling_detected={any_ceiling}")


if __name__ == "__main__":
    main()

