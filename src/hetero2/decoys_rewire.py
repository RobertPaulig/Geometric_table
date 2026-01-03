from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np

from hetero2.chemgraph import ChemGraph, _require_rdkit


def _bond_candidates(mol, *, lock_aromatic: bool, allow_ring_bonds: bool) -> List[Tuple[int, int]]:
    candidates: List[Tuple[int, int]] = []
    for bond in mol.GetBonds():
        if bond.GetBondType().name != "SINGLE":
            continue
        if lock_aromatic and (bond.GetIsAromatic() or bond.GetBeginAtom().GetIsAromatic() or bond.GetEndAtom().GetIsAromatic()):
            continue
        if not allow_ring_bonds and bond.IsInRing():
            continue
        u = bond.GetBeginAtomIdx()
        v = bond.GetEndAtomIdx()
        if u == v:
            continue
        candidates.append((u, v))
    return candidates


def _double_edge_swap(
    mol,
    rng: random.Random,
    candidates: Sequence[Tuple[int, int]],
) -> Tuple[bool, object]:
    if len(candidates) < 2:
        return False, mol
    for _ in range(20):
        (a, b) = rng.choice(candidates)
        (c, d) = rng.choice(candidates)
        if len({a, b, c, d}) < 4:
            continue
        if mol.GetBondBetweenAtoms(a, d) or mol.GetBondBetweenAtoms(c, b):
            continue
        rw = _require_rdkit()[0].RWMol(mol)
        rw.RemoveBond(a, b)
        rw.RemoveBond(c, d)
        rw.AddBond(a, d, order=_require_rdkit()[0].BondType.SINGLE)
        rw.AddBond(c, b, order=_require_rdkit()[0].BondType.SINGLE)
        return True, rw
    return False, mol


def _hash_smiles(smiles: str) -> str:
    return hashlib.sha256(smiles.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class DecoyResult:
    decoys: List[Dict[str, object]]
    warnings: List[str]
    stats: Dict[str, int]


def generate_rewire_decoys(
    smiles: str,
    *,
    k: int = 20,
    seed: int = 0,
    max_attempts: int | None = None,
    lock_aromatic: bool = True,
    allow_ring_bonds: bool = False,
) -> DecoyResult:
    Chem, _, _, _, _ = _require_rdkit()
    base = Chem.MolFromSmiles(smiles)
    if base is None:
        raise ValueError("Invalid SMILES")
    canonical_input = Chem.MolToSmiles(base)

    candidates = _bond_candidates(base, lock_aromatic=lock_aromatic, allow_ring_bonds=allow_ring_bonds)
    rng = random.Random(seed)
    max_attempts_eff = max_attempts if max_attempts is not None else max(100, k * 200)

    seen = {canonical_input}
    decoys: List[Dict[str, object]] = []
    warnings: List[str] = []
    stats = {"attempts": 0, "sanitize_fail": 0, "duplicate": 0, "no_candidate_swap": 0}

    for _ in range(max_attempts_eff):
        if len(decoys) >= k:
            break
        stats["attempts"] += 1
        ok, rw = _double_edge_swap(base, rng, candidates)
        if not ok:
            stats["no_candidate_swap"] += 1
            continue
        new_mol = rw.GetMol()
        try:
            Chem.SanitizeMol(new_mol)
        except Exception:
            stats["sanitize_fail"] += 1
            continue
        new_smiles = Chem.MolToSmiles(new_mol)
        if new_smiles in seen:
            stats["duplicate"] += 1
            continue
        seen.add(new_smiles)
        cg = ChemGraph(new_smiles)
        decoys.append(
            {
                "smiles": cg.canonical_smiles,
                "hash": _hash_smiles(cg.canonical_smiles),
                "ring_info": cg.ring_info(),
                "physchem": cg.physchem(),
            }
        )

    if len(decoys) < k:
        warnings.append(f"could_not_generate_k_decoys_under_constraints:{k-len(decoys)}")
    if stats["sanitize_fail"] > 0:
        warnings.append(f"rdkit_sanitize_failures:{stats['sanitize_fail']}")
    return DecoyResult(decoys=decoys, warnings=warnings, stats=stats)
