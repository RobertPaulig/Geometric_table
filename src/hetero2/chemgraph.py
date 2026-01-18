from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np


def _require_rdkit():
    try:
        from rdkit import Chem  # type: ignore
        from rdkit.Chem import Crippen, Descriptors, QED, rdMolDescriptors  # type: ignore
    except Exception as exc:  # pragma: no cover - executed only when rdkit missing
        raise ImportError("RDKit is required for hetero2 chemgraph. Install: pip install -e \".[dev,chem]\"") from exc
    return Chem, Crippen, Descriptors, QED, rdMolDescriptors


@dataclass(frozen=True)
class ChemGraph:
    smiles: str

    def __post_init__(self) -> None:
        Chem, _, _, _, _ = _require_rdkit()
        mol = Chem.MolFromSmiles(self.smiles)
        if mol is None:
            raise ValueError("Invalid SMILES")
        object.__setattr__(self, "_mol", mol)
        object.__setattr__(self, "_canonical_smiles", Chem.MolToSmiles(mol))

    @property
    def canonical_smiles(self) -> str:
        return str(getattr(self, "_canonical_smiles"))

    @property
    def mol(self):
        return getattr(self, "_mol")

    def _heavy_atom_map(self) -> Tuple[List[int], Dict[int, int]]:
        heavy = []
        mapping: Dict[int, int] = {}
        for idx, atom in enumerate(self.mol.GetAtoms()):
            if atom.GetAtomicNum() > 1:
                mapping[idx] = len(heavy)
                heavy.append(idx)
        return heavy, mapping

    def n_heavy_atoms(self) -> int:
        heavy, _ = self._heavy_atom_map()
        return int(len(heavy))

    def heavy_atom_types(self) -> Tuple[int, ...]:
        heavy, _ = self._heavy_atom_map()
        types: List[int] = []
        for idx in heavy:
            types.append(int(self.mol.GetAtomWithIdx(int(idx)).GetAtomicNum()))
        return tuple(types)

    def adjacency(self) -> np.ndarray:
        heavy, mapping = self._heavy_atom_map()
        n = len(heavy)
        adj = np.zeros((n, n), dtype=float)
        for bond in self.mol.GetBonds():
            u = bond.GetBeginAtomIdx()
            v = bond.GetEndAtomIdx()
            if u in mapping and v in mapping:
                i = mapping[u]
                j = mapping[v]
                adj[i, j] = 1.0
                adj[j, i] = 1.0
        return adj

    def laplacian(self) -> np.ndarray:
        adj = self.adjacency()
        deg = np.diag(adj.sum(axis=1))
        return deg - adj

    def physchem(self) -> Dict[str, float]:
        _, Crippen, Descriptors, QED, rdMolDescriptors = _require_rdkit()
        mol = self.mol
        out = {
            "mw": float(Descriptors.MolWt(mol)),
            "logp": float(Crippen.MolLogP(mol)),
            "tpsa": float(rdMolDescriptors.CalcTPSA(mol)),
            "hbd": float(rdMolDescriptors.CalcNumHBD(mol)),
            "hba": float(rdMolDescriptors.CalcNumHBA(mol)),
        }
        try:
            out["qed"] = float(QED.qed(mol))
        except Exception:
            pass
        return out

    def ring_info(self) -> Dict[str, int]:
        ring_info = self.mol.GetRingInfo()
        n_rings = int(ring_info.NumRings())
        n_aromatic = 0
        for ring in ring_info.AtomRings():
            if all(self.mol.GetAtomWithIdx(idx).GetIsAromatic() for idx in ring):
                n_aromatic += 1
        return {"n_rings": n_rings, "n_aromatic_rings": int(n_aromatic)}
