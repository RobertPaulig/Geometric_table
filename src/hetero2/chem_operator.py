from __future__ import annotations

import numpy as np

from analysis.chem.hetero_operator import build_operator_H

DEFAULT_ALPHA_H = 0.5
DEFAULT_RHO_BY_ATOMIC_NUM = {6: 0.0, 7: 0.2, 8: 0.5}
DEFAULT_VALENCE_BY_ATOMIC_NUM = {6: 4, 7: 3, 8: 2}


def laplacian_energy_from_edges(n: int, edges: tuple[tuple[int, int], ...]) -> float:
    n_i = int(n)
    adj = np.zeros((n_i, n_i), dtype=float)
    for u, v in edges:
        a, b = int(u), int(v)
        if a == b:
            continue
        adj[a, b] = 1.0
        adj[b, a] = 1.0
    deg = np.sum(adj, axis=1)
    lap = np.diag(deg) - adj
    vals = np.linalg.eigvalsh(lap)
    return float(np.mean(vals**2))


def h_operator_energy_from_edges(
    n: int,
    edges: tuple[tuple[int, int], ...],
    types: tuple[int, ...],
    *,
    alpha_H: float = DEFAULT_ALPHA_H,
    rho_by_atomic_num: dict[int, float] | None = None,
    valence_by_atomic_num: dict[int, int] | None = None,
) -> float:
    rho = dict(DEFAULT_RHO_BY_ATOMIC_NUM if rho_by_atomic_num is None else rho_by_atomic_num)
    valence = dict(DEFAULT_VALENCE_BY_ATOMIC_NUM if valence_by_atomic_num is None else valence_by_atomic_num)
    H = build_operator_H(
        int(n),
        edges,
        types,
        rho_by_type=rho,
        alpha_H=float(alpha_H),
        valence_by_type=valence,
    )
    vals = np.linalg.eigvalsh(H)
    return float(np.mean(vals**2))


def heavy_state_from_smiles(smiles: str) -> tuple[int, tuple[tuple[int, int], ...], tuple[int, ...]]:
    try:
        from rdkit import Chem  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise ImportError("RDKit is required for heavy_state_from_smiles. Install: pip install -e \".[dev,chem]\"") from exc

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("Invalid SMILES")
    heavy: list[int] = []
    mapping: dict[int, int] = {}
    types: list[int] = []
    for idx, atom in enumerate(mol.GetAtoms()):
        z = int(atom.GetAtomicNum())
        if z <= 1:
            continue
        mapping[idx] = len(heavy)
        heavy.append(idx)
        types.append(z)
    edges: list[tuple[int, int]] = []
    for bond in mol.GetBonds():
        u = int(bond.GetBeginAtomIdx())
        v = int(bond.GetEndAtomIdx())
        if u in mapping and v in mapping:
            a = int(mapping[u])
            b = int(mapping[v])
            if a == b:
                continue
            if a > b:
                a, b = b, a
            edges.append((a, b))
    edges_sorted = tuple(sorted(set(edges)))
    return len(types), edges_sorted, tuple(types)

