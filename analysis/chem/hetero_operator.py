from __future__ import annotations

from typing import Dict, Sequence, Tuple

import numpy as np

Edge = Tuple[int, int]


def build_operator_H(
    n: int,
    edges: Sequence[Edge],
    types: Sequence[int],
    *,
    rho_by_type: Dict[int, float],
    alpha_H: float,
    valence_by_type: Dict[int, int],
) -> np.ndarray:
    adj = np.zeros((n, n), dtype=float)
    for u, v in edges:
        if u == v:
            continue
        a, b = int(u), int(v)
        adj[a, b] = 1.0
        adj[b, a] = 1.0
    deg = np.sum(adj, axis=1)
    lap = np.diag(deg) - adj
    valence = np.array([valence_by_type.get(int(t), 0) for t in types], dtype=float)
    rho = np.array([rho_by_type.get(int(t), 0.0) for t in types], dtype=float)
    potential = rho + float(alpha_H) * (valence - deg)
    H = lap + np.diag(potential)
    return H


def hetero_fingerprint(H: np.ndarray, *, taus: Tuple[float, ...] = (0.5, 1.0)) -> np.ndarray:
    vals, vecs = np.linalg.eigh(H)
    feats = []
    for k in range(1, 5):
        feats.append(float(np.mean(vals ** k)))
    for tau in taus:
        exp_vals = np.exp(-float(tau) * vals)
        diag_heat = (vecs ** 2) @ exp_vals
        feats.extend(
            [
                float(np.mean(diag_heat)),
                float(np.var(diag_heat)),
                float(np.min(diag_heat)),
                float(np.max(diag_heat)),
            ]
        )
    return np.asarray(feats, dtype=float)


def hetero_energy_from_state(
    n: int,
    edges: Sequence[Edge],
    types: Sequence[int],
    *,
    rho_by_type: Dict[int, float],
    alpha_H: float,
    valence_by_type: Dict[int, int],
) -> float:
    H = build_operator_H(
        n,
        edges,
        types,
        rho_by_type=rho_by_type,
        alpha_H=float(alpha_H),
        valence_by_type=valence_by_type,
    )
    vals = np.linalg.eigvalsh(H)
    return float(np.mean(vals ** 2))
