from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable

import numpy as np

try:
    from rdkit import Chem
except Exception as exc:  # pragma: no cover
    raise ImportError("hetero2.phase_channel requires RDKit") from exc


@dataclass(frozen=True)
class PhaseChannelConfig:
    # Global cycle flux (holonomy) parameter. Internally normalized to [-pi, pi].
    flux_phi: float


def normalize_flux_phi(phi: float) -> float:
    """
    Normalize a flux parameter to the canonical interval [-pi, pi].
    This guarantees 2*pi periodicity for exp(i*A).
    """
    two_pi = 2.0 * math.pi
    return ((float(phi) + math.pi) % two_pi) - math.pi


def _order_simple_cycle_atoms(*, ring_edges: list[tuple[int, int]]) -> list[int]:
    """
    Given the undirected edges of a simple cycle, return a deterministic atom order
    [v0, v1, ..., v_{m-1}] around the cycle.

    Deterministic orientation rule (A3.2 contract):
    - rotate so the first vertex is the minimum atom index in the cycle
    - choose the direction by picking the lexicographically smaller of the two possible
      traversals starting from that vertex (clockwise vs counter-clockwise)
    """
    if not ring_edges:
        raise ValueError("ring_edges must be non-empty")

    adj: dict[int, list[int]] = {}
    for u, v in ring_edges:
        if u == v:
            raise ValueError("self-loop in ring_edges")
        adj.setdefault(int(u), []).append(int(v))
        adj.setdefault(int(v), []).append(int(u))

    for k, vs in adj.items():
        if len(vs) != 2:
            raise ValueError(f"expected simple cycle degree=2, got degree={len(vs)} for atom {k}")

    def _walk(*, start: int, nxt: int) -> list[int]:
        cycle = [start, nxt]
        prev = start
        cur = nxt
        while True:
            n0, n1 = adj[cur]
            nn = n0 if n0 != prev else n1
            if nn == start:
                break
            if nn in cycle:
                raise ValueError("cycle reconstruction failed (revisited atom)")
            cycle.append(nn)
            prev, cur = cur, nn
        if len(cycle) < 3:
            raise ValueError("cycle length must be >= 3")
        return cycle

    start = min(adj.keys())
    n_a, n_b = sorted(adj[start])

    cyc_a = _walk(start=start, nxt=n_a)
    cyc_b = _walk(start=start, nxt=n_b)

    return cyc_a if tuple(cyc_a) <= tuple(cyc_b) else cyc_b


def sssr_cycles_from_mol(mol: "Chem.Mol") -> list[list[int]]:
    """
    Return a deterministic list of SSSR cycles as ordered atom-index lists.
    Uses RDKit RingInfo (SSSR) and reconstructs per-ring ordered cycles from bond indices.
    """
    ri = mol.GetRingInfo()
    bond_rings: Iterable[Iterable[int]] = ri.BondRings()

    cycles: list[list[int]] = []
    for bond_ring in bond_rings:
        edges: list[tuple[int, int]] = []
        for bidx in bond_ring:
            b = mol.GetBondWithIdx(int(bidx))
            edges.append((int(b.GetBeginAtomIdx()), int(b.GetEndAtomIdx())))
        cycles.append(_order_simple_cycle_atoms(ring_edges=edges))

    # Deterministic ordering across rings.
    cycles.sort(key=lambda c: (len(c), tuple(c)))
    return cycles


def ring_edges_from_cycles(cycles: Iterable[Iterable[int]]) -> set[frozenset[int]]:
    edges: set[frozenset[int]] = set()
    for cyc in cycles:
        atoms = list(cyc)
        m = len(atoms)
        if m < 3:
            continue
        for k in range(m):
            a = int(atoms[k])
            b = int(atoms[(k + 1) % m])
            edges.add(frozenset((a, b)))
    return edges


def phase_matrix_flux_on_cycles(*, n: int, cycles: Iterable[Iterable[int]], phi: float) -> np.ndarray:
    """
    Build an antisymmetric phase matrix A for a graph with n nodes, by distributing
    a single global flux Phi across each cycle as +/-Phi/m per oriented edge (m = cycle length).

    If an undirected edge belongs to multiple SSSR cycles, contributions are summed.
    """
    phi_n = normalize_flux_phi(phi)
    A = np.zeros((int(n), int(n)), dtype=float)

    for cyc in cycles:
        atoms = [int(x) for x in cyc]
        m = len(atoms)
        if m < 3:
            continue
        delta = phi_n / float(m)
        for k in range(m):
            i = atoms[k]
            j = atoms[(k + 1) % m]
            A[i, j] += delta
            A[j, i] -= delta

    # Enforce antisymmetry exactly (robust against any accidental imbalance).
    return 0.5 * (A - A.T)


def phase_matrix_from_mol_sssr(*, mol: "Chem.Mol", config: PhaseChannelConfig) -> np.ndarray:
    cycles = sssr_cycles_from_mol(mol)
    return phase_matrix_flux_on_cycles(n=mol.GetNumAtoms(), cycles=cycles, phi=config.flux_phi)


def magnetic_laplacian(*, weights: np.ndarray, A: np.ndarray) -> np.ndarray:
    """
    Magnetic (phase) Laplacian:
      W^A_ij = w_ij * exp(i A_ij), with A_ji = -A_ij
      D_ii   = sum_j w_ij  (strictly real weights)
      L_A    = D - W^A
    """
    w = np.asarray(weights, dtype=float)
    if w.ndim != 2 or w.shape[0] != w.shape[1]:
        raise ValueError("weights must be square (n,n)")

    A0 = np.asarray(A, dtype=float)
    if A0.shape != w.shape:
        raise ValueError("A must match weights shape")

    if not np.allclose(w, w.T, atol=0.0):
        raise ValueError("weights must be symmetric (undirected graph)")

    # Antisymmetry is required for Hermiticity; enforce numerically.
    A0 = 0.5 * (A0 - A0.T)

    W = w.astype(np.complex128) * np.exp(1j * A0.astype(np.complex128))
    deg = np.sum(w, axis=1).astype(np.complex128)
    D = np.diag(deg)
    return D - W


def gauge_transform_A(*, A: np.ndarray, theta: np.ndarray) -> np.ndarray:
    """
    Gauge transform: A'_ij = A_ij + theta_i - theta_j.
    """
    A0 = np.asarray(A, dtype=float)
    t = np.asarray(theta, dtype=float).reshape((-1,))
    if A0.ndim != 2 or A0.shape[0] != A0.shape[1]:
        raise ValueError("A must be square (n,n)")
    if t.shape[0] != A0.shape[0]:
        raise ValueError("theta must have shape (n,)")
    A2 = A0 + t[:, None] - t[None, :]
    return 0.5 * (A2 - A2.T)

