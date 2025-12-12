from __future__ import annotations

import numpy as np

from core.nuclear_island import nuclear_functional


def build_radial_hamiltonian(
    R_max: float,
    R_well: float,
    V0: float,
    N_grid: int,
    ell: int,
):
    dr = R_max / (N_grid + 1)
    r = np.linspace(dr, R_max - dr, N_grid)

    V = np.where(r <= R_well, -V0, 0.0)
    V_eff = V + ell * (ell + 1) / (r ** 2)

    main_diag = 2.0 / dr**2 + V_eff
    off_diag = -1.0 / dr**2 * np.ones(N_grid - 1)

    H = (
        np.diag(main_diag)
        + np.diag(off_diag, k=1)
        + np.diag(off_diag, k=-1)
    )

    evals, evecs = np.linalg.eigh(H)
    return evals, evecs


def collect_levels(
    R_max: float = 12.0,
    R_well: float = 5.0,
    V0: float = 40.0,
    N_grid: int = 220,
    L_max: int = 5,
    levels_per_l: int = 10,
    energy_cut: float = 0.0,
):
    levels = []
    for ell in range(L_max + 1):
        evals, _ = build_radial_hamiltonian(
            R_max=R_max,
            R_well=R_well,
            V0=V0,
            N_grid=N_grid,
            ell=ell,
        )
        bound = evals[evals < energy_cut]
        if bound.size == 0:
            continue
        bound = bound[:levels_per_l]
        g = 2 * (2 * ell + 1)
        for E in bound:
            levels.append((float(E), int(g), int(ell)))

    levels.sort(key=lambda t: t[0])
    return levels


def get_toy_low_magic(
    R_max: float = 12.0,
    R_well: float = 5.0,
    V0: float = 40.0,
    N_grid: int = 220,
    L_max: int = 5,
    levels_per_l: int = 10,
    energy_cut: float = 0.0,
    n_prefix: int = 25,
    top_k_gaps: int = 4,
):
    levels = collect_levels(
        R_max=R_max,
        R_well=R_well,
        V0=V0,
        N_grid=N_grid,
        L_max=L_max,
        levels_per_l=levels_per_l,
        energy_cut=energy_cut,
    )
    if not levels:
        return []

    energies = np.array([E for (E, g, ell) in levels])
    g = np.array([g for (E, g, ell) in levels], dtype=int)
    N_cum = np.cumsum(g)

    n_use = min(n_prefix, len(levels))
    E_sub = energies[:n_use]
    N_sub = N_cum[:n_use]

    dE = np.diff(E_sub)
    if dE.size == 0:
        return []

    gap_indices_sorted = np.argsort(dE)[::-1]
    gap_indices_sorted = gap_indices_sorted[:top_k_gaps]
    gap_indices_unique = sorted(set(gap_indices_sorted.tolist()))

    toy_magic = [int(N_sub[idx]) for idx in gap_indices_unique]
    return toy_magic


def find_best_Z_for_N(
    N: int,
    Z_min: int = 20,
    Z_max: int = 130,
    step: int = 2,
):
    best_Z = None
    best_F = None
    for Z in range(Z_min, Z_max + 1, step):
        if (Z + N) % 2 != 0:
            continue
        F = nuclear_functional(Z, N)
        if best_F is None or F < best_F:
            best_F = F
            best_Z = Z
    return best_Z, best_F


def main():
    toy_magic = get_toy_low_magic(
        R_max=12.0,
        R_well=5.0,
        V0=40.0,
        N_grid=220,
        L_max=5,
        levels_per_l=10,
        energy_cut=0.0,
        n_prefix=25,
        top_k_gaps=4,
    )

    print("=== Spectral → F_nuc bridge ===")
    print("Toy low-N magic numbers from FDM operator:", toy_magic)
    print()

    print("Mapping toy magic N_toy to F_nuc minima over Z:")
    print(" N_toy |  Z_best  A=Z+N  N/Z    F_nuc(Z_best,N)")
    print("-------+----------------------------------------")

    for N in toy_magic:
        Z_best, F_best = find_best_Z_for_N(
            N=N,
            Z_min=20,
            Z_max=130,
            step=2,
        )
        if Z_best is None:
            print(f" {N:5d} |  (no even-even minimum found in range)")
            continue
        A = Z_best + N
        ratio = N / Z_best if Z_best != 0 else 0.0
        print(
            f" {N:5d} | {Z_best:7d}  {A:5d}  {ratio:4.2f}   {F_best:12.3f}"
        )


if __name__ == "__main__":
    main()
