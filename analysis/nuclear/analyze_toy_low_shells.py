from __future__ import annotations

import argparse

import numpy as np


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


def analyze_low_shells(
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
        print("No bound levels found.")
        return

    energies = np.array([E for (E, g, ell) in levels])
    g = np.array([g for (E, g, ell) in levels], dtype=int)
    ells = np.array([ell for (E, g, ell) in levels], dtype=int)
    N_cum = np.cumsum(g)

    n_use = min(n_prefix, len(levels))
    E_sub = energies[:n_use]
    N_sub = N_cum[:n_use]
    g_sub = g[:n_use]
    ell_sub = ells[:n_use]

    dE = np.diff(E_sub)
    if dE.size == 0:
        print("Not enough levels for gap analysis.")
        return

    gap_indices_sorted = np.argsort(dE)[::-1]
    gap_indices_sorted = gap_indices_sorted[:top_k_gaps]
    gap_indices_unique = sorted(set(gap_indices_sorted.tolist()))

    print("=== Low-N shell analysis (toy operator) ===")
    print(f"Parameters: R_well={R_well}, V0={V0}, R_max={R_max}, L_max={L_max}")
    print(f"Using first {n_use} bound levels.\n")

    print("First levels:")
    print(" idx |    E       g   ℓ   N_cum")
    print("-----+---------------------------")
    for i in range(n_use):
        print(
            f"{i:4d} | {E_sub[i]:7.3f}  {g_sub[i]:3d}  {ell_sub[i]:2d}  {N_sub[i]:5d}"
        )

    print("\nLargest gaps among first levels:")
    print(" gap_idx | dE       -> shell after level, N_cum")

    toy_magic = []
    for idx in gap_indices_unique:
        N_here = N_sub[idx]
        print(f" {idx:7d} | {dE[idx]:7.3f} -> N_toy ≈ {N_here}")
        toy_magic.append(int(N_here))

    print("\nToy low-N magic numbers (from largest early gaps):", toy_magic)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(
        description="Toy low-shell analysis for a simple radial well."
    )
    parser.add_argument("--R-max", type=float, default=12.0, help="Maximal radius.")
    parser.add_argument("--R-well", type=float, default=5.0, help="Well radius.")
    parser.add_argument("--V0", type=float, default=40.0, help="Well depth.")
    parser.add_argument("--N-grid", type=int, default=220, help="Number of radial grid points.")
    parser.add_argument("--L-max", type=int, default=5, help="Maximum orbital quantum number ℓ.")
    parser.add_argument(
        "--print-levels",
        action="store_true",
        help="If set, prints detailed level table (default behavior).",
    )
    args = parser.parse_args(argv)

    analyze_low_shells(
        R_max=args.R_max,
        R_well=args.R_well,
        V0=args.V0,
        N_grid=args.N_grid,
        L_max=args.L_max,
        levels_per_l=10,
        energy_cut=0.0,
        n_prefix=25,
        top_k_gaps=4,
    )


if __name__ == "__main__":
    main()
