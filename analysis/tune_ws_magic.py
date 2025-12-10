from __future__ import annotations

import numpy as np

from core.nuclear_spectrum_ws import collect_levels_ws


TARGET_MAGICS = np.array([2, 8, 20, 28, 50, 82], dtype=float)


def get_toy_magic_ws(
    R_max: float,
    R0: float,
    a: float,
    V0: float,
    N_grid: int = 220,
    L_max: int = 5,
    levels_per_l: int = 12,
    energy_cut: float = 0.0,
    n_prefix: int = 40,
    top_k_gaps: int = 8,
):
    levels = collect_levels_ws(
        R_max=R_max,
        R0=R0,
        a=a,
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

    gap_idx_sorted = np.argsort(dE)[::-1][:top_k_gaps]
    gap_idx_unique = sorted(set(gap_idx_sorted.tolist()))

    toy_magic = [int(N_sub[idx]) for idx in gap_idx_unique]
    toy_magic = sorted(set(toy_magic))
    return toy_magic


def cost_magic(toy_magic, target=None, n_compare: int = 4) -> float:
    if target is None:
        target = TARGET_MAGICS
    if len(toy_magic) == 0:
        return 1e9
    n = min(n_compare, len(toy_magic), len(target))
    tm = np.array(toy_magic[:n], dtype=float)
    tg = target[:n]
    return float(np.sum((tm - tg) ** 2))


def main():
    R0_grid = np.linspace(4.0, 7.0, 4)
    a_grid = np.linspace(0.4, 1.0, 4)
    V0_grid = np.linspace(40.0, 80.0, 5)

    R_max = 12.0
    N_grid = 220
    L_max = 5

    results = []

    print("Scanning WS parameter grid for toy magic numbers...")
    print(f"R0 in {R0_grid}, a in {a_grid}, V0 in {V0_grid}")

    for R0 in R0_grid:
        for a in a_grid:
            for V0 in V0_grid:
                toy_magic = get_toy_magic_ws(
                    R_max=R_max,
                    R0=R0,
                    a=a,
                    V0=V0,
                    N_grid=N_grid,
                    L_max=L_max,
                    levels_per_l=12,
                    energy_cut=0.0,
                    n_prefix=40,
                    top_k_gaps=10,
                )
                c = cost_magic(toy_magic, TARGET_MAGICS, n_compare=4)
                results.append((c, R0, a, V0, toy_magic))

    results.sort(key=lambda t: t[0])

    print("\n=== Top 10 parameter sets by cost (WS) ===")
    for i, (c, R0, a, V0, toy_magic) in enumerate(results[:10], start=1):
        print(
            f"{i:2d}) R0={R0:.2f}, a={a:.2f}, V0={V0:.1f}, "
            f"cost={c:.3f}, magic_N={toy_magic}"
        )


if __name__ == "__main__":
    main()

