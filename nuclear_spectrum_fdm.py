from __future__ import annotations

import numpy as np


def build_radial_hamiltonian(
    R_max: float,
    R_well: float,
    V0: float,
    N_grid: int,
    ell: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Строит радиальный Гамильтониан для заданного ℓ в квадратной яме.

    Уравнение для u(r) = r R(r):
        - d^2 u/dr^2 + V_eff(r) u = E u
    где V_eff(r) = V(r) + ℓ(ℓ+1)/r^2, и мы выбрали единицы ħ^2/2m = 1.

    Граничные условия:
        u(0) = 0, u(R_max) = 0  (реализуем как Дирихле на концах сетки)
    """
    dr = R_max / (N_grid + 1)
    r = np.linspace(dr, R_max - dr, N_grid)

    # квадратная яма: V = -V0 внутри R_well, 0 снаружи
    V = np.where(r <= R_well, -V0, 0.0)

    # эффективный потенциал: яма + центробежный член
    V_eff = V + ell * (ell + 1) / (r ** 2)

    # кинетический оператор: -d^2/dr^2
    main_diag = 2.0 / dr**2 + V_eff
    off_diag = -1.0 / dr**2 * np.ones(N_grid - 1)

    H = (
        np.diag(main_diag)
        + np.diag(off_diag, k=1)
        + np.diag(off_diag, k=-1)
    )

    evals, evecs = np.linalg.eigh(H)
    return evals, evecs


def compute_shells_from_spectrum(
    R_max: float = 12.0,
    R_well: float = 5.0,
    V0: float = 40.0,
    N_grid: int = 220,
    L_max: int = 5,
    levels_per_l: int = 10,
    energy_cut: float = 0.0,
    gap_factor: float = 3.0,
):
    """
    Считает спектр для ℓ = 0..L_max, оставляет bound-состояния (E < energy_cut),
    задаёт вырождение g = 2(2ℓ+1), сортирует уровни по энергии и ищет
    большие разрывы как игрушечные "магические" числа.
    """
    levels: list[tuple[float, int, int]] = []

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

    energies = np.array([E for (E, g, ell) in levels], dtype=float)
    degeneracies = np.array([g for (E, g, ell) in levels], dtype=int)
    ells = np.array([ell for (E, g, ell) in levels], dtype=int)

    N_cum = np.cumsum(degeneracies)

    dE = np.diff(energies)
    positive = dE[dE > 0]
    if positive.size == 0:
        return levels, N_cum, np.array([], dtype=int), np.array([], dtype=int)

    median_gap = np.median(positive)
    magic_indices = np.where(dE > gap_factor * median_gap)[0]
    magic_numbers = N_cum[magic_indices]

    return levels, N_cum, magic_indices, magic_numbers


def main():
    print("=== Toy spectral nuclear shells (FDM square well) ===")

    levels, N_cum, magic_idx, magic_N = compute_shells_from_spectrum(
        R_max=12.0,
        R_well=5.0,
        V0=40.0,
        N_grid=220,
        L_max=5,
        levels_per_l=10,
        energy_cut=0.0,
        gap_factor=3.0,
    )

    energies = np.array([E for (E, g, ell) in levels])
    degeneracies = np.array([g for (E, g, ell) in levels])
    ells = np.array([ell for (E, g, ell) in levels])

    print("\nFirst 15 bound levels (sorted):")
    print(" idx |    E       g   ℓ   N_cum")
    print("-----+---------------------------")
    for i in range(min(15, len(levels))):
        print(
            f"{i:4d} | {energies[i]:7.3f}  {degeneracies[i]:3d}  {ells[i]:2d}  {N_cum[i]:5d}"
        )

    print("\nApproximate shell closures (toy magic numbers):")
    if magic_N.size == 0:
        print("  (no large energy gaps detected)")
    else:
        for k, idx in enumerate(magic_idx):
            print(
                f"  shell {k+1:2d}: "
                f"after level {idx:3d}, E = {energies[idx]:7.3f} → N_toy ≈ {int(magic_N[k])}"
            )

    try:
        import csv

        out_name = "nuclear_spectrum_toy_levels.csv"
        with open(out_name, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["idx", "E", "degeneracy", "ell", "N_cum"])
            for i, (E, g, ell) in enumerate(levels):
                writer.writerow([i, E, g, ell, int(N_cum[i])])
        print(f"\nSaved levels to {out_name}")
    except Exception as e:
        print(f"\n[WARN] Failed to save CSV: {e}")


if __name__ == "__main__":
    main()

