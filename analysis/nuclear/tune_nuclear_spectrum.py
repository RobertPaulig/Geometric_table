from __future__ import annotations

import csv
from math import inf
from typing import List, Tuple

import numpy as np

from analysis.nuclear.tune_metrics import TARGET_MAGIC


def build_radial_hamiltonian(
    R_max: float,
    R_well: float,
    V0: float,
    N_grid: int,
    ell: int,
) -> Tuple[np.ndarray, np.ndarray]:
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


def compute_magic_numbers(
    R_max: float,
    R_well: float,
    V0: float,
    N_grid: int = 220,
    L_max: int = 5,
    levels_per_l: int = 8,
    energy_cut: float = 0.0,
    gap_factor: float = 3.0,
) -> Tuple[List[Tuple[float, int, int]], np.ndarray, List[int]]:
    levels: List[Tuple[float, int, int]] = []
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

    if not levels:
        return [], np.array([]), []

    levels.sort(key=lambda t: t[0])
    energies = np.array([E for (E, g, ell) in levels])
    degeneracies = np.array([g for (E, g, ell) in levels])

    N_cum = np.cumsum(degeneracies)
    dE = np.diff(energies)

    positive = dE[dE > 0]
    if positive.size == 0:
        return levels, N_cum, []

    median_gap = np.median(positive)
    magic_idx = np.where(dE > gap_factor * median_gap)[0]
    magic_N = N_cum[magic_idx]
    return levels, N_cum, magic_N.tolist()


def magic_cost(magic_N: List[int], target=None, k_use: int = 4) -> float:
    if target is None:
        target = TARGET_MAGIC
    if not magic_N:
        return inf
    k = min(len(magic_N), len(target), k_use)
    if k == 0:
        return inf
    magic_tr = magic_N[:k]
    targ_tr = target[:k]
    diffs = [
        (magic_tr[i] - targ_tr[i]) / max(targ_tr[i], 1) for i in range(k)
    ]
    cost = sum(d * d for d in diffs)
    if len(magic_N) < k_use:
        cost += (k_use - len(magic_N)) * 5.0
    return cost


def scan_parameter_grid() -> None:
    R_max = 12.0
    L_max = 5
    N_grid = 220
    levels_per_l = 8
    energy_cut = 0.0
    gap_factor = 3.0

    R_well_values = np.linspace(3.0, 7.0, 9)
    V0_values = np.linspace(25.0, 60.0, 8)

    results = []

    print("Scanning parameter grid for toy magic numbers...")
    print(
        f"R_well in [{R_well_values[0]}, {R_well_values[-1]}], "
        f"V0 in [{V0_values[0]}, {V0_values[-1]}]"
    )

    for R_well in R_well_values:
        for V0 in V0_values:
            _, _, magic_N = compute_magic_numbers(
                R_max=R_max,
                R_well=float(R_well),
                V0=float(V0),
                N_grid=N_grid,
                L_max=L_max,
                levels_per_l=levels_per_l,
                energy_cut=energy_cut,
                gap_factor=gap_factor,
            )
            cost = magic_cost(magic_N)
            results.append(
                {
                    "R_well": float(R_well),
                    "V0": float(V0),
                    "cost": float(cost),
                    "magic_N": magic_N,
                }
            )

    results.sort(key=lambda d: d["cost"])

    print("\n=== Top 10 parameter sets by cost ===")
    for i, res in enumerate(results[:10]):
        print(
            f"{i+1:2d}) R_well={res['R_well']:.2f}, V0={res['V0']:.1f}, "
            f"cost={res['cost']:.3f}, magic_N={res['magic_N']}"
        )

    out_name = "tuned_nuclear_magic_scan.csv"
    with open(out_name, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["R_well", "V0", "cost", "magic_N"])
        for res in results:
            w.writerow(
                [
                    res["R_well"],
                    res["V0"],
                    res["cost"],
                    " ".join(str(n) for n in res["magic_N"]),
                ]
            )

    print(f"\nSaved full scan to {out_name}")


if __name__ == "__main__":
    scan_parameter_grid()
