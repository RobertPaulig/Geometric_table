from __future__ import annotations

from core.nuclear_island import nuclear_functional


def find_best_N_for_Z(
    Z: int,
    N_min: int,
    N_max: int,
    **kwargs,
):
    best_F = None
    best_N = None
    for N in range(N_min, N_max + 1):
        F = nuclear_functional(Z, N, **kwargs)
        if (best_F is None) or (F < best_F):
            best_F = F
            best_N = N
    return best_N, best_F


def main():
    # лёгкие и средние ядра: от O до Pb
    for Z in range(8, 83):
        N_min = Z
        N_max = int(Z * 1.7)
        N_best, F_best = find_best_N_for_Z(Z, N_min, N_max)
        A = Z + (N_best or 0)
        ratio = (N_best / Z) if N_best is not None else 0.0
        print(
            f"Z={Z:3d}: N_best={N_best:3d}, "
            f"N/Z={ratio:5.2f}, A={A:4d}, F={F_best:8.2f}"
        )


if __name__ == "__main__":
    main()
