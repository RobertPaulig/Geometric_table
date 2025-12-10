from __future__ import annotations

from typing import List, Dict, Tuple

from nuclear_island import nuclear_functional


# Набор реально стабильных (или очень долгоживущих) ядер
NUCLIDES: List[Dict[str, int]] = [
    # лёгкие/средние
    {"label": "O-16",  "Z": 8,  "N": 8},
    {"label": "Ne-20", "Z": 10, "N": 10},
    {"label": "Mg-24", "Z": 12, "N": 12},
    {"label": "Si-28", "Z": 14, "N": 14},
    {"label": "Ca-40", "Z": 20, "N": 20},
    {"label": "Fe-56", "Z": 26, "N": 30},
    {"label": "Ni-58", "Z": 28, "N": 30},
    {"label": "Ni-60", "Z": 28, "N": 32},

    # средне-тяжёлые
    {"label": "Zr-90",  "Z": 40, "N": 50},
    {"label": "Mo-96",  "Z": 42, "N": 54},
    {"label": "Sn-120", "Z": 50, "N": 70},
    {"label": "Sn-118", "Z": 50, "N": 68},

    # тяжёлые
    {"label": "Xe-132", "Z": 54, "N": 78},
    {"label": "Sm-144", "Z": 62, "N": 82},
    {"label": "Sm-150", "Z": 62, "N": 88},
    {"label": "Pb-208", "Z": 82, "N": 126},
]


def find_best_N_local(
    Z: int,
    N_center: int,
    window: int = 20,
    lambda_shell: float = 30.0,
    sigma_p: float = 6.0,
    sigma_n: float = 8.0,
    a_p: float = 12.0,
) -> Tuple[int, float]:
    """
    Локальный поиск минимума F_nuc по N в окне [N_center-window, N_center+window].
    Возвращает (N_best, F_min).
    """
    N_min = max(1, N_center - window)
    N_max = max(N_min + 1, N_center + window)

    best_N = N_min
    best_F: float | None = None

    for N in range(N_min, N_max + 1):
        F = nuclear_functional(
            Z,
            N,
            lambda_shell=lambda_shell,
            sigma_p=sigma_p,
            sigma_n=sigma_n,
            a_p=a_p,
        )
        if best_F is None or F < best_F:
            best_F = F
            best_N = N

    assert best_F is not None
    return best_N, float(best_F)


def main() -> None:
    print("Comparing nuclear_functional v0.2 to a set of stable nuclides\n")

    rows: List[Dict[str, float]] = []

    for nucl in NUCLIDES:
        label = nucl["label"]
        Z = nucl["Z"]
        N_real = nucl["N"]

        # Энергия в реальной точке (по нашему функционалу)
        F_real = nuclear_functional(Z, N_real)

        # Локальный минимум вокруг реального N
        N_best, F_best = find_best_N_local(Z, N_real, window=20)

        dN = N_best - N_real
        dF = F_best - F_real

        rows.append(
            {
                "Z": float(Z),
                "N_real": float(N_real),
                "N_best": float(N_best),
                "dN": float(dN),
                "F_real": float(F_real),
                "F_best": float(F_best),
                "dF": float(dF),
                "label": label,
            }
        )

    # Печатаем таблицу
    print(
        f"{'Nuclide':>8s}  {'Z':>3s}  {'N_real':>6s}  "
        f"{'N_best':>6s}  {'dN':>4s}  {'F_real':>10s}  {'F_best':>10s}  {'dF':>8s}"
    )
    for r in rows:
        print(
            f"{r['label']:>8s}  "
            f"{int(r['Z']):3d}  "
            f"{int(r['N_real']):6d}  "
            f"{int(r['N_best']):6d}  "
            f"{int(r['dN']):4d}  "
            f"{r['F_real']:10.2f}  "
            f"{r['F_best']:10.2f}  "
            f"{r['dF']:8.2f}"
        )

    # Грубая статистика по |dN|
    abs_errors = [abs(r["dN"]) for r in rows]
    mean_abs_dN = sum(abs_errors) / len(abs_errors) if abs_errors else 0.0
    max_abs_dN = max(abs_errors) if abs_errors else 0.0

    print("\n[SUMMARY]")
    print(f"  mean |dN| = {mean_abs_dN:.2f}")
    print(f"  max  |dN| = {max_abs_dN:.2f}")


if __name__ == "__main__":
    main()

