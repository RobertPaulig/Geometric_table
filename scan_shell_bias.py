from __future__ import annotations

from typing import List, Dict, Tuple

from nuclear_island import nuclear_functional


# Тот же набор стабильных ядер, что и в compare_nuclear_v02_to_experiment.py
NUCLIDES: List[Dict[str, int]] = [
    {"label": "O-16",   "Z": 8,  "N": 8},
    {"label": "Ne-20",  "Z": 10, "N": 10},
    {"label": "Mg-24",  "Z": 12, "N": 12},
    {"label": "Si-28",  "Z": 14, "N": 14},
    {"label": "Ca-40",  "Z": 20, "N": 20},
    {"label": "Fe-56",  "Z": 26, "N": 30},
    {"label": "Ni-58",  "Z": 28, "N": 30},
    {"label": "Ni-60",  "Z": 28, "N": 32},
    {"label": "Zr-90",  "Z": 40, "N": 50},
    {"label": "Mo-96",  "Z": 42, "N": 54},
    {"label": "Sn-120", "Z": 50, "N": 70},
    {"label": "Sn-118", "Z": 50, "N": 68},
    {"label": "Xe-132", "Z": 54, "N": 78},
    {"label": "Sm-144", "Z": 62, "N": 82},
    {"label": "Sm-150", "Z": 62, "N": 88},
    {"label": "Pb-208", "Z": 82, "N": 126},
]


def find_best_N_local_mode(
    Z: int,
    N_center: int,
    window: int,
    mode: str,
    lambda_shell_base: float = 30.0,
    alpha: float = 1.0,
) -> Tuple[int, float]:
    """
    Локальный поиск минимума F_nuc по N в окне [N_center-window, N_center+window]
    в разных режимах оболочки.

    mode:
      - "no_shell"   : lambda_shell = 0
      - "weak_shell" : lambda_shell = 10
      - "base_shell" : lambda_shell = lambda_shell_base
      - "Adep_shell" : lambda_shell = lambda_shell_base * (A/200)^alpha
    """
    N_min = max(1, N_center - window)
    N_max = max(N_min + 1, N_center + window)

    best_N = N_min
    best_F: float | None = None

    for N in range(N_min, N_max + 1):
        A = Z + N

        if mode == "no_shell":
            lam = 0.0
        elif mode == "weak_shell":
            lam = 10.0
        elif mode == "base_shell":
            lam = lambda_shell_base
        elif mode == "Adep_shell":
            lam = lambda_shell_base * (A / 200.0) ** alpha
        else:
            raise ValueError(f"Unknown mode: {mode}")

        F = nuclear_functional(
            Z,
            N,
            lambda_shell=lam,
            sigma_p=6.0,
            sigma_n=8.0,
            a_p=12.0,
        )

        if best_F is None or F < best_F:
            best_F = F
            best_N = N

    assert best_F is not None
    return best_N, float(best_F)


def run_scan(window: int = 20) -> None:
    modes = ["no_shell", "weak_shell", "base_shell", "Adep_shell"]

    # Для Adep_shell выберем alpha=1.0 для начала
    lambda_shell_base = 30.0
    alpha = 1.0

    # Храним ошибки по модам
    errors: Dict[str, List[float]] = {m: [] for m in modes}

    print(
        f"{'Nuclide':>8s}  {'Z':>3s}  {'N_real':>6s}  "
        f"{'mode':>10s}  {'N_best':>6s}  {'dN':>4s}"
    )

    for nucl in NUCLIDES:
        label = nucl["label"]
        Z = nucl["Z"]
        N_real = nucl["N"]

        for mode in modes:
            N_best, _ = find_best_N_local_mode(
                Z,
                N_real,
                window=window,
                mode=mode,
                lambda_shell_base=lambda_shell_base,
                alpha=alpha,
            )
            dN = N_best - N_real
            errors[mode].append(abs(dN))

            print(
                f"{label:>8s}  {Z:3d}  {N_real:6d}  "
                f"{mode:>10s}  {N_best:6d}  {dN:4d}"
            )

    # Сводка по модам
    print("\n[SUMMARY |dN| by mode]")
    for mode in modes:
        errs = errors[mode]
        if not errs:
            continue
        mean_abs = sum(errs) / len(errs)
        max_abs = max(errs)
        print(
            f"  {mode:10s}: mean |dN| = {mean_abs:5.2f}, "
            f"max |dN| = {max_abs:5.2f}"
        )


def main() -> None:
    run_scan(window=20)


if __name__ == "__main__":
    main()

