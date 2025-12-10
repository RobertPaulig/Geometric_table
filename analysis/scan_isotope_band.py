from __future__ import annotations

import csv
from typing import Dict, Any, List, Tuple

from core.geom_atoms import compute_element_indices
from core.nuclear_island import nuclear_functional


def scan_isotope_band_for_Z(
    Z: int,
    N_min: int,
    N_max: int,
    delta_F: float = 5.0,
    lambda_shell: float = 30.0,
    sigma_p: float = 6.0,
    sigma_n: float = 8.0,
    a_p: float = 12.0,
) -> Tuple[int, float, List[int]]:
    """
    Для фиксированного Z:
      - ищем минимум F_nuc по N в [N_min, N_max]
      - считаем все N, у которых F_nuc <= F_min + delta_F

    Возвращает:
      (N_best, F_min, allowed_N_list)
    """
    best_N = N_min
    best_F: float | None = None
    values: Dict[int, float] = {}

    for N in range(N_min, N_max + 1):
        F = nuclear_functional(
            Z,
            N,
            lambda_shell=lambda_shell,
            sigma_p=sigma_p,
            sigma_n=sigma_n,
            a_p=a_p,
        )
        values[N] = F
        if best_F is None or F < best_F:
            best_F = F
            best_N = N

    assert best_F is not None
    threshold = best_F + delta_F

    allowed_N = [N for N, F in values.items() if F <= threshold]
    allowed_N.sort()
    return best_N, float(best_F), allowed_N


def scan_isotope_bands(
    Z_min: int = 1,
    Z_max: int = 40,
    delta_F: float = 5.0,
    N_corridor_factor: float = 1.8,
    lambda_shell: float = 30.0,
    sigma_p: float = 6.0,
    sigma_n: float = 8.0,
    a_p: float = 12.0,
) -> List[Dict[str, Any]]:
    """
    Для каждого элемента (Z) из compute_element_indices() в диапазоне
    [Z_min, Z_max] считаем:
      - N_best, F_min (как в map_geom_to_valley)
      - список 'почти стабильных' N (F <= F_min + delta_F)
      - ширину полосы: count_N = len(allowed_N)
    """
    indices = compute_element_indices()
    results: List[Dict[str, Any]] = []

    for row in indices:
        Z = int(row["Z"])
        if Z < Z_min or Z > Z_max:
            continue

        N_min = Z
        N_max = max(Z + 1, int(N_corridor_factor * Z))

        N_best, F_min, allowed_N = scan_isotope_band_for_Z(
            Z,
            N_min,
            N_max,
            delta_F=delta_F,
            lambda_shell=lambda_shell,
            sigma_p=sigma_p,
            sigma_n=sigma_n,
            a_p=a_p,
        )

        A_best = Z + N_best
        N_over_Z = float(N_best) / float(Z)
        band_width = len(allowed_N)

        D_index = float(row.get("D_index", 0.0))
        A_index = float(row.get("A_index", 0.0))
        role = str(row.get("role", ""))

        living_hub = (
            role == "hub"
            and A_index > 0.5
            and abs(A_index - 1.237) < 1e-3
        )

        result_row: Dict[str, Any] = dict(row)
        result_row.update(
            {
                "N_min": N_min,
                "N_max": N_max,
                "N_best": N_best,
                "A_best": A_best,
                "N_over_Z": N_over_Z,
                "F_min": F_min,
                "band_width": band_width,
                "allowed_N": ";".join(str(N) for N in allowed_N),
                "living_hub": living_hub,
            }
        )
        results.append(result_row)

    return results


def save_isotope_bands_csv(
    path: str = "geom_isotope_bands.csv",
    **kwargs,
) -> List[Dict[str, Any]]:
    rows = scan_isotope_bands(**kwargs)
    if not rows:
        print("No rows produced.")
        return rows

    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    print(f"Saved {len(rows)} rows to {path}")
    return rows


def print_band_summary(rows: List[Dict[str, Any]]) -> None:
    print("\n[ISOTOPE BANDS SUMMARY]")

    # Живые хабы
    hubs = [r for r in rows if r.get("living_hub")]
    if hubs:
        print("\nLiving hubs:")
        for r in hubs:
            print(
                f"  Z={int(r['Z']):2d} {str(r['El']):>2s}  "
                f"role={str(r['role']):<9}  "
                f"band_width={int(r['band_width']):2d}  "
                f"N_best={int(r['N_best']):3d}  "
                f"N_min={int(r['N_min']):3d}  "
                f"N_max={int(r['N_max']):3d}  "
                f"D={float(r['D_index']):.4f}  "
                f"A={float(r['A_index']):.4f}"
            )
    else:
        print("\nNo living hubs in Z-range.")

    # Доноры
    donors = [r for r in rows if float(r.get("D_index", 0.0)) > 0.0]
    if donors:
        print("\nDonors (D_index > 0):")
        for r in donors:
            print(
                f"  Z={int(r['Z']):2d} {str(r['El']):>2s}  "
                f"role={str(r['role']):<9}  "
                f"band_width={int(r['band_width']):2d}  "
                f"N_best={int(r['N_best']):3d}  "
                f"N_min={int(r['N_min']):3d}  "
                f"N_max={int(r['N_max']):3d}  "
                f"D={float(r['D_index']):.4f}  "
                f"A={float(r['A_index']):.4f}"
            )


def main() -> None:
    rows = save_isotope_bands_csv(
        path="geom_isotope_bands.csv",
        Z_min=1,
        Z_max=40,
        delta_F=5.0,       # порог 'почти стабильных' по F
        N_corridor_factor=1.8,
    )
    if rows:
        print_band_summary(rows)


if __name__ == "__main__":
    main()

