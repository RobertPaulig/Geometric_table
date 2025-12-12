from __future__ import annotations

import argparse
import csv
from typing import Any, Dict, List

from analysis.nuclear_cli import apply_nuclear_config_if_provided
from core.geom_atoms import compute_element_indices
from core.nuclear_bands import find_best_N_for_Z


def map_geom_to_valley(
    Z_min: int = 1,
    Z_max: int = 40,
    N_corridor_factor: float = 1.8,
) -> List[Dict[str, Any]]:
    """
    Для каждого элемента из compute_element_indices() с Z_min <= Z <= Z_max
    ищем N_best(Z) — минимум F_nuc — в коридоре
        N ∈ [Z, int(N_corridor_factor * Z)].

    Возвращает список словарей с полями:
      Z, El, role, period, chi_spec, E_port, D_index, A_index,
      N_best, A_best, N_over_Z, F_min, living_hub.
    """
    indices = compute_element_indices()

    results: List[Dict[str, Any]] = []

    for row in indices:
        Z = int(row["Z"])
        if Z < Z_min or Z > Z_max:
            continue

        # Коридор по N: от симметричного N≈Z до умеренно нейтронобогатого
        N_min = Z
        N_max = max(Z + 1, int(N_corridor_factor * Z))

        N_best, F_min = find_best_N_for_Z(Z, N_min, N_max)

        A_best = Z + N_best
        N_over_Z = float(N_best) / float(Z)

        D_index = float(row.get("D_index", 0.0))
        A_index = float(row.get("A_index", 0.0))
        role = str(row.get("role", ""))

        # "Живой хаб": hub с сильным акцепторным плато A_index ~ 1.237
        living_hub = (
            role == "hub"
            and A_index > 0.5
            and abs(A_index - 1.237) < 1e-3
        )

        result_row = dict(row)
        result_row.update(
            {
                "N_best": N_best,
                "A_best": A_best,
                "N_over_Z": N_over_Z,
                "F_min": F_min,
                "living_hub": living_hub,
            }
        )
        results.append(result_row)

    return results


def save_geom_valley_csv(
    path: str = "geom_nuclear_map.csv",
    **kwargs,
) -> List[Dict[str, Any]]:
    """
    Запустить map_geom_to_valley() и сохранить результат в CSV.
    Возвращает список строк (для дальнейшего анализа).
    """
    rows = map_geom_to_valley(**kwargs)

    if not rows:
        print("No rows produced; check Z_min/Z_max and base_atoms.")
        return rows

    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    print(f"Saved {len(rows)} rows to {path}")
    return rows


def print_summary(rows: List[Dict[str, Any]]) -> None:
    """
    Компактная сводка: живые хабы и доноры.
    """
    print("\n[GEOM–NUCLEAR MAP SUMMARY]")

    # Живые хабы
    hubs = [r for r in rows if r.get("living_hub")]
    if hubs:
        print("\nLiving hubs (A_index ≈ 1.237):")
        for r in hubs:
            print(
                f"  Z={int(r['Z']):2d} {str(r['El']):>2s}  "
                f"role={str(r['role']):<9}  "
                f"N_best={int(r['N_best']):3d}  "
                f"A_best={int(r['A_best']):3d}  "
                f"N/Z={float(r['N_over_Z']):.2f}  "
                f"D={float(r['D_index']):.4f}  "
                f"A={float(r['A_index']):.4f}"
            )
    else:
        print("\nNo living hubs found in the specified Z-range.")

    # Донорный сектор
    donors = [r for r in rows if float(r.get("D_index", 0.0)) > 0.0]
    if donors:
        print("\nDonor sector (D_index > 0):")
        for r in donors:
            print(
                f"  Z={int(r['Z']):2d} {str(r['El']):>2s}  "
                f"role={str(r['role']):<9}  "
                f"N_best={int(r['N_best']):3d}  "
                f"A_best={int(r['A_best']):3d}  "
                f"N/Z={float(r['N_over_Z']):.2f}  "
                f"D={float(r['D_index']):.4f}  "
                f"A={float(r['A_index']):.4f}"
            )


def main(argv=None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--nuclear-config",
        type=str,
        default=None,
        help=(
            "Path to nuclear shell config (YAML/JSON). "
            "If omitted, baseline NuclearConfig is used."
        ),
    )
    parser.add_argument(
        "--z-min",
        type=int,
        default=1,
        help="Minimal Z for geom–nuclear mapping (default: 1).",
    )
    parser.add_argument(
        "--z-max",
        type=int,
        default=40,
        help="Maximum Z for geom–nuclear mapping (default: 40).",
    )
    args = parser.parse_args(argv)

    apply_nuclear_config_if_provided(args.nuclear_config)

    rows = save_geom_valley_csv(
        path="geom_nuclear_map.csv",
        Z_min=args.z_min,
        Z_max=args.z_max,
    )
    if rows:
        print_summary(rows)


if __name__ == "__main__":
    main()
