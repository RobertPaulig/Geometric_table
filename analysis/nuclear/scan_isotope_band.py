from __future__ import annotations

import argparse
import csv
from typing import Any, Dict, List

from analysis.cli_common import script_banner
from core.geom_atoms import compute_element_indices
from core.nuclear_bands import scan_isotope_band_for_Z, make_default_corridor
from analysis.io_utils import data_path
from analysis.nuclear_cli import apply_nuclear_config_if_provided


def scan_isotope_bands(
    Z_min: int = 1,
    Z_max: int = 40,
    delta_F: float = 5.0,
    N_corridor_factor: float = 1.8,
) -> List[Dict[str, Any]]:
    """
    Для каждого элемента (Z) из compute_element_indices() в диапазоне
    [Z_min, Z_max] считаем:
      - N_best, F_min
      - список 'почти стабильных' N (F <= F_min + delta_F)
      - ширину полосы: count_N = len(allowed_N)
    """
    indices = compute_element_indices()
    results: List[Dict[str, Any]] = []

    for row in indices:
        Z = int(row["Z"])
        if Z < Z_min or Z > Z_max:
            continue

        N_min, N_max = make_default_corridor(Z, factor=N_corridor_factor)

        # базовый скан по F_nuc для данного Z
        band_points = scan_isotope_band_for_Z(Z, N_min, N_max)
        if not band_points:
            continue

        # ищем минимум и допустимую полосу по delta_F
        F_values = {p.N: p.F for p in band_points}
        N_best = min(F_values, key=F_values.get)
        F_min = F_values[N_best]
        threshold = F_min + delta_F
        allowed_N = sorted(N for N, F in F_values.items() if F <= threshold)

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


def save_isotope_bands_csv(path: str | None = None, **kwargs) -> List[Dict[str, Any]]:
    rows = scan_isotope_bands(**kwargs)
    if not rows:
        print("No rows produced.")
        return rows

    fieldnames = list(rows[0].keys())
    if path is None:
        path = data_path("geom_isotope_bands.csv")
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


def main(argv=None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--nuclear-config",
        type=str,
        default=None,
        help="Path to nuclear-shell config (YAML/JSON); baseline used if omitted.",
    )
    parser.add_argument(
        "--z-min",
        type=int,
        default=1,
        help="Minimal Z to include in isotope band scan.",
    )
    parser.add_argument(
        "--z-max",
        type=int,
        default=40,
        help="Maximum Z to include in isotope band scan.",
    )
    args = parser.parse_args(argv)

    apply_nuclear_config_if_provided(args.nuclear_config)

    with script_banner("scan_isotope_band"):
        rows = save_isotope_bands_csv(
            path="data/geom_isotope_bands.csv",
            Z_min=args.z_min,
            Z_max=args.z_max,
            delta_F=5.0,
            N_corridor_factor=1.8,
        )
        if rows:
            print_band_summary(rows)


if __name__ == "__main__":
    main()
