from __future__ import annotations

from collections import defaultdict
from typing import List, Tuple, Dict, Any

from core.geom_atoms import compute_element_indices
from core.nuclear_island import nuclear_functional

WS_MAGIC_N = [8, 20, 40, 112, 142, 184]
LIVING_HUBS = {"C", "N", "Si", "P", "Ge", "As"}
DONORS = {"Li", "Na", "K", "Be", "Mg", "Ca", "Rb", "Sr"}


def compute_isotope_band(
    Z: int,
    delta_F: float = 5.0,
    lambda_shell: float = 30.0,
    sigma_p: float = 6.0,
    sigma_n: float = 8.0,
    a_p: float = 12.0,
) -> Tuple[int, int, int]:
    N_min_scan = max(Z, 1)
    N_max_scan = max(int(1.7 * Z), N_min_scan)

    F_min = float("inf")
    N_best = N_min_scan
    values: List[Tuple[int, float]] = []

    for N in range(N_min_scan, N_max_scan + 1):
        F = nuclear_functional(
            Z,
            N,
            lambda_shell=lambda_shell,
            sigma_p=sigma_p,
            sigma_n=sigma_n,
            a_p=a_p,
        )
        values.append((N, F))
        if F < F_min:
            F_min = F
            N_best = N

    allowed = [N for (N, F) in values if F <= F_min + delta_F]
    if not allowed:
        return N_best, N_best, N_best

    band_N_min = min(allowed)
    band_N_max = max(allowed)
    return band_N_min, band_N_max, N_best


def main() -> None:
    rows = compute_element_indices()
    delta_F = 5.0

    records: List[Dict[str, Any]] = []

    for row in rows:
        Z = int(row["Z"])
        El = str(row["El"])
        role = str(row.get("role", ""))

        if Z < 2 or Z > 60:
            continue

        band_N_min, band_N_max, N_best = compute_isotope_band(Z, delta_F=delta_F)

        hits = [N for N in WS_MAGIC_N if band_N_min <= N <= band_N_max]
        n_hits = len(hits)

        if El in LIVING_HUBS:
            group = "living_hub"
        elif El in DONORS:
            group = "donor"
        else:
            group = "other"

        records.append(
            {
                "Z": Z,
                "El": El,
                "role": role,
                "group": group,
                "N_min": band_N_min,
                "N_max": band_N_max,
                "N_best": N_best,
                "hits": n_hits,
                "hit_list": hits,
            }
        )

    print(f"=== GEOM isotope bands vs WS magic N (delta_F = {delta_F}) ===")
    print("Z  El  role        group         N_min  N_best  N_max   hits  magic_in_band")
    print("--------------------------------------------------------------------------")
    for rec in sorted(records, key=lambda r: r["Z"]):
        Z = rec["Z"]
        El = rec["El"]
        role = rec["role"]
        group = rec["group"]
        N_min = rec["N_min"]
        N_max = rec["N_max"]
        N_best = rec["N_best"]
        hits = rec["hits"]
        hit_list = rec["hit_list"]
        hits_str = ",".join(str(h) for h in hit_list) if hit_list else "-"
        print(
            f"{Z:2d}  {El:2s}  {role:10s}  {group:11s}  "
            f"{N_min:5d}  {N_best:6d}  {N_max:5d}   {hits:3d}   {hits_str}"
        )

    by_group: Dict[str, List[int]] = defaultdict(list)
    for rec in records:
        by_group[rec["group"]].append(rec["hits"])

    print("\n[SUMMARY hits per group (WS magic in band)]")
    for group in ("living_hub", "donor", "other"):
        vals = by_group.get(group, [])
        if not vals:
            continue
        avg_hits = sum(vals) / len(vals)
        nonzero = sum(1 for v in vals if v > 0)
        print(
            f"  {group:11s}: n={len(vals):2d}, "
            f"avg hits = {avg_hits:4.2f}, "
            f"fraction with ≥1 hit = {nonzero}/{len(vals)}"
        )


if __name__ == "__main__":
    main()
