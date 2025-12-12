from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Tuple

from core.geom_atoms import compute_element_indices
from core.nuclear_island import nuclear_functional

# Woods–Saxon toy magic neutron numbers (from tune_ws_magic.py scan)
WS_MAGIC_N = [8, 20, 40, 112, 142, 184]

# "Living hubs" in geom table
LIVING_HUBS = {"C", "N", "Si", "P", "Ge", "As"}

# Donor sector (terminators + bridges on donor plateau)
DONORS = {"Li", "Na", "K", "Be", "Mg", "Ca", "Rb", "Sr"}


def find_best_N_for_Z(
    Z: int,
    N_min: int | None = None,
    N_max: int | None = None,
    lambda_shell: float = 30.0,
    sigma_p: float = 6.0,
    sigma_n: float = 8.0,
    a_p: float = 12.0,
) -> Tuple[int | None, float | None]:
    if N_min is None:
        N_min = max(Z, 1)
    if N_max is None:
        N_max = max(int(1.7 * Z), N_min)

    best_N = None
    best_F = float("inf")

    for N in range(N_min, N_max + 1):
        if (Z + N) % 2 != 0:
            continue
        F = nuclear_functional(
            Z,
            N,
            lambda_shell=lambda_shell,
            sigma_p=sigma_p,
            sigma_n=sigma_n,
            a_p=a_p,
        )
        if F < best_F:
            best_F = F
            best_N = N

    if best_N is None:
        return None, None
    return best_N, best_F


def nearest_magic_N(N_best: int, magic_list: List[int]) -> Tuple[int, int]:
    N_magic = min(magic_list, key=lambda m: abs(m - N_best))
    dN = N_best - N_magic
    return N_magic, dN


def main() -> None:
    rows = compute_element_indices()

    records: List[Tuple[int, str, str, int, int, int]] = []

    for row in rows:
        Z = int(row["Z"])
        el = str(row["El"])

        if Z < 2 or Z > 60:
            continue

        N_best, F_best = find_best_N_for_Z(Z)
        if N_best is None:
            continue

        N_magic, dN = nearest_magic_N(N_best, WS_MAGIC_N)

        if el in LIVING_HUBS:
            group = "living_hub"
        elif el in DONORS:
            group = "donor"
        else:
            group = "other"

        records.append((Z, el, group, N_best, N_magic, dN))

    print("=== GEOM vs WS magic N ===")
    print("Z  El  group         N_best  N_magic  dN")
    print("--------------------------------------------")
    for Z, el, group, N_best, N_magic, dN in sorted(records, key=lambda r: r[0]):
        print(
            f"{Z:2d}  {el:2s}  {group:11s}  "
            f"{N_best:6d}  {N_magic:7d}  {dN:3d}"
        )

    by_group: Dict[str, List[int]] = defaultdict(list)
    for _, _, group, _, _, dN in records:
        by_group[group].append(abs(dN))

    print("\n[SUMMARY |N_best - N_magic| by group]")
    for group in ("living_hub", "donor", "other"):
        vals = by_group.get(group, [])
        if not vals:
            continue
        avg_abs = sum(vals) / len(vals)
        print(f"  {group:11s}: n={len(vals):2d}, avg |dN| = {avg_abs:5.2f}")


if __name__ == "__main__":
    main()
