from __future__ import annotations

import argparse
from collections import defaultdict
from typing import Dict, List, Tuple

from analysis.nuclear_cli import apply_nuclear_config_if_provided
from core.geom_atoms import compute_element_indices
from core.nuclear_bands import find_best_N_for_Z

# Woods–Saxon toy magic neutron numbers (from tune_ws_magic.py scan)
WS_MAGIC_N = [8, 20, 40, 112, 142, 184]

# "Living hubs" in geom table
LIVING_HUBS = {"C", "N", "Si", "P", "Ge", "As"}

# Donor sector (terminators + bridges on donor plateau)
DONORS = {"Li", "Na", "K", "Be", "Mg", "Ca", "Rb", "Sr"}


def nearest_magic_N(N_best: int, magic_list: List[int]) -> Tuple[int, int]:
    N_magic = min(magic_list, key=lambda m: abs(m - N_best))
    dN = N_best - N_magic
    return N_magic, dN


def main(argv=None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--nuclear-config",
        type=str,
        default=None,
        help="Path to nuclear config (YAML/JSON); baseline used if omitted.",
    )
    args = parser.parse_args(argv)

    apply_nuclear_config_if_provided(args.nuclear_config)
    rows = compute_element_indices()

    records: List[Tuple[int, str, str, int, int, int]] = []

    for row in rows:
        Z = int(row["Z"])
        el = str(row["El"])

        if Z < 2 or Z > 60:
            continue

        N_min = max(Z, 1)
        N_max = max(int(1.7 * Z), N_min)
        N_best, F_best = find_best_N_for_Z(Z, N_min, N_max)

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
