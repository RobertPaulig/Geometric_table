from __future__ import annotations

import math
import statistics as stats

import core.geom_atoms as geom_atoms
from core.geom_atoms import compute_element_indices
from core.grower import GrowthParams, grow_molecule_christmas_tree
from core.complexity import compute_complexity_features


def scan_dblock_complexity(
    max_depth: int = 4,
    max_atoms: int = 25,
    n_trials: int = 20,
) -> None:
    if hasattr(geom_atoms, "SPECTRAL_MODE_V4"):
        geom_atoms.SPECTRAL_MODE = geom_atoms.SPECTRAL_MODE_V4

    indices = compute_element_indices()
    el_map = {row["El"]: row for row in indices}

    d_block_symbols = sorted(
        {row["El"] for row in indices if 21 <= row["Z"] <= 30},
        key=lambda s: el_map[s]["Z"],
    )

    donor_symbols = []
    hub_symbols = []
    for row in indices:
        Z = row["Z"]
        El = row["El"]
        role = row["role"]
        D = row["D_index"]
        A = row["A_index"]
        if D > 0 and abs(A) < 1e-6:
            donor_symbols.append(El)
        if role == "hub" and abs(A - 1.237) < 1e-3:
            hub_symbols.append(El)

    donor_symbols = sorted(set(donor_symbols), key=lambda s: el_map[s]["Z"])
    hub_symbols = sorted(set(hub_symbols), key=lambda s: el_map[s]["Z"])

    print("[GROUPS]")
    print("d-block:     ", d_block_symbols)
    print("donors:      ", donor_symbols)
    print("living hubs: ", hub_symbols)
    print()

    params = GrowthParams(max_depth=max_depth, max_atoms=max_atoms)

    def measure_for_group(name: str, symbols: list[str]):
        rows = []
        for el in symbols:
            row = el_map[el]
            Z = row["Z"]
            role = row["role"]
            D = row["D_index"]
            A = row["A_index"]

            complexities = []
            sizes = []
            for _ in range(n_trials):
                try:
                    mol = grow_molecule_christmas_tree(el, params)
                    adj = mol.adjacency_matrix()
                    feats = compute_complexity_features(adj)
                    complexities.append(feats.total)
                    sizes.append(len(mol.atoms))
                except Exception:
                    continue

            if not complexities:
                avg_c = float("nan")
                max_c = float("nan")
                avg_size = float("nan")
            else:
                avg_c = stats.mean(complexities)
                max_c = max(complexities)
                avg_size = stats.mean(sizes)

            rows.append(
                dict(
                    group=name,
                    Z=Z,
                    El=el,
                    role=role,
                    D_index=D,
                    A_index=A,
                    Avg_Complexity=avg_c,
                    Max_Complexity=max_c,
                    Avg_Size=avg_size,
                )
            )
        return rows

    all_rows = []
    all_rows += measure_for_group("d_block", d_block_symbols)
    all_rows += measure_for_group("donor", donor_symbols)
    all_rows += measure_for_group("living_hub", hub_symbols)

    print("\n[SUMMARY BY GROUP]")
    by_group: dict[str, list[dict]] = {}
    for r in all_rows:
        by_group.setdefault(r["group"], []).append(r)

    for g, rows in by_group.items():
        cs = [
            r["Avg_Complexity"]
            for r in rows
            if not math.isnan(r["Avg_Complexity"])
        ]
        if not cs:
            continue
        avg_c = stats.mean(cs)
        print(
            f"  group={g:11s}: n={len(rows):2d}, "
            f"avg Avg_Complexity = {avg_c:6.3f}"
        )

    print("\n[D-BLOCK DETAILS]")
    for r in sorted(
        [r for r in all_rows if r["group"] == "d_block"],
        key=lambda x: x["Z"],
    ):
        print(
            f"  Z={r['Z']:2d} {r['El']:>2s} role={r['role']:<9s} "
            f"D={r['D_index']:.4f} A={r['A_index']:.4f} "
            f"Avg_C={r['Avg_Complexity']:.3f} Max_C={r['Max_Complexity']:.3f}"
        )


if __name__ == "__main__":
    scan_dblock_complexity()

