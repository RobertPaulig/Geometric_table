import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from math import log2
from pathlib import Path

from core.geom_atoms import get_atom, SPECTRAL_MODE_V4, SPECTRAL_MODE, compute_element_indices
import core.geom_atoms as geom_atoms
from core.grower import GrowthParams, grow_molecule_christmas_tree
from core.complexity import compute_complexity_features, compute_complexity_features_v2

# Ensure V4 mode
geom_atoms.SPECTRAL_MODE = geom_atoms.SPECTRAL_MODE_V4

def run_living_sectors_scan():
    print("Starting Living Sectors Scan (Complexity vs D/A)...")
    
    # 1. Map existing elements to D/A
    indices = compute_element_indices()
    # Create a lookup: Element -> (D, A)
    el_map = {item['El']: (item['D_index'], item['A_index'], item['role']) for item in indices}
    
    # 2. Define Representative Sets (Seeds)
    # We want to enable growth for various "Hubs" using a standard "Terminator" pool.
    # Or test pair combinations: Hub + Terminator
    
    # Let's fix a "Soup" of standard terminators (H, Li, F)
    # And vary the ROOT element to see which roots produce complex trees.
    # OR: vary the "Environment" available for growth.
    
    # Simplified experiment:
    # Root = Element X (from across the table)
    # Pool = Standard organic mix (H, C, N, O) + X?
    # No, let's strictly follow the law text:
    # "Hubs from Strong Acceptor Plateau" + "Terminators from Donor Plateau" -> Complex
    
    # We will iterate over ALL elements Z=1..36 as ROOTS.
    # We will grow trees using a common "Feedstock" of [H, Li, F, Na, K, Cl] 
    # to serve as terminators, plus the Root element itself allowed in the body.
    
    # Params
    params = GrowthParams(max_depth=4, max_atoms=25)
    n_trials = 20

    results = []

    elements_to_scan = [e["El"] for e in indices if e["Z"] <= 36 and e["role"] != "inert"]

    for el in elements_to_scan:
        d_val, a_val, role = el_map[el]

        # Collect stats per element
        c_v1_list = []
        c_fdm_list = []
        c_norm_v1_list = []
        c_norm_fdm_list = []
        sizes = []

        for i_trial in range(n_trials):
            try:
                mol = grow_molecule_christmas_tree(el, params)
                adj = mol.adjacency_matrix()

                feats_v1 = compute_complexity_features(adj)
                feats_fdm = compute_complexity_features_v2(adj, backend="fdm")

                n = feats_v1.n
                if n <= 1:
                    continue

                c_v1 = feats_v1.total
                c_fdm = feats_fdm.total
                denom = n * log2(1.0 + n)

                c_v1_list.append(c_v1)
                c_fdm_list.append(c_fdm)
                c_norm_v1_list.append(c_v1 / denom)
                c_norm_fdm_list.append(c_fdm / denom)
                sizes.append(n)

                if el == "C" and i_trial < 2:
                    print(
                        f"DEBUG: {el} Trial {i_trial}: "
                        f"Atoms={n}, C_v1={c_v1:.3f}, C_fdm={c_fdm:.3f}"
                    )

            except Exception as e:
                if i_trial == 0:
                    print(f"DEBUG: Growth failed for {el}: {e}")
                    import traceback

                    traceback.print_exc()
                continue

        if not c_v1_list:
            avg_c = 0.0
            max_c = 0.0
            avg_s = 0.0
            c_v1_mean = c_v1_std = 0.0
            c_fdm_mean = c_fdm_std = 0.0
            c_norm_v1_mean = c_norm_v1_std = 0.0
            c_norm_fdm_mean = c_norm_fdm_std = 0.0
        else:
            c_v1_arr = np.array(c_v1_list, dtype=float)
            c_fdm_arr = np.array(c_fdm_list, dtype=float)
            c_norm_v1_arr = np.array(c_norm_v1_list, dtype=float)
            c_norm_fdm_arr = np.array(c_norm_fdm_list, dtype=float)
            sizes_arr = np.array(sizes, dtype=float)

            avg_c = float(c_v1_arr.mean())
            max_c = float(c_v1_arr.max())
            avg_s = float(sizes_arr.mean()) if sizes else 0.0

            c_v1_mean = float(c_v1_arr.mean())
            c_v1_std = float(c_v1_arr.std())
            c_fdm_mean = float(c_fdm_arr.mean())
            c_fdm_std = float(c_fdm_arr.std())
            c_norm_v1_mean = float(c_norm_v1_arr.mean())
            c_norm_v1_std = float(c_norm_v1_arr.std())
            c_norm_fdm_mean = float(c_norm_fdm_arr.mean())
            c_norm_fdm_std = float(c_norm_fdm_arr.std())

        results.append(
            {
                "Element": el,
                "D": d_val,
                "A": a_val,
                "Role": role,
                # legacy fields (v1)
                "Avg_Complexity": avg_c,
                "Max_Complexity": max_c,
                "Avg_Size": avg_s,
                # explicit v1/FDM stats
                "C_total_v1_mean": c_v1_mean,
                "C_total_v1_std": c_v1_std,
                "C_total_fdm_mean": c_fdm_mean,
                "C_total_fdm_std": c_fdm_std,
                "C_norm_v1_mean": c_norm_v1_mean,
                "C_norm_v1_std": c_norm_v1_std,
                "C_norm_fdm_mean": c_norm_fdm_mean,
                "C_norm_fdm_std": c_norm_fdm_std,
            }
        )

    df = pd.DataFrame(results)
    print("Scan complete.")
    print(df.sort_values("Max_Complexity", ascending=False).head(10))

    # Save complexity summary for downstream analysis
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    out_csv = data_dir / "complexity_summary.csv"
    df.to_csv(out_csv, index=False)
    print(f"Saved complexity summary to {out_csv}")
    
    # PLOTTING
    plt.figure(figsize=(10, 8))
    
    # Scatter plot: x=D, y=A, size/color = Complexity
    # Normalize complexity for sizing
    s_min, s_max = 50, 500
    c_vals = df["Max_Complexity"]
    if c_vals.max() > c_vals.min():
        sizes = s_min + (c_vals - c_vals.min()) / (c_vals.max() - c_vals.min()) * (s_max - s_min)
    else:
        sizes = [s_min] * len(df)
        
    sc = plt.scatter(
        df["D"], df["A"], 
        s=sizes, c=df["Max_Complexity"], 
        cmap="viridis", alpha=0.8, edgecolors='gray'
    )
    plt.colorbar(sc, label="Max Complexity")
    
    # Annotate elements
    for line in range(0, df.shape[0]):
        row = df.iloc[line]
        if row["Max_Complexity"] > 1.0 or row["Element"] in ["Li", "F", "He"]:
            plt.text(row["D"]+0.01, row["A"]+0.01, row["Element"], 
                     horizontalalignment='left', size='small', color='black')

    plt.title("Living Sectors: Graph Complexity vs D/A Indices")
    plt.xlabel("Donor Index D")
    plt.ylabel("Acceptor Index A")
    plt.grid(True, linestyle='--', alpha=0.5)
    
    # Highlight Sectors
    # Donor Plateau: D ~ 0.2-0.3, A ~ 0
    # Acceptor Plateau: A ~ 1.2, D ~ 0
    # Amphoteric: A ~ 0.1, D ~ 0
    
    plt.savefig("living_sectors_scan.png")
    print("Saved living_sectors_scan.png")

if __name__ == "__main__":
    run_living_sectors_scan()
