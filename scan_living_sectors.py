
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from geom_atoms import get_atom, SPECTRAL_MODE_V4, SPECTRAL_MODE, compute_element_indices
import geom_atoms
from grower import GrowthParams, grow_molecule_christmas_tree
from complexity import compute_complexity_features

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
    
    elements_to_scan = [e['El'] for e in indices if e['Z'] <= 36 and e['role'] != 'inert']
    
    for el in elements_to_scan:
        d_val, a_val, role = el_map[el]
        
        # Collect stats
        complexities = []
        sizes = []
        for i_trial in range(n_trials):
            # Grow tree starting from 'el'
            try:
                mol = grow_molecule_christmas_tree(el, params)
                # Compute C_graph
                adj = mol.adjacency_matrix()
                feats = compute_complexity_features(adj)
                complexities.append(feats.total)
                sizes.append(len(mol.atoms))
                
                # Debug print for first few trials of first element
                if el == "C" and i_trial < 2:
                    print(f"DEBUG: {el} Trial {i_trial}: Atoms={len(mol.atoms)}, Complexity={feats.total:.3f}")
                    
            except Exception as e:
                # Growth failed or 0 complexity
                complexities.append(0.0)
                if i_trial == 0:
                   print(f"DEBUG: Growth failed for {el}: {e}")
                   import traceback
                   traceback.print_exc()
        
        avg_c = np.mean(complexities)
        max_c = np.max(complexities)
        avg_s = np.mean(sizes) if sizes else 0.0
        
        results.append({
            "Element": el,
            "D": d_val,
            "A": a_val,
            "Role": role,
            "Avg_Complexity": avg_c,
            "Max_Complexity": max_c,
            "Avg_Size": avg_s
        })
        
    df = pd.DataFrame(results)
    print("Scan complete.")
    print(df.sort_values("Max_Complexity", ascending=False).head(10))
    
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
