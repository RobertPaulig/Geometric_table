import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
from geom_atoms import (
    AtomOverrideContext, PERIODIC_TABLE, get_atom, 
    make_virtual_molecule, SPECTRAL_MODE_V4, SPECTRAL_MODE
)
import geom_atoms

# Ensure V4 mode
geom_atoms.SPECTRAL_MODE = geom_atoms.SPECTRAL_MODE_V4

# Inject a dummy "X" into the table so AtomOverrideContext can find it
if "X" not in PERIODIC_TABLE:
    base_x = get_atom("B")  # copy B
    # Create a new AtomGraph for X
    from geom_atoms import AtomGraph
    if base_x:
        atom_x = AtomGraph(
            name="X",
            Z=5, # arbitrary
            nodes=base_x.nodes,
            edges=base_x.edges,
            ports=base_x.ports,
            symmetry_score=base_x.symmetry_score,
            port_geometry=base_x.port_geometry,
            role="hub", # Default to hub or copy from base_x? base_x is B (hub).
            epsilon=base_x.epsilon
        )
        PERIODIC_TABLE["X"] = atom_x

def run_scan(p_min=1, p_max=4, eps_min=-6.0, eps_max=-0.1, n_eps=20):
    print("Starting Virtual Island Scan...")
    print(f"Parameters: p=[{p_min}, {p_max}], ε=[{eps_min:.2f}, {eps_max:.2f}], n_eps={n_eps}")
    
    # Parameter Grid
    ports_metrics = list(range(p_min, p_max + 1))
    eps_values = np.linspace(eps_max, eps_min, n_eps)  # reversed: 0 to -10
    
    results = []
    
    # We'll fix Period=2 (Li-Ne) for the main heatmap to see the "generic" island
    target_period = 2
    z_base = 5 # Boron-like Z for sizing
    
    for ports in ports_metrics:
        for eps in eps_values:
            # Construct Virtual Atom X
            with AtomOverrideContext(PERIODIC_TABLE, "X", 
                                     Z=z_base, ports=ports, epsilon=eps):
                
                x = get_atom("X")
                chi_x = x.chi_geom_signed_spec() if x else np.nan
                
                # Build molecules
                try:
                    mol_hx = make_virtual_molecule("X", "H", 1)
                    mol_xo = make_virtual_molecule("X", "O", 1)
                    mol_xf = make_virtual_molecule("X", "F", 1)
                    
                    F_HX = mol_hx.total_molecular_energy() if mol_hx else np.nan
                    F_XO = mol_xo.total_molecular_energy() if mol_xo else np.nan
                    F_XF = mol_xf.total_molecular_energy() if mol_xf else np.nan
                    
                    # Robustness metric: max absolute energy
                    energies = [F_HX, F_XO, F_XF]
                    valid_energies = [e for e in energies if not np.isnan(e)]
                    
                    if valid_energies:
                        max_abs_F = max(abs(e) for e in valid_energies)
                        is_stable = max_abs_F < 20.0
                    else:
                        max_abs_F = np.nan
                        is_stable = False
                        
                except Exception as e:
                    chi_x = np.nan
                    max_abs_F = np.nan
                    is_stable = False
                
                results.append({
                    "Ports": ports,
                    "Epsilon": eps,
                    "Chi_Spec": chi_x,
                    "Max_Abs_F": max_abs_F,
                    "Stable": is_stable
                })

    df = pd.DataFrame(results)
    print(f"Scan complete. {len(df)} points.")
    print("DEBUG: First 5 rows:")
    print(df.head())
    print("DEBUG: Full stats:")
    print(df.describe())
    
    # PLOTTING
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Pivot for Chi Heatmap
    pivot_chi = df.pivot(index="Epsilon", columns="Ports", values="Chi_Spec")
    pivot_chi.sort_index(ascending=False, inplace=True)
    
    im0 = axes[0].imshow(pivot_chi, cmap="coolwarm", aspect='auto')
    axes[0].set_title("Spectral Electronegativity (Chi_X)")
    axes[0].set_ylabel("Epsilon")
    axes[0].set_xlabel("Ports")
    
    # Set ticks based on row/col indices
    yticks = np.arange(len(pivot_chi.index))
    ytictlabels = [f"{x:.2f}" for x in pivot_chi.index]
    # Reduce implementation density for ticks if too many
    if len(yticks) > 10:
        idx = np.round(np.linspace(0, len(yticks)-1, 10)).astype(int)
        yticks = yticks[idx]
        ytictlabels = [ytictlabels[i] for i in idx]
        
    axes[0].set_yticks(yticks)
    axes[0].set_yticklabels(ytictlabels)
    
    axes[0].set_xticks(np.arange(len(pivot_chi.columns)))
    axes[0].set_xticklabels(pivot_chi.columns)
    
    plt.colorbar(im0, ax=axes[0])
    
    # Pivot for Stability Heatmap
    pivot_stab = df.pivot(index="Epsilon", columns="Ports", values="Max_Abs_F")
    pivot_stab.sort_index(ascending=False, inplace=True)
    pivot_stab_capped = pivot_stab.clip(upper=10.0)
    
    im1 = axes[1].imshow(pivot_stab_capped, cmap="magma_r", aspect='auto')
    axes[1].set_title("Max Molecular Energy |F| (Lower is Stable)")
    axes[1].set_ylabel("Epsilon")
    axes[1].set_xlabel("Ports")
    
    axes[1].set_yticks(yticks)
    axes[1].set_yticklabels(ytictlabels)
    axes[1].set_xticks(np.arange(len(pivot_stab.columns)))
    axes[1].set_xticklabels(pivot_stab.columns)
    
    plt.colorbar(im1, ax=axes[1])
    
    plt.tight_layout()
    plt.savefig("virtual_island_scan.png")
    print("Saved virtual_island_scan.png")
    
    island = df[df["Max_Abs_F"] < 20.0]
    print("\n[ISLAND OF STABILITY SUMMARY]")
    print(island.describe())

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Scan virtual atom stability island")
    parser.add_argument("--p-min", type=int, default=1, help="Minimum ports")
    parser.add_argument("--p-max", type=int, default=4, help="Maximum ports")
    parser.add_argument("--eps-min", type=float, default=-6.0, help="Minimum epsilon")
    parser.add_argument("--eps-max", type=float, default=-0.1, help="Maximum epsilon")
    parser.add_argument("--n-eps", type=int, default=20, help="Number of epsilon steps")
    args = parser.parse_args()
    
    run_scan(args.p_min, args.p_max, args.eps_min, args.eps_max, args.n_eps)
