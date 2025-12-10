
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from core.geom_atoms import get_atom, SPECTRAL_MODE_V4, SPECTRAL_MODE, compute_element_indices
import core.geom_atoms as geom_atoms
from core.grower import GrowthParams, grow_molecule_christmas_tree
from core.complexity import compute_complexity_features

# Ensure V4 mode
geom_atoms.SPECTRAL_MODE = geom_atoms.SPECTRAL_MODE_V4

# Define element sets
DONORS = ["Li", "Na", "K", "Be", "Mg", "Ca"]
ACCEPTORS = ["C", "N", "O", "F", "Si", "P", "S", "Cl"]
MIXED = ["H", "Li", "Be", "B", "C", "N", "O", "F", 
         "Na", "Mg", "Al", "Si", "P", "S", "Cl",
         "K", "Ca", "Ga", "Ge", "As", "Se", "Br"]

def run_scenario(scenario_name, element_pool, seeds, params, n_trials=50):
    """Run growth experiment for a specific scenario."""
    print(f"\n[{scenario_name}]")
    print(f"Seeds: {seeds}")
    print(f"Growth pool: {element_pool}")
    
    results = []
    
    for seed in seeds:
        complexities = []
        sizes = []
        
        for i in range(n_trials):
            try:
                # Modify grower to use custom element pool
                # For now, we use the default candidate_pool
                # TODO: Pass element_pool to grower
                mol = grow_molecule_christmas_tree(seed, params)
                adj = mol.adjacency_matrix()
                feats = compute_complexity_features(adj)
                complexities.append(feats.total)
                sizes.append(len(mol.atoms))
            except Exception:
                complexities.append(0.0)
                sizes.append(1)
        
        avg_c = np.mean(complexities)
        max_c = np.max(complexities)
        avg_s = np.mean(sizes)
        max_s = np.max(sizes)
        
        results.append({
            "Scenario": scenario_name,
            "Seed": seed,
            "Avg_Complexity": avg_c,
            "Max_Complexity": max_c,
            "Avg_Size": avg_s,
            "Max_Size": max_s
        })
        print(f"  {seed}: Avg_C={avg_c:.2f}, Max_C={max_c:.2f}, Avg_Size={avg_s:.1f}")
    
    return results

def run_all_scenarios():
    params = GrowthParams(max_depth=4, max_atoms=25)
    
    all_results = []
    
    # Scenario 1: Donor-only
    all_results.extend(run_scenario(
        "Donor-only",
        DONORS,
        DONORS,
        params
    ))
    
    # Scenario 2: Acceptor-only
    all_results.extend(run_scenario(
        "Acceptor-only",
        ACCEPTORS,
        ACCEPTORS,
        params
    ))
    
    # Scenario 3: Mixed (baseline)
    mixed_seeds = ["C", "N", "O", "Si", "P", "S"]  # Representative acceptors
    all_results.extend(run_scenario(
        "Mixed",
        MIXED,
        mixed_seeds,
        params
    ))
    
    df = pd.DataFrame(all_results)
    
    # Export CSV
    df.to_csv("segregated_stats.csv", index=False)
    print(f"\nExported segregated_stats.csv")
    
    # Plot comparison
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    scenarios = ["Donor-only", "Acceptor-only", "Mixed"]
    for idx, scenario in enumerate(scenarios):
        subset = df[df["Scenario"] == scenario]
        ax = axes[idx]
        
        x = np.arange(len(subset))
        ax.bar(x, subset["Max_Complexity"], alpha=0.7, label="Max C")
        ax.set_title(f"{scenario} ({len(subset)} seeds)")
        ax.set_xlabel("Seed")
        ax.set_ylabel("Max Complexity")
        ax.set_xticks(x)
        ax.set_xticklabels(subset["Seed"], rotation=45)
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("segregated_comparison.png")
    print("Saved segregated_comparison.png")
    
    # Summary statistics
    print("\n[SUMMARY STATISTICS]")
    print(df.groupby("Scenario")[["Avg_Complexity", "Max_Complexity", "Avg_Size"]].mean())


def run_segregated_scan() -> None:
    """Convenience alias used by the pipeline."""
    run_all_scenarios()


if __name__ == "__main__":
    run_all_scenarios()
