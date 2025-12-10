
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

# Load data from scan_living_sectors
# Assuming user will run scan_living_sectors.py first and
# we'll use its console output or re-run it here

from core.geom_atoms import get_atom, SPECTRAL_MODE_V4, SPECTRAL_MODE, compute_element_indices
import core.geom_atoms as geom_atoms
from core.grower import GrowthParams, grow_molecule_christmas_tree
from core.complexity import compute_complexity_features

# Ensure V4 mode
geom_atoms.SPECTRAL_MODE = geom_atoms.SPECTRAL_MODE_V4

def collect_data():
    """Re-run scan_living_sectors to collect data."""
    print("Collecting data from living sectors...")
    
    indices = compute_element_indices()
    el_map = {item['El']: (item['D_index'], item['A_index'], item['role']) for item in indices}
    
    params = GrowthParams(max_depth=4, max_atoms=25)
    n_trials = 20
    
    results = []
    elements_to_scan = [e['El'] for e in indices if e['Z'] <= 36 and e['role'] != 'inert']
    
    for el in elements_to_scan:
        d_val, a_val, role = el_map[el]
        
        complexities = []
        sizes = []
        for i_trial in range(n_trials):
            try:
                mol = grow_molecule_christmas_tree(el, params)
                adj = mol.adjacency_matrix()
                feats = compute_complexity_features(adj)
                complexities.append(feats.total)
                sizes.append(len(mol.atoms))
            except Exception:
                complexities.append(0.0)
        
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
    
    return pd.DataFrame(results)

def analyze_correlations(df):
    """Perform correlation analysis."""
    print("\n[CORRELATION ANALYSIS]")
    
    # Spearman correlations
    corr_d = stats.spearmanr(df["D"], df["Max_Complexity"])
    corr_a = stats.spearmanr(df["A"], df["Max_Complexity"])
    
    print(f"D vs Max_Complexity: r={corr_d.correlation:.3f}, p={corr_d.pvalue:.4f}")
    print(f"A vs Max_Complexity: r={corr_a.correlation:.3f}, p={corr_a.pvalue:.4f}")
    
    # Group statistics by role
    print("\n[GROUP STATISTICS BY ROLE]")
    print(df.groupby("Role")[["Avg_Complexity", "Max_Complexity", "Avg_Size"]].mean())
    
    # Export statistics
    with open("correlation_stats.txt", "w") as f:
        f.write("[CORRELATION STATISTICS]\n\n")
        f.write(f"D vs Max_Complexity:\n")
        f.write(f"  Spearman r = {corr_d.correlation:.4f}\n")
        f.write(f"  p-value = {corr_d.pvalue:.6f}\n\n")
        f.write(f"A vs Max_Complexity:\n")
        f.write(f"  Spearman r = {corr_a.correlation:.4f}\n")
        f.write(f"  p-value = {corr_a.pvalue:.6f}\n\n")
        f.write("[ROLE SUMMARY]\n")
        f.write(df.groupby("Role")[["Avg_Complexity", "Max_Complexity"]].describe().to_string())
    
    print("Saved correlation_stats.txt")

def plot_correlations(df):
    """Generate correlation plots."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # D vs Complexity
    ax = axes[0]
    for role in df["Role"].unique():
        subset = df[df["Role"] == role]
        ax.scatter(subset["D"], subset["Max_Complexity"], label=role, alpha=0.7, s=100)
    ax.set_xlabel("Donor Index (D)")
    ax.set_ylabel("Max Complexity")
    ax.set_title("D vs Complexity")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # A vs Complexity
    ax = axes[1]
    for role in df["Role"].unique():
        subset = df[df["Role"] == role]
        ax.scatter(subset["A"], subset["Max_Complexity"], label=role, alpha=0.7, s=100)
    ax.set_xlabel("Acceptor Index (A)")
    ax.set_ylabel("Max Complexity")
    ax.set_title("A vs Complexity")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # D vs A colored by Complexity
    ax = axes[2]
    sc = ax.scatter(df["D"], df["A"], c=df["Max_Complexity"], cmap="viridis", s=100, alpha=0.8)
    plt.colorbar(sc, ax=ax, label="Max Complexity")
    
    # Annotate notable elements
    for i, row in df.iterrows():
        if row["Max_Complexity"] > 10 or row["Element"] in ["Li", "F", "He"]:
            ax.text(row["D"] + 0.01, row["A"] + 0.01, row["Element"], fontsize=8)
    
    ax.set_xlabel("Donor Index (D)")
    ax.set_ylabel("Acceptor Index (A)")
    ax.set_title("D/A Plane colored by Complexity")
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("correlation_plots.png")
    print("Saved correlation_plots.png")

def main():
    # Collect data
    df = collect_data()
    
    # Export CSV
    df.to_csv("complexity_summary.csv", index=False)
    print(f"\nExported complexity_summary.csv ({len(df)} elements)")
    
    # Analyze correlations
    analyze_correlations(df)
    
    # Plot
    plot_correlations(df)

if __name__ == "__main__":
    main()
