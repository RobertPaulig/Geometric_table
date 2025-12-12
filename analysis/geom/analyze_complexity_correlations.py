import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats

from analysis.io_utils import (
    read_data_csv,
    ensure_results_dir,
)


def load_complexity_summary() -> pd.DataFrame:
    """
    Load precomputed complexity summary from scan_living_sectors /
    scan_living_sectors_segregated.

    Ожидает data/complexity_summary.csv.
    """
    return read_data_csv(
        "complexity_summary.csv",
        required=True,
        expected_columns=[
            "Element",
            "Role",
            "D",
            "A",
            "Avg_Complexity",
            "Max_Complexity",
            "Avg_Size",
        ],
    )

def analyze_correlations(df):
    """Perform correlation analysis."""
    print("\n[CORRELATION ANALYSIS]")
    
    # Spearman correlations for legacy v1 complexity (Max_Complexity)
    corr_d = stats.spearmanr(df["D"], df["Max_Complexity"])
    corr_a = stats.spearmanr(df["A"], df["Max_Complexity"])
    
    print(f"D vs Max_Complexity: r={corr_d.correlation:.3f}, p={corr_d.pvalue:.4f}")
    print(f"A vs Max_Complexity: r={corr_a.correlation:.3f}, p={corr_a.pvalue:.4f}")

    # FDM-based correlations (if available)
    has_fdm = "C_total_fdm_mean" in df.columns and "C_norm_fdm_mean" in df.columns
    if has_fdm:
        corr_d_fdm = stats.spearmanr(df["D"], df["C_total_fdm_mean"])
        corr_a_fdm = stats.spearmanr(df["A"], df["C_total_fdm_mean"])
        corr_d_fdm_norm = stats.spearmanr(df["D"], df["C_norm_fdm_mean"])
        corr_a_fdm_norm = stats.spearmanr(df["A"], df["C_norm_fdm_mean"])

        print(
            f"D vs C_total_fdm_mean: r={corr_d_fdm.correlation:.3f}, "
            f"p={corr_d_fdm.pvalue:.4f}"
        )
        print(
            f"A vs C_total_fdm_mean: r={corr_a_fdm.correlation:.3f}, "
            f"p={corr_a_fdm.pvalue:.4f}"
        )
        print(
            f"D vs C_norm_fdm_mean: r={corr_d_fdm_norm.correlation:.3f}, "
            f"p={corr_d_fdm_norm.pvalue:.4f}"
        )
        print(
            f"A vs C_norm_fdm_mean: r={corr_a_fdm_norm.correlation:.3f}, "
            f"p={corr_a_fdm_norm.pvalue:.4f}"
        )
    
    # Group statistics by role
    print("\n[GROUP STATISTICS BY ROLE]")
    role_cols = ["Avg_Complexity", "Max_Complexity", "Avg_Size"]
    if has_fdm:
        role_cols += ["C_total_fdm_mean", "C_norm_fdm_mean"]
    print(df.groupby("Role")[role_cols].mean())
    
    # Export statistics
    results_dir = ensure_results_dir()
    stats_path = results_dir / "correlation_stats.txt"
    with stats_path.open("w", encoding="utf-8") as f:
        f.write("[CORRELATION STATISTICS]\n\n")
        f.write(f"D vs Max_Complexity:\n")
        f.write(f"  Spearman r = {corr_d.correlation:.4f}\n")
        f.write(f"  p-value = {corr_d.pvalue:.6f}\n\n")
        f.write(f"A vs Max_Complexity:\n")
        f.write(f"  Spearman r = {corr_a.correlation:.4f}\n")
        f.write(f"  p-value = {corr_a.pvalue:.6f}\n\n")

        if has_fdm:
            f.write("FDM complexity correlations\n")
            f.write("---------------------------\n")
            f.write(
                f"D vs C_total_fdm_mean: r={corr_d_fdm.correlation:.4f}, "
                f"p={corr_d_fdm.pvalue:.6f}\n"
            )
            f.write(
                f"A vs C_total_fdm_mean: r={corr_a_fdm.correlation:.4f}, "
                f"p={corr_a_fdm.pvalue:.6f}\n"
            )
            f.write(
                f"D vs C_norm_fdm_mean: r={corr_d_fdm_norm.correlation:.4f}, "
                f"p={corr_d_fdm_norm.pvalue:.6f}\n"
            )
            f.write(
                f"A vs C_norm_fdm_mean: r={corr_a_fdm_norm.correlation:.4f}, "
                f"p={corr_a_fdm_norm.pvalue:.6f}\n\n"
            )

        f.write("[ROLE SUMMARY]\n")
        f.write(
            df.groupby("Role")[role_cols].describe().to_string()
        )

    print(f"[ANALYSIS-IO] Saved stats: {stats_path}")

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
    out_png = ensure_results_dir() / "correlation_plots.png"
    plt.savefig(out_png, dpi=150, bbox_inches="tight")
    print(f"[ANALYSIS-IO] Saved figure: {out_png}")

def main():
    # Load precomputed summary from scan_living_sectors
    df = load_complexity_summary()
    print(f"\nLoaded complexity_summary.csv ({len(df)} elements)")
    
    # Analyze correlations
    analyze_correlations(df)
    
    # Plot
    plot_correlations(df)

if __name__ == "__main__":
    main()
