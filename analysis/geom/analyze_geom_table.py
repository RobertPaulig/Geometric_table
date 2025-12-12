import csv
import os
import sys

# Try to import matplotlib for optional plotting
try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

from analysis.io_utils import data_path, results_path
from core.geom_atoms import compute_element_indices, PERIODIC_TABLE

def get_element_indices():
    """
    Возвращает список словарей с полями:
      Z, El, role, period, chi_spec, E_port, D_index, A_index
    просто обёртка над compute_element_indices().
    """
    # compute_element_indices returns a list of dictionaries
    return compute_element_indices()

def analyze_DA_plateaus():
    """
    Analyzes and prints D/A index plateaus for elements H-Ar.
    """
    indices = get_element_indices()
    
    # Filter for H-Kr (Z=1..36)
    target_elements = [d for d in indices if 1 <= d['Z'] <= 36]
    
    # DEBUG: Check B and Al
    for d in target_elements:
        if d['El'] in ['B', 'Al']:
             print(f"DEBUG CHECK: {d['El']} epsilon derived from E_port/Chi? D={d['D_index']:.4f}, A={d['A_index']:.4f}")
    
    # 1. Donor Plateaus: D > 0.0001, A < 0.0001
    donors = [d for d in target_elements if d['D_index'] > 0.0001 and d['A_index'] < 0.0001]
    
    # Group by D_index rounded to 4 decimals
    donor_groups = {}
    for d in donors:
        val = round(d['D_index'], 4)
        if val not in donor_groups:
            donor_groups[val] = []
        donor_groups[val].append(d)
        
    # 2. Strong Acceptor Plateau: A > 0.5, D < 0.0001
    # Adjust threshold if needed, user said > 0.5. Let's check data. 
    # In rnd_master_v4_full.txt, F/Cl/H/N/O/P/S seem to share A_index ~ 1.2370 (normalized) or similar.
    strong_acceptors = [d for d in target_elements if d['A_index'] > 0.5 and d['D_index'] < 0.0001]
    
    strong_acc_groups = {}
    for d in strong_acceptors:
        val = round(d['A_index'], 4)
        if val not in strong_acc_groups:
            strong_acc_groups[val] = []
        strong_acc_groups[val].append(d)
        
    # 3. Weak Acceptors / Amphoteric: 0.05 <= A <= 0.3, D < 0.05
    # C and Si usually fall here.
    weak_acceptors = [d for d in target_elements if 0.05 <= d['A_index'] <= 0.3 and d['D_index'] < 0.05]
    
    # Print Report
    print("=== D/A plateaus (v4, H–Ar) ===\n")
    
    print("Donor plateaus (D_index):")
    for val in sorted(donor_groups.keys()):
        # Format list: "Li(3, terminator)"
        items = []
        for d in donor_groups[val]:
            # role might be full string, let's keep it as is
            items.append(f"{d['El']}({d['Z']}, {d['role']})")
        print(f"  {val:.4f} : {', '.join(items)}")
        
    print("\nStrong acceptor plateau (A_index):")
    for val in sorted(strong_acc_groups.keys()):
        items = []
        for d in strong_acc_groups[val]:
            items.append(f"{d['El']}({d['Z']}, {d['role']})")
        print(f"  {val:.4f} : {', '.join(items)}")

    print("\nWeak acceptors / amphoteric zone:")
    for d in weak_acceptors:
        # C(6, hub)  : D=0.0000, A=0.1237
        print(f"  {d['El']}({d['Z']}, {d['role']}): D={d['D_index']:.4f}, A={d['A_index']:.4f}")
    
    print("") # spacer

def dump_element_indices_to_csv(path: str | None = None):
    """
    Сохраняет таблицу индексов (Z,El,role,period,chi_spec,E_port,D_index,A_index)
    в CSV с заголовком. Разделитель — запятая, кодировка UTF-8.
    """
    if path is None:
        path = data_path("element_indices_v4.csv")

    indices = get_element_indices()
    # Sort by Z
    indices.sort(key=lambda x: x['Z'])

    headers = ["Z", "El", "role", "period", "chi_spec", "E_port", "D_index", "A_index"]

    try:
        with open(path, mode='w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=headers, extrasaction='ignore')
            writer.writeheader()
            for row in indices:
                writer.writerow(row)
        print(f"Successfully saved CSV to: {os.path.abspath(str(path))}")
    except Exception as e:
        print(f"Error saving CSV: {e}")

def plot_DA_scatter(outfile: str | None = None):
    """
    Рисует диаграмму рассеяния D_index vs A_index для Z=1..18,
    подписи — символы элементов. Цвет можно кодировать по роли (role).
    """
    if not MATPLOTLIB_AVAILABLE:
        print("Matplotlib not available, skipping scatter plot.")
        return

    indices = get_element_indices()
    target_elements = [d for d in indices if 1 <= d['Z'] <= 18]
    
    D_vals = [d['D_index'] for d in target_elements]
    A_vals = [d['A_index'] for d in target_elements]
    labels = [d['El'] for d in target_elements]
    roles = [d['role'] for d in target_elements]
    
    # Map roles to colors
    unique_roles = sorted(list(set(roles)))
    role_color_map = {}
    # Basic palette
    colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k']
    for i, r in enumerate(unique_roles):
        role_color_map[r] = colors[i % len(colors)]
        
    point_colors = [role_color_map[r] for r in roles]
    
    plt.figure(figsize=(10, 6))
    plt.scatter(D_vals, A_vals, c=point_colors, s=100, alpha=0.7, edgecolors='k')
    
    # Annotate points
    for i, label in enumerate(labels):
        plt.text(D_vals[i]+0.01, A_vals[i]+0.01, label, fontsize=9)
        
    plt.xlabel("Donor Index (D)")
    plt.ylabel("Acceptor Index (A)")
    plt.title("Geometric Table v4.0: D vs A Indices (H-Ar)")
    plt.grid(True, linestyle='--', alpha=0.5)
    
    # Create custom legend
    from matplotlib.lines import Line2D
    legend_elements = [Line2D([0], [0], marker='o', color='w', label=r,
                              markerfacecolor=c, markersize=10)
                       for r, c in role_color_map.items()]
    plt.legend(handles=legend_elements, title="Role")
    
    if outfile is None:
        outfile = results_path("DA_scatter.png")

    try:
        plt.savefig(outfile, dpi=150)
        print(f"Successfully saved scatter plot to: {os.path.abspath(str(outfile))}")
    except Exception as e:
        print(f"Error saving plot: {e}")
    finally:
        plt.close()

if __name__ == "__main__":
    analyze_DA_plateaus()
    dump_element_indices_to_csv()
    plot_DA_scatter()
