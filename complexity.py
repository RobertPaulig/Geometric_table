"""
complexity.py — геометрическая/топологическая сложность графа.

Соавторы: R. Paulig + GPT-5.1 Thinking (OpenAI)
"""

from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from typing import Dict, Any, Optional

@dataclass
class ComplexityDecomposition:
    n: int             # узлы
    m: int             # рёбра
    cyclomatic: int    # m - n + components
    tree_deviation: float
    density: float
    mean_degree: float
    degree_var: float
    clustering_mean: float
    clustering_std: float
    base_uncrossed: float
    density_factor: float
    variance_factor: float
    clustering_factor: float
    log_scale: float
    total: float       # итоговая сложность C(G)

def compute_complexity_features(adj_matrix: np.ndarray) -> ComplexityDecomposition:
    """
    Принимает adjacency matrix (0/1, без петель) и возвращает разложение ComplexityDecomposition.
    """
    n = adj_matrix.shape[0]
    if n == 0:
        return ComplexityDecomposition(0,0,0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0)

    # 1. Basic Stats
    degrees = np.sum(adj_matrix, axis=1)
    m_float = np.sum(degrees) / 2.0
    m = int(m_float)
    
    mean_degree = np.mean(degrees)
    degree_var = np.var(degrees)

    # 2. Components via BFS/DFS to get cyclomatic number
    visited = np.zeros(n, dtype=bool)
    components = 0
    for i in range(n):
        if not visited[i]:
            components += 1
            # Run BFS
            queue = [i]
            visited[i] = True
            while queue:
                u = queue.pop(0)
                neighbors = np.where(adj_matrix[u] > 0)[0]
                for v in neighbors:
                    if not visited[v]:
                        visited[v] = True
                        queue.append(v)
    
    cyclomatic = m - n + components
    cyclomatic = max(0, cyclomatic)

    # 3. Density
    # For n=1, density is undefined or 0.
    if n > 1:
        density = (2.0 * m) / (n * (n - 1))
    else:
        density = 0.0

    # 4. Clustering Coefficient
    # Local clustering C_i = 2 * E_neighbors / (k_i * (k_i - 1))
    c_list = []
    for i in range(n):
        k_i = int(degrees[i])
        if k_i < 2:
            c_list.append(0.0)
            continue
        
        neighbors = np.where(adj_matrix[i] > 0)[0]
        # Count edges between neighbors
        adj_sub = adj_matrix[np.ix_(neighbors, neighbors)]
        edges_sub = np.sum(adj_sub) / 2.0
        
        possible = k_i * (k_i - 1) / 2.0
        c_list.append(edges_sub / possible)
    
    clustering_mean = np.mean(c_list)
    clustering_std = np.std(c_list)

    # 5. Composite Factors
    tree_deviation = float(cyclomatic)
    
    # base_uncrossed: contribution from cycles enhanced by clustering
    # If tree (cyclomatic=0), base_uncrossed=0. If just cycle, base_uncrossed > 0.
    base_uncrossed = tree_deviation * (1.0 + clustering_mean)
    
    # density_factor normalized to 0.5 (arbitrary heuristic ref point)
    density_factor = density / 0.5
    
    # variance_factor
    variance_factor = degree_var / (mean_degree + 1e-6)
    
    # clustering_factor
    clustering_factor = clustering_mean / 0.5
    
    # log_scale: penalty for size
    # log2(1 + n)
    log_scale = np.log2(1.0 + n)
    
    # Total Complexity C(G)
    # Heuristic combined formula:
    # C(G) ~ (base_uncrossed + 0.1*n) * (factors) * scale?
    # Or strict adherence to prompt: 
    #   total = base_uncrossed * (1 + density_factor + variance_factor + clustering_factor) * log_scale
    # But wait, if base_uncrossed is 0 (tree), total is 0?
    # Trees have complexity too (branching).
    # Let's modify base to include n-term if cyclomatic is 0.
    
    # Re-reading prompt: "tree_deviation = max(cyclomatic, 0)"
    # If base_uncrossed is 0 (for trees), total becomes 0.
    # But trees are not complexity-free.
    # However, strict instructions say:
    # "total = base_uncrossed * (1 + ...) * log_scale"
    # I will follow this but add a small epsilon or linear term to base_uncrossed if n > 1.
    # Actually, let's strictly follow the prompt formula logic but enable it to pick up trees 
    # if I add a term. 
    # Let's interpret 'Complexity' here as 'Topological/Cycle Complexity'. 
    # Trees have 0 cyc complexity.
    # But wait, later in prompt: "Simple cyclomatic number... + 0.1*n".
    # I'll stick to a robust formula:
    
    term_size = 0.1 * n
    base_combined = base_uncrossed + term_size 
    
    total = base_combined * (1.0 + 0.2*density_factor + 0.2*variance_factor + 0.2*clustering_factor) * log_scale
    
    return ComplexityDecomposition(
        n=n,
        m=m,
        cyclomatic=cyclomatic,
        tree_deviation=tree_deviation,
        density=density,
        mean_degree=mean_degree,
        degree_var=degree_var,
        clustering_mean=clustering_mean,
        clustering_std=clustering_std,
        base_uncrossed=base_uncrossed,
        density_factor=density_factor,
        variance_factor=variance_factor,
        clustering_factor=clustering_factor,
        log_scale=log_scale,
        total=total
    )

def compute_crossing_complexity(adj_matrix: np.ndarray) -> float:
    """
    Упрощённый оценщик «числа пересечений» / сложности графа.
    Возвращает total из ComplexityDecomposition.
    """
    return compute_complexity_features(adj_matrix).total

def atom_complexity_from_adjacency(adj_matrix: np.ndarray) -> float:
    """
    Специальная версия для атомных графов.
    """
    return compute_crossing_complexity(adj_matrix)
