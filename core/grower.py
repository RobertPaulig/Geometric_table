"""
grower.py — молекулярный "Christmas Tree" grower.

Идея: расти молекулы как вероятностное дерево роста,
используя роли AtomGraph (terminator/bridge/hub) и χ_spec.

Соавторы: R. Paulig + GPT-5.1 Thinking (OpenAI)
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional, Any
import numpy as np
import random

# We import geom_atoms elements inside functions to avoid circular imports,
# or we assume they are passed/available.
# However, for typing, we might need forward refs.

@dataclass
class GrowthParams:
    max_depth: int = 4
    max_atoms: int = 20
    p_continue_base: float = 0.7  # базовая вероятность продолжать рост
    chi_sensitivity: float = 0.3  # как сильно χ_spec влияет на ветвление
    role_bonus_hub: float = 0.2   # надбавка к ветвлению для hubs
    role_penalty_terminator: float = -0.4 
    temperature: float = 1.0      # "температура" для софтмакса по χ
    # --- CY-1: loopy-режим (R&D QSG v6.x) ---
    allow_cycles: bool = False
    max_extra_bonds: int = 0      # сколько "добавочных" рёбер максимум
    p_extra_bond: float = 0.0     # вероятность попытки добавить каждую связь

@dataclass
class GrowthNode:
    atom_symbol: str
    atom_index: int       # индекс в Molecule.atoms
    depth: int
    free_ports: int


def _add_loopy_bonds(
    mol: Any,
    params: GrowthParams,
    rng: np.random.Generator,
) -> None:
    """
    CY-1: R&D-слой. Поверх уже выросшего дерева пытается добавить
    до max_extra_bonds дополнительных связей между существующими узлами,
    чтобы породить циклы.

    Инвариант: если allow_cycles=False или max_extra_bonds<=0, ничего не делает.
    """
    if (
        not params.allow_cycles
        or params.max_extra_bonds <= 0
        or params.p_extra_bond <= 0.0
    ):
        return

    n_atoms = len(mol.atoms)
    if n_atoms < 3:
        return

    adj = mol.adjacency_matrix()
    candidate_pairs: list[Tuple[int, int]] = []
    for i in range(n_atoms):
        for j in range(i + 1, n_atoms):
            if adj[i, j] == 0:
                candidate_pairs.append((i, j))

    if not candidate_pairs:
        return

    added = 0
    attempts = 0
    max_attempts = params.max_extra_bonds * 4

    while (
        added < params.max_extra_bonds
        and attempts < max_attempts
        and candidate_pairs
    ):
        attempts += 1
        if rng.random() > params.p_extra_bond:
            continue

        idx = rng.integers(low=0, high=len(candidate_pairs))
        i, j = candidate_pairs.pop(int(idx))

        # Добавляем ребро i-j в mol (как и в основном grower-е).
        mol.bonds.append((i, j))
        added += 1

def grow_molecule_christmas_tree(
    root_symbol: str,
    params: GrowthParams,
    rng: Optional[np.random.Generator] = None,
) -> Any:
    """
    Сгенерировать молекулу в стиле "рождественской ёлки".
    Returns: geom_atoms.Molecule
    """
    # Lazy import to avoid circular dependency
    from .geom_atoms import Molecule, get_atom, AtomGraph, PERIODIC_TABLE

    if rng is None:
        rng = np.random.default_rng()

    # 1. Initialize Molecule
    mol_name = f"GrowTree_{root_symbol}_{params.max_depth}"
    mol = Molecule(name=mol_name, atoms=[], bonds=[])
    
    # helper to add atom
    def add_atom_to_mol(symbol: str) -> int:
        idx = len(mol.atoms)
        mol.atoms.append(symbol)
        return idx

    root_idx = add_atom_to_mol(root_symbol)
    root_atom_data = get_atom(root_symbol)
    if root_atom_data is None:
        return mol  # Should not happen if symbol is valid

    # QSG v5.0: softness of the seed globally damps branching
    seed_softness = getattr(root_atom_data, "softness", 0.0)
    seed_softness = max(0.0, min(float(seed_softness), 0.95))

    # 2. Initialize Frontier
    frontier = [
        GrowthNode(
            atom_symbol=root_symbol, 
            atom_index=root_idx, 
            depth=0, 
            free_ports=root_atom_data.ports
        )
    ]
    
    # Candidate atoms for growth (simplified subset of common elements)
    candidate_pool = ["H", "C", "N", "O", "F", "Si", "P", "S", "Cl"]
    
    # Precompute chi for candidates to use in selection
    # (We could pull chi_spec from v4 model, but use rough Pauling or property from AtomGraph)
    # We will just use uniform or simple weighted selection for this demo R&D.
    
    while frontier and len(mol.atoms) < params.max_atoms:
        # Pop a node (BFS or DFS behavior depends on index)
        # Random pick or FIFO? Let's do FIFO for breadth-first "Christmas Tree" layers.
        # But "Christmas Tree" often implies depth. Let's try FIFO (pop(0)).
        current_node = frontier.pop(0)

        if current_node.depth >= params.max_depth:
            continue
        
        if current_node.free_ports <= 0:
            continue

        # Get parent atom data
        parent_atom = get_atom(current_node.atom_symbol)
        
        # Determine branching probability
        # P = base + role_bonus
        p_continue = params.p_continue_base
        if parent_atom.role == 'hub':
            p_continue += params.role_bonus_hub
        elif parent_atom.role == 'terminator':
            p_continue += params.role_penalty_terminator

        # Apply global softness penalty if the seed is soft (e.g. Si)
        if seed_softness > 0.0:
            p_continue *= (1.0 - seed_softness)

        # Temperature deformation: T=1 keeps baseline, T<1 lengthens trees, T>1 shortens
        T = max(params.temperature, 1e-6)
        p_continue = max(0.0, min(1.0, p_continue))
        p_continue_eff = max(0.0, min(1.0, p_continue ** (1.0 / T)))

        # Try to grow on each free port
        # Note: In a real geometric builder, we should track specific ports.
        # Here we just treat 'free_ports' as a counter.
        
        orig_free_ports = current_node.free_ports
        for _ in range(orig_free_ports):
            if len(mol.atoms) >= params.max_atoms:
                break
                
            # Roll dice
            if rng.random() > p_continue_eff:
                # Decide NOT to grow on this port -> connection to 'Nothing'?
                # Or just leave it open (radical)? 
                # In standard chemistry, open ports are radicals.
                # To make valid closed molecules, usually we cap with H.
                # "Christmas Tree" theorem usually implies filling.
                # Let's assume we cap with H if we don't extend, OR allow open ports as 'radicals'.
                # For this R&D, let's auto-cap with H if we stop, OR just leave it.
                # Let's leave it for now, or cap with H to make 'clean' molecules.
                # Let's cap with H so F_total makes sense.
                pass
            else:
                # Grow! Pick a child.
                # Simple random choice for now.
                child_sym = rng.choice(candidate_pool)
                child_atom_data = get_atom(child_sym)
                
                child_idx = add_atom_to_mol(child_sym)
                
                # Add bond
                mol.bonds.append((current_node.atom_index, child_idx))
                
                # Add to frontier
                # child uses 1 port to connect to parent
                child_free = child_atom_data.ports - 1
                if child_free > 0:
                    frontier.append(GrowthNode(
                        atom_symbol=child_sym,
                        atom_index=child_idx,
                        depth=current_node.depth + 1,
                        free_ports=child_free
                    ))
                
            current_node.free_ports -= 1

    # CY-1: R&D-слой циклов (loopy overlay) — по умолчанию выключен.
    _add_loopy_bonds(mol, params, rng)

    return mol


def grow_molecule_loopy(
    seed: str,
    *,
    params: Optional[GrowthParams] = None,
    rng: Optional[np.random.Generator] = None,
) -> Any:
    """
    CY-1: R&D-режим роста с циклами.

    Если params is None, используется канонический loopy-режим
    (близкий к конфигурации CY-1-A).
    """
    if rng is None:
        rng = np.random.default_rng()

    if params is None:
        params = GrowthParams(
            max_depth=4,
            max_atoms=25,
            p_continue_base=0.5,
            chi_sensitivity=0.3,
            role_bonus_hub=0.4,
            role_penalty_terminator=-0.6,
            temperature=1.0,
            allow_cycles=True,
            max_extra_bonds=3,
            p_extra_bond=0.3,
        )

    return grow_molecule_christmas_tree(seed, params=params, rng=rng)

def describe_molecule(mol: Any) -> str:
    """
    Return a short description of the molecule.
    """
    # No need to import compute_molecule_energy if we use the method directly
    # from .geom_atoms import compute_molecule_energy
    
    # Count atoms
    counts = {}
    for a in mol.atoms:
        if hasattr(a, 'name'):
            sym = a.name
        else:
            sym = str(a) # fallback
        counts[sym] = counts.get(sym, 0) + 1
    
    formula = "".join([f"{k}{v if v > 1 else ''}" for k, v in sorted(counts.items())])
    
    # Compute Energy
    try:
        # Assuming mol is instance of geom_atoms.Molecule
        if hasattr(mol, 'total_molecular_energy'):
            E_tot = mol.total_molecular_energy()
            # R_react calculation?
            # R_react = |F_flow| / (F_geom + F_angle)
            # We can get components from breakdown if available, 
            # but total_molecular_energy usually just returns float.
            # Let's check if we can get components.
            # Molecule.spectral_charges returns F_flow.
            # Molecule.F_mol returns approx sum.
            
            # Let's just print total energy for R&D report demo.
            R_react = 0.0 
        else:
            E_tot = 0.0
            R_react = 0.0
            
    except Exception as e:
        E_tot = 999.9
        R_react = 0.0
    
    return (f"Formula: {formula:<10} | Atoms: {len(mol.atoms):<3} | Bonds: {len(mol.bonds):<3} | "
            f"F_tot: {E_tot:6.2f}")

def run_grower_demo():
    """
    Demonstrate the grower.
    """
    params = GrowthParams(max_depth=3, max_atoms=15)
    seeds = ["C", "O", "Si"]
    
    print(f"--- Growth Params: {params} ---")
    
    for s in seeds:
        print(f"\nSeed: {s}")
        for i in range(2):
            mol = grow_molecule_christmas_tree(s, params)
            print(f"  Tree {i+1}: {describe_molecule(mol)}")
