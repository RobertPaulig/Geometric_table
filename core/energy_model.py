from __future__ import annotations

from typing import Any

import numpy as np

from core.complexity import compute_complexity_features_v2
from core.thermo_config import ThermoConfig, get_current_thermo_config


def molecule_to_adj(mol: Any) -> np.ndarray:
    """
    Построить матрицу смежности для Molecule-подобного объекта.
    Требуется, чтобы у mol были поля:
      - atoms: последовательность узлов
      - bonds: последовательность пар индексов (i,j).
    """
    n = len(getattr(mol, "atoms", []))
    adj = np.zeros((n, n), dtype=float)
    for i, j in getattr(mol, "bonds", []):
        if i == j:
            continue
        a, b = (int(i), int(j)) if i < j else (int(j), int(i))
        if 0 <= a < n and 0 <= b < n:
            adj[a, b] = 1.0
            adj[b, a] = 1.0
    return adj


def compute_energy(mol: Any, thermo: ThermoConfig | None = None) -> float:
    """
    Энергия молекулы как сложность графа, рассчитанная через выбранный backend.
    По умолчанию backend задаётся ThermoConfig.deltaG_backend.
    """
    if thermo is None:
        thermo = get_current_thermo_config()
    backend = getattr(thermo, "deltaG_backend", "fdm_entanglement")
    adj = molecule_to_adj(mol)
    feats = compute_complexity_features_v2(adj, backend=backend)
    return float(feats.total)


def compute_delta_G(old_mol: Any, new_mol: Any, thermo: ThermoConfig | None = None) -> float:
    """
    ΔG = coupling_delta_G * (E_new - E_old), где E — энергия из compute_energy().
    """
    if thermo is None:
        thermo = get_current_thermo_config()
    e_old = compute_energy(old_mol, thermo)
    e_new = compute_energy(new_mol, thermo)
    delta_e = float(e_new - e_old)
    coupling = float(getattr(thermo, "coupling_delta_G", 1.0))
    return coupling * delta_e

