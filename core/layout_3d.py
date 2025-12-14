from __future__ import annotations

from typing import List, Tuple, Optional, TYPE_CHECKING

import numpy as np

from core.thermo_config import ThermoConfig, get_current_thermo_config

if TYPE_CHECKING:
    from core.geom_atoms import Molecule


def force_directed_layout_3d(
    n: int,
    edges: List[Tuple[int, int]],
    *,
    n_steps: int = 500,
    step: float = 0.02,
    k_attract: float = 0.1,
    k_repulse: float = 0.01,
    seed: int = 0,
    init_pos: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Простейший 3D force-directed layout (Fruchterman–Reingold-подобный).
    Гарантии: конечный результат, детерминированность при фиксированном seed.
    """
    if n <= 0:
        return np.zeros((0, 3), dtype=float)

    rng = np.random.default_rng(seed)

    if init_pos is not None and init_pos.shape == (n, 3):
        pos = np.array(init_pos, dtype=float)
    else:
        pos = rng.normal(scale=0.1, size=(n, 3))

    edges_arr = [(int(i), int(j)) for (i, j) in edges if 0 <= i < n and 0 <= j < n]

    for _ in range(max(n_steps, 0)):
        disp = np.zeros_like(pos)

        # repulsive forces
        for i in range(n):
            for j in range(i + 1, n):
                delta = pos[i] - pos[j]
                dist2 = float(np.dot(delta, delta)) + 1e-9
                inv_dist = 1.0 / np.sqrt(dist2)
                force = k_repulse * inv_dist * inv_dist
                f_vec = force * delta * inv_dist
                disp[i] += f_vec
                disp[j] -= f_vec

        # attractive forces along edges
        for i, j in edges_arr:
            delta = pos[j] - pos[i]
            dist = float(np.linalg.norm(delta) + 1e-9)
            force = k_attract * dist
            f_vec = force * (delta / dist)
            disp[i] += f_vec
            disp[j] -= f_vec

        # update positions with clipping to avoid blow-up
        max_step = 0.1
        norms = np.linalg.norm(disp, axis=1, keepdims=True) + 1e-9
        step_vec = step * disp / norms
        step_vec = np.clip(step_vec, -max_step, max_step)
        pos += step_vec

        # optional centering to keep layout around origin
        pos -= np.mean(pos, axis=0, keepdims=True)

    return pos


def init_layout_from_ports(
    mol: "Molecule",
    *,
    thermo: Optional[ThermoConfig] = None,
    seed: int = 0,
) -> np.ndarray:
    """
    Инициализация позиций атомов по портовым векторам, если они есть,
    иначе — случайно на сфере.
    """
    if thermo is None:
        thermo = get_current_thermo_config()
    n = len(mol.atoms)
    rng = np.random.default_rng(seed)

    pos = np.zeros((n, 3), dtype=float)
    for idx, atom in enumerate(mol.atoms):
        vecs = atom.port_vectors(thermo)
        if vecs.size > 0:
            # усредняем портовые векторы как "направление связи"
            d = np.mean(vecs, axis=0)
            norm = float(np.linalg.norm(d))
            if norm > 0:
                pos[idx] = d / norm
                continue
        # fallback: случайная точка на сфере
        u = rng.normal(size=3)
        norm = float(np.linalg.norm(u))
        if norm > 0:
            pos[idx] = u / norm
        else:
            pos[idx] = np.array([1.0, 0.0, 0.0])
    return pos


def layout_molecule_3d(
    mol: "Molecule",
    *,
    thermo: Optional[ThermoConfig] = None,
    n_steps: int = 500,
    step: float = 0.02,
    k_attract: float = 0.1,
    k_repulse: float = 0.01,
    seed: int = 0,
) -> np.ndarray:
    """
    Полный цикл укладки: портовые вектора → init → force-directed релаксация.
    """
    init_pos = init_layout_from_ports(mol, thermo=thermo, seed=seed)
    return force_directed_layout_3d(
        n=len(mol.atoms),
        edges=list(mol.bonds),
        n_steps=n_steps,
        step=step,
        k_attract=k_attract,
        k_repulse=k_repulse,
        seed=seed,
        init_pos=init_pos,
    )
