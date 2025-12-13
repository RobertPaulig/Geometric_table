from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Tuple

import numpy as np

from core.nuclear_spectrum_ws import build_radial_hamiltonian_ws
from core.spectral_density_ws import WSRadialParams


def _best_bound_energy(params: WSRadialParams, ell: int) -> float:
    evals, _ = build_radial_hamiltonian_ws(
        R_max=params.R_max,
        R0=params.R_well,
        a=0.7,
        V0=params.V0,
        N_grid=params.N_grid,
        ell=ell,
    )
    bound = evals[evals < 0.0]
    if bound.size == 0:
        return float(evals[0])
    return float(bound[0])


def ws_sp_gap(Z: int, params: WSRadialParams) -> float:
    """
    Спектральный признак: разрыв между низшими s- и p-состояниями (WS-модель).
    Z включён для совместимости и будущей Z-зависимости параметров.
    """
    _ = Z
    E_s = _best_bound_energy(params, ell=0)
    E_p = _best_bound_energy(params, ell=1)
    return float(E_p - E_s)


def hybrid_strength(gap: float, gap_ref: float, scale: float) -> float:
    """
    Центрированное преобразование s-p разрыва в безразмерную "силу гибридизации" h∈(0,1).
    gap_ref задаёт точку, где h≈0.5, scale — ширину перехода.
    """
    s = max(float(scale), 1e-9)
    x = (float(gap_ref) - float(gap)) / s
    return float(1.0 / (1.0 + math.exp(-x)))


def infer_port_geometry(base_label: str, ports: int, symmetry_score: float, h: float) -> str:
    """
    Минимальная логика выбора геометрического ярлыка на основе числа портов и
    гибридизационного признака h. base_label используется как fallback.
    """
    base = str(base_label or "").strip() or "none"
    p = int(ports)
    h_clamped = max(0.0, min(float(h), 1.0))

    if p == 4:
        return "tetra"
    if p == 3:
        return "trigonal" if h_clamped > 0.5 else "pyramidal"
    if p == 2:
        return "linear" if h_clamped > 0.5 else "bent"
    if p == 1:
        return "single"
    return base


def canonical_port_vectors(label: str, ports: int) -> np.ndarray:
    """
    Вернуть детерминированный набор портовых направлений (k,3) для данного ярлыка.
    Векторы нормированы, k совпадает с ports, если возможно.
    """
    lab = (label or "").strip().lower()

    if ports <= 0:
        return np.zeros((0, 3), dtype=float)

    if lab == "linear":
        v = np.array([[0.0, 0.0, 1.0], [0.0, 0.0, -1.0]], dtype=float)
        return v[:ports]

    if lab == "trigonal":
        # три вектора в плоскости XY под 120°
        ang = 2.0 * math.pi / 3.0
        vs = []
        for k in range(3):
            theta = k * ang
            vs.append([math.cos(theta), math.sin(theta), 0.0])
        v = np.array(vs, dtype=float)
        return v[:ports]

    if lab == "tetra":
        # вершины правильного тетраэдра
        v = np.array(
            [
                [1.0, 1.0, 1.0],
                [1.0, -1.0, -1.0],
                [-1.0, 1.0, -1.0],
                [-1.0, -1.0, 1.0],
            ],
            dtype=float,
        )
        v /= np.linalg.norm(v, axis=1, keepdims=True)
        return v[:ports]

    if lab == "pyramidal":
        # три вектора вокруг оси z с небольшим наклоном (похож на NH3)
        ang = 2.0 * math.pi / 3.0
        tilt = math.radians(20.0)
        vs = []
        for k in range(3):
            theta = k * ang
            x = math.cos(theta) * math.cos(tilt)
            y = math.sin(theta) * math.cos(tilt)
            z = -math.sin(tilt)
            vs.append([x, y, z])
        v = np.array(vs, dtype=float)
        v /= np.linalg.norm(v, axis=1, keepdims=True)
        return v[:ports]

    if lab == "bent":
        # два вектора в плоскости XZ с углом ~104.5°
        theta = math.radians(52.25)
        v = np.array(
            [
                [math.sin(theta), 0.0, math.cos(theta)],
                [-math.sin(theta), 0.0, math.cos(theta)],
            ],
            dtype=float,
        )
        v /= np.linalg.norm(v, axis=1, keepdims=True)
        return v[:ports]

    if lab == "single":
        v = np.array([[0.0, 0.0, 1.0]], dtype=float)
        return v[:ports]

    # Fallback: равномерно на окружности в плоскости XY
    ang = 2.0 * math.pi / float(max(ports, 1))
    vs = []
    for k in range(ports):
        theta = k * ang
        vs.append([math.cos(theta), math.sin(theta), 0.0])
    v = np.array(vs, dtype=float)
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    return v
