from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple
from collections import defaultdict
from pathlib import Path
import json
import math

import numpy as np
from .fdm import IFS, FDMIntegrator, make_tensor_grid_ifs
from .complexity import atom_complexity_from_adjacency, compute_complexity_features
from core.density_models import beta_effective
from core.thermo_config import get_current_thermo_config, ThermoConfig
from core.spectral_density_ws import (
    WSRadialParams,
    make_ws_rho3d_interpolator,
    make_ws_rho3d_with_diagnostics,
)
from core.port_geometry_spectral import (
    ws_sp_gap,
    hybrid_strength,
    infer_port_geometry,
    canonical_port_vectors,
)
# We import grower inside the report function to avoid circular imports

# ============================================================
# FDM / NUMERICAL ENGINE HELPERS
# ============================================================

def toy_ldos_radial(r: np.ndarray, beta: float) -> np.ndarray:
    """
    Игрушечная 3D-плотность для атома:
    ρ(r) = exp(-beta * |r|^2).

    r: ndarray, shape (N, 3)
    beta: float, жёсткость "ямы".
    Возвращает: ndarray (N,)
    """
    # Sum of squares along last axis (x^2 + y^2 + z^2)
    norm2 = np.sum(r**2, axis=1)
    return np.exp(-beta * norm2)


def cube_to_ball(u: np.ndarray, R: float = 4.0) -> np.ndarray:
    """
    Простое отображение из [0,1]^3 в шар радиуса R (или просто масштабирование в ящик [-R, R]).
    Для FDM интеграла нам главное покрыть область, где функция не исчезает.
    
    Пусть будет линейное растяжение в куб [-R, R]^3 пока что, т.к. 
    гауссиана быстро падает.
    """
    # u in [0, 1] -> x in [-R, R]
    return (u - 0.5) * (2.0 * R)


def _ws_params_from_thermo(cfg: ThermoConfig) -> WSRadialParams:
    return WSRadialParams(
        R_max=float(getattr(cfg, "ws_R_max", 12.0)),
        R_well=float(getattr(cfg, "ws_R_well", 5.0)),
        V0=float(getattr(cfg, "ws_V0", 40.0)),
        N_grid=int(getattr(cfg, "ws_N_grid", 220)),
        ell=int(getattr(cfg, "ws_ell", 0)),
        state_index=int(getattr(cfg, "ws_state_index", 0)),
    )


def _I_box(R: float, beta: float) -> float:
    if beta <= 0.0:
        return 0.0
    # 1D интеграл по [-R, R]: sqrt(pi/beta) * erf(sqrt(beta)*R)
    sqrt_term = math.sqrt(math.pi / float(beta))
    erf_term = math.erf(math.sqrt(float(beta)) * float(R))
    Ix = sqrt_term * erf_term
    return Ix ** 3


def estimate_atom_energy_fdm(atom_z: int, e_port: float) -> float:
    """
    Оценка 'спектральной энергии' атома через FDM-интеграл по 3D-ядру.
    
    Параметры скейлинга (beta) зависят от Z (или E_port).
    
    Args:
        atom_z: Z атома
        e_port: E_port из модели v4
        
    Returns:
        Integral[ exp(-beta * r^2) ] (approx)
    """
    thermo = get_current_thermo_config()
    beta = beta_effective(
        atom_z,
        getattr(thermo, "coupling_density", 0.0),
        model=getattr(thermo, "density_model", "tf_radius"),
        blend=getattr(thermo, "density_blend", "linear"),
        Z_ref=getattr(thermo, "density_Z_ref", 10.0),
    )

    dim = 3
    ifs = make_tensor_grid_ifs(dim=dim, base=2)

    # Legacy Gaussian-only branch (density_source != ws_radial)
    def _integrate_gaussian(R: float) -> float:
        fdm = FDMIntegrator(ifs)
        def integrand(r: np.ndarray) -> np.ndarray:
            return toy_ldos_radial(r, beta)
        val = fdm.integrate(
            integrand, depth=4, dim=dim, transform=lambda u: cube_to_ball(u, R)
        )
        volume = (2.0 * R) ** 3
        return val * volume

    # Check if WS-ветка включена
    c_shape = float(getattr(thermo, "coupling_density_shape", 0.0))
    use_ws = (
        getattr(thermo, "density_source", "gaussian") == "ws_radial"
        and c_shape > 0.0
    )

    R_default = 4.0

    if not use_ws:
        return _integrate_gaussian(R_default)

    # WS branch with box-aware scaling and adaptive R_eff
    params = _ws_params_from_thermo(thermo)
    rho_ws_fn, diag = make_ws_rho3d_with_diagnostics(int(atom_z), params)

    R_eff = max(R_default, 1.2 * float(diag.r_99))

    # Gaussian integral in the same box
    I_box = _I_box(R_eff, beta)

    # Масса WS-плотности в кубе до масштабирования
    fdm_mass = FDMIntegrator(ifs)
    def integrand_ws(r: np.ndarray) -> np.ndarray:
        radii = np.sqrt(np.sum(r * r, axis=1))
        return rho_ws_fn(radii)
    mean_ws = fdm_mass.integrate(
        integrand_ws, depth=4, dim=dim, transform=lambda u: cube_to_ball(u, R_eff)
    )
    volume_eff = (2.0 * R_eff) ** 3
    M_ws_box = mean_ws * volume_eff

    scale_ws = I_box / max(M_ws_box, 1e-30) if I_box > 0.0 else 0.0

    fdm = FDMIntegrator(ifs)
    def integrand(r: np.ndarray) -> np.ndarray:
        rho_gauss = toy_ldos_radial(r, beta)
        radii = np.sqrt(np.sum(r * r, axis=1))
        rho_ws = rho_ws_fn(radii) * scale_ws

        c = max(0.0, min(c_shape, 1.0))
        blend_mode = getattr(thermo, "density_blend", "linear")
        rho_gauss = np.maximum(rho_gauss, 1e-30)
        rho_ws = np.maximum(rho_ws, 1e-30)

        if c <= 0.0:
            return rho_gauss
        if blend_mode == "log":
            return np.exp(
                (1.0 - c) * np.log(rho_gauss) + c * np.log(rho_ws)
            )
        return (1.0 - c) * rho_gauss + c * rho_ws

    val = fdm.integrate(
        integrand, depth=4, dim=dim, transform=lambda u: cube_to_ball(u, R_eff)
    )
    total_energy = val * volume_eff
    return total_energy



PAULING = {
    "H": 2.20,
    "Li": 0.98,
    "Be": 1.57,
    "B": 2.04,
    "C": 2.55,
    "N": 3.04,
    "O": 3.44,
    "F": 3.98,
    "Na": 0.93,
    "Mg": 1.31,
    "Al": 1.61,
    "Si": 1.90,
    "P": 2.19,
    "S": 2.58,
    "Cl": 3.16,
    # 4-й период: калибровка для геометрических клонов
    "K": 0.82,
    "Ca": 1.00,
    "Ga": 1.81,
    "Ge": 2.01,
    "As": 2.18,
    "Se": 2.55,
    "Br": 2.96,
    "Kr": 3.00,
    # 5-й период: клоны 4-го периода
    "Rb": 0.82,
    "Sr": 0.95,
    "In": 1.78,
    "Sn": 1.96,
    "Sb": 2.05,
    "Te": 2.10,
    "I": 2.66,
    # "Xe": 2.60,  # при необходимости можно включить
}

# Model version identifier
MODEL_VERSION = "geom-spec v4.0 (period + eps-coupled full)"

# Spectral modes:
#   "v1"              – baseline (Li~Na, F~Cl clones, no period scaling)
#   "v2_period_split" – period-based scaling of port energies (breaks clones)
#   "v3_eps_coupled"  – eps→chi coupling at FIXED geometry (Super-O experiment)
#   "v4_full"         – v2 + v3 combined (period scaling + eps coupling)
SPECTRAL_MODE_DEFAULT = "v4_full"  # Production default
SPECTRAL_MODE = SPECTRAL_MODE_DEFAULT
SPECTRAL_MODE_V3 = "v3_eps_coupled"
SPECTRAL_MODE_V4 = "v4_full"

# Calibrated spectral parameters (v1.0)
ALPHA_CALIBRATED = 1.237
GAMMA_DONOR_CALIBRATED = 3.0
KCENTER_CALIBRATED = 0.1
EPS_NEUTRAL = 0.06

# v2 period split parameters (R&D tuning)
V2_PERIOD_EXPONENT = 0.7  # E_port ~ period^(-k)

# v3/v4 epsilon–chi coupling strength (λ_ε)
EPS_COUPLING_STRENGTH = 0.4

FIT_ELEMENTS = [
    "H",
    "Li",
    "Be",
    "B",
    "C",
    "N",
    "O",
    "F",
    "Na",
    "Mg",
    "Al",
    "Si",
    "P",
    "S",
    "Cl",
]

from core.thermo_config import get_current_thermo_config

# Weight for complexity in F_geom
W_COMPLEXITY = 0.3


def compute_W_complexity_eff(
    W_base: float, coupling: float, temperature: float
) -> float:
    """
    Legacy: coupling<=0 -> W_base.
    Physics: coupling in (0..1] -> blend between W_base и W_base*T.
    """
    c = max(0.0, min(float(coupling), 1.0))
    T = max(float(temperature), 1e-9)
    return W_base * (1.0 - c) + (W_base * T) * c

@dataclass
class AtomGraph:
    """
    Игрушечное представление атома как графа.

    Здесь граф задаётся неявно через простые численные характеристики:
    число вершин/рёбер, число портов и условную меру симметрии, а также
    игрушечным спектральным параметром epsilon.
    """

    name: str               # символ элемента, например "C"
    Z: int                  # атомный номер
    nodes: int              # число вершин графа
    edges: int              # число рёбер
    ports: int              # число валентных "портов"
    symmetry_score: float   # чем меньше, тем ближе к идеальной симметрии
    port_geometry: str      # тип геометрии портов ('tetra', 'trigonal', ...)
    role: str               # роль в сетях: 'inert', 'hub', 'bridge', 'terminator'
    epsilon: float = 0.0    # игрушечное положение уровня ε_Z относительно Среды
    notes: str = ""         # произвольный комментарий
    softness: float = 0.0   # мягкость в росте деревьев (penalty factor)

    def effective_port_geometry(self, thermo: Optional[ThermoConfig] = None) -> str:
        """
        Опционально скорректированная геометрия портов с учётом спектрального признака.
        При coupling_port_geometry == 0 или legacy-режиме возвращает исходный ярлык.
        """
        base = self.port_geometry
        if thermo is None:
            thermo = get_current_thermo_config()
        c = max(0.0, min(float(getattr(thermo, "coupling_port_geometry", 0.0)), 1.0))
        if c <= 0.0 or getattr(thermo, "port_geometry_source", "legacy") == "legacy":
            return base

        params = WSRadialParams(
            R_max=float(getattr(thermo, "ws_geom_R_max", 25.0)),
            R_well=float(getattr(thermo, "ws_geom_R_well", 6.0)),
            V0=float(getattr(thermo, "ws_geom_V0", 45.0)),
            N_grid=int(getattr(thermo, "ws_geom_N_grid", 800)),
            ell=0,
            state_index=0,
        )
        gap = ws_sp_gap(self.Z, params)
        h_raw = hybrid_strength(
            gap,
            float(getattr(thermo, "ws_geom_gap_ref", 1.0)),
            float(getattr(thermo, "ws_geom_gap_scale", 1.0)),
        )

        # Лёгкий blend по h, если задан портовый blend режим
        mode = getattr(thermo, "port_geometry_blend", "linear")
        if mode == "log":
            # лог-бленд по odds(h), но здесь хватит линейного
            h_eff = h_raw
        else:
            h_eff = h_raw

        inferred = infer_port_geometry(base, self.ports, self.symmetry_score, h_eff)
        # Пока что делаем ступенчатый выбор: c<0.5 -> legacy, c>=0.5 -> spectral
        return inferred if c >= 0.5 else base

    def port_vectors(self, thermo: Optional[ThermoConfig] = None) -> np.ndarray:
        """
        Вернуть набор портовых направлений (ports,3) в зависимости от эффективной геометрии.
        """
        if thermo is None:
            thermo = get_current_thermo_config()
        label = self.effective_port_geometry(thermo)
        return canonical_port_vectors(label, self.ports)

    def adjacency_matrix(self) -> np.ndarray:
        """
        Вернуть каноническую матрицу смежности атомного графа.
        Прокси к canonical_adjacency_matrix.
        """
        return self.canonical_adjacency_matrix()

    def adjacency_matrix(self) -> np.ndarray:
        """
        Вернуть каноническую матрицу смежности атомного графа.
        Прокси к canonical_adjacency_matrix.
        """
        return self.canonical_adjacency_matrix()

    @property
    def period(self) -> int:
        """Период элемента в таблице Менделеева (приближённо по Z)."""
        if self.Z <= 2:
            return 1
        elif self.Z <= 10:
            return 2
        elif self.Z <= 18:
            return 3
        elif self.Z <= 36:
            return 4
        else:
            return 5

    def effective_softness(self, thermo) -> float:
        """
        Legacy: coupling_softness<=0 -> softness из atoms_db.
        Physics: при coupling_softness>0 подмешиваем оценку мягкости
        из спектрального масштаба epsilon_spec и периода.
        """
        base = max(0.0, min(float(getattr(self, "softness", 0.0)), 0.95))
        c = max(0.0, min(float(getattr(thermo, "coupling_softness", 0.0)), 1.0))

        if c <= 0.0:
            return base

        eps = float(self.epsilon_spec())
        spec_soft = 1.0 / (1.0 + eps)

        per = max(1.0, float(self.period))
        per_factor = (per - 1.0) / (per + 1.0)
        spec_soft = spec_soft * (0.5 + 0.5 * per_factor)
        spec_soft = max(0.0, min(float(spec_soft), 0.95))

        return max(0.0, min(base * (1.0 - c) + spec_soft * c, 0.95))

    def cyclomatic_number(self) -> int:
        """
        Цикломатическое число графа как приближение степени "завязанности".

        Для связного графа:  mu = E - V + 1.
        При дереве му = 0, при наличии p независимых циклов mu >= 1.
        """
        return self.edges - self.nodes + 1

    def canonical_adjacency_matrix(self) -> np.ndarray:
        """
        Канонический граф атома как матрица смежности A.

        - узлы: 0..nodes-1
        - сначала строим путь 0-1-2-...-(nodes-1),
          чтобы граф был связным,
        - затем добавляем дополнительные рёбра из 0 к 2,3,4,...
          пока не добьёмся заданного числа рёбер.

        Это простая игрушечная реализация структуры Среды
        с заданными V=nodes и E=edges.
        """
        n = self.nodes
        A = np.zeros((n, n), dtype=float)
        if n <= 1:
            return A

        # Связный путь
        for i in range(n - 1):
            A[i, i + 1] = 1.0
            A[i + 1, i] = 1.0

        extra = self.edges - (n - 1)
        j = 2
        # Добиваемся нужного числа рёбер, добавляя связи от 0 к j
        while extra > 0 and j < n:
            if A[0, j] == 0.0:
                A[0, j] = 1.0
                A[j, 0] = 1.0
                extra -= 1
            j += 1

        return A

    def laplacian_eigenvalues(self) -> np.ndarray:
        """
        Собственные значения лапласовского оператора L = D - A
        канонического графа атома.

        Возвращает отсортированный массив λ_i >= 0.
        """
        A = self.canonical_adjacency_matrix()
        if A.size == 0:
            return np.zeros(0, dtype=float)
        degrees = A.sum(axis=1)
        D = np.diag(degrees)
        L = D - A
        vals = np.linalg.eigvalsh(L)
        vals.sort()
        return vals

    def epsilon_spec(self, tol: float = 1e-8) -> float:
        """
        Игрушечный спектральный масштаб ε_spec(Z):
        первая ненулевая собственная величина лапласиана (значение Фидлера).

        Если все λ_i ≈ 0 (например, одиночный узел H), возвращает 0.
        """
        vals = self.laplacian_eigenvalues()
        for v in vals:
            if v > tol:
                return float(v)
        return 0.0

    def F_spec_toy(self, beta: float = 0.5) -> float:
        """
        Игрушечный спектральный функционал:

            F_spec^toy(Z) = Σ_i exp(-beta * λ_i),

        где λ_i — собственные значения лапласовского оператора.
        При малом beta вклад сильнее определяют малые λ_i.
        """
        vals = self.laplacian_eigenvalues()
        if vals.size == 0:
            return 0.0
        return float(np.exp(-beta * vals).sum())

    def ldos(
        self,
        omega_min: float = 0.0,
        omega_max: float = 6.0,
        n_points: int = 200,
        eta: float = 0.1,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Игрушечная локальная спектральная плотность (LDOS) для атомного графа.

        Строим дискретное распределение по собственным значениям λ_i
        лапласовского оператора и сглаживаем его лоренцианами:

            rho(ω) = (1/π) * Σ_i [ η / ((ω - λ_i)^2 + η^2) ].

        Возвращает:
            omegas: сетка по ω (np.ndarray длины n_points)
            rhos:   значения rho(ω) на этой сетке
        """
        vals = self.laplacian_eigenvalues()
        omegas = np.linspace(omega_min, omega_max, n_points)
        rhos = np.zeros_like(omegas)

        for lam in vals:
            rhos += (eta / np.pi) / ((omegas - lam) ** 2 + eta ** 2)

        return omegas, rhos

    def F_spec_integral(
        self,
        omega_min: float = 0.0,
        omega_max: float = 6.0,
        n_points: int = 400,
        eta: float = 0.1,
        kind: str = "exp",
        mu_env: float = 0.0,
        beta: float = 0.5,
    ) -> float:
        """
        Интегральная спектральная энергия F_spec через LDOS:

            F_spec = ∫ W(ω) * rho(ω) dω,

        где rho(ω) — лоренциановская LDOS, а вес W(ω) задаётся параметром kind:

        kind="exp":    W(ω) = exp(-beta * ω)
        kind="quad":   W(ω) = (ω - mu_env)^2
        kind="linear": W(ω) = ω
        """
        omegas, rhos = self.ldos(
            omega_min=omega_min,
            omega_max=omega_max,
            n_points=n_points,
            eta=eta,
        )

        if kind == "exp":
            W = np.exp(-beta * omegas)
        elif kind == "quad":
            W = (omegas - mu_env) ** 2
        elif kind == "linear":
            W = omegas
        else:
            raise ValueError(f"Unknown kind of spectral weight: {kind}")

        integrand = W * rhos
        F = np.trapezoid(integrand, omegas)
        return float(F)

    def F_geom(self, a: float = 0.5, b: float = 1.0, c: float = 1.5, use_complexity: bool = True) -> float:
        """
        Простейший геометрический функционал:

            F_geom = a * cyclomatic_number
                   + b * symmetry_score
                   + c * ports
                   + W_COMPLEXITY * C_complex (if enabled)
        """
        base_val = (
            a * self.cyclomatic_number()
            + b * self.symmetry_score
            + c * self.ports
        )
        if use_complexity:
            adj = self.adjacency_matrix()
            # If C_complex fails for some reason (empty graph), we handle inside
            C_complex = atom_complexity_from_adjacency(adj)
            thermo = get_current_thermo_config()
            W_eff = compute_W_complexity_eff(
                W_base=W_COMPLEXITY,
                coupling=getattr(thermo, "coupling_complexity", 0.0),
                temperature=thermo.temperature,
            )
            base_val += W_eff * C_complex
            
        return base_val

    def per_port_energy(
        self, a: float = 0.5, b: float = 1.0, c: float = 1.5
    ) -> Optional[float]:
        """
        Геометрическая "энергия" на один порт.

        Если портов нет (инертный газ), возвращается None.

        В режимах v2_period_split и v4_full энергия масштабируется по периоду:
            E_port_scaled = E_port_base * period^(-V2_PERIOD_EXPONENT)
        """
        if self.ports <= 0:
            return None
        base = self.F_geom(a=a, b=b, c=c) / self.ports

        if SPECTRAL_MODE in ("v2_period_split", "v4_full"):
            scale = self.period ** (-V2_PERIOD_EXPONENT)
            return base * scale
        else:
            return base

    def preferred_angle(self) -> Optional[float]:
        """
        ``Любимый'' угол между двумя портами (в градусах)
        для изолированного атома в базовом состоянии.

        Это не точный предсказанный угол связи, а
        геометрический прототип (аналог sp/sp2/sp3).
        """
        mapping = {
            "linear": 180.0,
            "trigonal": 120.0,
            "tetra": 109.5,
            "pyramidal": 107.0,
            "bent": 104.0,
        }
        return mapping.get(self.port_geometry)

    def chi_geom(
        self,
        a: float = 0.5,
        b: float = 1.0,
        c: float = 1.5,
        alpha: float = 1.0,
    ) -> Optional[float]:
        """
        Простая геометрическая электроотрицательность:
        chi_geom = alpha * E_port.
        """
        e = self.per_port_energy(a=a, b=b, c=c)
        if e is None:
            return None
        return alpha * e


    def chi_geom_signed(
        self,
        a: float = 0.5,
        b: float = 1.0,
        c: float = 1.5,
        alpha: float = 1.0,
        mu_env: float = 0.0,
        eps_neutral: float = 0.1,
    ) -> Optional[float]:
        """
        Знаковая геометрическая электроотрицательность.

        Знак определяется положением epsilon относительно химического
        потенциала Среды mu_env:

          epsilon << mu_env  -> акцептор (chi > 0),
          epsilon >> mu_env  -> донор   (chi < 0),
          epsilon ~= mu_env  -> структурный центр (chi ~ 0).
        """
        e = self.per_port_energy(a=a, b=b, c=c)
        if e is None or self.ports <= 0:
            return None

        delta = self.epsilon - mu_env
        if abs(delta) < eps_neutral:
            sign = 0.0
        elif delta > 0:
            sign = -1.0
        else:
            sign = 1.0

        return sign * alpha * e

    def chi_geom_signed_spec(
        self,
        a: float = 0.5,
        b: float = 1.0,
        c: float = 1.5,
        alpha: float = ALPHA_CALIBRATED,
        eps_neutral: float = EPS_NEUTRAL,
        gamma_donor: float = GAMMA_DONOR_CALIBRATED,
        k_center: float = KCENTER_CALIBRATED,
        eps_coupling: float = EPS_COUPLING_STRENGTH,
    ) -> Optional[float]:
        """
        Спектральная χ_geom с:
          - демпфером для доноров,
          - мягким "фоном" для hub-центров (почти нейтральные элементы).
        """
        e = self.per_port_energy(a=a, b=b, c=c)
        if e is None:
            return None

        chi_abs = alpha * e

        # Инертные: χ = 0
        if self.role == "inert":
            return 0.0

        mu_env_spec = compute_mu_env_spec()
        eps_spec = self.epsilon_spec()

        # v1/v2: старое поведение (без связи с геометрическим epsilon)
        # v3/v4: добавляем геометрический вклад epsilon
        if SPECTRAL_MODE in (SPECTRAL_MODE_V3, SPECTRAL_MODE_V4):
            # v3/v4: добавляем геометрический вклад epsilon (self.epsilon)
            # epsilon < 0 → более глубокая яма (сильнее акцептор)
            # epsilon > 0 → донорный сдвиг
            eps_geom = getattr(self, "epsilon", 0.0)
            eps_eff = eps_spec + eps_coupling * eps_geom
        else:
            eps_eff = eps_spec

        delta = eps_eff - mu_env_spec

        # Нейтральное окно: отдельная обработка hub-центров
        if abs(delta) <= eps_neutral:
            if self.role == "hub":
                if delta > 0.0:
                    # слабый донорный центр в нейтральном окне
                    sign_center = -1.0
                else:
                    # слабый акцепторный центр в нейтральном окне
                    sign_center = 1.0
                return sign_center * (k_center * chi_abs)
            else:
                return 0.0

        # Вне окна: обычные доноры / акцепторы
        if delta > 0.0:
            sign = -1.0
            s = 1.0 / (1.0 + gamma_donor * eps_eff)
        else:
            sign = 1.0
            if SPECTRAL_MODE == SPECTRAL_MODE_V3:
                # v3: более глубокие ямы (delta << 0) усиливают акцепторный характер
                s = 1.0 + eps_coupling * (-delta)
            else:
                s = 1.0

        return sign * (s * chi_abs)


def _load_atoms_from_json(path: Path) -> List[AtomGraph]:
    with path.open("r", encoding="utf-8") as f:
        raw = json.load(f)
    atoms: List[AtomGraph] = []
    for row in raw:
        atoms.append(AtomGraph(**row))
    return atoms


def _make_base_atoms_legacy() -> List[AtomGraph]:
    """
    Прототипы геометрической таблицы для H–Ne.

    Числа подобраны "на глаз", чтобы:
      - от H к Ne росла сложность (nodes / edges),
      - ports соответствовали привычной валентности,
      - He и Ne были близки к замкнутым симметричным конфигурациям.
    """

    return [
        # H: минимальная конфигурация с одним портом, осевая симметрия.
        AtomGraph(
            name="H",
            Z=1,
            nodes=1,
            edges=0,
            ports=1,
            symmetry_score=0.3,
            port_geometry="single",
            role="terminator",
            notes="Базовый тороид с одним портом",
            epsilon=-0.2,
        ),
        # He: замкнутая, очень симметричная конфигурация без портов.
        AtomGraph(
            name="He",
            Z=2,
            nodes=2,
            edges=1,
            ports=0,
            symmetry_score=0.1,
            port_geometry="none",
            role="inert",
            notes="Два сцепленных тора, портов нет (инертный газ)",
            epsilon=-3.0,
        ),
        # Li: He-ядро + один внешний узел.
        AtomGraph(
            name="Li",
            Z=3,
            nodes=3,
            edges=2,
            ports=1,
            symmetry_score=0.5,
            port_geometry="single",
            role="terminator",
            notes="Один внешний 'лепесток' над He-ядром",
            epsilon=2.0,
        ),
        # Be: два внешних узла более симметрично.
        AtomGraph(
            name="Be",
            Z=4,
            nodes=4,
            edges=3,
            ports=2,
            symmetry_score=0.4,
            port_geometry="linear",
            role="bridge",
            notes="Два порта, стремление к линейной/плоской геометрии",
            epsilon=1.0,
        ),
        # B: три внешних узла с тригональной тенденцией.
        AtomGraph(
            name="B",
            Z=5,
            nodes=5,
            edges=4,
            ports=3,
            symmetry_score=0.6,
            port_geometry="trigonal",
            role="hub",
            notes="Три порта, тригональная конфигурация",
            # лёгкий акцепторный сдвиг: B не должен быть чистым донором
            epsilon=-0.1,
        ),
        # C: четырёхпортовая, близкая к тетраэдрической симметрии.
        AtomGraph(
            name="C",
            Z=6,
            nodes=6,
            edges=6,  # первый независимый цикл: mu = 1
            ports=4,
            symmetry_score=0.2,
            port_geometry="tetra",
            role="hub",
            notes="Четыре порта, тетраэдрический прототип",
            epsilon=0.0,
        ),
        # N: три активных порта + одна "пара".
        AtomGraph(
            name="N",
            Z=7,
            nodes=7,
            edges=7,
            ports=3,
            symmetry_score=0.25,
            port_geometry="pyramidal",
            role="hub",
            notes="Три порта + одна пара, аналог lone pair",
            epsilon=-0.5,
        ),
        # O: два порта + две пары под ~90°.
        AtomGraph(
            name="O",
            Z=8,
            nodes=8,
            edges=8,
            ports=2,
            symmetry_score=0.3,
            port_geometry="bent",
            role="bridge",
            notes="Два порта, две неспаренные пары (угол ~104–110°)",
            epsilon=-1.0,
        ),
        # F: почти замкнутый, но с одной выразительной ямой/портом.
        AtomGraph(
            name="F",
            Z=9,
            nodes=9,
            edges=10,  # чуть более завязанная структура
            ports=1,
            symmetry_score=0.35,
            port_geometry="single",
            role="terminator",
            notes="Один сильный порт, почти замкнутая оболочка",
            epsilon=-1.5,
        ),
        # Ne: полностью замкнутая симметричная конфигурация.
        AtomGraph(
            name="Ne",
            Z=10,
            nodes=10,
            edges=11,
            ports=0,
            symmetry_score=0.1,
            port_geometry="none",
            role="inert",
            notes="Замкнутая конфигурация, инертный газ",
            epsilon=-3.0,
        ),
        # --- Третий период: геометрические аналоги Li–Ne ---
        AtomGraph(
            name="Na",
            Z=11,
            nodes=3,
            edges=2,
            ports=1,
            symmetry_score=0.5,
            port_geometry="single",
            role="terminator",
            notes="Геометрический аналог Li: один внешний порт, донор",
            epsilon=2.0,
        ),
        AtomGraph(
            name="Mg",
            Z=12,
            nodes=4,
            edges=3,
            ports=2,
            symmetry_score=0.4,
            port_geometry="linear",
            role="bridge",
            notes="Геометрический аналог Be: два линейных порта",
            epsilon=1.0,
        ),
        AtomGraph(
            name="Al",
            Z=13,
            nodes=5,
            edges=4,
            ports=3,
            symmetry_score=0.6,
            port_geometry="trigonal",
            role="hub",
            notes="Геометрический аналог B: тригональный плоский хаб",
            # тоже чуть акцепторный центр, а не металлический донор
            epsilon=-0.1,
        ),
        AtomGraph(
            name="Si",
            Z=14,
            nodes=6,
            edges=6,
            ports=4,
            symmetry_score=0.2,
            port_geometry="tetra",
            role="hub",
            epsilon=0.0,
            notes="Геометрический аналог C: тетраэдрический 3D-хаб",
            softness=0.30,  # QSG v5.0: silicon is softer than carbon in tree growth
        ),
        AtomGraph(
            name="P",
            Z=15,
            nodes=7,
            edges=7,
            ports=3,
            symmetry_score=0.25,
            port_geometry="pyramidal",
            role="hub",
            notes="Геометрический аналог N: три порта + виртуальный",
            epsilon=-0.5,
        ),
        AtomGraph(
            name="S",
            Z=16,
            nodes=8,
            edges=8,
            ports=2,
            symmetry_score=0.3,
            port_geometry="bent",
            role="bridge",
            notes="Геометрический аналог O: два изогнутых порта",
            epsilon=-1.0,
        ),
        AtomGraph(
            name="Cl",
            Z=17,
            nodes=9,
            edges=10,
            ports=1,
            symmetry_score=0.35,
            port_geometry="single",
            role="terminator",
            notes="Геометрический аналог F: сильный одиночный порт",
            epsilon=-1.5,
        ),
        AtomGraph(
            name="Ar",
            Z=18,
            nodes=10,
            edges=11,
            ports=0,
            symmetry_score=0.1,
            port_geometry="none",
            role="inert",
            notes="Геометрический аналог Ne: инертная замкнутая конфигурация",
            epsilon=-3.0,
        ),
        # --- 4-й период: главные группы (черновой прототип) ---

        # K: геометрический аналог Li/Na (щёлочной донор 4-го периода)
        AtomGraph(
            name="K",
            Z=19,
            nodes=3,
            edges=2,
            ports=1,
            symmetry_score=0.5,
            port_geometry="single",
            role="terminator",
            notes="Геометрический аналог Li/Na: щёлочной донор 4-го периода",
            epsilon=2.0,
        ),

        # Ca: геометрический аналог Be/Mg (щёлочноземельный донор)
        AtomGraph(
            name="Ca",
            Z=20,
            nodes=4,
            edges=3,
            ports=2,
            symmetry_score=0.4,
            port_geometry="linear",
            role="bridge",
            notes="Геометрический аналог Be/Mg: двухпортовый донор 4-го периода",
            epsilon=1.0,
        ),

        # --- 4-й период: d-блок (Sc–Zn) как металлические хабы/inert ---
        # Для простоты все d-металлы моделируются как 6‑портовые
        # октаэдрические графы одинаковой сложности. Роли согласованы
        # с таблицей индексов element_indices_with_dblock.csv.

        # Sc: спектрально почти инертный центр d-блока
        AtomGraph(
            name="Sc",
            Z=21,
            nodes=7,
            edges=9,
            ports=6,
            symmetry_score=0.25,
            port_geometry="octa",
            role="inert",
            notes="Прототип раннего d-металла: октаэдрическая 6‑портовая конфигурация",
            epsilon=-0.1,
        ),

        # Ti: слабый акцепторный центр, но по D/A близок к инертному сектору
        AtomGraph(
            name="Ti",
            Z=22,
            nodes=7,
            edges=9,
            ports=6,
            symmetry_score=0.25,
            port_geometry="octa",
            role="inert",
            notes="Ранний d-металл: октаэдрическая 6‑портовая конфигурация",
            epsilon=0.26,
        ),

        # V: первый выраженный d‑hub
        AtomGraph(
            name="V",
            Z=23,
            nodes=7,
            edges=9,
            ports=6,
            symmetry_score=0.25,
            port_geometry="octa",
            role="hub",
            notes="Ранний d-hub: октаэдрический металлический центр",
            epsilon=0.44,
        ),

        AtomGraph(
            name="Cr",
            Z=24,
            nodes=7,
            edges=9,
            ports=6,
            symmetry_score=0.25,
            port_geometry="octa",
            role="hub",
            notes="Cr: октаэдрический d-hub, χ_spec ~ 0.5",
            epsilon=0.50,
        ),

        AtomGraph(
            name="Mn",
            Z=25,
            nodes=7,
            edges=9,
            ports=6,
            symmetry_score=0.25,
            port_geometry="octa",
            role="inert",
            notes="Mn: геометрически похож на Sc/Ti, слабый акцептор",
            epsilon=0.28,
        ),

        AtomGraph(
            name="Fe",
            Z=26,
            nodes=7,
            edges=9,
            ports=6,
            symmetry_score=0.25,
            port_geometry="octa",
            role="hub",
            notes="Fe: классический d-hub (металлический центр)",
            epsilon=0.84,
        ),

        AtomGraph(
            name="Co",
            Z=27,
            nodes=7,
            edges=9,
            ports=6,
            symmetry_score=0.25,
            port_geometry="octa",
            role="hub",
            notes="Co: поздний d-hub с увеличенной χ_spec",
            epsilon=0.94,
        ),

        AtomGraph(
            name="Ni",
            Z=28,
            nodes=7,
            edges=9,
            ports=6,
            symmetry_score=0.25,
            port_geometry="octa",
            role="hub",
            notes="Ni: поздний d-hub, χ_spec ~ 1.0",
            epsilon=1.00,
        ),

        AtomGraph(
            name="Cu",
            Z=29,
            nodes=7,
            edges=9,
            ports=6,
            symmetry_score=0.25,
            port_geometry="octa",
            role="hub",
            notes="Cu: мягкий d-hub, близок к низкоакцепторному плато",
            epsilon=0.98,
        ),

        AtomGraph(
            name="Zn",
            Z=30,
            nodes=7,
            edges=9,
            ports=6,
            symmetry_score=0.25,
            port_geometry="octa",
            role="hub",
            notes="Zn: замыкающий d-hub, χ_spec ~ 0.48",
            epsilon=0.48,
        ),

        # Ga: геометрический аналог B/Al (слабый акцептор-хаб)
        AtomGraph(
            name="Ga",
            Z=31,
            nodes=5,
            edges=4,
            ports=3,
            symmetry_score=0.6,
            port_geometry="trigonal",
            role="hub",
            notes="Геометрический аналог B/Al: тригональный слабый акцептор",
            epsilon=-0.1,
        ),

        # Ge: геометрический аналог C/Si (четырёхпортовый центр)
        AtomGraph(
            name="Ge",
            Z=32,
            nodes=6,
            edges=6,
            ports=4,
            symmetry_score=0.2,
            port_geometry="tetra",
            role="hub",
            notes="Геометрический аналог C/Si: слабый акцепторный центр 4-го периода",
            epsilon=0.0,
        ),

        # As: геометрический аналог N/P (сильный акцептор-хаб)
        AtomGraph(
            name="As",
            Z=33,
            nodes=7,
            edges=7,
            ports=3,
            symmetry_score=0.25,
            port_geometry="pyramidal",
            role="hub",
            notes="Геометрический аналог N/P: трёхпортовый акцептор 4-го периода",
            epsilon=-0.5,
        ),

        # Se: геометрический аналог O/S (акцепторный мост)
        AtomGraph(
            name="Se",
            Z=34,
            nodes=8,
            edges=8,
            ports=2,
            symmetry_score=0.3,
            port_geometry="bent",
            role="bridge",
            notes="Геометрический аналог O/S: двухпортовый акцепторный мост",
            epsilon=-1.0,
        ),

        # Br: геометрический аналог F/Cl (сильный одиночный порт)
        AtomGraph(
            name="Br",
            Z=35,
            nodes=9,
            edges=10,
            ports=1,
            symmetry_score=0.35,
            port_geometry="single",
            role="terminator",
            notes="Геометрический аналог F/Cl: сильный одиночный порт 4-го периода",
            epsilon=-1.5,
        ),

          # Kr: геометрический аналог Ne/Ar (инертная замкнутая конфигурация)
          AtomGraph(
              name="Kr",
              Z=36,
              nodes=12,
              edges=13,
              ports=0,
              symmetry_score=0.1,
              port_geometry="none",
              role="inert",
              notes="Геометрический аналог Ne/Ar: инертный газ 4-го периода",
              epsilon=-3.0,
          ),

        # --- 5-й период: Rb–Xe как клоны 4-го периода ---------------------
        # Здесь мы копируем геометрию и роли с K–Kr, меняем только name и Z.

        AtomGraph(
            name="Rb",
            Z=37,
            nodes=3,              # как у K
            edges=2,              # как у K
            ports=1,              # как у K
            symmetry_score=0.5,   # как у K
            port_geometry="single",
            role="terminator",
            notes="Клон K (5-й период): щёлочной терминатор",
            epsilon=2.0,          # как у K
        ),

        AtomGraph(
            name="Sr",
            Z=38,
            nodes=4,              # как у Ca
            edges=3,              # как у Ca
            ports=2,              # как у Ca
            symmetry_score=0.4,   # как у Ca
            port_geometry="linear",
            role="bridge",
            notes="Клон Ca (5-й период): щёлочноземельный мост",
            epsilon=1.0,          # как у Ca
        ),

        AtomGraph(
            name="In",
            Z=49,
            nodes=5,              # как у Ga
            edges=4,              # как у Ga
            ports=3,              # как у Ga
            symmetry_score=0.6,   # как у Ga
            port_geometry="trigonal",
            role="hub",
            notes="Клон Ga (5-й период): мягкий p-hub",
            epsilon=-0.1,         # как у Ga
        ),

        AtomGraph(
            name="Sn",
            Z=50,
            nodes=6,              # как у Ge
            edges=6,              # как у Ge
            ports=4,              # как у Ge
            symmetry_score=0.2,   # как у Ge
            port_geometry="tetra",
            role="hub",
            notes="Клон Ge (5-й период): четырёхпортовый hub",
            epsilon=0.0,          # как у Ge
        ),

        AtomGraph(
            name="Sb",
            Z=51,
            nodes=7,              # как у As
            edges=7,              # как у As
            ports=3,              # как у As
            symmetry_score=0.25,  # как у As
            port_geometry="pyramidal",
            role="hub",
            notes="Клон As (5-й период): трёхпортовый hub",
            epsilon=-0.5,         # как у As
        ),

        AtomGraph(
            name="Te",
            Z=52,
            nodes=8,              # как у Se
            edges=8,              # как у Se
            ports=2,              # как у Se
            symmetry_score=0.3,   # как у Se
            port_geometry="bent",
            role="bridge",
            notes="Клон Se (5-й период): мост-акцептор",
            epsilon=-1.0,         # как у Se
        ),

        AtomGraph(
            name="I",
            Z=53,
            nodes=9,              # как у Br
            edges=10,             # как у Br
            ports=1,              # как у Br
            symmetry_score=0.35,  # как у Br
            port_geometry="single",
            role="terminator",
            notes="Клон Br (5-й период): терминатор",
            epsilon=-1.5,         # как у Br
        ),

        AtomGraph(
            name="Xe",
            Z=54,
            nodes=12,             # как у Kr
            edges=13,             # как у Kr
            ports=0,              # как у Kr
            symmetry_score=0.1,   # как у Kr
            port_geometry="none",
            role="inert",
            notes="Клон Kr (5-й период): инертный газ",
            epsilon=-3.0,         # как у Kr
        ),

        # --- 6-й период: Cs/Ba/Tl/Pb как клоны 5-го периода (черновой прототип) ---

        # Cs: клон Rb (щелочной донор 6-го периода)
        AtomGraph(
            name="Cs",
            Z=55,
            nodes=3,              # как у Rb
            edges=2,              # как у Rb
            ports=1,              # как у Rb
            symmetry_score=0.5,   # как у Rb
            port_geometry="single",
            role="terminator",
            notes="Клон Rb (6-й период): щелочной терминатор",
            epsilon=2.0,          # как у Rb/K
        ),

        # Ba: клон Sr (щелочноземельный мост 6-го периода)
        AtomGraph(
            name="Ba",
            Z=56,
            nodes=4,              # как у Sr
            edges=3,              # как у Sr
            ports=2,              # как у Sr
            symmetry_score=0.4,   # как у Sr
            port_geometry="linear",
            role="bridge",
            notes="Клон Sr (6-й период): щелочноземельный мост",
            epsilon=1.0,          # как у Sr/Ca
        ),

        # Tl: клон In (мягкий p-hub 6-го периода)
        AtomGraph(
            name="Tl",
            Z=81,
            nodes=5,              # как у In
            edges=4,              # как у In
            ports=3,              # как у In
            symmetry_score=0.6,   # как у In
            port_geometry="trigonal",
            role="hub",
            notes="Клон In (6-й период): мягкий p-hub",
            epsilon=-0.1,         # как у In/Ga
        ),

        # Pb: клон Sn (четырехпортовый p-hub 6-го периода)
        AtomGraph(
            name="Pb",
            Z=82,
            nodes=6,              # как у Sn
            edges=6,              # как у Sn
            ports=4,              # как у Sn
            symmetry_score=0.2,   # как у Sn
            port_geometry="tetra",
            role="hub",
            notes="Клон Sn (6-й период): четырехпортовый hub",
            epsilon=0.0,          # как у Sn/Ge
        ),
        # --- конец 5-го/6-го периода -------------------------------------------
    ]


def _make_base_atoms() -> List[AtomGraph]:
    """
    Создаёт базовый список атомов.

    Единственный источник правды — data/atoms_db_v1.json.
    Если файла нет или он битый, падаем с осмысленной ошибкой.
    """
    base = Path(__file__).resolve().parents[1]
    json_path = base / "data" / "atoms_db_v1.json"

    if not json_path.exists():
        raise RuntimeError(
            f"atoms_db_v1.json not found at {json_path}. "
            "Сгенерируй его скриптом "
            "`python -m analysis.data_tools.export_atoms_db` "
            "или восстанови из git."
        )

    try:
        return _load_atoms_from_json(json_path)
    except Exception as exc:
        raise RuntimeError(
            f"Failed to load atoms DB from {json_path}: {exc}"
        ) from exc


base_atoms: List[AtomGraph] = _make_base_atoms()


def get_atom(name: str) -> AtomGraph:
    """
    Найти прототип атома по символу элемента.

    Возвращает объект из PERIODIC_TABLE (поддерживает R&D переопределения).
    """
    # Use global PERIODIC_TABLE if available (runtime), otherwise fallback to base_atoms
    # This allows finding dynamically added atoms like "X"
    if "PERIODIC_TABLE" in globals() and name in PERIODIC_TABLE:
        return PERIODIC_TABLE[name]
    
    for atom in base_atoms:
        if atom.name == name:
            return atom
    raise ValueError(f"Unknown atom name: {name}")


# Global periodic table dictionary for R&D experiments
PERIODIC_TABLE: dict = {atom.name: atom for atom in base_atoms}


def get_periodic_table() -> dict:
    """Return the global periodic table dictionary."""
    return PERIODIC_TABLE


class AtomOverrideContext:
    """
    Контекстный менеджер для временного изменения параметров атома
    (например, epsilon и period) в спектральной таблице.
    
    Usage:
        with AtomOverrideContext(PERIODIC_TABLE, "O", epsilon=-5.0):
            mol = make_H2O()  # uses modified O
    """
    def __init__(self, periodic_table: dict, symbol: str, **overrides):
        self.pt = periodic_table
        self.symbol = symbol
        self.overrides = overrides
        self._backup: dict = {}

    def __enter__(self):
        atom = self.pt[self.symbol]
        # Backup and apply overrides
        for name, value in self.overrides.items():
            self._backup[name] = getattr(atom, name)
            # Use object.__setattr__ for frozen dataclass workaround
            object.__setattr__(atom, name, value)
        return atom

    def __exit__(self, exc_type, exc_val, exc_tb):
        atom = self.pt[self.symbol]
        # Restore original values
        for name, old_value in self._backup.items():
            object.__setattr__(atom, name, old_value)




def compute_mu_env_spec() -> float:
    """
    Спектральный химпотенциал среды:

        mu_env_spec = <epsilon_spec(Z)> по всем неинертным атомам.

    Используется как порог между донорами и акцепторами.
    """
    eps_values: List[float] = []
    for atom in base_atoms:
        if atom.role == "inert":
            continue
        eps = atom.epsilon_spec()
        eps_values.append(eps)
    if not eps_values:
        return 0.0
    return sum(eps_values) / len(eps_values)


@dataclass
class Molecule:
    """
    Игрушечное представление молекулы:

    atoms --- список прототипов атомов,
    bonds --- список пар индексов (i, j), задающих связи.
    """

    name: str
    atoms: List[AtomGraph]
    bonds: List[Tuple[int, int]]

    @property
    def nodes(self):
        """
        Backward-compatibility shim for older code/tests.

        New code should use `atoms` explicitly.
        """
        return self.atoms

    @property
    def depth(self) -> int:
        """
        Backward-compatibility shim.

        Depth определяется как максимальная длина пути от "корня"
        (первого атома) по списку связей bonds, если он есть.
        Для пустых/одиночных молекул возвращает 0.
        """
        n = len(self.atoms)
        if n <= 1:
            return 0
        if not self.bonds:
            return 0

        # Строим простую неориентированную смежность и считаем расстояния BFS.
        adj: dict[int, list[int]] = {i: [] for i in range(n)}
        for i, j in self.bonds:
            if 0 <= i < n and 0 <= j < n:
                adj[i].append(j)
                adj[j].append(i)

        visited = {0: 0}
        queue: list[int] = [0]
        while queue:
            v = queue.pop(0)
            for u in adj.get(v, []):
                if u not in visited:
                    visited[u] = visited[v] + 1
                    queue.append(u)

        return max(visited.values()) if visited else 0

    def adjacency_matrix(self) -> np.ndarray:
        """
        Build adjacency matrix of the molecule (NxN).
        """
        n = len(self.atoms)
        A = np.zeros((n, n), dtype=float)
        for i, j in self.bonds:
            A[i, j] = 1.0
            A[j, i] = 1.0
        return A

    def F_mol(self, a: float = 0.5, b: float = 1.0, c: float = 1.5) -> float:
        """
        Простейший молекулярный функционал: сумма F_geom по атомам.

        Угловой вклад F_angle пока не учитывается.
        """
        return sum(atom.F_geom(a=a, b=b, c=c) for atom in self.atoms)

    def bond_polarity(
        self,
        i: int,
        j: int,
        a: float = 0.5,
        b: float = 1.0,
        c: float = 1.5,
        alpha: float = 1.0,
        mu_env: float = 0.0,
        eps_neutral: float = 0.1,
    ) -> float:
        """
        Игрушечная полярность связи A_i--A_j:

          Δχ_geom = χ_sgn(j) - χ_sgn(i).

        Положительный знак: поляризация в сторону j.
        """
        atom_i = self.atoms[i]
        atom_j = self.atoms[j]
        chi_i = atom_i.chi_geom_signed(
            a=a, b=b, c=c, alpha=alpha, mu_env=mu_env, eps_neutral=eps_neutral
        )
        chi_j = atom_j.chi_geom_signed(
            a=a, b=b, c=c, alpha=alpha, mu_env=mu_env, eps_neutral=eps_neutral
        )
        if chi_i is None or chi_j is None:
            raise ValueError("chi_geom_signed is not defined for one of the atoms")
        return chi_j - chi_i

    def bond_polarity_spec(
        self,
        i: int,
        j: int,
        a: float = 0.5,
        b: float = 1.0,
        c: float = 1.5,
        alpha: float = 1.0,
        eps_neutral: float = 0.06,
        gamma_donor: float = 2.0,
    ) -> float:
        """
        Полярность связи A_i--A_j на базе спектральной χ_geom_signed_spec.

          Δχ_spec = χ_spec(j) - χ_spec(i).

        Положительный знак: поляризация в сторону j.
        """
        atom_i = self.atoms[i]
        atom_j = self.atoms[j]
        chi_i = atom_i.chi_geom_signed_spec(
            a=a,
            b=b,
            c=c,
            alpha=alpha,
            eps_neutral=eps_neutral,
            gamma_donor=gamma_donor,
        )
        chi_j = atom_j.chi_geom_signed_spec(
            a=a,
            b=b,
            c=c,
            alpha=alpha,
            eps_neutral=eps_neutral,
            gamma_donor=gamma_donor,
        )
        if chi_i is None or chi_j is None:
            raise ValueError("chi_geom_signed_spec is not defined for one of the atoms")
        return chi_j - chi_i

    def angular_tension_sp3(
        self,
        base_angle: float = 109.5,
        k_angle: float = 0.01,
    ) -> float:
        """
        Игрушечная оценка sp3-углового напряжения в молекуле.

        Для атомов с числом связей > 1 и определённым preferred_angle()
        оценивается вклад

          F_angle ≈ k_angle * n_virtual * (base_angle - theta0)^2,

        где n_virtual = max(0, 4 - ports) — число "виртуальных портов"
        относительно полного тетраэдра.
        """
        # Степени вершин (число связей у каждого атома).
        degrees = [0] * len(self.atoms)
        for i, j in self.bonds:
            degrees[i] += 1
            degrees[j] += 1

        total = 0.0
        for idx, atom in enumerate(self.atoms):
            # Атомы с одной связью (терминаторы) углового вклада не дают.
            if degrees[idx] <= 1:
                continue

            theta0 = atom.preferred_angle()
            if theta0 is None:
                continue

            n_virtual = max(0, 4 - atom.ports)
            if n_virtual <= 0:
                continue

            delta = base_angle - theta0
            total += k_angle * n_virtual * (delta ** 2)

        return total

    # -------------------------------
    # Спектральное перераспределение зарядов в молекуле
    # -------------------------------

    def _graph_distances(self) -> List[List[float]]:
        """
        Топологические расстояния между атомами по self.bonds.

        Возвращает матрицу d[i][j] (целые расстояния по графу, либо +inf).
        """
        import math
        from collections import deque

        n = len(self.atoms)
        if n == 0:
            return []

        adj: List[List[int]] = [[] for _ in range(n)]
        for i, j in self.bonds:
            adj[i].append(j)
            adj[j].append(i)

        dists: List[List[float]] = [[math.inf] * n for _ in range(n)]

        for s in range(n):
            dists[s][s] = 0.0
            q = deque([s])
            while q:
                u = q.popleft()
                for v in adj[u]:
                    if dists[s][v] is math.inf:
                        dists[s][v] = dists[s][u] + 1.0
                        q.append(v)

        return dists

    def total_molecular_energy(
        self,
        a: float = 0.5,
        b: float = 1.0,
        c: float = 1.5,
        alpha: float = ALPHA_CALIBRATED,
        eps_neutral: float = EPS_NEUTRAL,
        gamma_donor: float = GAMMA_DONOR_CALIBRATED,
        k_center: float = KCENTER_CALIBRATED,
        total_charge: float = 0.0,
        # параметры flow
        hardness_offset: float = 0.5,
        hardness_scale: float = 1.0,
        interaction_scale: float = 1.0,
        interaction_power: float = 1.0,
        interaction_floor: float = 0.5,
        # углы
        base_angle: float = 109.5,
        k_angle: float = 0.01,
    ) -> float:
        """
        Полная энергия молекулы:
        F_mol = F_geom (атомы) + F_angle (углы) + F_flow (потоки зарядов).
        """
        # 1. Атомная геометрическая энергия
        F_geom = sum(atom.F_geom(a=a, b=b, c=c) for atom in self.atoms)

        # 2. Угловое напряжение (sp3)
        F_angle = self.angular_tension_sp3(
            base_angle=base_angle,
            k_angle=k_angle,
        )

        # 3. Энергия потоков
        q, chi, eta, mu, F_flow = self.spectral_charges(
            total_charge=total_charge,
            a=a,
            b=b,
            c=c,
            alpha=alpha,
            eps_neutral=eps_neutral,
            gamma_donor=gamma_donor,
            k_center=k_center,
            hardness_offset=hardness_offset,
            hardness_scale=hardness_scale,
            interaction_scale=interaction_scale,
            interaction_power=interaction_power,
            interaction_floor=interaction_floor,
        )

        return F_geom + F_angle + F_flow

    def spectral_charges(
        self,
        total_charge: float = 0.0,
        a: float = 0.5,
        b: float = 1.0,
        c: float = 1.5,
        alpha: float = ALPHA_CALIBRATED,
        eps_neutral: float = EPS_NEUTRAL,
        gamma_donor: float = GAMMA_DONOR_CALIBRATED,
        k_center: float = KCENTER_CALIBRATED,
        hardness_offset: float = 0.5,
        hardness_scale: float = 1.0,
        interaction_scale: float = 1.0,
        interaction_power: float = 1.0,
        interaction_floor: float = 0.5,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
        """
        QEq-подобное уравнивание зарядов на основе χ_spec и E_port.

        Возвращает кортеж:
            q      — массив частичных зарядов (q_i > 0 ⇒ дефицит электронов),
            chi    — вектор χ_spec для атомов,
            eta    — диагональные "жёсткости" J_ii,
            mu     — общий химический потенциал λ,
            F_flow — энергия перераспределения зарядов.

        total_charge задаёт суммарный заряд молекулы (обычно 0.0).
        """
        import math

        n = len(self.atoms)
        if n == 0:
            return (
                np.zeros(0, dtype=float),
                np.zeros(0, dtype=float),
                np.zeros(0, dtype=float),
                0.0,
                0.0,
            )

        chi = np.zeros(n, dtype=float)
        eta = np.zeros(n, dtype=float)

        for i, atom in enumerate(self.atoms):
            chi_i = atom.chi_geom_signed_spec(
                a=a,
                b=b,
                c=c,
                alpha=alpha,
                eps_neutral=eps_neutral,
                gamma_donor=gamma_donor,
                k_center=k_center,
            )
            if chi_i is None:
                chi_i = 0.0
            chi[i] = float(chi_i)

            e_port = atom.per_port_energy(a=a, b=b, c=c)
            if e_port is None:
                e_port = 0.0
            eta[i] = float(hardness_offset + hardness_scale * max(e_port, 0.0))

        dists = self._graph_distances()
        J = np.zeros((n, n), dtype=float)

        for i in range(n):
            J[i, i] = eta[i]
            for j in range(i + 1, n):
                d = dists[i][j]
                if math.isinf(d):
                    val = 0.0
                else:
                    val = interaction_scale / (
                        (d + interaction_floor) ** interaction_power
                    )
                J[i, j] = val
                J[j, i] = val

        # Система Лагранжа:
        # [ J   -1 ] [ q ] = [ -chi ]
        # [ 1^T  0 ] [ λ ]   [ Q_tot ]
        M = np.zeros((n + 1, n + 1), dtype=float)
        M[:n, :n] = J
        M[:n, n] = -1.0
        M[n, :n] = 1.0

        rhs = np.zeros(n + 1, dtype=float)
        rhs[:n] = -chi
        rhs[n] = float(total_charge)

        try:
            sol = np.linalg.solve(M, rhs)
            q = sol[:n]
            mu = float(sol[n])
        except np.linalg.LinAlgError:
            q = np.zeros(n, dtype=float)
            mu = 0.0

        # Энергия потока: F_flow = chi·q + 1/2 q^T J q
        F_flow = float(chi.dot(q) + 0.5 * q.dot(J).dot(q))

        return q, chi, eta, mu, F_flow


def make_HF() -> Molecule:
    """
    HF: H--F, два атома, одна связь (0-1).
    """
    h = get_atom("H")
    f = get_atom("F")
    return Molecule(name="HF", atoms=[h, f], bonds=[(0, 1)])


def make_HCl() -> Molecule:
    """
    HCl: H--Cl, два атома, одна связь (0-1).
    """
    h = get_atom("H")
    cl = get_atom("Cl")
    return Molecule(name="HCl", atoms=[h, cl], bonds=[(0, 1)])


def make_LiF() -> Molecule:
    """
    LiF: Li--F, два атома, одна связь (0-1).
    """
    li = get_atom("Li")
    f = get_atom("F")
    return Molecule(name="LiF", atoms=[li, f], bonds=[(0, 1)])


def make_NaCl() -> Molecule:
    """
    NaCl: Na--Cl, два атома, одна связь (0-1).
    """
    na = get_atom("Na")
    cl = get_atom("Cl")
    return Molecule(name="NaCl", atoms=[na, cl], bonds=[(0, 1)])


def make_CH4() -> Molecule:
    """
    CH4: один C-хаб и четыре H-терминатора.
    """
    c = get_atom("C")
    h = get_atom("H")
    atoms = [c, h, h, h, h]
    bonds = [(0, 1), (0, 2), (0, 3), (0, 4)]
    return Molecule(name="CH4", atoms=atoms, bonds=bonds)


def make_NH3() -> Molecule:
    """
    NH3: N-хаб и три H-терминатора.
    """
    n = get_atom("N")
    h = get_atom("H")
    atoms = [n, h, h, h]
    bonds = [(0, 1), (0, 2), (0, 3)]
    return Molecule(name="NH3", atoms=atoms, bonds=bonds)


def make_H2O() -> Molecule:
    """
    H2O: O-bridge и два H-терминатора.
    """
    o = get_atom("O")
    h = get_atom("H")
    atoms = [o, h, h]
    bonds = [(0, 1), (0, 2)]
    return Molecule(name="H2O", atoms=atoms, bonds=bonds)


def make_CCOH() -> Molecule:
    """
    C–C–O–H: упрощённый фрагмент типа C–C–O–H (например, хвост этанола).
    atoms: C0 - C1 - O2 - H3
    """
    c = get_atom("C")
    o = get_atom("O")
    h = get_atom("H")
    atoms = [c, c, o, h]
    bonds = [(0, 1), (1, 2), (2, 3)]
    return Molecule(name="C-CO-H", atoms=atoms, bonds=bonds)


def make_SiOSi() -> Molecule:
    """
    Si–O–Si: мостиковый фрагмент силикатной сетки.
    atoms: Si0 - O1 - Si2
    """
    si = get_atom("Si")
    o = get_atom("O")
    atoms = [si, o, si]
    bonds = [(0, 1), (1, 2)]
    return Molecule(name="Si-O-Si", atoms=atoms, bonds=bonds)


def compute_alpha(a: float = 0.5, b: float = 1.0, c: float = 1.5) -> float:
    """
    Масштабный коэффициент alpha так, чтобы max(E_port) -> 4.0.
    """
    e_values: List[float] = []
    for atom in base_atoms:
        if atom.ports > 0:
            e = atom.per_port_energy(a=a, b=b, c=c)
            if e is not None:
                e_values.append(e)
    if not e_values:
        return 1.0
    max_E = max(e_values)
    return 4.0 / max_E


def print_table(a: float = 0.5, b: float = 1.0, c: float = 1.5) -> None:
    """
    Вывести таблицу H–Ne с геометрическими характеристиками и F_geom.
    """
    header = (
        f"{'El':<3} {'Z':>2} {'ports':>5} {'nodes':>5} {'edges':>5} "
        f"{'mu':>3} {'sym':>6} {'F_geom':>8}"
    )
    print(header)
    print("-" * len(header))

    for atom in sorted(base_atoms, key=lambda a_: a_.Z):
        mu = atom.cyclomatic_number()
        F = atom.F_geom(a=a, b=b, c=c)
        print(
            f"{atom.name:<3} {atom.Z:>2d} {atom.ports:>5d} {atom.nodes:>5d} "
            f"{atom.edges:>5d} {mu:>3d} {atom.symmetry_score:>6.2f} {F:>8.3f}"
        )


def print_port_energies(a: float = 0.5, b: float = 1.0, c: float = 1.5) -> None:
    """
    Вывести таблицу геометрической энергии на порт для элементов с ports > 0.
    """
    atoms_with_ports = [atom for atom in base_atoms if atom.ports > 0]
    atoms_with_ports.sort(
        key=lambda atom: atom.per_port_energy(a=a, b=b, c=c) or 0.0
    )

    print()
    print(f"Per-port geometric energy (a={a}, b={b}, c={c}):")
    header = f"{'El':<3} {'Z':>2} {'ports':>5} {'F_geom':>8} {'E_port':>8}"
    print(header)
    print("-" * len(header))

    for atom in atoms_with_ports:
        F = atom.F_geom(a=a, b=b, c=c)
        e_port = atom.per_port_energy(a=a, b=b, c=c)
        if e_port is None:
            continue
        print(
            f"{atom.name:<3} {atom.Z:>2d} {atom.ports:>5d} "
            f"{F:>8.3f} {e_port:>8.3f}"
        )


def print_port_geometries() -> None:
    """
    Вывести таблицу предпочтительной геометрии портов и углов.
    """
    print()
    header = (
        f"{'El':<3} {'Z':>2} {'ports':>5} {'geom':>10} "
        f"{'angle':>8} {'role':>10}"
    )
    print(header)
    print("-" * len(header))

    for atom in sorted(base_atoms, key=lambda a_: a_.Z):
        angle = atom.preferred_angle()
        angle_str = f"{angle:>8.1f}" if angle is not None else f"{'—':>8}"
        print(
            f"{atom.name:<3} {atom.Z:>2d} {atom.ports:>5d} "
            f"{atom.port_geometry:>10} {angle_str} {atom.role:>10}"
        )


def print_chi_comparison(a: float = 0.5, b: float = 1.0, c: float = 1.5) -> None:
    """
    Сравнение геометрической электроотрицательности с шкалой Полинга
    для элементов H–Cl.
    """
    alpha = compute_alpha(a=a, b=b, c=c)
    print(
        f"Geometric vs Pauling electronegativity "
        f"(a={a}, b={b}, c={c}, alpha={alpha:.3f})"
    )
    header = (
        f"{'El':<2} {'Z':>3} {'role':>10} "
        f"{'E_port':>8} {'chi_abs':>8} {'chi_sgn':>8} {'chi_Pauling':>12}"
    )
    print(header)
    print("-" * len(header))

    for atom in sorted(base_atoms, key=lambda at: at.Z):
        if atom.name not in PAULING:
            continue
        e = atom.per_port_energy(a=a, b=b, c=c)
        chi_abs = atom.chi_geom(a=a, b=b, c=c, alpha=alpha)
        chi_sgn = atom.chi_geom_signed(a=a, b=b, c=c, alpha=alpha)
        chi_p = PAULING[atom.name]
        if e is None or chi_abs is None:
            continue
        print(
            f"{atom.name:<2} {atom.Z:>3d} {atom.role:>10} "
            f"{e:8.3f} {chi_abs:8.3f} {chi_sgn:8.3f} {chi_p:12.2f}"
        )


def print_chi_spectral(
    a: float = 0.5,
    b: float = 1.0,
    c: float = 1.5,
    alpha: float = ALPHA_CALIBRATED,
    eps_neutral: float = EPS_NEUTRAL,
    gamma_donor: float = GAMMA_DONOR_CALIBRATED,
    k_center: float = KCENTER_CALIBRATED,
) -> None:
    """
    Таблица спектральной χ_geom (модуль + знак) на базе epsilon_spec,
    рядом с шкалой Полинга.
    """
    mu_env_spec = compute_mu_env_spec()

    print(
        f"Spectral geometric electronegativity "
        f"(a={a}, b={b}, c={c}, alpha={alpha:.3f}, "
        f"mu_env_spec={mu_env_spec:.3f}, eps_neutral={eps_neutral})"
    )
    header = (
        f"{'El':<2} {'Z':>3} "
        f"{'role':>10} {'eps_spec':>9} "
        f"{'chi_abs':>9} {'chi_spec':>10} {'chi_Pauling':>12}"
    )
    print(header)
    print("-" * len(header))

    for atom in sorted(base_atoms, key=lambda at: at.Z):
        if atom.name not in PAULING:
            continue

        e = atom.per_port_energy(a=a, b=b, c=c)
        if e is None:
            continue

        chi_abs = alpha * e
        chi_spec = atom.chi_geom_signed_spec(
            a=a,
            b=b,
            c=c,
            alpha=alpha,
            eps_neutral=eps_neutral,
            gamma_donor=gamma_donor,
            k_center=k_center,
        )
        chi_p = PAULING[atom.name]

        print(
            f"{atom.name:<2} {atom.Z:>3d} "
            f"{atom.role:>10} {atom.epsilon_spec():9.3f} "
            f"{chi_abs:9.3f} {chi_spec:10.3f} {chi_p:12.2f}"
        )


def print_bond_polarities(
    a: float = 0.5,
    b: float = 1.0,
    c: float = 1.5,
    mu_env: float = 0.0,
    eps_neutral: float = 0.1,
) -> None:
    """
    Вывести полярности связей для простых молекул HF и HCl.
    """
    alpha = compute_alpha(a=a, b=b, c=c)

    hf = make_HF()
    hcl = make_HCl()

    # В обоих случаях связь 0–1: H -> F или H -> Cl
    delta_hf = hf.bond_polarity(
        0, 1, a=a, b=b, c=c, alpha=alpha, mu_env=mu_env, eps_neutral=eps_neutral
    )
    delta_hcl = hcl.bond_polarity(
        0, 1, a=a, b=b, c=c, alpha=alpha, mu_env=mu_env, eps_neutral=eps_neutral
    )

    print()
    print(f"Bond polarities (a={a}, b={b}, c={c}, alpha={alpha:.3f}):")
    print("Molecule  bond   Δchi_sgn   comment")
    print("------------------------------------------")
    print(
        f"HF       H->F  {delta_hf:9.3f}   "
        f"(поляризация к F, акцептор)"
    )
    print(
        f"HCl      H->Cl {delta_hcl:9.3f}   "
        f"(поляризация к Cl, акцептор)"
    )


def classify_bond(delta_chi: float) -> str:
    """
    Грубая игрушечная шкала типа связи по |Δchi_spec|.

      |Δχ| < 0.2        → "ковалентная"
      0.2 ≤ |Δχ| < 3.8  → "полярная"
      |Δχ| ≥ 3.8        → "почти ионная"

    Пороговые значения подобраны феноменологически так, чтобы:
      - H–H, C–C, N–N, O–O и очень слабые гетероядерные связи (H–N, H–O)
        лежали в ковалентной области;
      - HF/HCl, C–O/Si–O, C–F/B–F читались как полярные (сильные
        ковалентные связи);
      - LiF/NaCl, MgF/LiCl и похожие комбинации металл–галоген
        относились к почти ионным связям.
    """
    x = abs(delta_chi)
    if x < 0.2:
        return "ковалентная"
    elif x < 3.8:
        return "полярная"
    else:
        return "почти ионная"


def print_bond_polarities_spec(
    a: float = 0.5,
    b: float = 1.0,
    c: float = 1.5,
    alpha: float = ALPHA_CALIBRATED,
    eps_neutral: float = EPS_NEUTRAL,
    gamma_donor: float = GAMMA_DONOR_CALIBRATED,
) -> None:
    """
    Полярности связей HF, HCl, LiF, NaCl на базе спектральной χ_spec.
    """
    hf = make_HF()
    hcl = make_HCl()
    lif = make_LiF()
    nacl = make_NaCl()

    def bond_delta(
        mol: Molecule,
    ) -> float:
        return mol.bond_polarity_spec(
            0,
            1,
            a=a,
            b=b,
            c=c,
            alpha=alpha,
            eps_neutral=eps_neutral,
            gamma_donor=gamma_donor,
        )

    delta_hf = bond_delta(hf)
    delta_hcl = bond_delta(hcl)
    delta_lif = bond_delta(lif)
    delta_nacl = bond_delta(nacl)

    print()
    print(
        f"Spectral bond polarities (a={a}, b={b}, c={c}, "
        f"alpha={alpha:.3f}, eps_neutral={eps_neutral}, "
        f"gamma_donor={gamma_donor}):"
    )
    header = (
        f"{'Molecule':<8} {'bond':<7} {'Δchi_spec':>10}   "
        f"{'type':<18} {'comment'}"
    )
    print(header)
    print("-" * len(header))

    rows = [
        ("HF", "H->F", delta_hf),
        ("HCl", "H->Cl", delta_hcl),
        ("LiF", "Li->F", delta_lif),
        ("NaCl", "Na->Cl", delta_nacl),
    ]
    for name, label, dchi in rows:
        bond_type = classify_bond(dchi)
        if label.startswith("H->"):
            comment = f"{label}, полярная ковалентная"
        else:
            comment = f"{label}, сильно ионная связь"
        print(
            f"{name:<8} {label:<7} {dchi:10.3f}   "
            f"{bond_type:<18} {comment}"
        )


def get_chi_spec(
    name: str,
    a: float = 0.5,
    b: float = 1.0,
    c: float = 1.5,
    alpha: float = ALPHA_CALIBRATED,
    eps_neutral: float = EPS_NEUTRAL,
    gamma_donor: float = GAMMA_DONOR_CALIBRATED,
    k_center: float = KCENTER_CALIBRATED,
) -> Optional[float]:
    """
    Удобный хелпер: вернуть спектральную χ_spec для данного элемента.
    """
    atom = get_atom(name)
    if atom is None:
        return None
    return atom.chi_geom_signed_spec(
        a=a,
        b=b,
        c=c,
        alpha=alpha,
        eps_neutral=eps_neutral,
        gamma_donor=gamma_donor,
        k_center=k_center,
    )


def print_chain_polarities(
    a: float = 0.5,
    b: float = 1.0,
    c: float = 1.5,
    mu_env: float = 0.0,
    eps_neutral: float = 0.1,
) -> None:
    """
    Вывести полярности связей для цепочек C–C–O–H и Si–O–Si.
    """
    alpha = compute_alpha(a=a, b=b, c=c)

    ccoh = make_CCOH()
    siosi = make_SiOSi()

    def print_for_mol(mol: Molecule) -> None:
        print(f"\nMolecule: {mol.name}")
        print("bond   atoms   Δchi_sgn   comment")
        print("------------------------------------------")
        for k, (i, j) in enumerate(mol.bonds):
            ai = mol.atoms[i]
            aj = mol.atoms[j]
            delta = mol.bond_polarity(
                i,
                j,
                a=a,
                b=b,
                c=c,
                alpha=alpha,
                mu_env=mu_env,
                eps_neutral=eps_neutral,
            )
            direction = (
                f"{ai.name}->{aj.name}"
                if delta > 0
                else f"{aj.name}->{ai.name}"
                if delta < 0
                else "none"
            )
            print(
                f"{k:>4d}   {ai.name}-{aj.name:2}  {delta:9.3f}   "
                f"({direction})"
            )

    print()
    print(f"Chain bond polarities (a={a}, b={b}, c={c}, alpha={alpha:.3f}):")
    print_for_mol(ccoh)
    print_for_mol(siosi)


def print_chain_polarities_spec(
    a: float = 0.5,
    b: float = 1.0,
    c: float = 1.5,
    alpha: float = ALPHA_CALIBRATED,
    eps_neutral: float = EPS_NEUTRAL,
    gamma_donor: float = GAMMA_DONOR_CALIBRATED,
) -> None:
    """
    Спектральные полярности связей для цепочек C–C–O–H и Si–O–Si.
    """
    chains = [
        ("C-CO-H", make_CCOH()),
        ("Si-O-Si", make_SiOSi()),
    ]

    print()
    print(
        f"Spectral chain bond polarities "
        f"(a={a}, b={b}, c={c}, alpha={alpha:.3f}, "
        f"eps_neutral={eps_neutral}, gamma_donor={gamma_donor}):"
    )

    for name, mol in chains:
        print()
        print(f"Molecule: {name}")
        header = (
            f"{'bond':>4} {'atoms':>7} {'Δchi_spec':>10}   "
            f"{'type':<18} {'comment'}"
        )
        print(header)
        print("-" * len(header))

        for idx, (i, j) in enumerate(mol.bonds):
            ai = mol.atoms[i]
            aj = mol.atoms[j]
            dchi = mol.bond_polarity_spec(
                i,
                j,
                a=a,
                b=b,
                c=c,
                alpha=alpha,
                eps_neutral=eps_neutral,
                gamma_donor=gamma_donor,
            )
            bond_type = classify_bond(dchi)
            if dchi > 0:
                direction = f"{ai.name}->{aj.name}"
            elif dchi < 0:
                direction = f"{aj.name}->{ai.name}"
            else:
                direction = "none"

            print(
                f"{idx:>4} {ai.name+'-'+aj.name:>7} {dchi:10.3f}   "
                f"{bond_type:<18} {direction}"
            )


def print_molecule_spectral_charges(
    mol: Molecule,
    name: str,
    total_charge: float = 0.0,
    a: float = 0.5,
    b: float = 1.0,
    c: float = 1.5,
    alpha: float = ALPHA_CALIBRATED,
    eps_neutral: float = EPS_NEUTRAL,
    gamma_donor: float = GAMMA_DONOR_CALIBRATED,
    k_center: float = KCENTER_CALIBRATED,
) -> None:
    """
    Диагностический вывод спектральных частичных зарядов в молекуле.
    """
    q, chi, eta, mu, F_flow = mol.spectral_charges(
        total_charge=total_charge,
        a=a,
        b=b,
        c=c,
        alpha=alpha,
        eps_neutral=eps_neutral,
        gamma_donor=gamma_donor,
        k_center=k_center,
    )

    print(
        f"Spectral charge equilibration for {name} "
        f"(a={a}, b={b}, c={c}, alpha={alpha:.3f}, Q_tot={total_charge:.3f})"
    )
    print(f"Chemical potential mu ≈ {mu:.3f}, F_flow ≈ {F_flow:.3f}")
    print("idx  El   chi_spec    eta(J_ii)          q")
    print("-----------------------------------------------")
    for i, atom in enumerate(mol.atoms):
        el = atom.name
        print(
            f"{i:3d}  {el:2s}  {chi[i]:8.3f}   {eta[i]:10.3f}   {q[i]:10.3f}"
        )
    print()

    print("Bond charge differences (approx dipoles q_j - q_i):")
    print("bond  i-j    Δq")
    print("-----------------")
    for k, (i, j) in enumerate(mol.bonds):
        dq = q[j] - q[i]
        print(f"{k:4d}  {i:2d}-{j:2d}  {dq:8.3f}")
    print()


def print_pair_polarity_map(
    a: float = 0.5,
    b: float = 1.0,
    c: float = 1.5,
    alpha: float = ALPHA_CALIBRATED,
    eps_neutral: float = EPS_NEUTRAL,
    gamma_donor: float = GAMMA_DONOR_CALIBRATED,
    k_center: float = KCENTER_CALIBRATED,
) -> None:
    """
    Диагностика всех пар A–B по χ_spec для элементов H–Cl (FIT_ELEMENTS).

    Для каждой пары (A,B) из FIT_ELEMENTS выводит:
        chi_spec(A), chi_spec(B), Δchi_spec = chi_B - chi_A и тип связи.
    """
    print()
    print(
        "Pairwise spectral bond polarity map "
        f"(a={a}, b={b}, c={c}, alpha={alpha:.3f}, "
        f"eps_neutral={eps_neutral}, gamma_donor={gamma_donor}, "
        f"k_center={k_center})"
    )
    header = (
        f"{'A':<3} {'B':<3} "
        f"{'chi_A':>9} {'chi_B':>9} "
        f"{'Δchi':>9} {'type':>14}"
    )
    print(header)
    print("-" * len(header))

    for i, A in enumerate(FIT_ELEMENTS):
        chi_A = get_chi_spec(
            A,
            a=a,
            b=b,
            c=c,
            alpha=alpha,
            eps_neutral=eps_neutral,
            gamma_donor=gamma_donor,
            k_center=k_center,
        )
        if chi_A is None:
            continue
        for B in FIT_ELEMENTS[i + 1 :]:
            chi_B = get_chi_spec(
                B,
                a=a,
                b=b,
                c=c,
                alpha=alpha,
                eps_neutral=eps_neutral,
                gamma_donor=gamma_donor,
                k_center=k_center,
            )
            if chi_B is None:
                continue
            delta = chi_B - chi_A
            bond_type = classify_bond(delta)
            print(
                f"{A:<3} {B:<3} "
                f"{chi_A:9.3f} {chi_B:9.3f} "
                f"{delta:9.3f} {bond_type:>14}"
            )


def print_sp3_tensions(
    a: float = 0.5,
    b: float = 1.0,
    c: float = 1.5,
    base_angle: float = 109.5,
    k_angle: float = 0.01,
) -> None:
    """
    Вывести игрушечное sp3-угловое напряжение для CH4, NH3 и H2O.
    """
    ch4 = make_CH4()
    nh3 = make_NH3()
    h2o = make_H2O()

    print()
    print(
        f"Approximate sp3 angular tension "
        f"(base_angle={base_angle}, k_angle={k_angle}):"
    )
    print("Molecule  F_geom_sum  F_angle_sp3")
    print("----------------------------------")

    for mol in (ch4, nh3, h2o):
        Fg = mol.F_mol(a=a, b=b, c=c)
        Fa = mol.angular_tension_sp3(base_angle=base_angle, k_angle=k_angle)
        print(f"{mol.name:<7} {Fg:11.3f} {Fa:12.3f}")


def print_spectral_summary(beta: float = 0.5) -> None:
    """
    Вывести спектральную сводку для атомных графов H–Ar:
    Fiedler-значение лапласиана, максимальное λ_max и F_spec^toy.
    """
    print()
    print(f"Spectral summary of atomic graphs (beta={beta}):")
    header = (
        f"{'El':<3} {'Z':>2} {'ports':>5} {'mu':>3} "
        f"{'lambda1':>9} {'lambda_max':>11} {'eps_spec':>9} {'F_spec':>9}"
    )
    print(header)
    print("-" * len(header))

    for atom in sorted(base_atoms, key=lambda a: a.Z):
        vals = atom.laplacian_eigenvalues()
        if len(vals) == 0:
            lam1 = 0.0
            lam_max = 0.0
        elif len(vals) == 1:
            lam1 = 0.0
            lam_max = float(vals[-1])
        else:
            lam1 = float(vals[1])
            lam_max = float(vals[-1])

        eps_spec = atom.epsilon_spec()
        Fspec = atom.F_spec_toy(beta=beta)

        print(
            f"{atom.name:<3} {atom.Z:>2d} {atom.ports:>5d} "
            f"{atom.cyclomatic_number():>3d} "
            f"{lam1:9.3f} {lam_max:11.3f} {eps_spec:9.3f} {Fspec:9.3f}"
        )


def print_spectral_energies(
    omega_min: float = 0.0,
    omega_max: float = 6.0,
    eta: float = 0.1,
    beta: float = 0.5,
) -> None:
    """
    Вывести интегральные спектральные энергии F_spec^int
    для H–Ar с разными весами W(ω).
    """
    print()
    print(
        f"Spectral integral energies "
        f"(omega in [{omega_min},{omega_max}], eta={eta}, beta={beta}):"
    )
    header = (
        f"{'El':<3} {'Z':>2} "
        f"{'F_exp':>10} {'F_quad':>10} {'F_lin':>10}"
    )
    print(header)
    print("-" * len(header))

    for atom in sorted(base_atoms, key=lambda a: a.Z):
        F_exp = atom.F_spec_integral(
            omega_min=omega_min,
            omega_max=omega_max,
            eta=eta,
            beta=beta,
            kind="exp",
        )
        F_quad = atom.F_spec_integral(
            omega_min=omega_min,
            omega_max=omega_max,
            eta=eta,
            kind="quad",
            mu_env=0.0,
        )
        F_lin = atom.F_spec_integral(
            omega_min=omega_min,
            omega_max=omega_max,
            eta=eta,
            kind="linear",
        )
        print(
            f"{atom.name:<3} {atom.Z:>2d} "
            f"{F_exp:10.3f} {F_quad:10.3f} {F_lin:10.3f}"
        )


def print_full_diagnostics(
    a: float = 0.5,
    b: float = 1.0,
    c: float = 1.5,
    alpha: float = ALPHA_CALIBRATED,
    eps_neutral: float = EPS_NEUTRAL,
    omega_min: float = 0.0,
    omega_max: float = 6.0,
    eta: float = 0.1,
    beta: float = 0.5,
) -> None:
    """
    Большая сводка по элементам H–Cl:

    - eps_spec (спектральный масштаб),
    - chi_spec (спектральная χ_geom),
    - chi_Pauling и разница,
    - F_spec^exp, F_spec^quad, F_spec^lin (интегральные спектральные энергии).
    """
    mu_env_spec = compute_mu_env_spec()

    print()
    print(
        "Full spectral diagnostics "
        f"(a={a}, b={b}, c={c}, alpha={alpha:.3f}, "
        f"mu_env_spec={mu_env_spec:.3f}, eps_neutral={eps_neutral}, "
        f"omega=[{omega_min},{omega_max}], eta={eta}, beta={beta})"
    )
    header = (
        f"{'El':<2} {'Z':>3} {'role':>10} "
        f"{'eps_spec':>9} "
        f"{'chi_spec':>9} {'chi_Paul':>9} {'Δchi':>9} "
        f"{'F_exp':>9} {'F_quad':>9} {'F_lin':>9}"
    )
    print(header)
    print("-" * len(header))

    for atom in sorted(base_atoms, key=lambda at: at.Z):
        name = atom.name
        if name not in PAULING:
            continue

        chi_spec = atom.chi_geom_signed_spec(
            a=a, b=b, c=c, alpha=alpha, eps_neutral=eps_neutral
        )
        if chi_spec is None:
            continue
        chi_p = PAULING[name]
        dchi = chi_spec - chi_p

        F_exp = atom.F_spec_integral(
            omega_min=omega_min,
            omega_max=omega_max,
            n_points=400,
            eta=eta,
            kind="exp",
            beta=beta,
        )
        F_quad = atom.F_spec_integral(
            omega_min=omega_min,
            omega_max=omega_max,
            n_points=400,
            eta=eta,
            kind="quad",
            mu_env=0.0,
        )
        F_lin = atom.F_spec_integral(
            omega_min=omega_min,
            omega_max=omega_max,
            n_points=400,
            eta=eta,
            kind="linear",
        )

        print(
            f"{name:<2} {atom.Z:>3d} {atom.role:>10} "
            f"{atom.epsilon_spec():9.3f} "
            f"{chi_spec:9.3f} {chi_p:9.3f} {dchi:9.3f} "
            f"{F_exp:9.3f} {F_quad:9.3f} {F_lin:9.3f}"
        )


def print_role_averages(
    omega_min: float = 0.0,
    omega_max: float = 6.0,
    eta: float = 0.1,
    beta: float = 0.5,
      a: float = 0.5,
      b: float = 1.0,
      c: float = 1.5,
      alpha: float = ALPHA_CALIBRATED,
      eps_neutral: float = EPS_NEUTRAL,
) -> None:
    """
    Усреднённые спектральные характеристики по ролям:

        inert / hub / bridge / terminator.

    Смотрим средние eps_spec, |chi_spec| и F_spec^exp.
    """
    buckets = defaultdict(list)

    for atom in base_atoms:
        role = atom.role

        eps = atom.epsilon_spec()
        chi_spec = atom.chi_geom_signed_spec(
            a=a, b=b, c=c, alpha=alpha, eps_neutral=eps_neutral
        )
        if chi_spec is None:
            chi_abs = 0.0
        else:
            chi_abs = abs(chi_spec)

        F_exp = atom.F_spec_integral(
            omega_min=omega_min,
            omega_max=omega_max,
            n_points=400,
            eta=eta,
            kind="exp",
            beta=beta,
        )

        buckets[role].append((eps, chi_abs, F_exp))

    print()
    print(
        f"Role-averaged spectral characteristics "
        f"(omega=[{omega_min},{omega_max}], eta={eta}, beta={beta})"
    )
    header = (
        f"{'role':>10} {'N':>3} "
        f"{'<eps_spec>':>12} {'<|chi_spec|>':>14} {'<F_exp>':>10}"
    )
    print(header)
    print("-" * len(header))

    for role in sorted(buckets.keys()):
        vals = buckets[role]
        n = len(vals)
        if n == 0:
            continue
        avg_eps = sum(v[0] for v in vals) / n
        avg_chi_abs = sum(v[1] for v in vals) / n
        avg_Fexp = sum(v[2] for v in vals) / n

        print(
            f"{role:>10} {n:>3d} "
            f"{avg_eps:12.3f} {avg_chi_abs:14.3f} {avg_Fexp:10.3f}"
        )


def period_of(Z: int) -> int:
    """
    Игрушечное разбиение на периоды для Z=1..18.
    """
    if Z <= 2:
        return 1
    if Z <= 10:
        return 2
    return 3


def print_spectral_periodic_table(
    a: float = 0.5,
    b: float = 1.0,
    c: float = 1.5,
    alpha: float = ALPHA_CALIBRATED,
    eps_neutral: float = EPS_NEUTRAL,
    gamma_donor: float = GAMMA_DONOR_CALIBRATED,
    omega_min: float = 0.0,
    omega_max: float = 6.0,
    eta: float = 0.1,
    beta: float = 0.5,
) -> None:
    """
    Спектральный "периодический закон" в табличной форме:
    Z, элемент, период, роль, eps_spec, F_exp, E_port, chi_spec.
    """
    print()
    print(
        f"Spectral periodic table "
        f"(a={a}, b={b}, c={c}, alpha={alpha:.3f}, "
        f"eps_neutral={eps_neutral}, gamma_donor={gamma_donor}, "
        f"omega=[{omega_min},{omega_max}], eta={eta}, beta={beta})"
    )
    header = (
        f"{'Z':>2} {'El':<2} {'per':>3} {'role':>10} "
        f"{'eps_spec':>9} {'F_exp':>9} {'E_port':>8} {'chi_spec':>9}"
    )
    print(header)
    print("-" * len(header))

    for atom in sorted(base_atoms, key=lambda at: at.Z):
        Z = atom.Z
        per = period_of(Z)
        eps = atom.epsilon_spec()
        F_exp = atom.F_spec_integral(
            omega_min=omega_min,
            omega_max=omega_max,
            n_points=400,
            eta=eta,
            kind="exp",
            beta=beta,
        )
        e_port = atom.per_port_energy(a=a, b=b, c=c) or 0.0
        chi = atom.chi_geom_signed_spec(
            a=a,
            b=b,
            c=c,
            alpha=alpha,
            eps_neutral=eps_neutral,
            gamma_donor=gamma_donor,
        ) or 0.0

        print(
            f"{Z:>2d} {atom.name:<2} {per:>3d} {atom.role:>10} "
            f"{eps:9.3f} {F_exp:9.3f} {e_port:8.3f} {chi:9.3f}"
        )


def fit_alpha_for_params(
    a: float,
    b: float,
    c: float,
    eps_neutral: float,
    gamma_donor: float,
    k_center: float,
    elements: Optional[List[str]] = None,
) -> float:
    """
    Подбор оптимального alpha в смысле МНК:

        chi_spec(Z; alpha) ≈ chi_Pauling(Z)

    при фиксированных eps_neutral, gamma_donor, k_center.

    Практически: сначала считаем chi_spec при alpha=1.0,
    потом alpha* = (sum g_Z * chi_Pauling) / (sum g_Z^2).
    """
    if elements is None:
        elements = FIT_ELEMENTS

    num = 0.0
    den = 0.0

    for atom in base_atoms:
        if atom.name not in elements:
            continue

        g = atom.chi_geom_signed_spec(
            a=a,
            b=b,
            c=c,
            alpha=1.0,
            eps_neutral=eps_neutral,
            gamma_donor=gamma_donor,
            k_center=k_center,
        )
        if g is None or abs(g) < 1e-12:
            continue

        chi_p = PAULING[atom.name]
        num += g * chi_p
        den += g * g

    if den == 0.0:
        return 1.0
    return num / den


def loss_for_params(
    a: float,
    b: float,
    c: float,
    eps_neutral: float,
    gamma_donor: float,
    k_center: float,
    elements: Optional[List[str]] = None,
) -> Tuple[float, float]:
    """
    Возвращает (loss, alpha_star) для данных параметров.

    loss = сумма квадратов (chi_spec - chi_Pauling)^2 по FIT_ELEMENTS.
    alpha_star подбирается оптимально для этих gamma_donor, k_center.
    """
    if elements is None:
        elements = FIT_ELEMENTS

    alpha_star = fit_alpha_for_params(
        a=a,
        b=b,
        c=c,
        eps_neutral=eps_neutral,
        gamma_donor=gamma_donor,
        k_center=k_center,
        elements=elements,
    )

    loss = 0.0

    for atom in base_atoms:
        if atom.name not in elements:
            continue

        g = atom.chi_geom_signed_spec(
            a=a,
            b=b,
            c=c,
            alpha=1.0,
            eps_neutral=eps_neutral,
            gamma_donor=gamma_donor,
            k_center=k_center,
        )
        if g is None:
            continue

        chi_model = alpha_star * g
        chi_p = PAULING[atom.name]
        diff = chi_model - chi_p
        loss += diff * diff

    return loss, alpha_star


def grid_fit_geom_spectral_params(
    a: float = 0.5,
    b: float = 1.0,
    c: float = 1.5,
    eps_neutral: float = 0.06,
) -> None:
    """
    Грубый перебор по gamma_donor и k_center для минимизации
    расхождения с χ Полинга по FIT_ELEMENTS.
    """
    gamma_grid = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
    kcenter_grid = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]

    best_loss: Optional[float] = None
    best_params: Optional[Tuple[float, float, float]] = None

    print("Grid search over (gamma_donor, k_center):")
    print(f"{'gamma':>7} {'k_center':>9} {'loss':>12} {'alpha*':>10}")
    print("-" * 42)

    for gamma in gamma_grid:
        for kc in kcenter_grid:
            loss, alpha_star = loss_for_params(
                a=a,
                b=b,
                c=c,
                eps_neutral=eps_neutral,
                gamma_donor=gamma,
                k_center=kc,
            )
            print(f"{gamma:7.3f} {kc:9.3f} {loss:12.4f} {alpha_star:10.4f}")

            if best_loss is None or loss < best_loss:
                best_loss = loss
                best_params = (gamma, kc, alpha_star)

    print("\nBest parameters found:")
    if best_params is not None:
        gamma, kc, alpha_star = best_params
        print(
            f"  gamma_donor = {gamma:.3f}, "
            f"k_center = {kc:.3f}, alpha* = {alpha_star:.3f}, "
            f"loss = {best_loss:.4f}"
        )
    else:
        print("  (no valid combination found)")


def print_molecule_energies() -> None:
    """
    Вывести полные энергии для набора тестовых молекул.
    """
    mols = [
        make_CH4(),
        make_NH3(),
        make_H2O(),
        make_HF(),
        make_HCl(),
        make_LiF(),
        make_NaCl(),
        make_CCOH(),
        make_SiOSi(),
    ]
    print()
    print("Molecule   F_total")
    print("-------------------")
    for m in mols:
        E = m.total_molecular_energy()
        print(f"{m.name:<8} {E:8.3f}")


def print_molecule_energy_breakdown() -> None:
    """
    Разложение полной энергии по вкладам:
    F_geom, F_angle, F_flow, F_total.
    """
    mols = [
        make_CH4(),
        make_NH3(),
        make_H2O(),
        make_HF(),
        make_HCl(),
        make_LiF(),
        make_NaCl(),
        make_CCOH(),
        make_SiOSi(),
    ]
    print()
    print("Molecule   F_geom    F_angle    F_flow    F_total")
    print("-------------------------------------------------")
    for m in mols:
        a, b, c = 0.5, 1.0, 1.5
        F_geom = sum(at.F_geom(a=a, b=b, c=c) for at in m.atoms)
        F_angle = m.angular_tension_sp3()
        q, chi, eta, mu, F_flow = m.spectral_charges()
        F_total = F_geom + F_angle + F_flow
        print(f"{m.name:<8} {F_geom:8.3f} {F_angle:9.3f} {F_flow:9.3f} {F_total:9.3f}")


def reaction_energy(reactants: List[Molecule], products: List[Molecule]) -> float:
    """
    Энергия реакции:
      ΔF = Σ F_mol(products) - Σ F_mol(reactants).

    Отрицательное ΔF означает экзотермическую реакцию (выгодную).
    """
    E_react = sum(m.total_molecular_energy() for m in reactants)
    E_prod = sum(m.total_molecular_energy() for m in products)
    return E_prod - E_react


def make_NaF() -> Molecule:
    """
    NaF: Na--F, два атома, одна связь (0-1).
    """
    na = get_atom("Na")
    f = get_atom("F")
    return Molecule(name="NaF", atoms=[na, f], bonds=[(0, 1)])


def make_LiCl() -> Molecule:
    """
    LiCl: Li--Cl, два атома, одна связь (0-1).
    """
    li = get_atom("Li")
    cl = get_atom("Cl")
    return Molecule(name="LiCl", atoms=[li, cl], bonds=[(0, 1)])


def make_CH3F() -> Molecule:
    """
    CH3F: тетраэдрический C с тремя H и одним F.
    Bonds: C(0)-H(1), C(0)-H(2), C(0)-H(3), C(0)-F(4).
    """
    c = get_atom("C")
    h1 = get_atom("H")
    h2 = get_atom("H")
    h3 = get_atom("H")
    f = get_atom("F")
    atoms = [c, h1, h2, h3, f]
    bonds = [(0, 1), (0, 2), (0, 3), (0, 4)]
    return Molecule(name="CH3F", atoms=atoms, bonds=bonds)


def make_CH3Cl() -> Molecule:
    """
    CH3Cl: тетраэдрический C с тремя H и одним Cl.
    Bonds: C(0)-H(1), C(0)-H(2), C(0)-H(3), C(0)-Cl(4).
    """
    c = get_atom("C")
    h1 = get_atom("H")
    h2 = get_atom("H")
    h3 = get_atom("H")
    cl = get_atom("Cl")
    atoms = [c, h1, h2, h3, cl]
    bonds = [(0, 1), (0, 2), (0, 3), (0, 4)]
    return Molecule(name="CH3Cl", atoms=atoms, bonds=bonds)


def make_H2S() -> Molecule:
    """
    H2S: bent, аналог H2O с серой вместо кислорода.
    Bonds: S(0)-H(1), S(0)-H(2).
    """
    s = get_atom("S")
    h1 = get_atom("H")
    h2 = get_atom("H")
    atoms = [s, h1, h2]
    bonds = [(0, 1), (0, 2)]
    return Molecule(name="H2S", atoms=atoms, bonds=bonds)


def print_reaction_examples() -> None:
    """
    Примеры расчёта энергий реакций.
    """
    print()
    print("Reaction energy examples (ΔF = products - reactants):")
    print("------------------------------------------------------")

    # Реакция 1: HF + NaCl → HCl + NaF
    hf = make_HF()
    nacl = make_NaCl()
    hcl = make_HCl()
    naf = make_NaF()
    dF1 = reaction_energy([hf, nacl], [hcl, naf])
    print(f"HF + NaCl → HCl + NaF:  ΔF = {dF1:+.3f}")

    # Реакция 2: LiF + HCl → LiCl + HF
    lif = make_LiF()
    licl = make_LiCl()
    dF2 = reaction_energy([lif, hcl], [licl, hf])
    print(f"LiF + HCl → LiCl + HF:  ΔF = {dF2:+.3f}")

    # Реакция 3: 2 HF → H2 + F2 (игрушечно, без H2/F2)
    # Просто показываем, что можно считать
    print()
    print("(Note: absolute F values are arbitrary; focus on relative ΔF signs)")


def run_v1_baseline_checks() -> None:
    """
    Фиксируем поведение модели v1.0 как baseline.
    Эти тесты НЕ для исправления, а для документирования текущего состояния.
    """
    print()
    print(f"[BASELINE] {MODEL_VERSION}")
    print("=" * 60)

    # 1. Близнецы по χ_spec (F~Cl, Li~Na)
    chi_F = get_chi_spec("F")
    chi_Cl = get_chi_spec("Cl")
    chi_Li = get_chi_spec("Li")
    chi_Na = get_chi_spec("Na")

    print(f"F vs Cl:  χ_spec(F)  = {chi_F:.3f}, χ_spec(Cl) = {chi_Cl:.3f}, Δ = {abs(chi_F - chi_Cl):.6f}")
    print(f"Li vs Na: χ_spec(Li) = {chi_Li:.3f}, χ_spec(Na) = {chi_Na:.3f}, Δ = {abs(chi_Li - chi_Na):.6f}")

    # 2. Реакция обмена HF + NaCl ↔ HCl + NaF (должна быть ≈ 0)
    hf = make_HF()
    hcl = make_HCl()
    naf = make_NaF()
    nacl = make_NaCl()
    dF_exchange = reaction_energy([hf, nacl], [hcl, naf])
    print(f"HF + NaCl → HCl + NaF: ΔF = {dF_exchange:.6f}")

    # 3. Вода — слабый поток в v1.0
    h2o = make_H2O()
    _, _, _, _, F_flow_h2o = h2o.spectral_charges()
    print(f"H2O: F_flow = {F_flow_h2o:.6f}")

    # 4. HF — заметный поток
    _, _, _, _, F_flow_hf = hf.spectral_charges()
    print(f"HF:  F_flow = {F_flow_hf:.6f}")

    print()
    print("[v1.0 baseline complete]")
    print()


def print_organic_testbench() -> None:
    """
    Тест-стенд для органических и полярных молекул.
    Показывает разложение энергии для быстрой диагностики.
    """
    molecules = [
        ("H2O", make_H2O()),
        ("H2S", make_H2S()),
        ("HF", make_HF()),
        ("HCl", make_HCl()),
        ("LiF", make_LiF()),
        ("NaCl", make_NaCl()),
        ("NaF", make_NaF()),
        ("LiCl", make_LiCl()),
        ("CH4", make_CH4()),
        ("CH3F", make_CH3F()),
        ("CH3Cl", make_CH3Cl()),
        ("NH3", make_NH3()),
        ("C-CO-H", make_CCOH()),
        ("Si-O-Si", make_SiOSi()),
    ]

    print()
    print(f"Organic / polar testbench ({MODEL_VERSION}):")
    print("Molecule   F_geom    F_angle    F_flow    F_total")
    print("--------------------------------------------------")
    for name, mol in molecules:
        a, b, c = 0.5, 1.0, 1.5
        F_geom = sum(at.F_geom(a=a, b=b, c=c) for at in mol.atoms)
        F_angle = mol.angular_tension_sp3()
        _, _, _, _, F_flow = mol.spectral_charges()
        F_total = F_geom + F_angle + F_flow
        print(f"{name:8s} {F_geom:8.3f} {F_angle:9.3f} {F_flow:9.3f} {F_total:9.3f}")
    print()


def run_v2_diagnostics() -> None:
    """
    Диагностика v2_period_split режима.
    Показывает как ломается симметрия Li~Na и F~Cl.
    """
    global SPECTRAL_MODE, MODEL_VERSION
    old_mode = SPECTRAL_MODE
    old_version = MODEL_VERSION

    SPECTRAL_MODE = "v2_period_split"
    MODEL_VERSION = "geom-spec v2.0 (period split)"

    print()
    print(f"[DIAGNOSTICS] {MODEL_VERSION}")
    print("=" * 60)
    print(f"V2_PERIOD_EXPONENT = {V2_PERIOD_EXPONENT}")
    print()

    # Сравнение χ_spec для пар F/Cl, Li/Na
    chi_F = get_chi_spec("F")
    chi_Cl = get_chi_spec("Cl")
    chi_Li = get_chi_spec("Li")
    chi_Na = get_chi_spec("Na")

    print("Spectral electronegativity comparison (period split):")
    print(f"  F  (period 2): χ_spec = {chi_F:.3f}")
    print(f"  Cl (period 3): χ_spec = {chi_Cl:.3f}")
    print(f"  Δ(F - Cl) = {chi_F - chi_Cl:+.3f}")
    print()
    print(f"  Li (period 2): χ_spec = {chi_Li:.3f}")
    print(f"  Na (period 3): χ_spec = {chi_Na:.3f}")
    print(f"  Δ(Li - Na) = {chi_Li - chi_Na:+.3f}")
    print()

    # Реакции обмена (теперь ΔF ≠ 0!)
    hf = make_HF()
    hcl = make_HCl()
    naf = make_NaF()
    nacl = make_NaCl()
    lif = make_LiF()
    licl = make_LiCl()

    dF1 = reaction_energy([hf, nacl], [hcl, naf])
    dF2 = reaction_energy([lif, hcl], [licl, hf])

    print("Exchange reactions in v2:")
    print(f"  HF + NaCl → HCl + NaF:  ΔF = {dF1:+.3f}")
    print(f"  LiF + HCl → LiCl + HF:  ΔF = {dF2:+.3f}")
    print()

    # Organic testbench в режиме v2
    molecules = [
        ("H2O", make_H2O()),
        ("H2S", make_H2S()),
        ("HF", make_HF()),
        ("HCl", make_HCl()),
        ("LiF", make_LiF()),
        ("NaCl", make_NaCl()),
        ("NaF", make_NaF()),
        ("LiCl", make_LiCl()),
        ("CH4", make_CH4()),
        ("CH3F", make_CH3F()),
        ("CH3Cl", make_CH3Cl()),
        ("NH3", make_NH3()),
    ]

    print(f"Organic / polar testbench ({MODEL_VERSION}):")
    print("Molecule   F_geom    F_angle    F_flow    F_total")
    print("--------------------------------------------------")
    for name, mol in molecules:
        a, b, c = 0.5, 1.0, 1.5
        F_geom = sum(at.F_geom(a=a, b=b, c=c) for at in mol.atoms)
        F_angle = mol.angular_tension_sp3()
        _, _, _, _, F_flow = mol.spectral_charges()
        F_total = F_geom + F_angle + F_flow
        print(f"{name:8s} {F_geom:8.3f} {F_angle:9.3f} {F_flow:9.3f} {F_total:9.3f}")
    print()

    print("[v2.0 diagnostics complete]")
    print()

    # Восстанавливаем v1 режим
    SPECTRAL_MODE = old_mode
    MODEL_VERSION = old_version


def scan_period_exponent() -> None:
    """
    Сканирование параметра V2_PERIOD_EXPONENT для калибровки v2 модели.
    Показывает χ_spec для F/Cl/Li/Na и ΔF для обменной реакции.
    """
    global V2_PERIOD_EXPONENT, SPECTRAL_MODE
    old_mode = SPECTRAL_MODE
    old_k = V2_PERIOD_EXPONENT

    SPECTRAL_MODE = "v2_period_split"

    print()
    print("Period exponent scan (v2_period_split):")
    print("k       chi_F   chi_Cl   chi_Li   chi_Na   dF(HF+NaCl)")
    print("--------------------------------------------------------")

    ks = [0.3, 0.5, 0.7, 0.9, 1.1, 1.3]
    for k in ks:
        V2_PERIOD_EXPONENT = k

        chi_F = get_chi_spec("F") or 0.0
        chi_Cl = get_chi_spec("Cl") or 0.0
        chi_Li = get_chi_spec("Li") or 0.0
        chi_Na = get_chi_spec("Na") or 0.0

        dF1 = reaction_energy([make_HF(), make_NaCl()],
                              [make_HCl(), make_NaF()])

        print(f"{k:4.1f}   {chi_F:7.3f} {chi_Cl:8.3f} {chi_Li:8.3f} {chi_Na:8.3f} {dF1:11.3f}")

    print()
    print("Interpretation:")
    print("  - Higher k → bigger F/Cl and Li/Na split")
    print("  - Look for dF sign matching known chemistry (NaF more stable than NaCl)")
    print()

    # Восстанавливаем
    SPECTRAL_MODE = old_mode
    V2_PERIOD_EXPONENT = old_k


# ============================================================
# R&D EXPERIMENTS
# ============================================================

def print_single_molecule_breakdown(mol, label: str) -> None:
    """
    Печатает разложение энергии для одной молекулы в удобном формате.
    """
    a, b, c = 0.5, 1.0, 1.5
    F_geom = sum(at.F_geom(a=a, b=b, c=c) for at in mol.atoms)
    F_angle = mol.angular_tension_sp3()
    _, _, _, _, F_flow = mol.spectral_charges()
    F_total = F_geom + F_angle + F_flow
    print(f"  {label:20s}  F_geom={F_geom:7.3f}  F_angle={F_angle:6.3f}  "
          f"F_flow={F_flow:7.3f}  F_total={F_total:7.3f}")


def run_super_O_vs_S_experiment(
    k_period: float = 0.7,
    super_eps: float = -5.0,
) -> None:
    """
    R&D-эксперимент: Super-O vs S.
    
    1) Сравниваем обычную H2O и H2S
    2) Вводим 'Super-O' (очень глубокая epsilon) и сравниваем H2O* с H2O/H2S.
    """
    global SPECTRAL_MODE, V2_PERIOD_EXPONENT
    old_mode = SPECTRAL_MODE
    old_k = V2_PERIOD_EXPONENT

    SPECTRAL_MODE = "v2_period_split"
    V2_PERIOD_EXPONENT = k_period

    print()
    print("=" * 60)
    print("===== R&D EXPERIMENT: SUPER-O vs S =====")
    print("=" * 60)
    print(f"[PARAMS] SPECTRAL_MODE=v2_period_split, k_period={k_period:.3f}, "
          f"super_eps={super_eps:.3f}")

    # Baseline: normal O and S
    print()
    print("--- BASELINE (normal O and S) ---")
    mol_h2o = make_H2O()
    mol_h2s = make_H2S()
    print_single_molecule_breakdown(mol_h2o, "H2O (baseline)")
    print_single_molecule_breakdown(mol_h2s, "H2S (baseline)")

    # Super-O variant: temporarily modify O's epsilon
    print()
    print("--- SUPER-O VARIANT ---")
    print(f"[INFO] Setting O.epsilon = {super_eps} (was {get_atom('O').epsilon})")
    
    with AtomOverrideContext(PERIODIC_TABLE, "O", epsilon=super_eps):
        mol_h2o_super = make_H2O()
        print_single_molecule_breakdown(mol_h2o_super, "H2O* (Super-O)")

    # Comparison summary
    print()
    print("--- COMPARISON ---")
    _, _, _, _, flow_h2o = mol_h2o.spectral_charges()
    _, _, _, _, flow_h2s = mol_h2s.spectral_charges()
    print(f"  F_flow(H2O)  = {flow_h2o:+.4f}")
    print(f"  F_flow(H2S)  = {flow_h2s:+.4f}")
    print(f"  Δ(H2O - H2S) = {flow_h2o - flow_h2s:+.4f}")

    # Restore
    SPECTRAL_MODE = old_mode
    V2_PERIOD_EXPONENT = old_k


def scan_donor_acceptor_decay(
    k_values: tuple = (0.0, 0.3, 0.5, 0.7, 0.9, 1.1, 1.3),
) -> None:
    """
    Сканируем k_period для пары доноров (Li, Na) и акцепторов (F, Cl).
    Показывает, как быстро разваливается симметрия по k.
    """
    global SPECTRAL_MODE, V2_PERIOD_EXPONENT
    old_mode = SPECTRAL_MODE
    old_k = V2_PERIOD_EXPONENT

    SPECTRAL_MODE = "v2_period_split"

    print()
    print("=" * 75)
    print("===== R&D EXPERIMENT: DONOR vs ACCEPTOR DECAY =====")
    print("=" * 75)
    print(" k_period   chi_Li   chi_Na  Δchi_donor   chi_F   chi_Cl  Δchi_acceptor")
    print("-" * 75)

    for k in k_values:
        V2_PERIOD_EXPONENT = k

        chi_Li = get_chi_spec("Li") or 0.0
        chi_Na = get_chi_spec("Na") or 0.0
        chi_F = get_chi_spec("F") or 0.0
        chi_Cl = get_chi_spec("Cl") or 0.0

        delta_d = chi_Li - chi_Na
        delta_a = chi_F - chi_Cl

        print(f"{k:7.3f}  {chi_Li:8.3f} {chi_Na:8.3f} {delta_d:11.3f}  "
              f"{chi_F:8.3f} {chi_Cl:8.3f} {delta_a:14.3f}")

    print()
    print("[INTERPRETATION]")
    print("  - Δchi_donor  = chi_Li - chi_Na  (positive means Li is 'softer donor')")
    print("  - Δchi_acceptor = chi_F - chi_Cl (positive means F is 'stronger acceptor')")
    print()

    # Restore
    SPECTRAL_MODE = old_mode
    V2_PERIOD_EXPONENT = old_k


def find_zero_chemistry_point_for_Cl_vs_H(
    k_min: float = 0.0,
    k_max: float = 2.0,
    k_step: float = 0.1,
    threshold: float = 0.05,
) -> None:
    """
    Сканируем k_period так, чтобы найти область, где chi_spec(Cl) ~= chi_spec(H).
    Отмечаем, где знак delta меняется (примерная 'точка нулевой химии'), и
    ищем k*, при котором |chi_Cl - chi_H| < threshold.
    """
    global SPECTRAL_MODE, V2_PERIOD_EXPONENT
    old_mode = SPECTRAL_MODE
    old_k = V2_PERIOD_EXPONENT

    SPECTRAL_MODE = "v2_period_split"

    print()
    print("=" * 60)
    print("===== R&D EXPERIMENT: ZERO-CHEMISTRY POINT (Cl ~ H) =====")
    print("=" * 60)
    print(" k_period    chi_H      chi_Cl    delta(Cl-H)")
    print("-" * 50)

    prev_delta = None
    prev_k = None
    best_k = None
    best_delta = None
    n_steps = int((k_max - k_min) / k_step) + 1

    for i in range(n_steps):
        k = k_min + i * k_step
        V2_PERIOD_EXPONENT = k

        chi_H = get_chi_spec("H") or 0.0
        chi_Cl = get_chi_spec("Cl") or 0.0
        delta = chi_Cl - chi_H

        print(f"{k:7.3f}  {chi_H:9.3f}  {chi_Cl:9.3f}  {delta:11.3f}")

        if prev_delta is not None and delta * prev_delta < 0:
            print(f"  >>> SIGN CHANGE between k={prev_k:.3f} and k={k:.3f}")

        if best_delta is None or abs(delta) < abs(best_delta):
            best_delta = delta
            best_k = k

        prev_delta = delta
        prev_k = k

    print()
    print("[INTERPRETATION]")
    print("  - When delta(Cl-H) = 0, Cl and H have equal electronegativity")
    print("  - Sign change marks the 'horizon' where Cl crosses H")
    if best_k is not None:
        print(
            f"  - Closest zero-chemistry point: k* ≈ {best_k:.3f}, "
            f"delta(Cl-H) = {best_delta:+.3f}"
        )
        if abs(best_delta) < threshold:
            print(f"    (|delta| < {threshold:.3f} → near-equality region)")
    print()

    # Restore
    SPECTRAL_MODE = old_mode
    V2_PERIOD_EXPONENT = old_k


# ============================================================
# R&D MASTER REPORT
# ============================================================

def collect_element_chi(mode: str) -> dict:
    """
    Возвращает {symbol -> chi_spec} для указанного режима (v1 или v2).
    """
    global SPECTRAL_MODE
    old_mode = SPECTRAL_MODE
    SPECTRAL_MODE = mode if mode == "v1" else "v2_period_split"
    
    result = {}
    for atom in base_atoms:
        if atom.role == "inert":
            continue
        chi = atom.chi_geom_signed_spec()
        if chi is not None:
            result[atom.name] = chi
    
    SPECTRAL_MODE = old_mode
    return result


def print_compact_spectral_table(mode_label: str) -> None:
    """Печатает компактную спектральную таблицу."""
    print(f"\n[{mode_label}] Spectral table (compact):")
    print("Z  El   role        eps_spec   E_port   chi_spec")
    print("-" * 55)
    
    for atom in base_atoms:
        if atom.role == "inert":
            continue
        eps = atom.epsilon_spec()
        e_port = atom.per_port_energy() or 0.0
        chi = atom.chi_geom_signed_spec()
        chi_str = f"{chi:8.3f}" if chi is not None else "    N/A"
        print(f"{atom.Z:2d} {atom.name:3s}  {atom.role:10s}  {eps:8.3f}  {e_port:7.3f}  {chi_str}")


def print_organic_testbench_with_reactivity(mode_label: str) -> None:
    """Печатает organic testbench с индексом реактивности R."""
    molecules = [
        ("H2O", make_H2O()),
        ("H2S", make_H2S()),
        ("HF", make_HF()),
        ("HCl", make_HCl()),
        ("LiF", make_LiF()),
        ("NaCl", make_NaCl()),
        ("CH4", make_CH4()),
        ("CH3F", make_CH3F()),
        ("CH3Cl", make_CH3Cl()),
        ("NH3", make_NH3()),
    ]

    print(f"\n[{mode_label}] Organic testbench with reactivity index:")
    print("Molecule   F_geom   F_angle    F_flow   F_total  R_react")
    print("-" * 60)
    
    for name, mol in molecules:
        a, b, c = 0.5, 1.0, 1.5
        F_geom = sum(at.F_geom(a=a, b=b, c=c) for at in mol.atoms)
        F_angle = mol.angular_tension_sp3()
        _, _, _, _, F_flow = mol.spectral_charges()
        F_total = F_geom + F_angle + F_flow
        denom = max(F_geom + F_angle, 1e-9)
        R = abs(F_flow) / denom
        print(f"{name:8s} {F_geom:8.3f} {F_angle:9.3f} {F_flow:9.3f} {F_total:9.3f} {R:7.4f}")


def print_reaction_block(mode_label: str) -> None:
    """Печатает блок реакций с энергиями."""
    hf   = make_HF()
    hcl  = make_HCl()
    naf  = make_NaF()
    nacl = make_NaCl()
    lif  = make_LiF()
    licl = make_LiCl()
    
    dF1 = reaction_energy([hf, nacl], [hcl, naf])
    dF2 = reaction_energy([lif, hcl], [licl, hf])
    
    print(f"\n[{mode_label}] Reaction energies:")
    print(f"  HF + NaCl → HCl + NaF:  ΔF = {dF1:+.4f}")
    print(f"  LiF + HCl → LiCl + HF:  ΔF = {dF2:+.4f}")
    if abs(dF1) < 0.001:
        print("  (twins: ΔF ≈ 0)")
    else:
        sign = "exothermic" if dF1 < 0 else "endothermic"
        print(f"  (non-zero: {sign})")


def scan_period_exponent_compact(
    k_values: tuple = (0.3, 0.5, 0.7, 0.9, 1.1, 1.3)
) -> None:
    """Компактный k-scan с chi и ΔF."""
    global SPECTRAL_MODE, V2_PERIOD_EXPONENT
    old_mode = SPECTRAL_MODE
    old_k = V2_PERIOD_EXPONENT
    SPECTRAL_MODE = "v2_period_split"

    print("\n[K-SCAN] Period exponent scan (χ and ΔF):")
    print("k      chi_F   chi_Cl  dchi_FCl   chi_Li   chi_Na  dchi_LiNa   dF_exch")
    print("-" * 75)

    for k in k_values:
        V2_PERIOD_EXPONENT = k
        
        chi_F  = get_chi_spec("F") or 0.0
        chi_Cl = get_chi_spec("Cl") or 0.0
        chi_Li = get_chi_spec("Li") or 0.0
        chi_Na = get_chi_spec("Na") or 0.0
        
        dF = reaction_energy([make_HF(), make_NaCl()], [make_HCl(), make_NaF()])
        
        dchi_FCl = chi_F - chi_Cl
        dchi_LiNa = chi_Li - chi_Na
        
        print(f"{k:4.2f}  {chi_F:7.3f} {chi_Cl:8.3f} {dchi_FCl:9.3f}  "
              f"{chi_Li:8.3f} {chi_Na:8.3f} {dchi_LiNa:10.3f} {dF:9.3f}")

    SPECTRAL_MODE = old_mode
    V2_PERIOD_EXPONENT = old_k


def run_super_O_vs_S_enhanced(k_period: float = 0.7, super_eps: float = -5.0) -> None:
    """Расширенный Super-O эксперимент с chi и q."""
    global SPECTRAL_MODE, V2_PERIOD_EXPONENT
    old_mode = SPECTRAL_MODE
    old_k = V2_PERIOD_EXPONENT
    SPECTRAL_MODE = "v2_period_split"
    V2_PERIOD_EXPONENT = k_period

    print(f"\n[SUPER-O] Parameters: k_period={k_period:.3f}, super_eps={super_eps:.3f}")
    
    # Baseline O
    O_atom = get_atom("O")
    S_atom = get_atom("S")
    chi_O_base = O_atom.chi_geom_signed_spec() or 0.0
    chi_S = S_atom.chi_geom_signed_spec() or 0.0
    
    mol_h2o = make_H2O()
    mol_h2s = make_H2S()
    
    q_h2o, _, _, _, flow_h2o = mol_h2o.spectral_charges()
    q_h2s, _, _, _, flow_h2s = mol_h2s.spectral_charges()
    
    # q values: index 0 is central atom (O or S), 1,2 are H
    q_O = q_h2o[0] if len(q_h2o) > 0 else 0.0
    q_S = q_h2s[0] if len(q_h2s) > 0 else 0.0
    
    print(f"\nBaseline O : chi_spec={chi_O_base:+.3f}, eps={O_atom.epsilon:.2f}")
    print(f"             q_O={q_O:+.4f}, F_flow(H2O)={flow_h2o:+.4f}")
    
    print(f"\nSulfur     : chi_spec={chi_S:+.3f}, eps={S_atom.epsilon:.2f}")
    print(f"             q_S={q_S:+.4f}, F_flow(H2S)={flow_h2s:+.4f}")
    
    # Super-O
    with AtomOverrideContext(PERIODIC_TABLE, "O", epsilon=super_eps):
        chi_O_super = get_atom("O").chi_geom_signed_spec() or 0.0
        mol_h2o_super = make_H2O()
        q_super, _, _, _, flow_super = mol_h2o_super.spectral_charges()
        q_O_super = q_super[0] if len(q_super) > 0 else 0.0
        
        print(f"\nSuper-O    : chi_spec={chi_O_super:+.3f}, eps={super_eps:.2f}")
        print(f"             q_O={q_O_super:+.4f}, F_flow(H2O*)={flow_super:+.4f}")
    
    print(f"\n[COMPARISON]")
    print(f"  Δ(F_flow): H2O* - H2O = {flow_super - flow_h2o:+.4f}")
    print(f"  Δ(F_flow): H2O  - H2S = {flow_h2o - flow_h2s:+.4f}")
    print(f"  Δ(q_central): O* - O = {q_O_super - q_O:+.4f}")

    print()
    print("[NOTE] Super-O experiment in v2.0: overriding epsilon(O)")
    print("       does NOT change chi_spec or QEq charges in the current calibration.")
    print("       This is an intentional NEGATIVE result and a marker for future v3.0 work.")

    SPECTRAL_MODE = old_mode
    V2_PERIOD_EXPONENT = old_k


def run_super_O_vs_S_v3(k_period: float = 0.7, super_eps: float = -5.0) -> None:
    """Super-O эксперимент в режиме v3_eps_coupled (без принудительного v2-режима)."""
    global V2_PERIOD_EXPONENT
    old_k = V2_PERIOD_EXPONENT
    V2_PERIOD_EXPONENT = k_period

    print(f"\n[SUPER-O v3] Parameters: k_period={k_period:.3f}, super_eps={super_eps:.3f}")

    # Baseline O/S in current (v3) mode
    O_atom = get_atom("O")
    S_atom = get_atom("S")
    chi_O_base = O_atom.chi_geom_signed_spec() or 0.0
    chi_S = S_atom.chi_geom_signed_spec() or 0.0

    mol_h2o = make_H2O()
    mol_h2s = make_H2S()

    q_h2o, _, _, _, flow_h2o = mol_h2o.spectral_charges()
    q_h2s, _, _, _, flow_h2s = mol_h2s.spectral_charges()

    q_O = q_h2o[0] if len(q_h2o) > 0 else 0.0
    q_S = q_h2s[0] if len(q_h2s) > 0 else 0.0

    print(f"\nBaseline O (v3): chi_spec={chi_O_base:+.3f}, eps={O_atom.epsilon:.2f}")
    print(f"                 q_O={q_O:+.4f}, F_flow(H2O)={flow_h2o:+.4f}")

    print(f"\nSulfur (v3)    : chi_spec={chi_S:+.3f}, eps={S_atom.epsilon:.2f}")
    print(f"                 q_S={q_S:+.4f}, F_flow(H2S)={flow_h2s:+.4f}")

    # Super-O in v3
    with AtomOverrideContext(PERIODIC_TABLE, "O", epsilon=super_eps):
        chi_O_super = get_atom("O").chi_geom_signed_spec() or 0.0
        mol_h2o_super = make_H2O()
        q_super, _, _, _, flow_super = mol_h2o_super.spectral_charges()
        q_O_super = q_super[0] if len(q_super) > 0 else 0.0

        print(f"\nSuper-O (v3)   : chi_spec={chi_O_super:+.3f}, eps={super_eps:.2f}")
        print(f"                 q_O={q_O_super:+.4f}, F_flow(H2O*)={flow_super:+.4f}")

    print(f"\n[COMPARISON v3]")
    print(f"  Δ(F_flow): H2O* - H2O = {flow_super - flow_h2o:+.4f}")
    print(f"  Δ(F_flow): H2O  - H2S = {flow_h2o - flow_h2s:+.4f}")
    print(f"  Δ(q_central): O* - O = {q_O_super - q_O:+.4f}")

    V2_PERIOD_EXPONENT = old_k


# ============================================================
# CSV EXPORT FUNCTIONS (for book tables)
# ============================================================

def export_periodic_table_v3_csv(out=None) -> None:
    """
    Печатает компактную v4.0-таблицу в CSV-формате:
    Z,El,period,role,eps_spec,E_port,chi_spec
    """
    import sys
    if out is None:
        out = sys.stdout
    
    global SPECTRAL_MODE
    old_mode = SPECTRAL_MODE
    SPECTRAL_MODE = SPECTRAL_MODE_V4
    
    try:
        print("Z,El,period,role,eps_spec,E_port,chi_spec", file=out)
        for atom in base_atoms:
            eps_spec = atom.epsilon_spec()
            e_port = atom.per_port_energy() or 0.0
            chi = atom.chi_geom_signed_spec()
            chi_val = chi if chi is not None else 0.0
            print(f"{atom.Z},{atom.name},{atom.period},{atom.role},"
                  f"{eps_spec:.4f},{e_port:.4f},{chi_val:.4f}", file=out)
    finally:
        SPECTRAL_MODE = old_mode


def export_organic_testbench_v3_csv(out=None) -> None:
    """
    Экспортирует органический тестбенч для v3.0:
    mol,F_geom,F_angle,F_flow,F_total,R_react
    """
    import sys
    if out is None:
        out = sys.stdout
    
    global SPECTRAL_MODE
    old_mode = SPECTRAL_MODE
    SPECTRAL_MODE = SPECTRAL_MODE_V4
    
    try:
        mols = [
            ("CH4", make_CH4()),
            ("NH3", make_NH3()),
            ("H2O", make_H2O()),
            ("H2S", make_H2S()),
            ("HF", make_HF()),
            ("HCl", make_HCl()),
            ("LiF", make_LiF()),
            ("NaCl", make_NaCl()),
            ("NaF", make_NaF()),
            ("LiCl", make_LiCl()),
            ("CH3F", make_CH3F()),
            ("CH3Cl", make_CH3Cl()),
        ]
        
        print("mol,F_geom,F_angle,F_flow,F_total,R_react", file=out)
        for name, mol in mols:
            a, b, c = 0.5, 1.0, 1.5
            F_geom = sum(at.F_geom(a=a, b=b, c=c) for at in mol.atoms)
            F_angle = mol.angular_tension_sp3()
            _, _, _, _, F_flow = mol.spectral_charges()
            F_total = F_geom + F_angle + F_flow
            denom = max(F_geom + F_angle, 1e-9)
            R = abs(F_flow) / denom
            print(f"{name},{F_geom:.4f},{F_angle:.4f},{F_flow:.4f},"
                  f"{F_total:.4f},{R:.4f}", file=out)
    finally:
        SPECTRAL_MODE = old_mode


def export_reactions_v3_csv(out=None) -> None:
    """
    Экспортирует ключевые реакции в v4.0:
    reaction,DeltaF
    """
    import sys
    if out is None:
        out = sys.stdout
    
    global SPECTRAL_MODE
    old_mode = SPECTRAL_MODE
    SPECTRAL_MODE = SPECTRAL_MODE_V4
    
    try:
        reactions = [
            ("HF + NaCl -> HCl + NaF",
             [make_HF(), make_NaCl()],
             [make_HCl(), make_NaF()]),
            ("LiF + HCl -> LiCl + HF",
             [make_LiF(), make_HCl()],
             [make_LiCl(), make_HF()]),
            ("HF + LiCl -> HCl + LiF",
             [make_HF(), make_LiCl()],
             [make_HCl(), make_LiF()]),
            ("NaF + HCl -> NaCl + HF",
             [make_NaF(), make_HCl()],
             [make_NaCl(), make_HF()]),
        ]
        
        print("reaction,DeltaF", file=out)
        for label, reactants, products in reactions:
            dF = reaction_energy(reactants, products)
            print(f'"{label}",{dF:.6f}', file=out)
    finally:
        SPECTRAL_MODE = old_mode


# ============================================================
# ELEMENT INDICES (donor/acceptor characterization)
# ============================================================

def compute_element_indices(
    a: float = 0.5,
    b: float = 1.0,
    c: float = 1.5,
) -> list:
    """
    Для каждого элемента из base_atoms считает в v4-режиме:
      - chi_spec (подписанная)
      - E_port
      - DonorIndex  = max(-chi_spec, 0) / max(E_port, 1e-6)
      - AcceptorIndex = max(chi_spec, 0) / max(E_port, 1e-6)
    Возвращает список словарей.
    """
    global SPECTRAL_MODE
    old_mode = SPECTRAL_MODE
    SPECTRAL_MODE = SPECTRAL_MODE_V4
    
    results = []
    for atom in base_atoms:
        chi = atom.chi_geom_signed_spec()
        e_port = atom.per_port_energy(a=a, b=b, c=c)
        
        if chi is None:
            chi = 0.0
        if e_port is None:
            e_port = 0.0
        
        denom = max(e_port, 1e-6)
        D_index = max(-chi, 0.0) / denom  # donor: negative chi
        A_index = max(chi, 0.0) / denom   # acceptor: positive chi
        
        results.append({
            "Z": atom.Z,
            "El": atom.name,
            "role": atom.role,
            "period": atom.period,
            "chi_spec": chi,
            "E_port": e_port,
            "D_index": D_index,
            "A_index": A_index,
        })
    
    SPECTRAL_MODE = old_mode
    return results


def print_element_indices_table(label: str = "v4.0 indices") -> None:
    """
    Печатает компактную таблицу индексов:
    Z  El  role  per  chi_spec  E_port  D_index  A_index
    """
    indices = compute_element_indices()
    
    print(f"\n[{label}] Element donor/acceptor indices:")
    print("Z  El   role       per  chi_spec  E_port  D_index  A_index")
    print("-" * 65)
    
    for item in indices:
        print(f"{item['Z']:2d} {item['El']:3s}  {item['role']:10s} "
              f"{item['period']:2d}  {item['chi_spec']:+7.3f}  "
              f"{item['E_port']:6.3f}  {item['D_index']:7.4f}  {item['A_index']:7.4f}")
    
    print()
    print("[LEGEND]")
    print("  D_index = max(-χ, 0) / E_port  → higher = stronger donor (metals)")
    print("  A_index = max(χ, 0) / E_port   → higher = stronger acceptor (halogens)")


def print_element_quadrants(label: str = "v4.0 classes") -> None:
    """
    Классифицирует элементы по D/A индексам:
      - Metals: D >> 0, A ~ 0
      - Non-metals (Oxidizers): A >> 0, D ~ 0
      - Amphoteric/Borderline: D ~ A or both small
      - Inert: both ~ 0
    """
    indices = compute_element_indices()
    
    print(f"\n[{label}] Element Classification by D/A Index")
    print("Z  El   D_index  A_index  Class")
    print("-" * 45)
    
    for item in indices:
        D = item['D_index']
        A = item['A_index']
        
        # Simple heuristic classification
        if item['role'] == 'inert':
            cls = "Inert"
        elif D > 0.1 and A < 0.05:
            cls = "Metal (Donor)"
        elif A > 0.1 and D < 0.05:
            cls = "Non-metal (Acceptor)"
        elif D > 0.05 and A > 0.05:
            cls = "Amphoteric"
        else:
            cls = "Borderline / Weak"
            
        print(f"{item['Z']:2d} {item['El']:3s}  {D:7.4f}  {A:7.4f}  {cls}")


def predict_bond_polarity_and_type(atom1_symbol: str, atom2_symbol: str) -> None:
    """
    Предсказывает тип связи между двумя атомами на основе v4-параметров.
    """
    global SPECTRAL_MODE
    old_mode = SPECTRAL_MODE
    SPECTRAL_MODE = SPECTRAL_MODE_V4
    
    try:
        a1 = get_atom(atom1_symbol)
        a2 = get_atom(atom2_symbol)
        
        chi1 = a1.chi_geom_signed_spec() or 0.0
        chi2 = a2.chi_geom_signed_spec() or 0.0
        
        e1 = a1.per_port_energy() or 0.1
        e2 = a2.per_port_energy() or 0.1
        
        delta_chi = chi2 - chi1
        abs_delta = abs(delta_chi)
        
        print(f"\n[BOND PREDICTION v4.0] {atom1_symbol} -- {atom2_symbol}")
        print(f"  {atom1_symbol}: chi={chi1:+.3f}, E_port={e1:.3f}")
        print(f"  {atom2_symbol}: chi={chi2:+.3f}, E_port={e2:.3f}")
        print(f"  Delta_chi = {delta_chi:+.3f}")
        
        # Polarity
        if abs_delta < 0.1:
            polarity = "Non-polar / Covalent"
        elif delta_chi > 0:
            polarity = f"Polar: {atom1_symbol}(δ+) -> {atom2_symbol}(δ-)"
        else:
            polarity = f"Polar: {atom2_symbol}(δ+) -> {atom1_symbol}(δ-)"
            
        # Type estimation
        # Heuristic: Ionic if large delta_chi AND one is strong donor, other strong acceptor
        # Covalent if small delta_chi OR both are hard/soft similar
        
        if abs_delta > 1.5:
            bond_type = "Ionic (Strong)"
        elif abs_delta > 0.8:
            bond_type = "Ionic / Polar Covalent"
        else:
            bond_type = "Covalent"
            
        print(f"  > Polarity: {polarity}")
        print(f"  > Est. Type: {bond_type}")
        
    except Exception as e:
        print(f"Error predicting bond: {e}")
    finally:
        SPECTRAL_MODE = old_mode


# ============================================================
# ISLAND OF STABILITY SCAN (Virtual Atom X)
# ============================================================

def make_virtual_molecule(central_symbol: str, ligand_symbol: str, n_ligands: int = 1):
    """Создаёт простую молекулу: центральный атом + лиганды."""
    # Always get fresh copies to avoid graph pollution
    central = get_atom(central_symbol)
    ligand = get_atom(ligand_symbol)
    
    if central is None or ligand is None:
        return None
    
    # Deep copy atoms to ensure they are independent nodes in the new graph
    import copy
    atoms = [copy.deepcopy(central)] + [copy.deepcopy(ligand) for _ in range(n_ligands)]
    bonds = [(0, i+1) for i in range(n_ligands)]
    
    try:
        mol_name = f"{central_symbol}-{ligand_symbol}_{n_ligands}"
        mol = Molecule(name=mol_name, atoms=atoms, bonds=bonds)
        return mol
    except Exception as e:
        print(f"DEBUG: Molecule init failed for {central_symbol}-{ligand_symbol}: {e}")
        return None


def run_virtual_atom_island_scan(
    period_values: tuple = (2, 3, 4, 5),
    epsilon_values: tuple = (-0.5, -1.0, -2.0, -3.0, -5.0),
    ports_values: tuple = (1, 2, 3, 4),
) -> None:
    """
    Для виртуального атома X перебирает сетку параметров (period, epsilon, ports),
    через AtomOverrideContext подставляет их в PERIODIC_TABLE,
    и для каждого варианта:
      - собирает молекулы HX, XO, XF
      - считает F_total в v4 режиме
      - пишет компактный лог
    """
    global SPECTRAL_MODE
    old_mode = SPECTRAL_MODE
    SPECTRAL_MODE = SPECTRAL_MODE_V4
    
    # Use Si as base template for X
    if "X" not in PERIODIC_TABLE:
        # Create virtual atom X (Z=14 gives period=3 like Si)
        # AtomGraph fields: name, Z, nodes, edges, ports, symmetry_score, port_geometry, role, notes, epsilon
        x_atom = AtomGraph(
            name="X", Z=14,  # Z=14 → period=3
            nodes=4, edges=6, ports=4,
            symmetry_score=0.0, port_geometry="tetrahedral",
            role="hub", notes="R&D virtual", epsilon=-1.0
        )
        PERIODIC_TABLE["X"] = x_atom
    
    print("\n[ISLAND SCAN] Virtual atom X parameter sweep")
    print("period  ports   eps     F(HX)    F(XO)    F(XF)   chi_X")
    print("-" * 65)
    
    scanned = 0
    for period in period_values:
        for ports in ports_values:
            for eps in epsilon_values:
                # Compute Z that gives the desired period
                # period 1: Z=1-2, period 2: Z=3-10, period 3: Z=11-18, period 4: Z=19-36
                z_for_period = {1: 1, 2: 5, 3: 14, 4: 25, 5: 40}
                target_z = z_for_period.get(period, 14)
                
                # Apply overrides to X (use Z instead of period)
                with AtomOverrideContext(PERIODIC_TABLE, "X", 
                                         Z=target_z, ports=ports, epsilon=eps):
                    x = get_atom("X")
                    chi_x = x.chi_geom_signed_spec() if x else 0.0
                    
                    # Try to build molecules
                    F_HX = F_XO = F_XF = float('nan')
                    
                    # HX (X + 1 H)
                    mol_hx = make_virtual_molecule("X", "H", 1)
                    if mol_hx:
                        try:
                            F_HX = mol_hx.total_molecular_energy(a=0.5, b=1.0, c=1.5)
                        except Exception as e:
                            print(f"DEBUG: HX energy failed: {e}")
                            pass
                    
                    # XO (X + 1 O) - linear
                    mol_xo = make_virtual_molecule("X", "O", 1)
                    if mol_xo:
                        try:
                            F_XO = mol_xo.total_molecular_energy(a=0.5, b=1.0, c=1.5)
                        except Exception as e:
                            print(f"DEBUG: XO energy failed: {e}")
                            pass
                    
                    # XF (X + 1 F) - linear
                    mol_xf = make_virtual_molecule("X", "F", 1)
                    if mol_xf:
                        try:
                            F_XF = mol_xf.total_molecular_energy(a=0.5, b=1.0, c=1.5)
                        except Exception as e:
                            print(f"DEBUG: XF energy failed: {e}")
                            pass
                    
                    # Format output
                    def fmt(x):
                        return f"{x:8.3f}" if not (x != x) else "     N/A"
                    
                    print(f"{period:5d}  {ports:5d}  {eps:6.2f}  "
                          f"{fmt(F_HX)}  {fmt(F_XO)}  {fmt(F_XF)}  {chi_x:+7.3f}")
                    scanned += 1
    
    print()
    print(f"[INFO] Scanned {scanned} configurations")
    print("[INTERPRETATION]")
    print("  - Low |F_total| → stable molecule")
    print("  - Very negative F → super-stable (possibly unphysical)")
    print("  - Very positive F or N/A → unstable/impossible configuration")
    print("  - Look for 'sweet spot' where F values are moderate")
    
    SPECTRAL_MODE = old_mode


def run_rnd_master_report() -> None:
    """
    Большой R&D отчёт:
      - SECTION 1: v1 baseline (таблица + органика + реакции)
      - SECTION 2: v2 baseline (сравнение chi, органика с R, реакции)
      - SECTION 3: k_period scan (доноры/акцепторы + ΔF)
      - SECTION 4: Super-O vs S
      - SECTION 5: zero-chemistry point для Cl~H
    """
    global SPECTRAL_MODE, V2_PERIOD_EXPONENT, MODEL_VERSION
    
    # Save original state
    orig_mode = SPECTRAL_MODE
    orig_k = V2_PERIOD_EXPONENT
    
    # Reset to default v4 for fresh start
    SPECTRAL_MODE = SPECTRAL_MODE_DEFAULT
    
    print("=" * 70)
    print("===== RND MASTER REPORT =====")
    print("=" * 70)
    print(f"[MODEL] {MODEL_VERSION}")
    print(f"[MODES] v1=twins, v2=period, v3=eps, v4=full (production)")
    print(f"[PARAMS] k_period={V2_PERIOD_EXPONENT}, eps_coupling={EPS_COUPLING_STRENGTH}")
    print()
    
    # ========== SECTION 1: BASELINE v1.0 ==========
    print("\n--- SECTION 1: BASELINE v1.0 ---")
    
    SPECTRAL_MODE = "v1"
    
    print_compact_spectral_table("v1.0")
    print_organic_testbench_with_reactivity("v1.0")
    print_reaction_block("v1.0")
    
    # ========== SECTION 2: BASELINE v2.0 ==========
    print("\n--- SECTION 2: BASELINE v2.0 (period split) ---")
    
    SPECTRAL_MODE = "v2_period_split"
    V2_PERIOD_EXPONENT = orig_k
    
    # Chi comparison v1 vs v2
    chi_v1 = collect_element_chi("v1")
    chi_v2 = collect_element_chi("v2")
    
    print(f"\n[CHI COMPARISON] v1 vs v2 (k={orig_k:.2f}):")
    print("Z  El   role        chi_v1    chi_v2    dchi")
    print("-" * 55)
    
    for atom in base_atoms:
        if atom.role == "inert":
            continue
        name = atom.name
        c1 = chi_v1.get(name, 0.0)
        c2 = chi_v2.get(name, 0.0)
        d = c2 - c1
        print(f"{atom.Z:2d} {name:3s}  {atom.role:10s}  {c1:8.3f}  {c2:8.3f}  {d:+7.3f}")
    
    print_compact_spectral_table("v2.0")
    print_organic_testbench_with_reactivity("v2.0")
    print_reaction_block("v2.0")
    
    # ========== SECTION 3: PERIOD EXPONENT SCAN ==========
    print("\n--- SECTION 3: PERIOD EXPONENT SCAN (donor/acceptor decay) ---")
    
    scan_period_exponent_compact()
    
    # ========== SECTION 4: SUPER-O vs S ==========
    print("\n--- SECTION 4: SUPER-O vs S EXPERIMENT ---")
    
    run_super_O_vs_S_enhanced(k_period=orig_k, super_eps=-5.0)
    print("\n[SECTION 4 NOTE] In v2.0 this is a NEGATIVE control experiment:")
    print("                 epsilon-override for O does not affect chi_spec/QEq yet.")
    
    # ========== SECTION 5: ZERO-CHEMISTRY POINT ==========
    print("\n--- SECTION 5: ZERO-CHEMISTRY POINT (Cl ~ H) ---")
    
    find_zero_chemistry_point_for_Cl_vs_H(k_min=0.0, k_max=2.0, k_step=0.1)

    # ========== SECTION 6: v3.0 epsilon–chi coupling ==========
    print("\n--- SECTION 6: v3.0 (epsilon–chi coupling) ---")

    # Switch to v3 mode locally
    SPECTRAL_MODE = SPECTRAL_MODE_V3

    # v3: ключевые элементы, органика, реакции, Super-O
    print_compact_spectral_table("v3.0 (eps-coupled)")
    print_organic_testbench_with_reactivity("v3.0 (eps-coupled)")
    print_reaction_block("v3.0 (eps-coupled)")
    run_super_O_vs_S_v3(k_period=orig_k, super_eps=-5.0)

    # ========== SECTION 7: v4.0 FULL MODEL ==========
    print("\n" + "=" * 70)
    print("--- SECTION 7: v4.0 FULL (period + eps-coupled) ---")
    print("=" * 70)

    SPECTRAL_MODE = SPECTRAL_MODE_V4

    print(f"\n[v4.0] Mode: {SPECTRAL_MODE_V4}")
    print(f"[v4.0] period exponent k = {orig_k}")
    print(f"[v4.0] eps coupling λ = {EPS_COUPLING_STRENGTH}")

    print_compact_spectral_table("v4.0 (full)")
    print_organic_testbench_with_reactivity("v4.0 (full)")
    print_reaction_block("v4.0 (full)")

    # Super-O in v4 (both period AND eps effects active)
    run_super_O_vs_S_v3(k_period=orig_k, super_eps=-5.0)

    # ========== SECTION 7: ELEMENT INDICES ==========
    print("\n" + "=" * 70)
    print("--- SECTION 7: ELEMENT INDICES (v4.0) ---")
    print("=" * 70)
    
    SPECTRAL_MODE = SPECTRAL_MODE_V4
    print_element_indices_table("v4.0 (period+eps)")
    print_element_quadrants("v4.0 classes")

    # ========== SECTION 8: ISLAND OF STABILITY SCAN ==========
    print("\n" + "=" * 70)
    print("--- SECTION 8: VIRTUAL ATOM X - ISLAND SCAN (coarse) ---")
    print("=" * 70)
    
    run_virtual_atom_island_scan()

    # ========== SECTION 9: NUMERICAL ENGINE CHECK (FDM) ==========
    print("\n" + "=" * 70)
    print("--- SECTION 9: NUMERICAL ENGINE CHECK (FDM) ---")
    print("=" * 70)
    
    print("[FDM] Estimating 'spectral energy' integral via FDM (3D, depth=4)")
    print("      Model: rho(r) = exp(-beta * r^2), beta = 0.5 + 0.05*Z")
    print("Atom  Z   E_port(v4)   E_fdm_integral (approx)")
    print("-" * 55)
    
    test_atoms = ["H", "C", "O", "F", "Si", "S", "Cl"]
    for sym in test_atoms:
        a = get_atom(sym)
        if a:
            e_port = a.per_port_energy() or 0.0
            e_fdm = estimate_atom_energy_fdm(a.Z, e_port)
            print(f"{sym:4s} {a.Z:2d}   {e_port:6.3f}       {e_fdm:8.4f}")
            
    print("\n[NOTE] E_fdm reflects total spectral volume; E_port is per-bond Potential.")

    # ========== SECTION 9: GRAPH COMPLEXITY ==========
    print("\n" + "=" * 70)
    print("--- SECTION 9: GRAPH COMPLEXITY ---")
    print("=" * 70)
    
    print(f"{'Z':<3} {'El':<3} {'Role':<10} {'Ports':<5}  {'Cyclomatic':<10}  {'C_graph':<8}")
    print("-" * 65)
    
    # Sort atoms by Z
    all_atoms = sorted(PERIODIC_TABLE.values(), key=lambda x: x.Z)
    for a in all_atoms:
        if a.Z > 18: continue
        adj = a.adjacency_matrix()
        feats = compute_complexity_features(adj)
        print(f"{a.Z:<3} {a.name:<3} {a.role:<10} {a.ports:<5}  {feats.cyclomatic:<10}  {feats.total:8.3f}")
        
    print("-" * 65)
    
    # ========== SECTION 10: CHRISTMAS TREE GROWTH (R&D) ==========
    print("\n" + "=" * 70)
    print("--- SECTION 10: CHRISTMAS TREE GROWTH (R&D) ---")
    print("=" * 70)
    
    # Import locally
    from .grower import GrowthParams, grow_molecule_christmas_tree, describe_molecule
    
    # Demo parameters
    params = GrowthParams(max_depth=4, max_atoms=16)
    seeds = ["C", "Si", "O"]
    
    for root in seeds:
        print(f"\n[ROOT = {root} | MaxDepth=4]")
        # Fixed seed for reproducibility in R&D report
        rng_seed = 42 + sum(ord(c) for c in root)
        rng = np.random.default_rng(rng_seed)
        
        for i in range(2):
            mol = grow_molecule_christmas_tree(root, params, rng=rng)
            print(f"  Tree #{i+1}: {describe_molecule(mol)}")


    # ========== FINAL SUMMARY ==========
    print("\n" + "=" * 70)
    print("===== ИТОГОВАЯ СВОДКА: ЭВОЛЮЦИЯ МОДЕЛИ =====")
    print("=" * 70)
    print("""
v1.0 (базовый режим):
  - Li~Na, F~Cl — спектральные «близнецы» (одинаковые E_port, χ_spec)
  - Обменные реакции изоэнергетичны: ΔF ≈ 0
  - Нет дифференциации по периоду или ε

v2.0 (разделение по периоду):
  - E_port ~ period^(-k), где k ≈ 0.7
  - F ≠ Cl (F — «жёстче»), Li ≠ Na
  - Обменные реакции неизоэнергетичны: ΔF ≠ 0 (предсказательные)
  - Только геометрия: ε не влияет на χ

v3.0 (ε-связь при фиксированной геометрии):
  - E_port как в v1 (без периодного скейлинга)
  - χ_spec зависит от ε через связь λ_ε
  - Близнецы возвращаются (F~Cl, O~S), но Super-O работает
  - Ортогональный эксперимент к v2: чистый эффект ε

v4.0 (полная модель):
  - Объединяет v2 (period-scaling) + v3 (ε-coupling)
  - Лучшее от обоих: период ломает близнецов, ε добавляет характер
  - Готова для химических предсказаний
""")

    print("===== КОНЕЦ R&D МАСТЕР-ОТЧЁТА =====")
    print("=" * 70)
    
    # Restore original state
    SPECTRAL_MODE = orig_mode
    V2_PERIOD_EXPONENT = orig_k


if __name__ == "__main__":
    import sys

    args = sys.argv[1:]

    if "--rnd" in args:
        # Полный R&D-отчёт (все эксперименты v1/v2/v3)
        run_rnd_master_report()
    elif "--export-periodic" in args:
        # CSV периодической таблицы v3
        export_periodic_table_v3_csv()
    elif "--export-organic" in args:
        # CSV органического тестбенча v3
        export_organic_testbench_v3_csv()
    elif "--export-reactions" in args:
        # CSV реакций v3
        export_reactions_v3_csv()
    elif "--bond" in args:
        # Predict bond: python geom_atoms.py --bond H Cl
        try:
            idx = args.index("--bond")
            if idx + 2 < len(args):
                a1 = args[idx+1]
                a2 = args[idx+2]
                predict_bond_polarity_and_type(a1, a2)
            else:
                print("Error: --bond requires two element symbols (e.g. --bond H Cl)")
        except Exception as e:
            print(f"Error: {e}")
    else:
        # По умолчанию – короткая справка
        print(f"geom_atoms.py :: {MODEL_VERSION}")
        print()
        print("Usage:")
        print("  python geom_atoms.py --rnd               # полный R&D-отчёт v1+v2+v3")
        print("  python geom_atoms.py --export-periodic   # CSV периодической таблицы v3")
        print("  python geom_atoms.py --export-organic    # CSV органического тестбенча v3")
        print("  python geom_atoms.py --export-reactions  # CSV реакций v3")
        print()
        print("Examples:")
        print("  python geom_atoms.py --rnd             > rnd_master_report.txt")
        print("  python geom_atoms.py --export-periodic > table_periodic_v3.csv")
        print("  python geom_atoms.py --export-organic  > organic_v3.csv")
        print("  python geom_atoms.py --export-reactions > reactions_v3.csv")
