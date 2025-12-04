from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple
from collections import defaultdict

import numpy as np

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
}

# Calibrated spectral parameters (v1.0)
ALPHA_CALIBRATED = 1.237
GAMMA_DONOR_CALIBRATED = 3.0
KCENTER_CALIBRATED = 0.1
EPS_NEUTRAL = 0.06

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
    notes: str = ""         # произвольный комментарий
    epsilon: float = 0.0    # игрушечное положение уровня ε_Z относительно Среды

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

    def F_geom(self, a: float = 0.5, b: float = 1.0, c: float = 1.5) -> float:
        """
        Простейший геометрический функционал:

            F_geom = a * cyclomatic_number
                   + b * symmetry_score
                   + c * ports
        """
        return (
            a * self.cyclomatic_number()
            + b * self.symmetry_score
            + c * self.ports
        )

    def per_port_energy(
        self, a: float = 0.5, b: float = 1.0, c: float = 1.5
    ) -> Optional[float]:
        """
        Геометрическая "энергия" на один порт.

        Если портов нет (инертный газ), возвращается None.
        """
        if self.ports <= 0:
            return None
        return self.F_geom(a=a, b=b, c=c) / self.ports

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
    ) -> Optional[float]:
        """
        Спектральная χ_geom с:
          - демпфером для доноров,
          - мягким "фоном" для hub-центров (B/C/Al/Si).
        """
        e = self.per_port_energy(a=a, b=b, c=c)
        if e is None:
            return None

        chi_abs = alpha * e

        # Инертные: χ = 0
        if self.role == "inert":
            return 0.0

        mu_env_spec = compute_mu_env_spec()
        eps = self.epsilon_spec()
        delta = eps - mu_env_spec

        # Нейтральное окно: отдельная обработка hub-центров
        if abs(delta) <= eps_neutral:
            if self.role == "hub":
                if delta > 0.0:
                    sign_center = -1.0  # слабый донор (B, Al)
                else:
                    sign_center = 1.0   # слабый акцептор (C, Si)
                return sign_center * (k_center * chi_abs)
            else:
                return 0.0

        # Вне окна: обычные доноры / акцепторы
        if delta > 0.0:
            sign = -1.0
            s = 1.0 / (1.0 + gamma_donor * eps)
        else:
            sign = 1.0
            s = 1.0

        return sign * (s * chi_abs)


def _make_base_atoms() -> List[AtomGraph]:
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
            epsilon=0.2,
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
            epsilon=0.2,
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
            notes="Геометрический аналог C: тетраэдрический 3D-хаб",
            epsilon=0.0,
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
    ]


base_atoms: List[AtomGraph] = _make_base_atoms()


def get_atom(name: str) -> AtomGraph:
    """
    Найти прототип атома по символу элемента.

    Возвращает объект из base_atoms.
    """
    for atom in base_atoms:
        if atom.name == name:
            return atom
    raise ValueError(f"Unknown atom name: {name}")


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

      |Δχ| < 0.5        → "ковалентная"
      0.5 ≤ |Δχ| < 2.0  → "полярная"
      |Δχ| ≥ 2.0        → "почти ионная"
    """
    x = abs(delta_chi)
    if x < 0.5:
        return "ковалентная"
    elif x < 2.0:
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


if __name__ == "__main__":
    # Пример: базовые веса a=0.5, b=1.0, c=1.5.
    print_table(a=0.5, b=1.0, c=1.5)
    print_port_energies(a=0.5, b=1.0, c=1.5)
    print_port_geometries()
    print()
    print_chi_comparison(a=0.5, b=1.0, c=1.5)
    print()
    print_chi_spectral(a=0.5, b=1.0, c=1.5)
    print()
    print_bond_polarities(a=0.5, b=1.0, c=1.5)
    print()
    print_bond_polarities_spec(a=0.5, b=1.0, c=1.5)
    print()
    print_chain_polarities_spec(a=0.5, b=1.0, c=1.5, eps_neutral=0.06)
    print()
    print_sp3_tensions(a=0.5, b=1.0, c=1.5)
    print()
    print_spectral_summary(beta=0.5)
    print()
    print_spectral_energies(
        omega_min=0.0, omega_max=6.0, eta=0.1, beta=0.5
    )
    print()
    print_full_diagnostics(
        a=0.5,
        b=1.0,
        c=1.5,
        eps_neutral=0.06,
        omega_min=0.0,
        omega_max=6.0,
        eta=0.1,
        beta=0.5,
    )
    print()
    print_role_averages(
        omega_min=0.0,
        omega_max=6.0,
        eta=0.1,
        beta=0.5,
        a=0.5,
        b=1.0,
        c=1.5,
        eps_neutral=0.06,
    )
    print()
    print_spectral_periodic_table(
        a=0.5,
        b=1.0,
        c=1.5,
        omega_min=0.0,
        omega_max=6.0,
        eta=0.1,
        beta=0.5,
    )
    print()
    grid_fit_geom_spectral_params(
        a=0.5,
        b=1.0,
        c=1.5,
        eps_neutral=0.06,
    )
