from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple
import math


# --- Magic numbers for protons / neutrons (incl. superheavy region) ---
# Legacy (v0.2) lists used when spectral H_nuc is unavailable.
MAGIC_Z_LEGACY = [2, 8, 20, 28, 50, 82, 114]
MAGIC_N_LEGACY = [2, 8, 20, 28, 50, 82, 126, 184]

# By default we try to import spectral magic numbers from the
# Woods–Saxon operator implemented in nuclear_spectrum_ws.py.
USE_SPECTRAL_MAGIC_N = True

try:
    from nuclear_spectrum_ws import get_magic_numbers_ws_cached
except ImportError:
    get_magic_numbers_ws_cached = None  # type: ignore[assignment]
    USE_SPECTRAL_MAGIC_N = False

if USE_SPECTRAL_MAGIC_N and get_magic_numbers_ws_cached is not None:
    MAGIC_Z = MAGIC_Z_LEGACY
    MAGIC_N = get_magic_numbers_ws_cached()
else:
    MAGIC_Z = MAGIC_Z_LEGACY
    MAGIC_N = MAGIC_N_LEGACY


def _min_sq_distance(x: int, magic_list, sigma: float) -> float:
    """Minimal squared (x - magic)/sigma over given magic_list."""
    return min(((x - m) / sigma) ** 2 for m in magic_list)


@dataclass
class NuclearConfig:
    Z: int
    N: int

    @property
    def A(self) -> int:
        return self.Z + self.N


def shell_penalty(
    Z: int,
    N: int,
    sigma_p: float = 6.0,
    sigma_n: float = 8.0,
) -> float:
    """
    Shell penalty as sum of proton and neutron distances
    to nearest magic numbers. Each term is a smooth 'bowl'
    of width sigma_p / sigma_n.

    P_shell = P_p(Z) + P_n(N)
    """
    Pp = _min_sq_distance(Z, MAGIC_Z, sigma_p)
    Pn = _min_sq_distance(N, MAGIC_N, sigma_n)
    return Pp + Pn


def pairing_penalty(Z: int, N: int, a_p: float = 12.0) -> float:
    """
    Simple pairing term in Weizsaecker spirit.

    Even-even  : more binding  -> negative correction
    Odd-odd    : less binding  -> positive correction
    Even-odd   : ~ 0
    """
    A = Z + N
    if A <= 1:
        return 0.0

    delta = a_p / (A ** 0.5)

    if (Z % 2 == 0) and (N % 2 == 0):
        # even-even: extra binding (lower F)
        return -delta
    elif (Z % 2 == 1) and (N % 2 == 1):
        # odd-odd: penalty (higher F)
        return +delta
    else:
        # even-odd: intermediate
        return 0.0


def liquid_drop_binding(Z: int, N: int) -> float:
    """
    Semi-empirical liquid-drop binding energy (MeV, up to an overall scale).

    B = a_v A - a_s A^{2/3} - a_c Z(Z-1)/A^{1/3} - a_a (N-Z)^2/A

    Coefficients are mild fit to a handful of stable nuclei.
    """
    A = Z + N
    if A <= 1:
        return 0.0

    # Coefficients fitted by fit_nuclear_coeffs.py
    a_v, a_s, a_c, a_a = 15.062, 15.379, 0.730, 15.728

    volume = a_v * A
    surface = -a_s * A ** (2.0 / 3.0)
    coulomb = -a_c * Z * (Z - 1) / (A ** (1.0 / 3.0))
    asym = -a_a * (N - Z) ** 2 / A

    return volume + surface + coulomb + asym


def nuclear_functional(
    Z: int,
    N: int,
    lambda_shell: float = 30.0,
    sigma_p: float = 6.0,
    sigma_n: float = 8.0,
    a_p: float = 12.0,
) -> float:
    """
    Toy nuclear functional:

        F = -B_drop + lambda_shell * P_shell + F_pair

    Spherical approximation, no deformation.
    Minima of F correspond to 'more stable' configurations.
    """
    B = liquid_drop_binding(Z, N)
    shell = shell_penalty(Z, N, sigma_p=sigma_p, sigma_n=sigma_n)
    pair = pairing_penalty(Z, N, a_p=a_p)
    return -B + lambda_shell * shell + pair


def scan_island(
    Z_range: Tuple[int, int] = (80, 130),
    N_range: Tuple[int, int] = (120, 210),
    lambda_shell: float = 30.0,
    sigma_p: float = 6.0,
    sigma_n: float = 8.0,
    a_p: float = 12.0,
    only_even_even: bool = True,
    top_k: int = 10,
) -> List[Tuple[int, int, float]]:
    """
    Scan nuclear functional F over a (Z,N) grid and return top_k minima.

    Returns list of (Z, N, F) sorted by F ascending.
    """
    best: List[Tuple[int, int, float]] = []

    Z_min, Z_max = Z_range
    N_min, N_max = N_range

    for Z in range(Z_min, Z_max + 1):
        for N in range(N_min, N_max + 1):
            if only_even_even and ((Z % 2 == 1) or (N % 2 == 1)):
                continue

            F = nuclear_functional(
                Z,
                N,
                lambda_shell=lambda_shell,
                sigma_p=sigma_p,
                sigma_n=sigma_n,
                a_p=a_p,
            )

            best.append((Z, N, F))

    best.sort(key=lambda t: t[2])
    return best[:top_k]


if __name__ == "__main__":
    # Пример: грубый скан тяжёлой области для поиска "островка"
    best = scan_island()
    print("Top candidates (even-even):")
    for Z, N, F in best:
        A = Z + N
        print(f"  Z={Z:3d}, N={N:3d}, A={A:3d}, F={F:8.2f}")
