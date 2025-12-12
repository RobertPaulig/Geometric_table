from __future__ import annotations

from core.nuclear_bands import find_best_N_for_Z, scan_isotope_band_for_Z


def _best_from_scan(Z: int, N_min: int, N_max: int) -> int:
    points = scan_isotope_band_for_Z(Z, N_min, N_max)
    assert points, "scan_isotope_band_for_Z returned no points"
    best = min(points, key=lambda p: p.F)
    return best.N


def test_valley_best_N_matches_band_scan_for_C() -> None:
    Z = 6
    N_min = Z
    N_max = int(1.7 * Z)
    N_best, _ = find_best_N_for_Z(Z, N_min, N_max)
    N_best_scan = _best_from_scan(Z, N_min, N_max)
    assert N_best == N_best_scan


def test_valley_best_N_matches_band_scan_for_Si() -> None:
    Z = 14
    N_min = Z
    N_max = int(1.7 * Z)
    N_best, _ = find_best_N_for_Z(Z, N_min, N_max)
    N_best_scan = _best_from_scan(Z, N_min, N_max)
    assert N_best == N_best_scan


