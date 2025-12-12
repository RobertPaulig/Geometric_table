from __future__ import annotations

from types import SimpleNamespace

from analysis.nuclear.scan_isotope_band import compute_delta_F


def test_compute_deltaF_prefers_cli_override() -> None:
    val = compute_delta_F(args_deltaF=7.5, delta_F_base=5.0, coupling_delta_F=1.0, temperature=2.0)
    assert val == 7.5


def test_compute_deltaF_legacy_when_coupling_zero() -> None:
    val = compute_delta_F(args_deltaF=None, delta_F_base=5.0, coupling_delta_F=0.0, temperature=2.0)
    assert val == 5.0


def test_compute_deltaF_scaled_when_coupling_positive() -> None:
    val = compute_delta_F(args_deltaF=None, delta_F_base=5.0, coupling_delta_F=1.0, temperature=2.0)
    assert val == 10.0

