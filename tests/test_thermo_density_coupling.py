from __future__ import annotations

from core.density_models import beta_legacy, beta_tf, beta_effective


def test_beta_effective_legacy_and_tf() -> None:
    Z = 10
    b0 = beta_legacy(Z)
    b1 = beta_tf(Z)

    assert b0 > 0.0
    assert b1 > 0.0
    assert beta_effective(Z, coupling=0.0, model="tf") == b0
    assert beta_effective(Z, coupling=1.0, model="tf") == b1


def test_beta_effective_monotonic_in_Z() -> None:
    b0_10 = beta_tf(10)
    b0_20 = beta_tf(20)
    assert b0_20 > b0_10


def test_beta_effective_positive_for_all_Z() -> None:
    for Z in [1, 2, 10, 20, 50]:
        assert beta_effective(Z, coupling=0.0) > 0.0
        assert beta_effective(Z, coupling=1.0) > 0.0

