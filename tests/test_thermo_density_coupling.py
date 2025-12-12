from __future__ import annotations

from core.density_models import beta_legacy, beta_tf_radius, beta_effective


def test_beta_effective_legacy_and_tf_radius() -> None:
    Z = 10
    b0 = beta_legacy(Z)
    b1 = beta_tf_radius(Z)

    assert b0 > 0.0
    assert b1 > 0.0
    assert beta_effective(Z, coupling=0.0, model="tf_radius") == b0
    # при coupling=1 beta_effective(Z) равна beta_legacy(Z) на Z_ref по определению,
    # но для другого Z остаётся положительной и сопоставимой по масштабу
    val = beta_effective(Z, coupling=1.0, model="tf_radius")
    assert val > 0.0


def test_beta_tf_radius_monotonic_in_Z() -> None:
    b0_10 = beta_tf_radius(10)
    b0_20 = beta_tf_radius(20)
    assert b0_20 > b0_10


def test_beta_effective_positive_for_all_Z() -> None:
    for Z in [1, 2, 10, 20, 50]:
        assert beta_effective(Z, coupling=0.0) > 0.0
        assert beta_effective(Z, coupling=1.0) > 0.0
