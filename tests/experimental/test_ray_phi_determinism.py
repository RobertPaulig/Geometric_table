import numpy as np

from analysis.experimental.ray_audit import phi_from_eigs


def test_phi_from_eigs_is_deterministic_and_scale_sensitive() -> None:
    eigs = np.array([0.1, 0.5, 1.0, 2.0])
    a = phi_from_eigs(eigs, scale=50, auditor_size=256)
    b = phi_from_eigs(eigs, scale=50, auditor_size=256)
    c = phi_from_eigs(eigs, scale=200, auditor_size=256)
    assert a == b
    assert c != a
