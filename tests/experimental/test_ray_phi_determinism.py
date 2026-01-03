import numpy as np

from analysis.experimental.ray_audit import phi_from_eigs


def test_phi_from_eigs_is_deterministic() -> None:
    eigs = np.array([0.1, 0.5, 1.0, 2.0])
    a = phi_from_eigs(eigs, scale=50)
    b = phi_from_eigs(eigs, scale=50)
    assert a == b
