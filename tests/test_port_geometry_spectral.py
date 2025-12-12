from __future__ import annotations

import numpy as np

from core.port_geometry_spectral import (
    hybrid_strength,
    infer_port_geometry,
    canonical_port_vectors,
)
from core.geom_atoms import AtomGraph
from core.thermo_config import ThermoConfig
from core.thermo_config import override_thermo_config


def test_effective_port_geometry_legacy_unchanged() -> None:
    atom = AtomGraph(
        name="N",
        Z=7,
        nodes=4,
        edges=3,
        ports=3,
        symmetry_score=0.0,
        port_geometry="trigonal",
        role="hub",
    )
    cfg = ThermoConfig(coupling_port_geometry=0.0, port_geometry_source="legacy")
    with override_thermo_config(cfg):
        assert atom.effective_port_geometry(cfg) == "trigonal"


def test_infer_port_geometry_basic_cases() -> None:
    assert infer_port_geometry("trigonal", 4, 0.0, 0.0) == "tetra"
    assert infer_port_geometry("trigonal", 3, 0.0, 0.9) == "trigonal"
    assert infer_port_geometry("trigonal", 3, 0.0, 0.1) == "pyramidal"
    assert infer_port_geometry("bent", 2, 0.0, 0.9) == "linear"
    assert infer_port_geometry("bent", 2, 0.0, 0.1) == "bent"


def test_canonical_port_vectors_norms_and_shape() -> None:
    v_tetra = canonical_port_vectors("tetra", ports=4)
    assert v_tetra.shape == (4, 3)
    norms = np.linalg.norm(v_tetra, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-6)

    v_linear = canonical_port_vectors("linear", ports=2)
    assert v_linear.shape == (2, 3)
    norms_lin = np.linalg.norm(v_linear, axis=1)
    assert np.allclose(norms_lin, 1.0, atol=1e-6)
    # два направления должны быть различимы
    assert not np.allclose(v_linear[0], v_linear[1])

