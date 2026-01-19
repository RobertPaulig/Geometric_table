import numpy as np
import pytest

from hetero2.physics_operator import AtomsDbV1, POTENTIAL_UNIT_MODEL, compute_operator_payload


@pytest.mark.parametrize("gamma", [1.0, 2.0])
def test_potential_scale_gamma_scales_v0_and_is_reported(gamma: float) -> None:
    # Path graph on 4 nodes: 0-1-2-3 (no RDKit needed).
    adj = np.array(
        [
            [0.0, 1.0, 0.0, 0.0],
            [1.0, 0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0, 1.0],
            [0.0, 0.0, 1.0, 0.0],
        ],
        dtype=float,
    )
    atoms_db = AtomsDbV1(
        source_path="test",
        potential_by_atomic_num={6: 1.23},
        chi_by_atomic_num={},
        symbol_by_atomic_num={6: "C"},
    )

    out = compute_operator_payload(
        adjacency=adj,
        bonds=None,
        types=(6, 6, 6, 6),
        physics_mode="hamiltonian",
        edge_weight_mode="unweighted",
        potential_mode="both",
        potential_scale_gamma=float(gamma),
        scf_max_iter=1,
        scf_tol=1e-30,
        scf_damping=1.0,
        atoms_db=atoms_db,
    )

    assert out["potential_unit_model"] == POTENTIAL_UNIT_MODEL
    assert float(out["potential_scale_gamma"]) == float(gamma)

    scf = out.get("scf", {})
    assert isinstance(scf, dict)
    assert scf["potential_unit_model"] == POTENTIAL_UNIT_MODEL
    assert float(scf["potential_scale_gamma"]) == float(gamma)

    vecs = scf.get("vectors", [])
    assert isinstance(vecs, list) and vecs
    for v in vecs:
        assert isinstance(v, dict)
        assert float(v["V0"]) == pytest.approx(1.23)
        assert float(v["V_scaled"]) == pytest.approx(float(gamma) * 1.23)
        assert float(v["gamma"]) == pytest.approx(float(gamma))
