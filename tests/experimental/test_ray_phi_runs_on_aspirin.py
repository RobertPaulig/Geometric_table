import pytest

from analysis.experimental.ray_audit import phi_from_smiles


def test_phi_from_smiles_runs() -> None:
    pytest.importorskip("rdkit")
    smiles = "CC(=O)OC1=CC=CC=C1C(=O)O"
    val = phi_from_smiles(smiles, scale=100)
    assert val == val  # not NaN
