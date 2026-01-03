import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hetero2.chemgraph import ChemGraph  # noqa: E402
from hetero2.decoys_rewire import generate_rewire_decoys  # noqa: E402


def test_hetero2_decoys_rewire_generates_valid_smiles() -> None:
    pytest.importorskip("rdkit")
    smiles = "CC(=O)OC1=CC=CC=C1C(=O)O"
    orig = ChemGraph(smiles)
    dec = generate_rewire_decoys(smiles, k=5, seed=0)
    assert len(dec.decoys) >= 1
    seen = set()
    for d in dec.decoys:
        s = d["smiles"]
        assert s not in seen
        seen.add(s)
        assert s != orig.canonical_smiles
        assert "ring_info" in d
        assert "physchem" in d
    # aromatic ring count preserved (lock_aromatic default True)
    orig_arom = orig.ring_info().get("n_aromatic_rings", 0)
    for d in dec.decoys:
        assert d["ring_info"].get("n_aromatic_rings", 0) == orig_arom
