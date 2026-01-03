import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hetero2.chemgraph import ChemGraph  # noqa: E402


def test_hetero2_chemgraph_aspirin() -> None:
    pytest.importorskip("rdkit")
    smiles = "CC(=O)OC1=CC=CC=C1C(=O)O"
    g = ChemGraph(smiles)
    assert g.n_heavy_atoms() == 13

    phys = g.physchem()
    for key in ["mw", "logp", "tpsa", "hbd", "hba"]:
        val = float(phys[key])
        assert not math.isnan(val)

    adj = g.adjacency()
    assert adj.shape[0] == adj.shape[1]
    assert float(adj.trace()) == 0.0
    assert (adj == adj.T).all()
