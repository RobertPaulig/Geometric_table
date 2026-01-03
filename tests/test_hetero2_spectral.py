import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hetero2.chemgraph import ChemGraph  # noqa: E402
from hetero2.spectral import spectral_fp_from_laplacian  # noqa: E402


def test_hetero2_spectral_fp_invariant_to_permutation() -> None:
    pytest.importorskip("rdkit")
    smiles = "CC(=O)OC1=CC=CC=C1C(=O)O"
    g = ChemGraph(smiles)
    lap = g.laplacian()
    fp1 = spectral_fp_from_laplacian(lap, round_decimals=10)

    n = lap.shape[0]
    perm = np.arange(n)
    perm = perm[::-1]
    lap2 = lap[np.ix_(perm, perm)]
    fp2 = spectral_fp_from_laplacian(lap2, round_decimals=10)

    assert fp1 == fp2
    assert len(fp1) == n
    assert all(not math.isnan(x) for x in fp1)
