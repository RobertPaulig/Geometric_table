import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hetero2.chemgraph import ChemGraph  # noqa: E402
from hetero2.spectral import compute_stability_metrics, laplacian_eigvals, spectral_fp_from_laplacian  # noqa: E402


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


def test_hetero2_stability_metrics_invariant_to_permutation() -> None:
    pytest.importorskip("rdkit")
    smiles = "CC(=O)OC1=CC=CC=C1C(=O)O"
    g = ChemGraph(smiles)
    lap = g.laplacian()
    metrics1 = compute_stability_metrics(laplacian_eigvals(lap))

    n = lap.shape[0]
    perm = np.arange(n)[::-1]
    lap2 = lap[np.ix_(perm, perm)]
    metrics2 = compute_stability_metrics(laplacian_eigvals(lap2))

    for key in ("spectral_gap", "spectral_entropy", "spectral_entropy_norm"):
        v1 = float(metrics1[key])
        v2 = float(metrics2[key])
        assert math.isclose(v1, v2, rel_tol=1e-8, abs_tol=1e-8)


def test_hetero2_stability_metrics_sanity() -> None:
    pytest.importorskip("rdkit")
    g = ChemGraph("CCC")
    lap = g.laplacian()
    metrics = compute_stability_metrics(laplacian_eigvals(lap))

    assert metrics["spectral_gap"] > 0.0
    assert not math.isnan(metrics["spectral_entropy"])
    assert not math.isnan(metrics["spectral_entropy_norm"])
    assert 0.0 <= metrics["spectral_entropy_norm"] <= 1.0 + 1e-8
