import csv
import json
from pathlib import Path

import pytest

from hetero2.batch import run_batch


def test_batch_emits_hardness_curve_and_operator_features(tmp_path: Path) -> None:
    pytest.importorskip("rdkit")
    input_csv = tmp_path / "input.csv"
    with input_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["id", "smiles"])
        w.writeheader()
        w.writerow({"id": "aspirin", "smiles": "CC(=O)OC1=CC=CC=C1C(=O)O"})

    out_dir = tmp_path / "out"
    run_batch(
        input_csv=input_csv,
        out_dir=out_dir,
        artifacts="light",
        seed=0,
        k_decoys=5,
        score_mode="mock",
        physics_mode="hamiltonian",
        workers=1,
        zip_pack=True,
    )

    assert (out_dir / "hardness_curve.csv").exists()
    assert (out_dir / "hardness_curve.md").exists()
    assert (out_dir / "operator_features.csv").exists()

    metrics = json.loads((out_dir / "metrics.json").read_text(encoding="utf-8"))
    assert "decoy_realism" in metrics
    decoy_realism = metrics["decoy_realism"]
    assert "auc_interpretation" in decoy_realism
    assert "auc_tie_aware_by_bin" in decoy_realism

