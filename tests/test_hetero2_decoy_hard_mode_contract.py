import pytest

from hetero2.pipeline import run_pipeline_v2


def test_hard_decoys_mode_changes_skip_reason_when_no_decoys() -> None:
    pytest.importorskip("rdkit")
    smiles = "CC"

    out_normal = run_pipeline_v2(
        smiles,
        k_decoys=5,
        seed=0,
        max_attempts=0,
        score_mode="mock",
        scores_input=None,
        decoy_hard_mode=False,
    )
    assert out_normal.get("skip", {}).get("reason") == "no_decoys_generated"

    out_hard = run_pipeline_v2(
        smiles,
        k_decoys=5,
        seed=0,
        max_attempts=0,
        score_mode="mock",
        scores_input=None,
        decoy_hard_mode=True,
    )
    assert out_hard.get("skip", {}).get("reason") == "no_hard_decoys_generated"

