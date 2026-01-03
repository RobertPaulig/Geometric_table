import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hetero2.pipeline import run_pipeline_v2  # noqa: E402
from hetero2.report import render_report_v2  # noqa: E402


def test_hetero2_pipeline_and_report_are_deterministic(tmp_path: Path) -> None:
    pytest.importorskip("rdkit")
    smiles = "CC(=O)OC1=CC=CC=C1C(=O)O"
    kwargs = dict(smiles=smiles, k_decoys=5, seed=0, timestamp="2026-01-02T00:00:00+00:00", score_mode="external_scores")

    p1 = run_pipeline_v2(**kwargs)
    p2 = run_pipeline_v2(**kwargs)
    assert p1 == p2

    md1 = tmp_path / "r1.md"
    md2 = tmp_path / "r2.md"
    render_report_v2(p1, out_path=md1)
    render_report_v2(p2, out_path=md2)

    assert md1.read_text(encoding="utf-8") == md2.read_text(encoding="utf-8")
    text = md1.read_text(encoding="utf-8")
    assert "Rings" in text
    assert "PhysChem" in text
    assert "Hardness" in text
    assert "Verdict" in text
