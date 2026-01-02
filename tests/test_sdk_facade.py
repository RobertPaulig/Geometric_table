import json
from pathlib import Path

import hetero1a
from hetero1a.api import render_report, run_pipeline


def test_sdk_facade_pipeline_and_report(tmp_path: Path) -> None:
    tree_payload = json.loads(Path("tests/data/hetero_tree_min.json").read_text(encoding="utf-8"))
    result = run_pipeline(
        tree_payload,
        k=10,
        seed=0,
        timestamp="2026-01-02T00:00:00+00:00",
        select_k=5,
        selection="maxmin",
    )
    assert result["schema_version"] == "hetero_pipeline.v1"
    assert "decoys" in result and "selection" in result and "audit" in result
    assert result["score_mode"] == "toy_edge_dist"

    md_path, csv_path = render_report(result, out_dir=str(tmp_path), stem="sdk")
    md_text = Path(md_path).read_text(encoding="utf-8")
    csv_text = Path(csv_path).read_text(encoding="utf-8")
    assert "HETERO-1A Report" in md_text
    assert "index,hash,selected,dist_to_original,edge_overlap_to_original" in csv_text
