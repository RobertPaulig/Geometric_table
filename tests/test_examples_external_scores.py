import json
from pathlib import Path

from hetero1a.api import run_pipeline


def test_examples_external_scores(tmp_path: Path) -> None:
    tree_payload = json.loads(Path("tests/data/hetero_tree_min.json").read_text(encoding="utf-8"))
    base = run_pipeline(
        tree_payload,
        k=10,
        seed=0,
        timestamp="2026-01-02T00:00:00+00:00",
        select_k=5,
        selection="maxmin",
    )
    hashes = [d["hash"] for d in sorted(base["decoys"]["decoys"], key=lambda d: d["hash"])]

    scores_payload = {
        "schema_version": "hetero_scores.v1",
        "original": {"score": 1.0, "weight": 1.0},
        "decoys": {h: {"score": 0.001 * i, "weight": 1.0} for i, h in enumerate(hashes)},
    }
    # Drop one score to trigger warning.
    scores_payload["decoys"].pop(hashes[0])
    scores_path = tmp_path / "scores.json"
    scores_path.write_text(json.dumps(scores_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    ext = run_pipeline(
        tree_payload,
        k=10,
        seed=0,
        timestamp="2026-01-02T00:00:00+00:00",
        select_k=5,
        selection="maxmin",
        score_mode="external_scores",
        scores_input=str(scores_path),
    )

    assert ext["audit_input"]["score_mode"] == "external_scores"
    assert ext["audit_input"]["n_decoys_missing_scores"] == 1
    assert "missing_scores_for_some_decoys:1" in ext["warnings"]
