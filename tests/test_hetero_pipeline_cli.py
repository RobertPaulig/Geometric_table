import json
import subprocess
import sys
from pathlib import Path


def test_pipeline_deterministic_and_selection_improves(tmp_path: Path) -> None:
    tree_input = Path("tests/data/hetero_tree_min.json")

    out_a = tmp_path / "a.json"
    out_b = tmp_path / "b.json"
    cmd_base = [
        sys.executable,
        "-m",
        "analysis.chem.pipeline",
        "--tree_input",
        str(tree_input),
        "--k",
        "10",
        "--seed",
        "0",
        "--timestamp",
        "2026-01-02T00:00:00+00:00",
        "--neg_control_reps",
        "20",
        "--select_k",
        "5",
    ]

    subprocess.run([*cmd_base, "--selection", "maxmin", "--out", str(out_a)], check=True)
    subprocess.run([*cmd_base, "--selection", "maxmin", "--out", str(out_b)], check=True)
    a = json.loads(out_a.read_text(encoding="utf-8"))
    b = json.loads(out_b.read_text(encoding="utf-8"))
    assert a == b
    assert a["score_mode"] == "toy_edge_dist"
    assert a["score_definition"] == "score=1 - dist_to_original"

    out_first = tmp_path / "first.json"
    out_maxmin = tmp_path / "maxmin.json"
    subprocess.run([*cmd_base, "--selection", "firstk", "--out", str(out_first)], check=True)
    subprocess.run([*cmd_base, "--selection", "maxmin", "--out", str(out_maxmin)], check=True)
    first = json.loads(out_first.read_text(encoding="utf-8"))
    maxmin = json.loads(out_maxmin.read_text(encoding="utf-8"))

    f_min = first["selection"]["metrics"]["min_pairwise_dist"]
    m_min = maxmin["selection"]["metrics"]["min_pairwise_dist"]
    assert m_min >= f_min

    def _selected_hashes(payload: dict) -> list[str]:
        indices = payload["selection"]["selected_indices"]
        decoys_sorted = sorted(payload["decoys"]["decoys"], key=lambda d: d["hash"])
        return [decoys_sorted[i]["hash"] for i in indices]

    assert first["selection"]["selected_hashes"] == _selected_hashes(first)
    assert maxmin["selection"]["selected_hashes"] == _selected_hashes(maxmin)


def test_pipeline_warnings_aggregate(tmp_path: Path) -> None:
    tree_input = Path("tests/data/hetero_tree_min.json")
    out = tmp_path / "w.json"
    cmd = [
        sys.executable,
        "-m",
        "analysis.chem.pipeline",
        "--tree_input",
        str(tree_input),
        "--k",
        "10",
        "--seed",
        "0",
        "--timestamp",
        "2026-01-02T00:00:00+00:00",
        "--min_pair_dist",
        "0.5",
        "--min_dist_to_original",
        "0.5",
        "--max_attempts",
        "20",
        "--select_k",
        "5",
        "--selection",
        "maxmin",
        "--out",
        str(out),
    ]
    subprocess.run(cmd, check=True)
    data = json.loads(out.read_text(encoding="utf-8"))

    warnings = set(data["warnings"])
    decoy_warnings = set(data["decoys"].get("warnings", []))
    audit_warnings = set(data["audit"].get("warnings", []))
    assert decoy_warnings.issubset(warnings)
    assert audit_warnings.issubset(warnings)


def test_pipeline_external_scores_mode(tmp_path: Path) -> None:
    tree_input = Path("tests/data/hetero_tree_min.json")
    base = [
        sys.executable,
        "-m",
        "analysis.chem.pipeline",
        "--tree_input",
        str(tree_input),
        "--k",
        "10",
        "--seed",
        "0",
        "--timestamp",
        "2026-01-02T00:00:00+00:00",
        "--select_k",
        "5",
    ]

    toy_out = tmp_path / "toy.json"
    subprocess.run([*base, "--selection", "maxmin", "--out", str(toy_out)], check=True)
    toy = json.loads(toy_out.read_text(encoding="utf-8"))
    hashes = [d["hash"] for d in sorted(toy["decoys"]["decoys"], key=lambda d: d["hash"])]

    scores_input = tmp_path / "scores.json"
    scores_payload = {
        "schema_version": "hetero_scores.v1",
        "original": {"score": 1.0, "weight": 1.0},
        "decoys": {h: {"score": 0.001 * i, "weight": 1.0} for i, h in enumerate(hashes)},
    }
    scores_input.write_text(json.dumps(scores_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    ext_out = tmp_path / "ext.json"
    subprocess.run(
        [
            *base,
            "--selection",
            "maxmin",
            "--score_mode",
            "external_scores",
            "--scores_input",
            str(scores_input),
            "--out",
            str(ext_out),
        ],
        check=True,
    )
    ext = json.loads(ext_out.read_text(encoding="utf-8"))
    assert ext["score_mode"] == "external_scores"
    assert ext["audit"]["audit_input"]["score_mode"] == "external_scores"

    scores_payload["decoys"].pop(hashes[0])
    scores_input.write_text(json.dumps(scores_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    ext_out2 = tmp_path / "ext2.json"
    subprocess.run(
        [
            *base,
            "--selection",
            "maxmin",
            "--score_mode",
            "external_scores",
            "--scores_input",
            str(scores_input),
            "--out",
            str(ext_out2),
        ],
        check=True,
    )
    ext2 = json.loads(ext_out2.read_text(encoding="utf-8"))
    assert "missing_scores_for_some_decoys:1" in ext2["warnings"]
    assert ext2["audit_input"]["n_decoys_missing_scores"] == 1
