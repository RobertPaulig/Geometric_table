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
