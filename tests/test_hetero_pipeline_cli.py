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

    out_first = tmp_path / "first.json"
    out_maxmin = tmp_path / "maxmin.json"
    subprocess.run([*cmd_base, "--selection", "firstk", "--out", str(out_first)], check=True)
    subprocess.run([*cmd_base, "--selection", "maxmin", "--out", str(out_maxmin)], check=True)
    first = json.loads(out_first.read_text(encoding="utf-8"))
    maxmin = json.loads(out_maxmin.read_text(encoding="utf-8"))

    f_min = first["selection"]["metrics"]["min_pairwise_dist"]
    m_min = maxmin["selection"]["metrics"]["min_pairwise_dist"]
    assert m_min >= f_min

