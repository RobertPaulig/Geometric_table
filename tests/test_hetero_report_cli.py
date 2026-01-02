import csv
import json
import subprocess
import sys
from pathlib import Path


def test_report_deterministic_and_sanity(tmp_path: Path) -> None:
    tree_input = Path("tests/data/hetero_tree_min.json")
    pipeline_json = tmp_path / "pipeline.json"
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
        "--select_k",
        "5",
        "--selection",
        "maxmin",
        "--out",
        str(pipeline_json),
    ]
    subprocess.run(cmd, check=True)

    out_dir = tmp_path / "report"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "analysis.chem.report",
            "--input",
            str(pipeline_json),
            "--out_dir",
            str(out_dir),
            "--stem",
            "min",
        ],
        check=True,
    )
    md1 = (out_dir / "min.report.md").read_text(encoding="utf-8")
    csv1 = (out_dir / "min.decoys.csv").read_text(encoding="utf-8")

    subprocess.run(
        [
            sys.executable,
            "-m",
            "analysis.chem.report",
            "--input",
            str(pipeline_json),
            "--out_dir",
            str(out_dir),
            "--stem",
            "min2",
        ],
        check=True,
    )
    md2 = (out_dir / "min2.report.md").read_text(encoding="utf-8")
    csv2 = (out_dir / "min2.decoys.csv").read_text(encoding="utf-8")

    assert md1 == md2
    assert csv1 == csv2

    assert "score_mode: toy_edge_dist" in md1
    assert "schema_version:" in md1

    with (out_dir / "min.decoys.csv").open(encoding="utf-8", newline="") as f:
        rows = list(csv.reader(f))
    assert rows[0] == ["index", "hash", "selected", "dist_to_original", "edge_overlap_to_original"]
    assert len(rows) == 1 + 5  # header + k_generated for the small tree
