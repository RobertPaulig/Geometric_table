import json
import subprocess
import sys
from pathlib import Path


def test_examples_smoke_cli(tmp_path: Path) -> None:
    out_pipeline = tmp_path / "smoke_pipeline.json"
    out_dir = tmp_path
    stem = "smoke"

    subprocess.run(
        [
            sys.executable,
            "-m",
            "analysis.chem.pipeline",
            "--tree_input",
            "tests/data/hetero_tree_min.json",
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
            str(out_pipeline),
        ],
        check=True,
    )
    subprocess.run(
        [
            sys.executable,
            "-m",
            "analysis.chem.report",
            "--input",
            str(out_pipeline),
            "--out_dir",
            str(out_dir),
            "--stem",
            stem,
        ],
        check=True,
    )

    report_md = out_dir / f"{stem}.report.md"
    report_csv = out_dir / f"{stem}.decoys.csv"
    assert out_pipeline.exists()
    assert report_md.exists()
    assert report_csv.exists()

    payload = json.loads(out_pipeline.read_text(encoding="utf-8"))
    assert "schema_version" in payload
    assert payload["score_mode"] == "toy_edge_dist"
    warnings = payload.get("warnings", [])
    assert isinstance(warnings, list)
    assert warnings == sorted(set(warnings))
