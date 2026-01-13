import ast
import hashlib
from pathlib import Path

import pytest

from scripts.generate_proxy_truth import TRUTH_SOURCE, TRUTH_VERSION, generate_proxy_truth_rows, main, write_proxy_truth_csv


def test_proxy_truth_rule_v1_deterministic_bytes(tmp_path: Path) -> None:
    input_csv = tmp_path / "input.csv"
    input_csv.write_text("id,smiles\nm1,c1ccccc1\nm2,CCO\n", encoding="utf-8")

    rows = generate_proxy_truth_rows(input_csv=input_csv)
    assert [r["molecule_id"] for r in rows] == ["m1", "m2"]
    assert all(r["truth_source"] == TRUTH_SOURCE for r in rows)
    assert all(r["truth_version"] == TRUTH_VERSION for r in rows)

    out1 = tmp_path / "truth1.csv"
    out2 = tmp_path / "truth2.csv"
    write_proxy_truth_csv(rows=rows, out_csv=out1)
    write_proxy_truth_csv(rows=rows, out_csv=out2)

    b1 = out1.read_bytes()
    b2 = out2.read_bytes()
    assert b1 == b2
    assert b"\r\n" not in b1  # enforce byte-for-byte with LF line endings


def test_proxy_truth_rule_v1_matches_spec(tmp_path: Path) -> None:
    canonical_smiles = "c1ccccc1"
    h = hashlib.sha256(canonical_smiles.encode("utf-8")).hexdigest()
    bucket = int(h[:8], 16) % 100
    expected = "PASS" if bucket < 5 else "FAIL"

    input_csv = tmp_path / "input.csv"
    input_csv.write_text(f"id,canonical_smiles\nmol_0,{canonical_smiles}\n", encoding="utf-8")
    rows = generate_proxy_truth_rows(input_csv=input_csv)
    assert rows[0]["expensive_label"] == expected


def test_proxy_truth_rule_v1_no_leakage_imports() -> None:
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "generate_proxy_truth.py"
    tree = ast.parse(script_path.read_text(encoding="utf-8"))
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)

    forbidden_prefixes = ("analysis", "hetero")
    assert not any(any(name == p or name.startswith(p + ".") for p in forbidden_prefixes) for name in imported)


def test_proxy_truth_rejects_unexpected_summary_args() -> None:
    with pytest.raises(SystemExit):
        main(["--summary_csv", "out/summary.csv", "--truth_csv", "out/truth.csv"])
