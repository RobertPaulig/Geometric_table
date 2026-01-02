import json
import subprocess
import sys
from pathlib import Path


def test_hetero_audit_is_deterministic(tmp_path: Path) -> None:
    input_path = Path("tests/data/hetero_audit_min.json")
    out1 = tmp_path / "out1.json"
    out2 = tmp_path / "out2.json"

    cmd = [
        sys.executable,
        "-m",
        "analysis.chem.audit",
        "--input",
        str(input_path),
        "--seed",
        "0",
        "--timestamp",
        "2026-01-02T00:00:00+00:00",
        "--neg_control_reps",
        "50",
        "--out",
        str(out1),
    ]
    subprocess.run(cmd, check=True)
    subprocess.run([*cmd[:-1], str(out2)], check=True)

    a = json.loads(out1.read_text(encoding="utf-8"))
    b = json.loads(out2.read_text(encoding="utf-8"))
    assert a == b

    for key in ["version", "dataset_id", "n_pos", "n_neg", "auc_tie_aware", "neg_controls", "run"]:
        assert key in a
    for key in ["null_q", "perm_q", "rand_q", "neg_auc_max", "gate", "margin", "slack", "verdict"]:
        assert key in a["neg_controls"]
    for key in ["seed", "timestamp", "cmd"]:
        assert key in a["run"]


def test_hetero_audit_is_invariant_to_item_order(tmp_path: Path) -> None:
    input_path = Path("tests/data/hetero_audit_min.json")
    payload = json.loads(input_path.read_text(encoding="utf-8"))
    items = list(payload["items"])
    payload["items"] = list(reversed(items))
    input_rev = tmp_path / "input_rev.json"
    input_rev.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    out1 = tmp_path / "out1.json"
    out2 = tmp_path / "out2.json"

    base = [
        sys.executable,
        "-m",
        "analysis.chem.audit",
        "--seed",
        "0",
        "--timestamp",
        "2026-01-02T00:00:00+00:00",
        "--neg_control_reps",
        "50",
    ]
    subprocess.run([*base, "--input", str(input_path), "--out", str(out1)], check=True)
    subprocess.run([*base, "--input", str(input_rev), "--out", str(out2)], check=True)

    a = json.loads(out1.read_text(encoding="utf-8"))
    b = json.loads(out2.read_text(encoding="utf-8"))
    assert a == b
