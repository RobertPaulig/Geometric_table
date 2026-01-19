import json
import zipfile
from pathlib import Path

from hetero1a import schemas as hetero_schemas
from hetero2 import batch as hetero2_batch


def test_standard_claims_frozen_schema_versions() -> None:
    assert hetero_schemas.SCORES_SCHEMA == "hetero_scores.v1"
    assert hetero_schemas.AUDIT_SCHEMA == "hetero_audit.v2"


def test_standard_claims_evidence_pack_required_files(tmp_path: Path) -> None:
    out_dir = tmp_path / "out"
    out_dir.mkdir(parents=True, exist_ok=True)

    (out_dir / "summary.csv").write_text("id,smiles,status,reason\nx,,SKIP,missing_smiles\n", encoding="utf-8")
    metrics_path = out_dir / "metrics.json"
    metrics_path.write_text(json.dumps({"counts": {"ERROR": 0}}, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    (out_dir / "index.md").write_text("# Evidence Index\n", encoding="utf-8")

    file_infos = hetero2_batch._compute_file_infos(out_dir, skip_names={"manifest.json", "checksums.sha256", "evidence_pack.zip"})
    manifest_files = list(file_infos)
    manifest_files.append({"path": "./manifest.json", "size_bytes": None, "sha256": None})
    manifest_files.append(
        {
            "path": "./metrics.json",
            "size_bytes": metrics_path.stat().st_size,
            "sha256": hetero2_batch._sha256_of_file(metrics_path),
        }
    )
    hetero2_batch._write_manifest(
        out_dir,
        seed=0,
        seed_strategy="global",
        score_mode="mock",
        scores_provenance={},
        guardrails_max_atoms=200,
        guardrails_require_connected=True,
        potential_unit_model=str(hetero2_batch.POTENTIAL_UNIT_MODEL),
        potential_scale_gamma=1.0,
        files=manifest_files,
    )
    file_infos_final = hetero2_batch._compute_file_infos(out_dir, skip_names={"checksums.sha256", "evidence_pack.zip"})
    hetero2_batch._write_checksums(out_dir, file_infos_final)
    hetero2_batch._write_zip_pack(out_dir)

    required_paths = {
        out_dir / "summary.csv",
        out_dir / "metrics.json",
        out_dir / "index.md",
        out_dir / "manifest.json",
        out_dir / "checksums.sha256",
        out_dir / "evidence_pack.zip",
    }
    for p in required_paths:
        assert p.exists(), f"missing required evidence pack file: {p.name}"

    with zipfile.ZipFile(out_dir / "evidence_pack.zip") as zf:
        names = set(zf.namelist())
    assert "summary.csv" in names
    assert "metrics.json" in names
    assert "index.md" in names
    assert "manifest.json" in names
    assert "checksums.sha256" in names

