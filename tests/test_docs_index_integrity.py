from __future__ import annotations

import re
from pathlib import Path


def test_docs_index_integrity_required_refs_exist() -> None:
    repo_root = Path(__file__).resolve().parents[1]

    index_path = repo_root / "docs" / "99_index.md"
    assert index_path.exists(), "docs/99_index.md missing"

    index_text = index_path.read_text(encoding="utf-8")

    # Heuristic: required repo paths should be referenced in backticks in the index.
    backticked_paths = set(re.findall(r"`([^`]+)`", index_text))

    required_paths = [
        "CONTEXT.md",
        "docs/ROADMAP.md",
        "docs/90_lineage.md",
        "docs/95_release_checklist.md",
        "docs/artefacts_registry.md",
        "docs/contracts/hetero_scores.v1.md",
        "VERSION",
        "docs/name3.md",
        "docs/name4.tex",
    ]

    missing_refs = [path for path in required_paths if path not in backticked_paths]
    assert not missing_refs, f"docs/99_index.md missing required refs: {missing_refs}"

    missing_files = [path for path in required_paths if not (repo_root / path).exists()]
    assert not missing_files, f"required files missing in repo: {missing_files}"

