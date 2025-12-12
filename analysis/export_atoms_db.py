from __future__ import annotations

"""
export_atoms_db.py — DATA-1/step1:
экспорт текущей базы AtomGraph из geom_atoms в JSON.
"""

from pathlib import Path
import json
import dataclasses

from core.geom_atoms import _make_base_atoms


def main() -> None:
    base = Path(".")
    data_dir = base / "data"
    data_dir.mkdir(exist_ok=True)

    atoms = _make_base_atoms()
    rows = [dataclasses.asdict(a) for a in atoms]

    out_path = data_dir / "atoms_db_v1.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)

    print(f"Exported {len(rows)} atoms to {out_path}")


if __name__ == "__main__":
    main()

