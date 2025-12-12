from __future__ import annotations

"""
export_atoms_db.py — DATA-1/step1:
экспорт текущей базы AtomGraph из geom_atoms в JSON.
"""

import json
import dataclasses

from analysis.io_utils import ensure_data_dir, data_path
from core.geom_atoms import _make_base_atoms


def main() -> None:
    data_dir = ensure_data_dir()

    atoms = _make_base_atoms()
    rows = [dataclasses.asdict(a) for a in atoms]

    out_path = data_path("atoms_db_v1.json")
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)

    print(f"Exported {len(rows)} atoms to {out_path}")


if __name__ == "__main__":
    main()
