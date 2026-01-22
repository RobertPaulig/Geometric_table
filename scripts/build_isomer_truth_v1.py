from __future__ import annotations

import argparse
import csv
import hashlib
from pathlib import Path


class IsomerTruthError(ValueError):
    pass


TRUTH_SOURCE = "spice2_0_1_isomers_v2"
TRUTH_VERSION = "isomer_truth.v1"


DEFAULT_RAW_CSV = Path("data/accuracy/raw/dft_golden_isomers_v2_spice2_0_1.csv")
DEFAULT_OUT_CSV = Path("data/accuracy/isomer_truth.v1.csv")


_REQUIRED_INPUT_COLS = [
    "id",
    "group_id",
    "smiles",
    "energy_rel_kcalmol",
]


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def _read_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise IsomerTruthError(f"missing csv: {path}")
    reader = csv.DictReader(path.read_text(encoding="utf-8").splitlines())
    if not reader.fieldnames:
        raise IsomerTruthError(f"empty csv or missing header: {path}")
    return [dict(r) for r in reader]


def build_isomer_truth_rows(*, raw_csv: Path) -> tuple[list[str], list[dict[str, str]]]:
    rows = _read_rows(raw_csv)
    if not rows:
        raise IsomerTruthError("raw_csv has no data rows")

    header = list(rows[0].keys())
    missing = [c for c in _REQUIRED_INPUT_COLS if c not in header]
    if missing:
        raise IsomerTruthError(f"raw_csv missing required columns: {missing}")

    seen_ids: set[str] = set()
    group_counts: dict[str, int] = {}

    out_rows: list[dict[str, str]] = []
    for row in rows:
        rid = (row.get("id") or "").strip()
        if not rid:
            raise IsomerTruthError("empty id in raw_csv row")
        if rid in seen_ids:
            raise IsomerTruthError(f"duplicate id in raw_csv: {rid}")
        seen_ids.add(rid)

        gid = (row.get("group_id") or "").strip()
        if not gid:
            raise IsomerTruthError(f"empty group_id for id={rid}")
        group_counts[gid] = int(group_counts.get(gid, 0)) + 1

        smiles = (row.get("smiles") or "").strip()
        if not smiles:
            raise IsomerTruthError(f"empty smiles for id={rid}")

        rel = (row.get("energy_rel_kcalmol") or "").strip()
        try:
            float(rel)
        except Exception as exc:
            raise IsomerTruthError(f"energy_rel_kcalmol not a float for id={rid}: {rel!r}") from exc

        out_row = {k: (row.get(k) or "").strip() for k in header}
        out_row["truth_source"] = TRUTH_SOURCE
        out_row["truth_version"] = TRUTH_VERSION
        out_rows.append(out_row)

    small_groups = sorted([gid for gid, n in group_counts.items() if int(n) < 2])
    if small_groups:
        raise IsomerTruthError(f"groups with <2 rows: {small_groups}")

    out_header = header + ["truth_source", "truth_version"]
    out_rows_sorted = sorted(out_rows, key=lambda r: (str(r.get("group_id", "")), float(r.get("energy_rel_kcalmol") or 0.0), str(r.get("id", ""))))
    return out_header, out_rows_sorted


def write_isomer_truth_csv(*, fieldnames: list[str], rows: list[dict[str, str]], out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: str(row.get(k, "")) for k in fieldnames})


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build canonical isomer truth CSV (isomer_truth.v1) from raw SPICE isomers CSV.")
    p.add_argument("--raw_csv", type=Path, default=DEFAULT_RAW_CSV, help="Raw input CSV path.")
    p.add_argument("--out_csv", type=Path, default=DEFAULT_OUT_CSV, help="Output canonical truth CSV path.")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    fieldnames, rows = build_isomer_truth_rows(raw_csv=args.raw_csv)
    write_isomer_truth_csv(fieldnames=fieldnames, rows=rows, out_csv=args.out_csv)
    out_hash = _sha256_file(args.out_csv)
    print(f"wrote: {args.out_csv.as_posix()}")
    print(f"rows: {len(rows)}")
    print(f"sha256: {out_hash}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
