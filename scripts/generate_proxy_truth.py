from __future__ import annotations

import argparse
import csv
import hashlib
from pathlib import Path
from typing import Iterable


class ProxyTruthError(ValueError):
    pass


TRUTH_SOURCE = "proxy_rule_v1"
TRUTH_VERSION = "customer_truth.v1"


def _read_input_rows(input_csv: Path) -> list[dict[str, str]]:
    if not input_csv.exists():
        raise ProxyTruthError(f"missing input_csv: {input_csv}")
    reader = csv.DictReader(input_csv.read_text(encoding="utf-8").splitlines())
    if not reader.fieldnames:
        raise ProxyTruthError(f"empty input_csv or missing header: {input_csv}")
    return [dict(r) for r in reader]


def _canonical_smiles_from_row(row: dict[str, str]) -> str:
    # Canonical SMILES must be derived only from input (no-leakage).
    raw = (row.get("canonical_smiles") or row.get("smiles") or "").strip()
    if not raw:
        raise ProxyTruthError("missing canonical_smiles/smiles in input row")
    return raw


def _label_from_canonical_smiles(canonical_smiles: str) -> str:
    h = hashlib.sha256(canonical_smiles.encode("utf-8")).hexdigest()
    bucket = int(h[:8], 16) % 100
    return "PASS" if bucket < 5 else "FAIL"


def generate_proxy_truth_rows(*, input_csv: Path) -> list[dict[str, str]]:
    rows = _read_input_rows(input_csv)
    if not rows:
        raise ProxyTruthError("input_csv has no data rows")
    if "id" not in rows[0]:
        raise ProxyTruthError("input_csv missing required column: id")

    seen_ids: set[str] = set()
    out: list[dict[str, str]] = []
    for row in rows:
        molecule_id = (row.get("id") or "").strip()
        if not molecule_id:
            raise ProxyTruthError("input_csv row has empty id")
        if molecule_id in seen_ids:
            raise ProxyTruthError(f"duplicate molecule_id in input_csv: {molecule_id}")
        seen_ids.add(molecule_id)

        canonical_smiles = _canonical_smiles_from_row(row)
        expensive_label = _label_from_canonical_smiles(canonical_smiles)

        out.append(
            {
                "molecule_id": molecule_id,
                "canonical_smiles": canonical_smiles,
                "expensive_label": expensive_label,
                "truth_source": TRUTH_SOURCE,
                "truth_version": TRUTH_VERSION,
            }
        )
    return out


def write_proxy_truth_csv(*, rows: Iterable[dict[str, str]], out_csv: Path) -> None:
    fieldnames = ["molecule_id", "canonical_smiles", "expensive_label", "truth_source", "truth_version"]
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: str(row.get(k, "")) for k in fieldnames})


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate deterministic proxy truth CSV (customer_truth.v1).")
    p.add_argument("--input_csv", type=Path, required=True, help="Input CSV with at least: id,smiles (or canonical_smiles).")
    p.add_argument("--out_csv", type=Path, default=Path("truth.csv"), help="Output truth CSV path.")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    rows = generate_proxy_truth_rows(input_csv=args.input_csv)
    write_proxy_truth_csv(rows=rows, out_csv=args.out_csv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

