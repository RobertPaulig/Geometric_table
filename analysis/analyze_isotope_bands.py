from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

from analysis.io_utils import data_path
from analysis.nuclear_cli import apply_nuclear_config_if_provided


def load_bands(path: str | None = None) -> List[Dict[str, Any]]:
    if path is None:
        csv_path = data_path("geom_isotope_bands.csv")
    else:
        csv_path = Path(path)

    rows: List[Dict[str, Any]] = []
    with csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)
    return rows


def to_float(row: Dict[str, Any], key: str, default: float = 0.0) -> float:
    try:
        return float(row.get(key, default))
    except Exception:
        return default


def to_int(row: Dict[str, Any], key: str, default: int = 0) -> int:
    try:
        return int(row.get(key, default))
    except Exception:
        return default


def summarize_group(name: str, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        print(f"\n[{name}] — empty group")
        return

    ws: List[float] = []
    rel_ws: List[float] = []
    Ns: List[int] = []
    Zs: List[int] = []

    for r in rows:
        bw = to_int(r, "band_width", 0)
        N_min = to_int(r, "N_min", 0)
        N_max = to_int(r, "N_max", 0)
        Z = to_int(r, "Z", 0)
        N_best = to_int(r, "N_best", 0)

        if Z <= 0:
            continue

        span = max(1, N_max - N_min + 1)
        rel = float(bw) / float(span)

        ws.append(float(bw))
        rel_ws.append(rel)
        Ns.append(N_best)
        Zs.append(Z)

    def avg(x: List[float]) -> float:
        return sum(x) / len(x) if x else 0.0

    if not Zs:
        print(f"\n[{name}] — empty group (after filtering invalid Z)")
        return

    print(f"\n[{name}]")
    print(f"  count          = {len(Zs)}")
    print(f"  avg band_width = {avg(ws):.2f}")
    print(f"  avg rel_width  = {avg(rel_ws):.3f}")
    print(f"  avg Z          = {avg([float(z) for z in Zs]):.1f}")
    print(f"  avg N/Z        = {avg([Ns[i] / Zs[i] for i in range(len(Zs))]):.2f}")


def main(argv=None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--nuclear-config",
        type=str,
        default=None,
        help="Path to nuclear config (YAML/JSON); baseline used if omitted.",
    )
    args = parser.parse_args(argv)

    apply_nuclear_config_if_provided(args.nuclear_config)

    rows = load_bands()

    # Группы по роли
    by_role: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in rows:
        role = (r.get("role") or "").strip()
        by_role[role].append(r)

    print("[BY ROLE]")
    for role, rlist in by_role.items():
        summarize_group(f"role={role or 'unknown'}", rlist)

    # Живые хабы
    living = [r for r in rows if (r.get("living_hub") or "").lower() == "true"]
    summarize_group("living_hubs", living)

    # Доноры (D_index > 0)
    donors = [r for r in rows if to_float(r, "D_index") > 0.0]
    summarize_group("donors", donors)

    # Амфотерные (hub, A_index ~ 0.1237)
    amph = [
        r
        for r in rows
        if (r.get("role") or "").strip() == "hub"
        and abs(to_float(r, "A_index") - 0.1237) < 1e-3
    ]
    summarize_group("amphoteric_hubs", amph)


if __name__ == "__main__":
    main()
