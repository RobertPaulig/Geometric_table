from __future__ import annotations

import argparse
from collections import defaultdict
from typing import Any, Dict, List

from analysis.scan_isotope_band import scan_isotope_bands


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


def summarize_group(rows: List[Dict[str, Any]]) -> Dict[str, float]:
    if not rows:
        return {
            "count": 0,
            "avg_band": 0.0,
            "avg_rel": 0.0,
        }

    ws: List[float] = []
    rel_ws: List[float] = []

    for r in rows:
        bw = to_int(r, "band_width", 0)
        N_min = to_int(r, "N_min", 0)
        N_max = to_int(r, "N_max", 0)
        span = max(1, N_max - N_min + 1)
        rel = float(bw) / float(span)

        ws.append(float(bw))
        rel_ws.append(rel)

    def avg(x: List[float]) -> float:
        return sum(x) / len(x) if x else 0.0

    return {
        "count": len(rows),
        "avg_band": avg(ws),
        "avg_rel": avg(rel_ws),
    }


def run_for_deltaF(
    delta_F: float,
    Z_min: int = 10,
    Z_max: int = 40,
) -> None:
    print(f"\n=== delta_F = {delta_F:.1f}, Z in [{Z_min},{Z_max}] ===")

    rows = scan_isotope_bands(
        Z_min=Z_min,
        Z_max=Z_max,
        delta_F=delta_F,
        N_corridor_factor=1.8,
    )

    # По ролям
    by_role: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in rows:
        role = (r.get("role") or "").strip()
        by_role[role].append(r)

    for role in ["terminator", "bridge", "hub", "inert"]:
        grp = by_role.get(role, [])
        stats = summarize_group(grp)
        print(
            f"  role={role:10s}: "
            f"n={stats['count']:2d}, "
            f"avg_band={stats['avg_band']:.2f}, "
            f"avg_rel={stats['avg_rel']:.3f}"
        )

    # Живые хабы и доноры
    living = [
        r for r in rows
        if bool(r.get("living_hub"))
    ]
    donors = [r for r in rows if to_float(r, "D_index") > 0.0]

    live_stats = summarize_group(living)
    don_stats = summarize_group(donors)

    print(
        f"  living_hubs : n={live_stats['count']:2d}, "
        f"avg_band={live_stats['avg_band']:.2f}, "
        f"avg_rel={live_stats['avg_rel']:.3f}"
    )
    print(
        f"  donors      : n={don_stats['count']:2d}, "
        f"avg_band={don_stats['avg_band']:.2f}, "
        f"avg_rel={don_stats['avg_rel']:.3f}"
    )


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(
        description="Scan isotope band widths vs delta_F threshold."
    )
    parser.add_argument(
        "--deltaF",
        type=float,
        nargs="*",
        default=[2.0, 3.0, 5.0, 8.0, 10.0],
        help="List of delta_F thresholds to scan.",
    )
    parser.add_argument(
        "--Z-min",
        type=int,
        default=10,
        help="Minimal Z to include in scan.",
    )
    parser.add_argument(
        "--Z-max",
        type=int,
        default=40,
        help="Maximal Z to include in scan.",
    )
    args = parser.parse_args(argv)

    for dF in args.deltaF:
        run_for_deltaF(dF, Z_min=args.Z_min, Z_max=args.Z_max)


if __name__ == "__main__":
    main()
