from __future__ import annotations

import sys
from pathlib import Path

from analysis.geom.analyze_geom_table import compute_element_indices
from analysis.extend_d_block_from_pauling import main as extend_d_block_main
from analysis.growth.scan_living_sectors import run_living_sectors_scan
from analysis.growth.scan_living_sectors_segregated import run_segregated_scan
from analysis.scan_dblock_complexity import scan_dblock_complexity
from analysis.nuclear.scan_isotope_band import main as scan_isotope_band_main
from analysis.map_geom_to_valley import main as map_geom_to_valley_main
from analysis.nuclear.analyze_geom_nuclear_complexity import main as analyze_geom_nuclear_complexity_main


ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"


def ensure_dirs() -> None:
    DATA_DIR.mkdir(exist_ok=True)
    RESULTS_DIR.mkdir(exist_ok=True)


def main() -> None:
    ensure_dirs()

    # 1. Geom indices
    indices_path = DATA_DIR / "element_indices_v4.csv"
    if not indices_path.exists():
        df = compute_element_indices()
        df.to_csv(indices_path, index=False)

    # 2. Extend d-block (writes element_indices_with_dblock.csv next to indices)
    extend_d_block_main(indices_path=str(indices_path))

    # 3. Complexity scans
    run_living_sectors_scan()
    run_segregated_scan()
    scan_dblock_complexity()

    # 4. Nuclear scans
    scan_isotope_band_main()
    # map_geom_to_valley already writes geom_nuclear_map.csv; use UTF-8 mode
    import subprocess

    subprocess.run(
        [sys.executable, "-X", "utf8", "-m", "analysis.map_geom_to_valley"],
        check=True,
    )
    analyze_geom_nuclear_complexity_main()

    print("\n=== PIPELINE DONE ===")
    print("Key artifacts:")
    print(f"  D/A indices (with d-block): data/element_indices_with_dblock.csv")
    print(f"  Living sectors plot: results/living_sectors_scan.png")
    print(f"  Geom–nuclear summary: data/geom_nuclear_complexity_summary.csv")
    print(f"  Geom–nuclear stats: results/geom_nuclear_complexity_stats.txt")


if __name__ == "__main__":
    sys.exit(main())
