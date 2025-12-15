from __future__ import annotations

import csv
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from analysis.io_utils import results_path
from analysis.growth.rng import make_rng
from analysis.growth.reporting import write_growth_txt
from core.grower import GrowthParams, grow_molecule_christmas_tree
from core.complexity import compute_complexity_features_v2
from core.geom_atoms import compute_element_indices
from core.thermo_config import ThermoConfig, override_thermo_config


@dataclass
class SmartGrowthBenchConfig:
    z_elements: Tuple[int, ...] = (6, 8, 14, 26)
    n_trees_per_Z: int = 50
    max_depth: int = 4
    max_atoms: int = 25
    # shape couplings enabled to stress ShapeObs path
    coupling_shape_softness: float = 1.0
    coupling_shape_chi: float = 1.0
    # MH/growth controls
    grower_use_mh: bool = True
    deltaG_backend: str = "fdm_entanglement"
    temperature_T: float = 1.0


def _make_thermo(ws_integrator: str) -> ThermoConfig:
    return ThermoConfig(
        ws_integrator=ws_integrator,
        ws_fdm_base=2,
        ws_fdm_depth=5,
        coupling_shape_softness=1.0,
        coupling_shape_chi=1.0,
        grower_use_mh=True,
        deltaG_backend="fdm_entanglement",
        temperature_T=1.0,
    )


def _element_symbol(Z: int) -> str:
    indices = [(item["Z"], item["El"]) for item in compute_element_indices()]
    for z, el in indices:
        if int(z) == int(Z):
            return str(el)
    raise ValueError(f"No element symbol found for Z={Z}")


def _run_growth_for_mode(
    cfg: SmartGrowthBenchConfig,
    ws_integrator: str,
) -> List[Dict[str, float]]:
    thermo = _make_thermo(ws_integrator)
    results: List[Dict[str, float]] = []

    params = GrowthParams(
        max_depth=cfg.max_depth,
        max_atoms=cfg.max_atoms,
    )

    base_rng = make_rng(f"smart_growth_1_{ws_integrator}")

    with override_thermo_config(thermo):
        for Z in cfg.z_elements:
            sym = _element_symbol(Z)
            # фиксируем отдельный RNG на каждый (mode, Z) чтобы сценарий повторялся
            rng = np.random.default_rng(base_rng.integers(0, 2**32 - 1))

            runtimes: List[float] = []
            sizes: List[int] = []
            c_total: List[float] = []

            for _ in range(cfg.n_trees_per_Z):
                t0 = time.perf_counter()
                mol = grow_molecule_christmas_tree(sym, params, rng=rng)
                dt = time.perf_counter() - t0
                runtimes.append(dt)

                n_atoms = len(mol.atoms)
                sizes.append(n_atoms)

                adj = mol.adjacency_matrix()
                feats_fdm = compute_complexity_features_v2(adj, backend="fdm")
                c_total.append(float(feats_fdm.total))

            runtimes_arr = np.array(runtimes, dtype=float)
            sizes_arr = np.array(sizes, dtype=float)
            c_arr = np.array(c_total, dtype=float)

            results.append(
                {
                    "mode": ws_integrator,
                    "Z": Z,
                    "Z_symbol": sym,
                    "n_trees": cfg.n_trees_per_Z,
                    "runtime_total_sec": float(runtimes_arr.sum()),
                    "runtime_per_tree_sec_mean": float(runtimes_arr.mean()),
                    "runtime_per_tree_sec_median": float(np.median(runtimes_arr)),
                    "size_mean": float(sizes_arr.mean()),
                    "size_median": float(np.median(sizes_arr)),
                    "complexity_fdm_mean": float(c_arr.mean()),
                    "complexity_fdm_max": float(c_arr.max()),
                }
            )

    return results


def run_smart_growth_bench(cfg: SmartGrowthBenchConfig) -> Tuple[Path, Path]:
    rows: List[Dict[str, float]] = []

    for mode in ["trapz", "fdm"]:
        rows.extend(_run_growth_for_mode(cfg, mode))

    csv_path = results_path("smart_growth_1_bench.csv")
    fieldnames = [
        "mode",
        "Z",
        "Z_symbol",
        "n_trees",
        "runtime_total_sec",
        "runtime_per_tree_sec_mean",
        "runtime_per_tree_sec_median",
        "size_mean",
        "size_median",
        "complexity_fdm_mean",
        "complexity_fdm_max",
    ]

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    # summary
    lines: List[str] = []
    lines.append("SMART-GROWTH-1 end-to-end MH growth benchmark")
    lines.append(f"Z_list={list(cfg.z_elements)}")
    lines.append(f"n_trees_per_Z={cfg.n_trees_per_Z}")
    lines.append(f"GrowthParams: max_depth={cfg.max_depth}, max_atoms={cfg.max_atoms}")
    lines.append("Thermo: ws_fdm_base=2, ws_fdm_depth=5, coupling_shape_softness=1.0, coupling_shape_chi=1.0")
    lines.append("")

    # group by mode/Z
    by_key: Dict[Tuple[str, int], Dict[str, float]] = {}
    for r in rows:
        key = (str(r["mode"]), int(r["Z"]))
        by_key[key] = r

    for Z in cfg.z_elements:
        row_trapz = by_key.get(("trapz", Z))
        row_fdm = by_key.get(("fdm", Z))
        if row_trapz is None or row_fdm is None:
            continue
        speedup = row_trapz["runtime_per_tree_sec_mean"] / row_fdm["runtime_per_tree_sec_mean"]
        lines.append(
            f"Z={Z}: trapz_mean={row_trapz['runtime_per_tree_sec_mean']:.6f}s, "
            f"fdm_mean={row_fdm['runtime_per_tree_sec_mean']:.6f}s, "
            f"speedup={speedup:.2f}x, "
            f"size_mean_trapz={row_trapz['size_mean']:.2f}, "
            f"size_mean_fdm={row_fdm['size_mean']:.2f}, "
            f"Cfdm_mean_trapz={row_trapz['complexity_fdm_mean']:.2f}, "
            f"Cfdm_mean_fdm={row_fdm['complexity_fdm_mean']:.2f}"
        )

    summary_path = write_growth_txt("smart_growth_1_bench", lines)
    return csv_path, summary_path


def main() -> None:
    cfg = SmartGrowthBenchConfig()
    csv_path, summary_path = run_smart_growth_bench(cfg)
    print("[SMART-GROWTH-1] benchmark done.")
    print(f"CSV: {csv_path}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
