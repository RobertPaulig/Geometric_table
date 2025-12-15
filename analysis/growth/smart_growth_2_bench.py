from __future__ import annotations

import csv
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np

from analysis.io_utils import results_path
from analysis.growth.rng import make_rng
from analysis.growth.reporting import write_growth_txt
from core.grower import GrowthParams, grow_molecule_christmas_tree
from core.complexity import compute_complexity_features_v2
from core.geom_atoms import compute_element_indices
from core.thermo_config import ThermoConfig, override_thermo_config
from core.shape_observables import get_shape_observables as _cached_get_shape_observables


@dataclass
class ProfileConfig:
    name: str
    max_depth: int
    max_atoms: int
    n_trees_per_Z: int


@dataclass
class SmartGrowth2BenchConfig:
    z_elements: Tuple[int, ...] = (6, 8, 14, 26)
    profiles: Tuple[ProfileConfig, ...] = (
        ProfileConfig(name="SMALL", max_depth=4, max_atoms=25, n_trees_per_Z=50),
        ProfileConfig(name="HEAVY", max_depth=8, max_atoms=80, n_trees_per_Z=100),
    )


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
    indices = compute_element_indices()
    for item in indices:
        if int(item["Z"]) == int(Z):
            return str(item["El"])
    raise ValueError(f"No element symbol found for Z={Z}")


def _make_seeds(z_list: Iterable[int], n_trees_per_Z: int, label: str) -> Dict[Tuple[int, int], int]:
    rng = make_rng(label)
    seeds: Dict[Tuple[int, int], int] = {}
    for Z in z_list:
        for i in range(n_trees_per_Z):
            seeds[(int(Z), int(i))] = int(rng.integers(0, 2**32 - 1))
    return seeds


@contextmanager
def profile_shape_and_complexity(
    timing_acc: Dict[str, float],
    call_counts: Dict[str, int],
):
    """
    Monkeypatch get_shape_observables / compute_complexity_features_v2
    to accumulate timings and counts in timing_acc / call_counts.
    """
    from core import shape_observables as so
    from core import complexity as cx

    orig_get_shape = so.get_shape_observables
    orig_complexity = cx.compute_complexity_features_v2

    def wrapped_get_shape(Z, fp):
        t0 = time.perf_counter()
        result = orig_get_shape(Z, fp)
        dt = time.perf_counter() - t0
        timing_acc["t_shape_total"] = timing_acc.get("t_shape_total", 0.0) + dt
        call_counts["n_shape_calls"] = call_counts.get("n_shape_calls", 0) + 1
        return result

    def wrapped_complexity(adj, backend="fdm"):
        t0 = time.perf_counter()
        result = orig_complexity(adj, backend=backend)
        dt = time.perf_counter() - t0
        timing_acc["t_complexity_total"] = timing_acc.get("t_complexity_total", 0.0) + dt
        call_counts["n_complexity_calls"] = call_counts.get("n_complexity_calls", 0) + 1
        return result

    so.get_shape_observables = wrapped_get_shape  # type: ignore[assignment]
    cx.compute_complexity_features_v2 = wrapped_complexity  # type: ignore[assignment]
    try:
        yield
    finally:
        so.get_shape_observables = orig_get_shape  # type: ignore[assignment]
        cx.compute_complexity_features_v2 = orig_complexity  # type: ignore[assignment]


def _run_profile_for_mode(
    cfg: SmartGrowth2BenchConfig,
    profile: ProfileConfig,
    ws_integrator: str,
) -> List[Dict[str, float]]:
    thermo = _make_thermo(ws_integrator)
    params = GrowthParams(max_depth=profile.max_depth, max_atoms=profile.max_atoms)

    # Shared seeds per (Z, tree_idx) — одинаковые для trapz/fdm.
    seeds = _make_seeds(cfg.z_elements, profile.n_trees_per_Z, label=f"smart_growth_2_{profile.name}")

    rows: List[Dict[str, float]] = []

    # Сброс кэша ShapeObs и измерение hit/miss после профиля.
    _cached_get_shape_observables.cache_clear()

    timing_acc: Dict[str, float] = {}
    call_counts: Dict[str, int] = {}

    with override_thermo_config(thermo), profile_shape_and_complexity(timing_acc, call_counts):
        for Z in cfg.z_elements:
            sym = _element_symbol(Z)

            runtimes: List[float] = []
            sizes: List[int] = []
            c_total: List[float] = []
            mh_proposals = 0
            mh_accepted = 0
            mh_rejected = 0

            for i_tree in range(profile.n_trees_per_Z):
                seed = seeds[(int(Z), int(i_tree))]
                rng = np.random.default_rng(seed)

                t0 = time.perf_counter()
                mol = grow_molecule_christmas_tree(sym, params, rng=rng)
                dt = time.perf_counter() - t0
                runtimes.append(dt)

                n_atoms = len(mol.atoms)
                sizes.append(n_atoms)

                adj = mol.adjacency_matrix()
                feats_fdm = compute_complexity_features_v2(adj, backend="fdm")
                c_total.append(float(feats_fdm.total))

                stats = getattr(mol, "mh_stats", None)
                if isinstance(stats, dict):
                    mh_proposals += int(stats.get("proposals", 0))
                    mh_accepted += int(stats.get("accepted", 0))
                    mh_rejected += int(stats.get("rejected", 0))

            runtimes_arr = np.array(runtimes, dtype=float)
            sizes_arr = np.array(sizes, dtype=float)
            c_arr = np.array(c_total, dtype=float)

            mh_total = mh_proposals if mh_proposals > 0 else 1
            mh_accept_rate = float(mh_accepted) / float(mh_total)

            rows.append(
                {
                    "profile": profile.name,
                    "mode": ws_integrator,
                    "Z": Z,
                    "Z_symbol": sym,
                    "n_trees": profile.n_trees_per_Z,
                    "runtime_total_sec": float(runtimes_arr.sum()),
                    "runtime_per_tree_sec_mean": float(runtimes_arr.mean()),
                    "runtime_per_tree_sec_median": float(np.median(runtimes_arr)),
                    "size_mean": float(sizes_arr.mean()),
                    "size_median": float(np.median(sizes_arr)),
                    "complexity_fdm_mean": float(c_arr.mean()),
                    "complexity_fdm_max": float(c_arr.max()),
                    "mh_proposals": float(mh_proposals),
                    "mh_accepted": float(mh_accepted),
                    "mh_rejected": float(mh_rejected),
                    "mh_accept_rate": mh_accept_rate,
                }
            )

    cache_info = _cached_get_shape_observables.cache_info()
    # Добавляем агрегированные тайминги/счётчики в каждую строку профиля/режима.
    for r in rows:
        r["t_shape_total_sec"] = float(timing_acc.get("t_shape_total", 0.0))
        r["t_complexity_total_sec"] = float(timing_acc.get("t_complexity_total", 0.0))
        r["t_topo3d_total_sec"] = 0.0  # на текущем стенде topo3d/layout не используются
        r["n_shape_calls"] = float(call_counts.get("n_shape_calls", 0))
        r["n_complexity_calls"] = float(call_counts.get("n_complexity_calls", 0))
        r["shape_cache_hits"] = float(cache_info.hits)
        r["shape_cache_misses"] = float(cache_info.misses)

    return rows


def run_smart_growth_2_bench(cfg: SmartGrowth2BenchConfig) -> Tuple[Path, Path]:
    rows: List[Dict[str, float]] = []

    for profile in cfg.profiles:
        for mode in ("trapz", "fdm"):
            rows.extend(_run_profile_for_mode(cfg, profile, ws_integrator=mode))

    csv_path = results_path("smart_growth_2_bench.csv")
    fieldnames = [
        "profile",
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
        "mh_proposals",
        "mh_accepted",
        "mh_rejected",
        "mh_accept_rate",
        "t_shape_total_sec",
        "t_complexity_total_sec",
        "t_topo3d_total_sec",
        "n_shape_calls",
        "n_complexity_calls",
        "shape_cache_hits",
        "shape_cache_misses",
    ]

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    # summary txt
    lines: List[str] = []
    lines.append("SMART-GROWTH-2 end-to-end MH growth benchmark with profiling")
    lines.append(f"Z_list={list(cfg.z_elements)}")
    lines.append("")

    # группировка: (profile, mode, Z)
    by_key: Dict[Tuple[str, str, int], Dict[str, float]] = {}
    for r in rows:
        key = (str(r["profile"]), str(r["mode"]), int(r["Z"]))
        by_key[key] = r

    for profile in cfg.profiles:
        lines.append(f"Profile={profile.name} (max_depth={profile.max_depth}, max_atoms={profile.max_atoms}, n_trees_per_Z={profile.n_trees_per_Z})")
        for Z in cfg.z_elements:
            key_trapz = (profile.name, "trapz", int(Z))
            key_fdm = (profile.name, "fdm", int(Z))
            row_trapz = by_key.get(key_trapz)
            row_fdm = by_key.get(key_fdm)
            if row_trapz is None or row_fdm is None:
                continue

            speedup_total = row_trapz["runtime_per_tree_sec_mean"] / row_fdm["runtime_per_tree_sec_mean"]
            # shape компонента
            t_shape_trapz = row_trapz["t_shape_total_sec"]
            t_shape_fdm = row_fdm["t_shape_total_sec"]
            speedup_shape = (t_shape_trapz / t_shape_fdm) if t_shape_fdm > 0 else 1.0

            # доли времени
            t_tot_trapz = row_trapz["runtime_total_sec"]
            t_tot_fdm = row_fdm["runtime_total_sec"]
            frac_shape_trapz = (t_shape_trapz / t_tot_trapz) if t_tot_trapz > 0 else 0.0
            frac_shape_fdm = (t_shape_fdm / t_tot_fdm) if t_tot_fdm > 0 else 0.0

            t_cx_trapz = row_trapz["t_complexity_total_sec"]
            t_cx_fdm = row_fdm["t_complexity_total_sec"]
            frac_cx_trapz = (t_cx_trapz / t_tot_trapz) if t_tot_trapz > 0 else 0.0
            frac_cx_fdm = (t_cx_fdm / t_tot_fdm) if t_tot_fdm > 0 else 0.0

            lines.append(
                f"Profile={profile.name}, Z={Z}: "
                f"speedup_total={speedup_total:.2f}x, "
                f"speedup_shape={speedup_shape:.2f}x, "
                f"shape_frac_trapz={frac_shape_trapz:.2f}, "
                f"shape_frac_fdm={frac_shape_fdm:.2f}, "
                f"complexity_frac_trapz={frac_cx_trapz:.2f}, "
                f"complexity_frac_fdm={frac_cx_fdm:.2f}, "
                f"mh_accept_rate_trapz={row_trapz['mh_accept_rate']:.2f}, "
                f"mh_accept_rate_fdm={row_fdm['mh_accept_rate']:.2f}, "
                f"shape_hits_trapz={row_trapz['shape_cache_hits']:.0f}, "
                f"shape_hits_fdm={row_fdm['shape_cache_hits']:.0f}, "
                f"shape_misses_trapz={row_trapz['shape_cache_misses']:.0f}, "
                f"shape_misses_fdm={row_fdm['shape_cache_misses']:.0f}"
            )
        lines.append("")

    summary_path = write_growth_txt("smart_growth_2_bench", lines)
    return csv_path, summary_path


def main() -> None:
    cfg = SmartGrowth2BenchConfig()
    csv_path, summary_path = run_smart_growth_2_bench(cfg)
    print("[SMART-GROWTH-2] benchmark done.")
    print(f"CSV: {csv_path}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()

