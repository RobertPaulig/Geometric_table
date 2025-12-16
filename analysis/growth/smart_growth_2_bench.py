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


def _make_thermo(
    ws_integrator: str,
    *,
    coupling_topo_3d: float = 0.0,
    topo_3d_beta: float = 0.0,
    topo3d_prefilter_tree: bool = False,
    topo3d_prefilter_min_n: int = 0,
    deltaG_backend: str = "fdm_entanglement",
) -> ThermoConfig:
    return ThermoConfig(
        ws_integrator=ws_integrator,
        ws_fdm_base=2,
        ws_fdm_depth=5,
        coupling_shape_softness=1.0,
        coupling_shape_chi=1.0,
        coupling_topo_3d=coupling_topo_3d,
        topo_3d_beta=topo_3d_beta,
        topo3d_prefilter_tree=topo3d_prefilter_tree,
        topo3d_prefilter_min_n=topo3d_prefilter_min_n,
        grower_use_mh=True,
        deltaG_backend=deltaG_backend,
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
def profile_shape_complexity_layout(
    timing_acc: Dict[str, float],
    call_counts: Dict[str, int],
) -> None:
    """
    Monkeypatch:
    - core.shape_observables.get_shape_observables
    - core.complexity.compute_complexity_features_v2
    - core.energy_model.compute_complexity_features_v2
    - core.layout_3d.force_directed_layout_3d

    to аккумулировать t_shape_total_sec / t_complexity_total_sec / t_layout_total_sec
    и счётчики вызовов в timing_acc / call_counts.
    """
    from core import shape_observables as so
    from core import complexity as cx
    from core import energy_model as em
    from core import layout_3d as l3d

    orig_get_shape = so.get_shape_observables
    orig_complexity = cx.compute_complexity_features_v2
    orig_energy_complexity = em.compute_complexity_features_v2
    orig_layout = l3d.force_directed_layout_3d

    def wrapped_get_shape(Z, fp):
        t0 = time.perf_counter()
        result = orig_get_shape(Z, fp)
        dt = time.perf_counter() - t0
        timing_acc["t_shape_total"] = timing_acc.get("t_shape_total", 0.0) + dt
        call_counts["n_shape_calls"] = call_counts.get("n_shape_calls", 0) + 1
        return result

    def wrapped_complexity(adj, backend="fdm", *args, **kwargs):
        t0 = time.perf_counter()
        result = orig_complexity(adj, backend=backend, *args, **kwargs)
        dt = time.perf_counter() - t0
        timing_acc["t_complexity_total"] = timing_acc.get("t_complexity_total", 0.0) + dt
        call_counts["n_complexity_calls"] = call_counts.get("n_complexity_calls", 0) + 1
        return result

    def wrapped_layout(*args, **kwargs):
        t0 = time.perf_counter()
        result = orig_layout(*args, **kwargs)
        dt = time.perf_counter() - t0
        timing_acc["t_layout_total"] = timing_acc.get("t_layout_total", 0.0) + dt
        call_counts["n_layout_calls"] = call_counts.get("n_layout_calls", 0) + 1
        return result

    so.get_shape_observables = wrapped_get_shape  # type: ignore[assignment]
    cx.compute_complexity_features_v2 = wrapped_complexity  # type: ignore[assignment]
    em.compute_complexity_features_v2 = wrapped_complexity  # type: ignore[assignment]
    l3d.force_directed_layout_3d = wrapped_layout  # type: ignore[assignment]
    try:
        yield
    finally:
        so.get_shape_observables = orig_get_shape  # type: ignore[assignment]
        cx.compute_complexity_features_v2 = orig_complexity  # type: ignore[assignment]
        em.compute_complexity_features_v2 = orig_energy_complexity  # type: ignore[assignment]
        l3d.force_directed_layout_3d = orig_layout  # type: ignore[assignment]


def _run_profile_for_mode(
    cfg: SmartGrowth2BenchConfig,
    profile: ProfileConfig,
    mode: str,
    thermo: ThermoConfig,
) -> List[Dict[str, float]]:
    params = GrowthParams(max_depth=profile.max_depth, max_atoms=profile.max_atoms)

    # Shared seeds per (Z, tree_idx) — одинаковые для trapz/fdm.
    seeds = _make_seeds(cfg.z_elements, profile.n_trees_per_Z, label=f"smart_growth_2_{profile.name}")

    rows: List[Dict[str, float]] = []

    # Сброс кэша ShapeObs и измерение hit/miss после профиля.
    _cached_get_shape_observables.cache_clear()

    with override_thermo_config(thermo):
        for Z in cfg.z_elements:
            sym = _element_symbol(Z)

            timing_acc: Dict[str, float] = {}
            call_counts: Dict[str, int] = {}

            runtimes: List[float] = []
            sizes: List[int] = []
            c_total: List[float] = []
            mh_proposals = 0
            mh_accepted = 0
            mh_rejected = 0

            mols: List[Tuple[int, "object"]] = []

            with profile_shape_complexity_layout(timing_acc, call_counts):
                for i_tree in range(profile.n_trees_per_Z):
                    seed = seeds[(int(Z), int(i_tree))]
                    rng = np.random.default_rng(seed)

                    t0 = time.perf_counter()
                    mol = grow_molecule_christmas_tree(sym, params, rng=rng)
                    dt = time.perf_counter() - t0
                    runtimes.append(dt)

                    n_atoms = len(mol.atoms)
                    sizes.append(n_atoms)

                    mols.append((n_atoms, mol))

                    stats = getattr(mol, "mh_stats", None)
                    if isinstance(stats, dict):
                        mh_proposals += int(stats.get("proposals", 0))
                        mh_accepted += int(stats.get("accepted", 0))
                        mh_rejected += int(stats.get("rejected", 0))

            # post-hoc complexity (fdm) считаем отдельно, вне профилирования entanglement/layout
            for _n_atoms, mol in mols:
                adj = mol.adjacency_matrix()
                feats_fdm = compute_complexity_features_v2(adj, backend="fdm")
                c_total.append(float(feats_fdm.total))

            runtimes_arr = np.array(runtimes, dtype=float)
            sizes_arr = np.array(sizes, dtype=float)
            c_arr = np.array(c_total, dtype=float)

            mh_total = mh_proposals if mh_proposals > 0 else 1
            mh_accept_rate = float(mh_accepted) / float(mh_total)

            cache_info = _cached_get_shape_observables.cache_info()

            rows.append(
                {
                    "profile": profile.name,
                    "mode": mode,
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
                    "t_shape_total_sec": float(timing_acc.get("t_shape_total", 0.0)),
                    "t_complexity_total_sec": float(timing_acc.get("t_complexity_total", 0.0)),
                    "t_layout_total_sec": float(timing_acc.get("t_layout_total", 0.0)),
                    "n_shape_calls": float(call_counts.get("n_shape_calls", 0)),
                    "n_complexity_calls": float(call_counts.get("n_complexity_calls", 0)),
                    "n_layout_calls": float(call_counts.get("n_layout_calls", 0)),
                    "shape_cache_hits": float(cache_info.hits),
                    "shape_cache_misses": float(cache_info.misses),
                }
            )

    return rows


def run_smart_growth_2_bench(cfg: SmartGrowth2BenchConfig) -> Tuple[Path, Path]:
    rows: List[Dict[str, float]] = []

    for profile in cfg.profiles:
        if profile.name == "SMALL":
            # Для SMALL сохраняем сравнение trapz vs fdm как раньше.
            for mode in ("trapz", "fdm"):
                thermo = _make_thermo(ws_integrator=mode)
                rows.extend(_run_profile_for_mode(cfg, profile, mode=mode, thermo=thermo))
        elif profile.name == "HEAVY":
            # HEAVY: baseline vs optimized для entanglement/layout.
            thermo_baseline = _make_thermo(
                ws_integrator="fdm",
                coupling_topo_3d=1.0,
                topo_3d_beta=1.0,
                topo3d_prefilter_tree=False,
                topo3d_prefilter_min_n=0,
                deltaG_backend="fdm_entanglement",
            )
            thermo_optimized = _make_thermo(
                ws_integrator="fdm",
                coupling_topo_3d=1.0,
                topo_3d_beta=1.0,
                topo3d_prefilter_tree=True,
                topo3d_prefilter_min_n=10,
                deltaG_backend="fdm_entanglement",
            )
            rows.extend(_run_profile_for_mode(cfg, profile, mode="baseline", thermo=thermo_baseline))
            rows.extend(_run_profile_for_mode(cfg, profile, mode="optimized", thermo=thermo_optimized))
        else:
            for mode in ("trapz", "fdm"):
                thermo = _make_thermo(ws_integrator=mode)
                rows.extend(_run_profile_for_mode(cfg, profile, mode=mode, thermo=thermo))

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
        "t_layout_total_sec",
        "n_shape_calls",
        "n_complexity_calls",
        "n_layout_calls",
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
        if profile.name == "SMALL":
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
        elif profile.name == "HEAVY":
            lines.append("HEAVY baseline vs optimized (per Z):")
            for Z in cfg.z_elements:
                key_baseline = (profile.name, "baseline", int(Z))
                key_opt = (profile.name, "optimized", int(Z))
                row_b = by_key.get(key_baseline)
                row_o = by_key.get(key_opt)
                if row_b is None or row_o is None:
                    continue

                speedup_total = row_b["runtime_per_tree_sec_mean"] / row_o["runtime_per_tree_sec_mean"]
                t_cx_b = row_b["t_complexity_total_sec"]
                t_cx_o = row_o["t_complexity_total_sec"]
                speedup_complexity = (t_cx_b / t_cx_o) if t_cx_o > 0 else float("inf")

                t_layout_b = row_b["t_layout_total_sec"]
                t_layout_o = row_o["t_layout_total_sec"]
                n_layout_b = row_b["n_layout_calls"]
                n_layout_o = row_o["n_layout_calls"]

                if t_layout_o > 0 and n_layout_o > 0:
                    speedup_layout_str = f"speedup_layout={t_layout_b / t_layout_o:.2f}x"
                else:
                    speedup_layout_str = f"layout_eliminated (n_layout_calls_baseline={int(n_layout_b)}, optimized=0)"

                lines.append(
                    f"Z={Z}: "
                    f"speedup_total={speedup_total:.2f}x, "
                    f"speedup_complexity={speedup_complexity:.2f}x, "
                    f"{speedup_layout_str}; "
                    f"mh_accept_rate_baseline={row_b['mh_accept_rate']:.2f}, "
                    f"mh_accept_rate_optimized={row_o['mh_accept_rate']:.2f}"
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
