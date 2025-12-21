from __future__ import annotations

import argparse
import csv
import pickle
import platform
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from analysis.chem.alkane_expected_counts import expected_unique_alkane_tree_topologies
from analysis.chem.mixing_diagnostics_1 import MixingDiagnosticsSummary, compute_mixing_diagnostics
from analysis.chem.topology_mcmc import Edge, run_fixed_n_tree_mcmc, tree_topology_edge_key_from_edges
from analysis.io_utils import results_path
from analysis.utils.progress import progress_iter
from analysis.utils.timing import now_iso


def _start_edges_for_spec(n: int, spec: str) -> List[Edge]:
    spec = str(spec)
    if spec == "path":
        return [(int(i), int(i + 1)) for i in range(int(n) - 1)]
    if spec == "max_branch":
        if int(n) <= 5:
            return [(0, i) for i in range(1, int(n))]
        edges: List[Edge] = [(0, 1), (0, 2), (0, 3), (0, 4)]
        last = 4
        for v in range(5, int(n)):
            edges.append((last, v))
            last = v
        return edges
    raise ValueError(f"Unknown start_spec: {spec!r}")


@dataclass
class Cfg:
    n: int = 15
    steps_grid: Tuple[int, ...] = (2_000_000, 4_000_000, 8_000_000, 16_000_000)
    chains: int = 3
    thin: int = 10
    start_specs: Tuple[str, ...] = ("path", "max_branch")
    seed: int = 0
    progress: bool = True
    step_progress: bool = True
    step_heartbeat_every: int = 200_000
    out_stub: Optional[str] = None
    resume: bool = False
    append: bool = False
    energy_cache_path: Optional[str] = None
    save_cache_every_points: int = 1
    emit_run_blocks: bool = True


def _load_existing_steps(path: Path) -> List[int]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        steps: List[int] = []
        for row in reader:
            if "steps" in row and row["steps"]:
                try:
                    steps.append(int(row["steps"]))
                except ValueError:
                    continue
        return steps


def _load_energy_cache(path: Path) -> Dict[Tuple[str, object], float]:
    if not path.exists():
        return {}
    with path.open("rb") as f:
        return pickle.load(f)


def _save_energy_cache(path: Path, cache: Mapping[Tuple[str, object], float]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("wb") as f:
        pickle.dump(dict(cache), f)
    tmp.replace(path)


def _git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "UNKNOWN"


def _safe_nanmax(values: Sequence[float]) -> float:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0 or np.isnan(arr).all():
        return float("nan")
    return float(np.nanmax(arr))


def _safe_nanmin(values: Sequence[float]) -> float:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0 or np.isnan(arr).all():
        return float("nan")
    return float(np.nanmin(arr))


def _parse_args(argv: Optional[Sequence[str]] = None) -> Cfg:
    ap = argparse.ArgumentParser(description="EQ-TARGET-3: scan steps for arbitrary N using mixing diagnostics (Mode A).")
    ap.add_argument("--N", type=int, required=True)
    ap.add_argument("--steps_grid", type=int, nargs="*", default=[2_000_000, 4_000_000, 8_000_000, 16_000_000])
    ap.add_argument("--chains", type=int, default=3)
    ap.add_argument("--thin", type=int, default=10)
    ap.add_argument("--start_specs", type=str, nargs="*", default=["path", "max_branch"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--progress", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--step_progress", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--step_heartbeat_every", type=int, default=200_000)
    ap.add_argument("--out_stub", type=str, default=None)
    ap.add_argument("--resume", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--append", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--energy_cache_path", type=str, default=None)
    ap.add_argument("--save_cache_every_points", type=int, default=1)
    ap.add_argument("--emit_run_blocks", action=argparse.BooleanOptionalAction, default=True)
    args = ap.parse_args(argv)
    return Cfg(
        n=int(args.N),
        steps_grid=tuple(int(x) for x in args.steps_grid),
        chains=int(args.chains),
        thin=int(args.thin),
        start_specs=tuple(str(x) for x in args.start_specs),
        seed=int(args.seed),
        progress=bool(args.progress),
        step_progress=bool(args.step_progress),
        step_heartbeat_every=int(args.step_heartbeat_every),
        out_stub=str(args.out_stub) if args.out_stub is not None else None,
        resume=bool(args.resume),
        append=bool(args.append),
        energy_cache_path=str(args.energy_cache_path) if args.energy_cache_path else None,
        save_cache_every_points=max(1, int(args.save_cache_every_points)),
        emit_run_blocks=bool(args.emit_run_blocks),
    )


def main(argv: Optional[Sequence[str]] = None) -> None:
    argv_list = list(argv) if argv is not None else sys.argv[1:]
    cfg = _parse_args(argv_list)
    out_stub = cfg.out_stub or f"eq_target_3_N{cfg.n}_modeA"
    out_csv = results_path(f"{out_stub}.csv")
    out_txt = results_path(f"{out_stub}.txt")
    cache_path = Path(cfg.energy_cache_path) if cfg.energy_cache_path else results_path(f"{out_stub}_energy_cache.pkl")

    start_ts = now_iso()
    t0 = time.perf_counter()
    ensure_dir = out_csv.parent
    ensure_dir.mkdir(parents=True, exist_ok=True)
    if not cfg.append and not cfg.resume:
        if out_csv.exists():
            out_csv.unlink()
        if out_txt.exists():
            out_txt.unlink()

    csv_header_written = out_csv.exists()
    txt_initialized = out_txt.exists()
    csv_fieldnames: Optional[List[str]] = None

    existing_steps = set(_load_existing_steps(out_csv)) if cfg.resume and out_csv.exists() else set()

    try:
        energy_cache: Dict[Tuple[str, object], float] = _load_energy_cache(cache_path)
    except Exception:
        energy_cache = {}

    expected_unique = expected_unique_alkane_tree_topologies(int(cfg.n))
    git_sha = _git_sha()
    host = platform.node() or "UNKNOWN"
    date_utc = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    cmd_str = "python -m analysis.chem.eq_target_3_scan " + " ".join(argv_list)

    def append_txt_lines(lines: Sequence[str]) -> None:
        with out_txt.open("a", encoding="utf-8") as f:
            for line in lines:
                f.write(line + "\n")

    def ensure_txt_header() -> None:
        nonlocal txt_initialized
        if txt_initialized:
            return
        append_txt_lines(
            [
                f"EQ-TARGET-3: N={cfg.n}, mode=A, chains={cfg.chains}, thin={cfg.thin}, starts={list(cfg.start_specs)}",
                f"steps_grid={list(cfg.steps_grid)}",
                f"out_stub={out_stub}",
                "",
            ]
        )
        txt_initialized = True

    ensure_txt_header()
    append_txt_lines([f"RUN_START_TS={start_ts}", f"CMD={cmd_str}", ""])

    if cfg.emit_run_blocks:
        run_header_lines = [
            "RUN_HEADER",
            f"N={cfg.n}",
            "MODE=A",
            f"GIT_SHA={git_sha}",
            f"HOST={host}",
            f"DATE_UTC={date_utc}",
            f"CMD={cmd_str}",
            f"START_SPECS={','.join(cfg.start_specs)}",
            f"CHAINS={cfg.chains}",
            f"THIN={cfg.thin}",
            "",
        ]
        print("\n".join(run_header_lines))

    points_since_cache_save = 0

    def append_csv_row(row: Dict[str, object]) -> None:
        nonlocal csv_header_written, csv_fieldnames
        if csv_fieldnames is None:
            csv_fieldnames = list(row.keys())
        with out_csv.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=csv_fieldnames)
            if not csv_header_written:
                writer.writeheader()
                csv_header_written = True
            writer.writerow(row)

    for steps_per_chain in cfg.steps_grid:
        if steps_per_chain in existing_steps:
            print(f"[EQ-TARGET-3] Skipping steps={steps_per_chain} (resume)")
            continue

        burnin = int(max(0, round(0.1 * float(steps_per_chain))))
        per_start_diags: Dict[str, MixingDiagnosticsSummary] = {}
        total_elapsed = 0.0
        total_steps_done = 0
        total_samples_recorded = 0
        total_proposals = 0
        total_accepted = 0
        cache_hits_total = 0
        cache_misses_total = 0
        unique_eqs_seen: set[str] = set()

        for start_spec in cfg.start_specs:
            seqs: List[List[str]] = []
            energies: List[List[float]] = []
            t_start = time.perf_counter()

            chains_iter = progress_iter(
                range(int(cfg.chains)),
                total=int(cfg.chains),
                desc=f"EQ-TARGET-3 N{cfg.n} steps={steps_per_chain} start={start_spec}",
                enabled=bool(cfg.progress) and not bool(cfg.step_progress),
            )
            for chain_idx in chains_iter:
                edges0 = _start_edges_for_spec(cfg.n, start_spec)

                pbar = None
                hb_every = max(1, int(cfg.step_heartbeat_every))
                if bool(cfg.progress) and bool(cfg.step_progress):
                    try:
                        from tqdm import tqdm  # type: ignore

                        pbar = tqdm(
                            total=int(steps_per_chain),
                            desc=f"EQ N{cfg.n} start={start_spec} chain={int(chain_idx)}",
                            unit="step",
                            mininterval=1.0,
                            smoothing=0.1,
                        )
                    except Exception:
                        pbar = None

                last_hb_step = 0

                def _heartbeat(info: Mapping[str, float]) -> None:
                    nonlocal last_hb_step
                    step = int(info.get("step", 0))
                    if pbar is not None:
                        delta = max(0, step - last_hb_step)
                        last_hb_step = step
                        pbar.update(delta)
                        pbar.set_postfix(
                            steps_per_sec=f"{info.get('heartbeat_steps_per_sec', 0.0):.0f}",
                            acc=f"{info.get('accept_rate', 0.0):.3f}",
                            hit=f"{info.get('energy_cache_hit_rate', 0.0):.3f}",
                            misses=str(int(info.get("energy_cache_misses_seen", 0.0))),
                        )
                    else:
                        print(
                            f"[EQ-TARGET-3 N={cfg.n} start={start_spec} chain={int(chain_idx)}] "
                            f"step={step}/{int(steps_per_chain)} "
                            f"steps/sec={info.get('heartbeat_steps_per_sec', 0.0):.0f} "
                            f"acc={info.get('accept_rate', 0.0):.3f} "
                            f"hit={info.get('energy_cache_hit_rate', 0.0):.3f} "
                            f"misses_seen={int(info.get('energy_cache_misses_seen', 0.0))}"
                        )

                try:
                    samples, summary = run_fixed_n_tree_mcmc(
                        n=int(cfg.n),
                        steps=int(steps_per_chain),
                        burnin=int(burnin),
                        thin=int(cfg.thin),
                        backend="fdm",
                        lam=1.0,
                        temperature_T=1.0,
                        seed=int(cfg.seed) + 101 * int(chain_idx) + 10_000 * int(steps_per_chain),
                        max_valence=4,
                        topology_key_fn_edges=tree_topology_edge_key_from_edges,
                        start_edges=edges0,
                        progress=None,
                        step_heartbeat_every=hb_every if bool(cfg.progress) else 0,
                        step_heartbeat=_heartbeat if bool(cfg.progress) else None,
                        energy_cache=energy_cache,
                    )
                finally:
                    if pbar is not None:
                        try:
                            if last_hb_step < int(steps_per_chain):
                                pbar.update(int(steps_per_chain) - int(last_hb_step))
                        finally:
                            pbar.close()
                seqs.append([str(s["topology"]) for s in samples])
                energies.append([float(s["energy"]) for s in samples])
                cache_hits_total += int(summary.energy_cache_hits)
                cache_misses_total += int(summary.energy_cache_misses)
                total_steps_done += int(summary.steps)
                total_proposals += int(summary.proposals)
                total_accepted += int(summary.accepted)

            dt = time.perf_counter() - t_start
            total_elapsed += dt
            per_start_diags[str(start_spec)] = compute_mixing_diagnostics(
                n=int(cfg.n),
                steps=int(steps_per_chain),
                burnin=int(burnin),
                thin=int(cfg.thin),
                start_spec=str(start_spec),
                topology_sequences_by_chain=seqs,
                energy_traces_by_chain=energies,
            )
            total_samples_recorded += sum(len(seq) for seq in seqs)
            for seq in seqs:
                unique_eqs_seen.update(seq)

        diag_values = list(per_start_diags.values())
        kl_max = _safe_nanmax([d.kl_pairwise_max for d in diag_values])
        kl_split_max = _safe_nanmax([d.kl_split_max for d in diag_values])
        rhat_max = _safe_nanmax([d.rhat_energy for d in diag_values])
        ess_min = _safe_nanmin([d.ess_energy_min for d in diag_values])
        steps_per_sec_total = float(total_steps_done) / float(total_elapsed) if total_elapsed > 0 else 0.0
        cache_total = cache_hits_total + cache_misses_total
        cache_hit_rate = float(cache_hits_total) / float(cache_total) if cache_total > 0 else 0.0
        accept_rate = float(total_accepted) / float(total_proposals) if total_proposals > 0 else 0.0
        n_unique_eq = int(len(unique_eqs_seen))
        unique_frac_samples = float(n_unique_eq) / float(total_samples_recorded) if total_samples_recorded > 0 else 0.0
        coverage_unique = float(n_unique_eq) / float(expected_unique) if expected_unique > 0 else 0.0

        row = {
            "N": int(cfg.n),
            "steps": int(steps_per_chain),
            "burnin": int(burnin),
            "thin": int(cfg.thin),
            "chains": int(cfg.chains),
            "starts": "|".join(cfg.start_specs),
            "elapsed_sec": float(total_elapsed),
            "steps_total": int(total_steps_done),
            "steps_per_sec_total": float(steps_per_sec_total),
            "energy_cache_hit_rate": float(cache_hit_rate),
            "energy_cache_hits": int(cache_hits_total),
            "energy_cache_misses": int(cache_misses_total),
            "accept_rate_total": float(accept_rate),
            "n_unique_eq": int(n_unique_eq),
            "expected_unique_eq": int(expected_unique),
            "coverage_unique_eq": float(coverage_unique),
            "unique_frac_samples": float(unique_frac_samples),
            "KL_max_pairwise": float(kl_max),
            "KL_split_max": float(kl_split_max),
            "Rhat_energy_max": float(rhat_max),
            "ESS_energy_min": float(ess_min),
        }
        append_csv_row(row)

        summary_line = (
            f"steps={int(steps_per_chain)}: KL_max={kl_max:.6g} KL_split_max={kl_split_max:.6g} "
            f"Rhat_max={rhat_max:.4g} ESS_min={ess_min:.1f} "
            f"elapsed={total_elapsed:.1f}s steps/s={steps_per_sec_total:.0f} hit={cache_hit_rate:.3f} "
            f"ACC_RATE={accept_rate:.3f} UNIQUE_EQ={n_unique_eq} EXPECTED_UNIQUE_EQ={expected_unique} "
            f"COVERAGE_UNIQUE_EQ={coverage_unique:.6f} UNIQUE_FRAC_SAMPLES={unique_frac_samples:.6f} "
            f"HITS={cache_hits_total} MISSES={cache_misses_total} "
            f"STEPS_TOTAL={int(total_steps_done)} STEPS_PER_SEC_TOTAL={steps_per_sec_total:.1f} "
            f"ELAPSED_SEC={total_elapsed:.3f}"
        )
        append_txt_lines([f"[EQ-TARGET-3] {summary_line}"])
        print(f"[EQ-TARGET-3] {summary_line}")

        if cfg.emit_run_blocks:
            block_lines = [
                "POINT",
                f"STEPS_PER_CHAIN={int(steps_per_chain)}",
                f"KL_MAX_PAIRWISE={kl_max:.6g}",
                f"KL_SPLIT_MAX={kl_split_max:.6g}",
                f"RHAT_ENERGY_MAX={rhat_max:.4g}",
                f"ESS_ENERGY_MIN={ess_min:.1f}",
                f"N_UNIQUE_EQ={n_unique_eq}",
                f"EXPECTED_UNIQUE_EQ={expected_unique}",
                f"COVERAGE_UNIQUE_EQ={coverage_unique:.6f}",
                f"UNIQUE_FRAC_SAMPLES={unique_frac_samples:.6f}",
                f"ACCEPT_RATE={accept_rate:.3f}" if total_proposals > 0 else "ACCEPT_RATE=NA",
                f"HIT_RATE={cache_hit_rate:.3f}",
                f"MISSES_SEEN={cache_misses_total}",
                f"STEPS_TOTAL={int(total_steps_done)}",
                f"STEPS_PER_SEC_TOTAL={steps_per_sec_total:.1f}",
                f"ELAPSED_SEC={total_elapsed:.3f}",
                f"RAW=[EQ-TARGET-3] {summary_line}",
                "",
            ]
            print("\n".join(block_lines))

        points_since_cache_save += 1
        if cache_path and energy_cache and points_since_cache_save >= int(cfg.save_cache_every_points):
            _save_energy_cache(cache_path, energy_cache)
            points_since_cache_save = 0

    end_ts = now_iso()
    elapsed_total = time.perf_counter() - t0

    append_txt_lines([f"RUN_END_TS={end_ts}", f"ELAPSED_TOTAL_SEC={elapsed_total:.6f}", ""])
    if cache_path and energy_cache:
        _save_energy_cache(cache_path, energy_cache)

    print(f"[EQ-TARGET-3] wrote {out_csv}")
    print(f"[EQ-TARGET-3] wrote {out_txt}")
    print(f"Wall-clock: start={start_ts} end={end_ts} elapsed_total_sec={elapsed_total:.3f}")


if __name__ == "__main__":
    main()
