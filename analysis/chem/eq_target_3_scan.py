from __future__ import annotations

import argparse
import csv
import math
import multiprocessing as mp
import pickle
import platform
import queue
import subprocess
import sys
import time
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from analysis.chem import eq_worker_pool
from analysis.chem.alkane_expected_counts import expected_unique_alkane_tree_topologies
from analysis.chem.topology_mcmc import Edge, run_fixed_n_tree_mcmc  # re-export legacy helper for tests
from analysis.io_utils import results_path
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
    workers: int = 1


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


def _print_progress_event(kind: str, payload: Mapping[str, object]) -> None:
    task_id = payload.get("task_id", "?")
    start_spec = payload.get("start_spec", "?")
    chain_idx = payload.get("chain_idx", "?")
    steps_total = payload.get("steps_total", "?")
    step = payload.get("step", 0)
    accept_rate = payload.get("accept_rate")
    hit_rate = payload.get("hit_rate")
    pid = payload.get("pid")
    prefix = f"[WORKER {task_id}] start={start_spec} chain={chain_idx} steps_total={steps_total} pid={pid}"

    if kind == "start":
        msg = f"{prefix} START"
    elif kind == "heartbeat":
        if isinstance(accept_rate, (int, float)) and isinstance(hit_rate, (int, float)):
            msg = (
                f"{prefix} HEARTBEAT step={step} "
                f"accept_rate={float(accept_rate):.3f} hit_rate={float(hit_rate):.3f}"
            )
        else:
            msg = f"{prefix} HEARTBEAT step={step}"
    elif kind == "done":
        msg = f"{prefix} DONE step={step}"
    else:
        msg = f"{prefix} EVENT={kind} step={step}"
    print(msg, flush=True)


def _drain_progress_queue(progress_queue: Optional[queue.Queue]) -> None:
    if progress_queue is None:
        return
    while True:
        try:
            item = progress_queue.get_nowait()
        except queue.Empty:
            break
        except Exception:
            break
        else:
            if not isinstance(item, tuple) or len(item) != 2:
                continue
            kind, payload = item
            if isinstance(kind, str) and isinstance(payload, Mapping):
                _print_progress_event(kind, payload)


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


def _counter_to_prob(counter: Counter[str]) -> Dict[str, float]:
    total = float(sum(counter.values()))
    if total <= 0:
        return {}
    return {k: float(v) / total for k, v in counter.items()}


def _kl(p: Mapping[str, float], q: Mapping[str, float], eps: float = 1e-12) -> float:
    if not p:
        return 0.0
    out = 0.0
    for key, pv in p.items():
        qv = q.get(key, 0.0) + eps
        out += pv * math.log(max(pv, eps) / qv)
    return float(out)


def _pairwise_kl_max(probs: Sequence[Mapping[str, float]]) -> float:
    if len(probs) < 2:
        return float("nan")
    best = 0.0
    for i in range(len(probs)):
        for j in range(i + 1, len(probs)):
            kij = _kl(probs[i], probs[j])
            kji = _kl(probs[j], probs[i])
            best = max(best, kij, kji)
    return float(best)


def _rhat_from_chain_stats(means: Sequence[float], vars_: Sequence[float], ns: Sequence[int]) -> float:
    counts = [n for n in ns if n > 1]
    if len(means) < 2 or len(counts) < 2:
        return float("nan")
    n = min(counts)
    m = len(means)
    mean_all = sum(means) / m
    B = (n / (m - 1)) * sum((mi - mean_all) ** 2 for mi in means)
    W = sum(vars_) / m
    if W <= 0:
        return float("nan")
    var_hat = ((n - 1) / n) * W + (1 / n) * B
    return float(math.sqrt(max(var_hat / W, 0.0)))


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
    ap.add_argument("--workers", type=int, default=1)
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
        workers=max(1, int(args.workers)),
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
    progress_enabled = bool(cfg.step_progress)
    eq_worker_pool.set_run_fixed_mcmc_fn(run_fixed_n_tree_mcmc)
    mp_ctx = mp.get_context("spawn") if (cfg.workers > 1 or progress_enabled) else None

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

        point_t0 = time.perf_counter()
        burnin = int(max(0, round(0.1 * float(steps_per_chain))))
        samples_per_chain = max(0, (steps_per_chain - burnin) // max(1, int(cfg.thin)))

        tasks: List[eq_worker_pool.EqTask] = []
        start_edges_by_spec: Dict[str, Tuple[Edge, ...]] = {}
        progress_every = int(cfg.step_heartbeat_every) if (cfg.progress or cfg.step_progress) else 0
        for start_spec_idx, start_spec in enumerate(cfg.start_specs):
            start_edges_by_spec[start_spec] = tuple(_start_edges_for_spec(cfg.n, start_spec))
            for chain_idx in range(int(cfg.chains)):
                seed = (
                    int(cfg.seed)
                    + 101 * int(chain_idx)
                    + 10_000 * int(steps_per_chain)
                    + 1_000_000 * int(start_spec_idx)
                )
                task_id = f"steps{int(steps_per_chain)}_{start_spec}_c{int(chain_idx)}"
                tasks.append(
                    eq_worker_pool.EqTask(
                        n=int(cfg.n),
                        steps_per_chain=int(steps_per_chain),
                        burnin_steps=int(burnin),
                        thin=int(cfg.thin),
                        start_spec=str(start_spec),
                        chain_idx=int(chain_idx),
                        seed=seed,
                        start_edges=start_edges_by_spec[start_spec],
                        task_id=task_id,
                        progress_every=int(progress_every),
                        backend="fdm",
                        lam=1.0,
                        temperature=1.0,
                        max_valence=4,
                        n_samples_expected=int(samples_per_chain),
                    )
                )

        if not tasks:
            continue

        results: List[eq_worker_pool.EqTaskResult] = []
        workers = max(1, int(cfg.workers))
        use_process_pool = workers > 1 or progress_enabled
        progress_queue_obj: Optional[queue.Queue] = None

        if not use_process_pool:
            eq_worker_pool.configure_for_sequential(energy_cache, None)
            for task in tasks:
                results.append(eq_worker_pool.run_task(task))
        else:
            assert mp_ctx is not None
            if progress_enabled:
                progress_queue_obj = mp_ctx.Queue()
            cache_arg = str(cache_path) if cache_path is not None else None
            with ProcessPoolExecutor(
                max_workers=workers,
                mp_context=mp_ctx,
                initializer=eq_worker_pool.init_worker,
                initargs=(cache_arg, progress_queue_obj),
            ) as ex:
                futures = [ex.submit(eq_worker_pool.run_task, task) for task in tasks]
                pending = set(futures)
                while pending:
                    done, pending = wait(pending, timeout=0.2, return_when=FIRST_COMPLETED)
                    _drain_progress_queue(progress_queue_obj)
                    for fut in done:
                        results.append(fut.result())
                _drain_progress_queue(progress_queue_obj)
            if progress_queue_obj is not None:
                try:
                    progress_queue_obj.close()
                except Exception:
                    pass
                try:
                    progress_queue_obj.join_thread()
                except Exception:
                    pass

        total_elapsed = time.perf_counter() - point_t0
        total_steps_done = sum(int(r.steps_total) for r in results) or int(steps_per_chain) * len(tasks)
        total_samples_recorded = sum(int(r.n_samples_recorded) for r in results)

        accepted_total = sum(int(r.accepted) for r in results)
        proposed_total = sum(int(r.proposed) for r in results)
        cache_hits_total = sum(int(r.cache_hits) for r in results)
        cache_misses_total = sum(int(r.cache_misses) for r in results)

        probs = []
        split_vals = []
        means = []
        vars_ = []
        ns = []
        ess_vals = []
        unique_eqs_seen: set[str] = set()

        for res in results:
            probs.append(_counter_to_prob(res.topo_counts))
            split_first = _counter_to_prob(res.topo_counts_first)
            split_second = _counter_to_prob(res.topo_counts_second)
            split_vals.append(max(_kl(split_first, split_second), _kl(split_second, split_first)))
            unique_eqs_seen.update(res.topo_counts.keys())
            if res.energy_samples > 1:
                means.append(res.energy_mean)
                vars_.append(max(res.energy_var, 0.0))
                ns.append(int(res.energy_samples))
            if not math.isnan(res.ess_energy_est):
                ess_vals.append(res.ess_energy_est)
            if cache_path and res.cache_delta:
                energy_cache.update(res.cache_delta)

        kl_max = _pairwise_kl_max(probs)
        kl_split_max = _safe_nanmax(split_vals)
        rhat_max = _rhat_from_chain_stats(means, vars_, ns) if len(means) >= 2 else float("nan")
        ess_min = _safe_nanmin(ess_vals)
        steps_per_sec_total = float(total_steps_done) / float(total_elapsed) if total_elapsed > 0 else 0.0
        cache_total = cache_hits_total + cache_misses_total
        cache_hit_rate = float(cache_hits_total) / float(cache_total) if cache_total > 0 else 0.0
        accept_rate = float(accepted_total) / float(proposed_total) if proposed_total > 0 else 0.0
        n_unique_eq = len(unique_eqs_seen)
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
                f"ACCEPT_RATE={accept_rate:.3f}" if proposed_total > 0 else "ACCEPT_RATE=NA",
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
