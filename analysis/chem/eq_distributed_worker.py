from __future__ import annotations

import argparse
import hashlib
import json
import socket
import pickle
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Tuple, Any

import numpy as np

from analysis.chem.topology_mcmc import (
    run_fixed_n_tree_mcmc,
    tree_topology_edge_key_from_edges,
    Edge,
)


def now_utc_iso() -> str:
    return (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def _start_edges_for_spec(n: int, spec: str) -> Tuple[Edge, ...]:
    spec = str(spec)
    if spec == "path":
        return tuple((int(i), int(i + 1)) for i in range(int(n) - 1))
    if spec == "max_branch":
        if int(n) <= 5:
            return tuple((0, i) for i in range(1, int(n)))
        edges = [(0, 1), (0, 2), (0, 3), (0, 4)]
        last = 4
        for v in range(5, int(n)):
            edges.append((last, v))
            last = v
        return tuple(edges)
    raise ValueError(f"Unknown start_spec: {spec!r}")


def sha_counter(counter: Counter) -> str:
    h = hashlib.sha256()
    for key, val in sorted(counter.items(), key=lambda kv: kv[0]):
        h.update(str(key).encode("utf-8"))
        h.update(b"\0")
        h.update(str(int(val)).encode("utf-8"))
        h.update(b"\n")
    return "sha256:" + h.hexdigest()


def estimate_ess_fft(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    n = x.size
    if n < 10:
        return float(max(1, n))
    x = x - x.mean()
    m = 1 << (2 * n - 1).bit_length()
    f = np.fft.rfft(x, m)
    acf = np.fft.irfft(f * np.conj(f), m)[:n]
    if acf.size == 0 or acf[0] == 0:
        return float(max(1, n))
    acf /= acf[0]
    tau = 1.0
    for k in range(1, n):
        if acf[k] <= 0:
            break
        tau += 2.0 * acf[k]
    ess = n / tau if tau > 0 else float(n)
    return float(max(1.0, min(float(n), ess)))


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_energy_cache(path: Path | None) -> Dict[Any, Any]:
    if path is None or not path.exists():
        return {}
    with path.open("rb") as f:
        try:
            return pickle.load(f)
        except Exception:
            return {}


def _save_energy_cache(path: Path, cache: Dict[Any, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("wb") as f:
        pickle.dump(cache, f)
    tmp.replace(path)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Run one distributed EQ-TARGET-3 task.")
    ap.add_argument("--task", type=str, required=True, help="Path to task_XXXXXX.json")
    ap.add_argument("--out_dir", type=str, required=True, help="Where to write submission json")
    ap.add_argument("--energy_cache_path", type=str, default="", help="Optional pickle cache path per host")
    ap.add_argument("--git_sha", type=str, default="", help="Git SHA of current code (for provenance)")
    args = ap.parse_args(argv)

    task_path = Path(args.task)
    task = _load_json(task_path)

    host = socket.gethostname() or "UNKNOWN"
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cache_path = Path(args.energy_cache_path) if args.energy_cache_path else None
    energy_cache: Dict[Any, Any] = _load_energy_cache(cache_path)

    steps = int(task["steps_per_chain"])
    thin = int(task["thin"])
    burnin_frac = float(task.get("burnin_frac", 0.1))
    burnin = int(burnin_frac * steps)

    # streaming counters
    topo_all: Counter[str] = Counter()
    topo_first: Counter[str] = Counter()
    topo_second: Counter[str] = Counter()
    energies: list[float] = []

    n_target = max(1, (steps - burnin) // thin)
    half = n_target // 2
    sample_idx = 0

    def on_sample(topo: str, energy: float) -> None:
        nonlocal sample_idx
        topo_all[topo] += 1
        if sample_idx < half:
            topo_first[topo] += 1
        else:
            topo_second[topo] += 1
        energies.append(float(energy))
        sample_idx += 1

    start_edges = list(_start_edges_for_spec(task["N"], task["start_spec"]))

    _, summary = run_fixed_n_tree_mcmc(
        n=int(task["N"]),
        steps=steps,
        burnin=burnin,
        thin=thin,
        backend="fdm",
        lam=1.0,
        temperature_T=1.0,
        seed=int(task["seed"]),
        max_valence=4,
        topology_key_fn_edges=tree_topology_edge_key_from_edges,
        start_edges=start_edges,
        energy_cache=energy_cache,
        progress=None,
        profile_every=0,
        step_heartbeat_every=0,
        step_heartbeat=None,
        sample_callback=on_sample,
        collect_samples=False,
    )

    accepted = getattr(summary, "accepted", None)
    if accepted is None:
        accepted = getattr(summary, "accepted_total", 0)
    proposals = getattr(summary, "proposed", None)
    if proposals is None:
        proposals = getattr(summary, "proposals", None)
    if proposals is None:
        proposals = getattr(summary, "proposals_total", 0)
    cache_hits = getattr(summary, "energy_cache_hits", getattr(summary, "cache_hits", 0))
    cache_misses = getattr(summary, "energy_cache_misses_seen", getattr(summary, "cache_misses", 0))
    elapsed_sec = getattr(summary, "elapsed_sec", getattr(summary, "elapsed", 0.0))

    accept_rate = float(accepted) / float(proposals) if proposals else 0.0
    hit_rate = float(cache_hits) / float(cache_hits + cache_misses) if (cache_hits + cache_misses) else 0.0

    energies_np = np.asarray(energies, dtype=np.float64)
    e_mean = float(energies_np.mean()) if energies_np.size else 0.0
    e_var = float(energies_np.var(ddof=1)) if energies_np.size > 1 else 0.0
    ess = estimate_ess_fft(energies_np) if energies_np.size else 1.0

    result = {
        "task_id": task["task_id"],
        "git_sha": args.git_sha or task.get("git_sha_expected", ""),
        "host": host,
        "date_utc": now_utc_iso(),
        "N": int(task["N"]),
        "mode": str(task["mode"]),
        "steps_per_chain": steps,
        "thin": thin,
        "burnin_frac": burnin_frac,
        "start_spec": str(task["start_spec"]),
        "chain_idx": int(task["chain_idx"]),
        "seed": int(task["seed"]),
        "elapsed_sec": float(elapsed_sec),
        "steps_total": steps,
        "accept_rate": float(accept_rate),
        "hit_rate": float(hit_rate),
        "misses_seen": int(cache_misses),
        "n_samples": int(sample_idx),
        "energy_mean": e_mean,
        "energy_var": e_var,
        "ess_energy": float(ess),
        "topology_counter": dict(topo_all),
        "topology_counter_first_half": dict(topo_first),
        "topology_counter_second_half": dict(topo_second),
        "counter_hash": sha_counter(topo_all),
    }

    out_name = f'{task["task_id"]}__{host}__{result["git_sha"] or "UNKNOWN"}.json'
    out_path = out_dir / out_name
    out_path.write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")
    print(f"WROTE {out_path}")

    if cache_path is not None:
        _save_energy_cache(cache_path, energy_cache)


if __name__ == "__main__":
    main()
