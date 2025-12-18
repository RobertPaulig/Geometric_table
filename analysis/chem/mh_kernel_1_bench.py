from __future__ import annotations

import argparse
import csv
import re
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from analysis.io_utils import results_path
from analysis.chem.core2_fit import fit_lambda, compute_p_pred, kl_divergence
from analysis.chem.topology_mcmc import run_fixed_n_tree_mcmc
from analysis.growth.reporting import write_growth_txt
from core.complexity import compute_complexity_features_v2
from core.thermo_config import ThermoConfig, override_thermo_config


def _make_thermo_for_mode(mode: str) -> ThermoConfig:
    from analysis.chem.chem_validation_1a_pentane import _make_thermo_for_mode as _t

    return _t(mode)


def _degeneracy_for_n(n: int) -> Dict[str, int]:
    if n == 4:
        return {"n_butane": 12, "isobutane": 4}
    if n == 5:
        return {"n_pentane": 60, "isopentane": 60, "neopentane": 5}
    raise ValueError("Supported N: 4 or 5")


def _ref_energies_for_n_mode(n: int, mode: str) -> Dict[str, float]:
    import numpy as np

    from analysis.chem.chem_validation_0_butane import classify_butane_topology
    from analysis.chem.chem_validation_1a_pentane import classify_pentane_topology

    mode = mode.upper()
    thermo = _make_thermo_for_mode(mode)

    def score(adj: np.ndarray) -> float:
        with override_thermo_config(thermo):
            feats = compute_complexity_features_v2(adj, backend="fdm_entanglement")
        return float(feats.total)

    if n == 4:
        # path
        adj_path = np.zeros((4, 4), dtype=float)
        for a, b in ((0, 1), (1, 2), (2, 3)):
            adj_path[a, b] = adj_path[b, a] = 1.0
        # star
        adj_star = np.zeros((4, 4), dtype=float)
        for j in (1, 2, 3):
            adj_star[0, j] = adj_star[j, 0] = 1.0
        return {"n_butane": score(adj_path), "isobutane": score(adj_star)}

    if n == 5:
        adj_n = np.zeros((5, 5), dtype=float)
        for a, b in ((0, 1), (1, 2), (2, 3), (3, 4)):
            adj_n[a, b] = adj_n[b, a] = 1.0
        adj_iso = np.zeros((5, 5), dtype=float)
        for a, b in ((1, 0), (1, 2), (1, 4), (2, 3)):
            adj_iso[a, b] = adj_iso[b, a] = 1.0
        adj_neo = np.zeros((5, 5), dtype=float)
        for j in (1, 2, 3, 4):
            adj_neo[0, j] = adj_neo[j, 0] = 1.0
        return {
            "n_pentane": score(adj_n),
            "isopentane": score(adj_iso),
            "neopentane": score(adj_neo),
        }

    raise ValueError("Supported N: 4 or 5")


def _load_lambda_star_from_chem_validation(n: int, mode: str) -> Optional[float]:
    """
    Read lambda* from the latest chem_validation TXT artifacts.

    Expected format in TXT:
      CORE-2: Fit lambda (degeneracy-aware)
        [Mode A] ...
          lambda*=0.9225, ...
    """
    mode = mode.upper()
    if n == 4:
        path = results_path("chem_validation_0_butane.txt")
    elif n == 5:
        path = results_path("chem_validation_1a_pentane.txt")
    else:
        return None

    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return None

    # Find the CORE-2 section and then the matching [Mode X] block.
    m_core = re.search(r"CORE-2: Fit lambda \(degeneracy-aware\)(.*)\Z", text, flags=re.S)
    if not m_core:
        return None
    tail = m_core.group(1)
    m_mode = re.search(rf"\[Mode {re.escape(mode)}\].*?lambda\*=([0-9.]+)", tail, flags=re.S)
    if not m_mode:
        return None
    try:
        return float(m_mode.group(1))
    except Exception:
        return None


@dataclass
class BenchConfig:
    n: int = 5
    mode: str = "A"
    steps: int = 200_000
    burnin: int = 20_000
    thin: int = 10
    seed: int = 0
    lam: Optional[float] = None
    progress: bool = True


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="MH-KERNEL-1: fixed-N tree MCMC benchmark.")
    parser.add_argument("--N", type=int, choices=[4, 5], default=5)
    parser.add_argument("--mode", type=str, default="A", choices=["A", "B", "C"])
    parser.add_argument("--steps", type=int, default=200_000)
    parser.add_argument("--burnin", type=int, default=20_000)
    parser.add_argument("--thin", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--lambda", dest="lam", type=float, default=None)
    parser.add_argument(
        "--progress",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Show progress bar / periodic progress output.",
    )
    args = parser.parse_args(argv)
    cfg = BenchConfig(
        n=int(args.N),
        mode=str(args.mode).upper(),
        steps=int(args.steps),
        burnin=int(args.burnin),
        thin=int(args.thin),
        seed=int(args.seed),
        lam=(float(args.lam) if args.lam is not None else None),
        progress=bool(args.progress),
    )

    thermo = _make_thermo_for_mode(cfg.mode)
    T = float(getattr(thermo, "temperature_T", 1.0))
    backend = "fdm_entanglement"

    g = _degeneracy_for_n(cfg.n)
    e_ref = _ref_energies_for_n_mode(cfg.n, cfg.mode)

    if cfg.lam is None:
        lam_star = _load_lambda_star_from_chem_validation(cfg.n, cfg.mode)
        if lam_star is None:
            # Fallback: neutral scale if CORE-2 artifact not found.
            p_obs_uniform = {k: 1.0 / float(len(g)) for k in g.keys()}
            fit = fit_lambda(p_obs_uniform, g, e_ref, T=T)
            lam = float(fit.lam_star)
        else:
            lam = float(lam_star)
    else:
        lam = float(cfg.lam)

    out_csv = results_path(f"mh_kernel_1_mcmc_N{cfg.n}_mode{cfg.mode}.csv")
    out_txt = results_path(f"mh_kernel_1_mcmc_N{cfg.n}_mode{cfg.mode}.txt")

    t0 = time.perf_counter()

    pbar = None
    if cfg.progress:
        try:
            from tqdm import tqdm  # type: ignore

            pbar = tqdm(total=cfg.steps, desc=f"MCMC N={cfg.n} mode={cfg.mode}", leave=True)
        except Exception:
            pbar = None

    last_print = {"t": time.perf_counter(), "n": 0}

    def step_progress(n: int = 1) -> None:
        if not cfg.progress:
            return
        if pbar is not None:
            try:
                pbar.update(int(n))  # type: ignore[attr-defined]
                return
            except Exception:
                pass
        # Fallback: print periodically.
        last_print["n"] += int(n)
        if last_print["n"] == 1 or last_print["n"] % 1000 == 0 or last_print["n"] >= cfg.steps:
            dt = time.perf_counter() - last_print["t"]
            last_print["t"] = time.perf_counter()
            print(f"[MCMC] {last_print['n']}/{cfg.steps} (+{dt:.2f}s)")

    # Use the MCMC runner's internal loop; we just pass a progress callback and let
    # our callback update tqdm/fallback prints.
    samples, summary = run_fixed_n_tree_mcmc(
        n=cfg.n,
        steps=cfg.steps,
        burnin=cfg.burnin,
        thin=cfg.thin,
        backend=backend,
        lam=lam,
        temperature_T=T,
        seed=cfg.seed,
        progress=step_progress,
    )
    elapsed = time.perf_counter() - t0
    if pbar is not None and hasattr(pbar, "close"):
        try:
            pbar.close()  # type: ignore[attr-defined]
        except Exception:
            pass

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=sorted(samples[0].keys()) if samples else ["step"])
        w.writeheader()
        for r in samples:
            w.writerow(r)

    # Compute P_pred(lam)
    p_pred = compute_p_pred(g, e_ref, T=T, lam=lam)

    # Normalize observed from MCMC samples
    p_mcmc = summary.p_topology

    lines: List[str] = []
    lines.append("MH-KERNEL-1: fixed-N tree MCMC benchmark")
    lines.append(f"N={cfg.n}, mode={cfg.mode}, backend={backend}")
    lines.append(f"steps={cfg.steps}, burnin={cfg.burnin}, thin={cfg.thin}, seed={cfg.seed}")
    lines.append(f"temperature_T={T:.3f}, lambda={lam:.4f}")
    lines.append(f"elapsed_sec={elapsed:.3f}")
    lines.append(f"accepted={summary.accepted}, proposals={summary.proposals}, accept_rate={(summary.accepted/summary.proposals if summary.proposals else 0.0):.4f}")
    lines.append(f"mean_moves={summary.mean_moves:.2f}, p90_moves={summary.p90_moves:.2f}")
    lines.append(f"mean_log(q_rev/q_fwd)={summary.mean_log_qratio:.4f}, p90_log(q_rev/q_fwd)={summary.p90_log_qratio:.4f}")
    lines.append("")
    lines.append(f"P_mcmc={p_mcmc}")
    lines.append(f"P_pred={p_pred}")
    lines.append(f"KL(P_mcmc||P_pred)={kl_divergence(p_mcmc, p_pred):.6f}")
    lines.append("")

    out_txt = write_growth_txt(f"mh_kernel_1_mcmc_N{cfg.n}_mode{cfg.mode}", lines)
    print("[MH-KERNEL-1] done.")
    print(f"CSV: {out_csv}")
    print(f"Summary: {out_txt}")


if __name__ == "__main__":
    main()
