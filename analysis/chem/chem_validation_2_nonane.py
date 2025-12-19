from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

from analysis.chem.chem_validation_2_common import EqRunConfig, p_pred_from_energy, run_equilibrium_with_guardrail, write_report_and_csv


@dataclass
class Config:
    mode: str = "A"
    lam: float = 1.0
    temperature_T: float = 1.0
    steps: int = 20_000
    burnin: int = 2_000
    thin: int = 10
    chains: int = 5
    start_specs: Tuple[str, ...] = ("path", "max_branch")
    seed: int = 0
    progress: bool = True
    top_k: int = 20


def _parse_args(argv: Optional[Sequence[str]] = None) -> Config:
    ap = argparse.ArgumentParser(description="CHEM-VALIDATION-2: C9 (nonane) equilibrium-first + self-consistency.")
    ap.add_argument("--mode", type=str, default="A", choices=["A", "B", "C"])
    ap.add_argument("--lambda", dest="lam", type=float, default=1.0)
    ap.add_argument("--T", dest="temperature_T", type=float, default=1.0)
    ap.add_argument("--steps", type=int, default=20_000)
    ap.add_argument("--burnin", type=int, default=2_000)
    ap.add_argument("--thin", type=int, default=10)
    ap.add_argument("--chains", type=int, default=5)
    ap.add_argument("--start_specs", type=str, nargs="*", default=["path", "max_branch"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--progress", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--top_k", type=int, default=20)
    args = ap.parse_args(argv)
    return Config(
        mode=str(args.mode).upper(),
        lam=float(args.lam),
        temperature_T=float(args.temperature_T),
        steps=int(args.steps),
        burnin=int(args.burnin),
        thin=int(args.thin),
        chains=int(args.chains),
        start_specs=tuple(str(x) for x in args.start_specs),
        seed=int(args.seed),
        progress=bool(args.progress),
        top_k=int(args.top_k),
    )


def main(argv: Optional[Sequence[str]] = None) -> None:
    cfg = _parse_args(argv)
    mode = str(cfg.mode).upper()
    backend = "fdm" if mode == "A" else "fdm_entanglement"

    eq_cfg = EqRunConfig(
        n=9,
        expected_n_topologies=35,
        mode=mode,
        backend=backend,
        lam=float(cfg.lam),
        temperature_T=float(cfg.temperature_T),
        steps=int(cfg.steps),
        burnin=int(cfg.burnin),
        thin=int(cfg.thin),
        chains=int(cfg.chains),
        start_specs=tuple(cfg.start_specs),
        seed=int(cfg.seed),
        progress=bool(cfg.progress),
        top_k=int(cfg.top_k),
        guardrail_kl_max_target=0.002,
    )

    p_eq, meta = run_equilibrium_with_guardrail(eq_cfg)
    topo_keys = sorted(p_eq.keys())
    p_pred, energies, aut_sizes, g_vals = p_pred_from_energy(
        topo_keys, n=9, backend=backend, lam=float(cfg.lam), temperature_T=float(cfg.temperature_T)
    )
    write_report_and_csv(
        out_stub="chem_validation_2_nonane",
        cfg=eq_cfg,
        p_eq=p_eq,
        meta=meta,
        p_pred=p_pred,
        energies=energies,
        aut_sizes=aut_sizes,
        g_vals=g_vals,
    )


if __name__ == "__main__":
    main()
