from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

from analysis.chem.alkane_expected_counts import expected_unique_alkane_tree_topologies
from analysis.chem.chem_validation_2_common import (
    EqRunConfig,
    p_pred_from_energy,
    run_equilibrium_with_guardrail,
    write_report_and_csv,
)


@dataclass
class Config:
    mode: str = "A"
    lam: float = 1.0
    temperature_T: float = 1.0
    steps: int = 200_000
    burnin: int = 20_000
    thin: int = 10
    chains: int = 5
    start_specs: Tuple[str, ...] = ("path", "max_branch")
    seed: int = 0
    progress: bool = True
    top_k: int = 20
    profile_every: int = 100


def _parse_args(argv: Optional[Sequence[str]] = None) -> Config:
    ap = argparse.ArgumentParser(
        description="CHEM-VALIDATION-4: C14 (tetradecane) equilibrium-first (tree-only)."
    )
    ap.add_argument("--mode", type=str, default="A", choices=["A", "B", "C"])
    ap.add_argument("--lambda", dest="lam", type=float, default=1.0)
    ap.add_argument("--T", dest="temperature_T", type=float, default=1.0)
    ap.add_argument("--steps", type=int, default=200_000)
    ap.add_argument("--burnin", type=int, default=20_000)
    ap.add_argument("--thin", type=int, default=10)
    ap.add_argument("--chains", type=int, default=5)
    ap.add_argument("--start_specs", type=str, nargs="*", default=["path", "max_branch"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--progress", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--top_k", type=int, default=20)
    ap.add_argument("--profile_every", type=int, default=100)
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
        profile_every=int(args.profile_every),
    )


def main(argv: Optional[Sequence[str]] = None) -> None:
    cfg0 = _parse_args(argv)
    expected = expected_unique_alkane_tree_topologies(14)

    cfg = EqRunConfig(
        n=14,
        expected_n_topologies=expected,
        mode=cfg0.mode,
        backend={"A": "fdm", "B": "fdm_entanglement", "C": "fdm_entanglement"}[cfg0.mode],
        lam=cfg0.lam,
        temperature_T=cfg0.temperature_T,
        steps=cfg0.steps,
        burnin=cfg0.burnin,
        thin=cfg0.thin,
        chains=cfg0.chains,
        start_specs=cfg0.start_specs,
        seed=cfg0.seed,
        progress=cfg0.progress,
        top_k=cfg0.top_k,
        max_attempts=4,  # base + up to 3 escalations
        guardrail_kl_max_target=0.005,
        profile_every=cfg0.profile_every,
    )

    p_eq, meta = run_equilibrium_with_guardrail(cfg)
    p_pred, energies, aut_sizes, g_vals = p_pred_from_energy(
        n=int(cfg.n),
        backend=str(cfg.backend),
        topo_keys=sorted(p_eq.keys()),
        lam=float(cfg.lam),
        temperature_T=float(cfg.temperature_T),
    )

    write_report_and_csv(
        out_stub="chem_validation_4_tetradecane",
        cfg=cfg,
        p_eq=p_eq,
        meta=meta,
        p_pred=p_pred,
        energies=energies,
        aut_sizes=aut_sizes,
        g_vals=g_vals,
    )


if __name__ == "__main__":
    main()
