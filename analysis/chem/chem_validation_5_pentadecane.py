from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

from analysis.chem.chem_validation_5_common import EqCfg, GrowthCfg, run_equilibrium_distribution_mode_a, run_growth_distribution, write_report


@dataclass
class Config:
    n_runs: int = 5000
    growth_seeds: Tuple[int, ...] = (0, 1, 2, 3, 4)
    chains: int = 5
    steps_init: int = 40_000_000
    max_steps: int = 160_000_000
    thin: int = 10
    eq_seed: int = 0
    start_specs: Tuple[str, ...] = ("path", "max_branch")
    guardrail_target: float = 0.005
    profile_every: int = 100
    progress: bool = True
    auto_escalate: bool = True
    burnin_frac: float = 0.30
    lambda_scale: float = 1.0


def _parse_args(argv: Optional[Sequence[str]] = None) -> Config:
    ap = argparse.ArgumentParser(description="CHEM-VALIDATION-5: C15 (pentadecane) growth vs equilibrium, Mode A only.")
    ap.add_argument("--mode", type=str, default="A", choices=["A"])
    ap.add_argument("--n_runs", type=int, default=5000)
    ap.add_argument("--growth_seeds", type=int, nargs="*", default=[0, 1, 2, 3, 4])
    ap.add_argument("--chains", type=int, default=5)
    ap.add_argument("--steps_init", type=int, default=40_000_000)
    ap.add_argument("--max_steps", type=int, default=160_000_000)
    ap.add_argument("--thin", type=int, default=10)
    ap.add_argument("--seed", dest="eq_seed", type=int, default=0)
    ap.add_argument("--start_specs", type=str, nargs="*", default=["path", "max_branch"])
    ap.add_argument("--guardrail_target", type=float, default=0.005)
    ap.add_argument("--profile_every", type=int, default=100)
    ap.add_argument("--progress", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--auto_escalate", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--burnin_frac", type=float, default=0.30)
    ap.add_argument("--lambda_scale", type=float, default=1.0)
    args = ap.parse_args(argv)
    return Config(
        n_runs=int(args.n_runs),
        growth_seeds=tuple(int(x) for x in args.growth_seeds),
        chains=int(args.chains),
        steps_init=int(args.steps_init),
        max_steps=int(args.max_steps),
        thin=int(args.thin),
        eq_seed=int(args.eq_seed),
        start_specs=tuple(str(x) for x in args.start_specs),
        guardrail_target=float(args.guardrail_target),
        profile_every=int(args.profile_every),
        progress=bool(args.progress),
        auto_escalate=bool(args.auto_escalate),
        burnin_frac=float(args.burnin_frac),
        lambda_scale=float(args.lambda_scale),
    )


def main(argv: Optional[Sequence[str]] = None) -> None:
    cfg = _parse_args(argv)
    p_growth, growth_counts, t_growth = run_growth_distribution(
        n=15,
        cfg=GrowthCfg(n_runs=int(cfg.n_runs), seeds=cfg.growth_seeds, progress=bool(cfg.progress)),
    )
    p_eq, meta, _, _ = run_equilibrium_distribution_mode_a(
        n=15,
        cfg=EqCfg(
            n=15,
            steps_init=int(cfg.steps_init),
            max_steps=int(cfg.max_steps),
            chains=int(cfg.chains),
            thin=int(cfg.thin),
            seed=int(cfg.eq_seed),
            start_specs=cfg.start_specs,
            guardrail_target=float(cfg.guardrail_target),
            profile_every=int(cfg.profile_every),
            progress=bool(cfg.progress),
            burnin_frac=float(cfg.burnin_frac),
        ),
    )
    meta["lambda_scale"] = float(cfg.lambda_scale)
    meta["temperature_T"] = 1.0
    meta["growth_total_sec"] = float(t_growth)
    write_report(
        out_stub="chem_validation_5_pentadecane",
        n=15,
        p_growth=p_growth,
        growth_counts=growth_counts,
        p_eq=p_eq,
        meta=meta,
    )
    if bool(meta.get("FAIL", False)) and bool(cfg.auto_escalate):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
