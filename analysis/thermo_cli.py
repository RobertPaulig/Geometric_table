from __future__ import annotations

from dataclasses import replace
from typing import Optional

from core.thermo_config import (
    ThermoConfig,
    load_thermo_config,
    set_current_thermo_config,
)


def apply_thermo_config_if_provided(config_path: Optional[str]) -> None:
    """
    Если config_path задан:
    - читает либо секцию thermo: из общего YAML,
    - либо "чистый" thermo YAML/JSON,
    и ставит как текущее глобальное состояние.
    """
    if not config_path:
        return
    cfg = load_thermo_config(config_path)
    set_current_thermo_config(cfg)


def add_thermo_args(parser) -> None:
    parser.add_argument(
        "--thermo-config",
        type=str,
        default=None,
        help="Thermo config (YAML/JSON or embedded).",
    )
    parser.add_argument(
        "--thermo-temp",
        type=float,
        default=None,
        help="Override thermo temperature.",
    )
    parser.add_argument(
        "--coupling-delta_F",
        type=float,
        default=None,
        help="Override coupling_delta_F.",
    )
    parser.add_argument(
        "--coupling-complexity",
        type=float,
        default=None,
        help="Override coupling_complexity.",
    )
    parser.add_argument(
        "--coupling-softness",
        type=float,
        default=None,
        help="Override coupling_softness.",
    )
    parser.add_argument(
        "--coupling-density",
        type=float,
        default=None,
        help="Override coupling_density.",
    )


def apply_thermo_from_args(args, fallback_config_path: Optional[str] = None) -> None:
    # Resolution order: defaults -> YAML -> CLI overrides
    if getattr(args, "thermo_config", None):
        cfg = load_thermo_config(args.thermo_config)
    elif fallback_config_path:
        cfg = load_thermo_config(fallback_config_path)
    else:
        cfg = ThermoConfig()

    if getattr(args, "thermo_temp", None) is not None:
        cfg = replace(cfg, temperature=float(args.thermo_temp))
    if getattr(args, "coupling_delta_F", None) is not None:
        cfg = replace(cfg, coupling_delta_F=float(args.coupling_delta_F))
    if getattr(args, "coupling_complexity", None) is not None:
        cfg = replace(cfg, coupling_complexity=float(args.coupling_complexity))
    if getattr(args, "coupling_softness", None) is not None:
        cfg = replace(cfg, coupling_softness=float(args.coupling_softness))
    if getattr(args, "coupling_density", None) is not None:
        cfg = replace(cfg, coupling_density=float(args.coupling_density))

    set_current_thermo_config(cfg)
