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
    parser.add_argument(
        "--coupling-density-shape",
        type=float,
        default=None,
        help="Override coupling_density_shape (WS density mixing).",
    )
    parser.add_argument(
        "--coupling-port-geometry",
        type=float,
        default=None,
        help="Override coupling_port_geometry (spectral port geometry).",
    )
    parser.add_argument(
        "--coupling-ws-z",
        type=float,
        default=None,
        help="Override coupling_ws_Z (Z-coupling for WS radial problem).",
    )
    parser.add_argument(
        "--density-model",
        type=str,
        default=None,
        help="Density model for beta(Z): tf_radius | tf_energy | hydrogenic.",
    )
    parser.add_argument(
        "--density-blend",
        type=str,
        default=None,
        help="Blend mode for beta(Z): linear | log.",
    )
    parser.add_argument(
        "--density-z-ref",
        type=float,
        default=None,
        help="Reference Z for matching beta_legacy in physical model.",
    )
    parser.add_argument(
        "--density-source",
        type=str,
        choices=["gaussian", "ws_radial"],
        default=None,
        help="Source for 3D density in FDM: gaussian | ws_radial.",
    )
    parser.add_argument(
        "--ws-r-max",
        type=float,
        default=None,
        help="WS radial grid R_max (fm-like units, R&D).",
    )
    parser.add_argument(
        "--ws-r-well",
        type=float,
        default=None,
        help="WS potential radius R_well (R0).",
    )
    parser.add_argument(
        "--ws-v0",
        type=float,
        default=None,
        help="WS potential depth V0 (>0).",
    )
    parser.add_argument(
        "--ws-n-grid",
        type=int,
        default=None,
        help="WS radial grid size N_grid.",
    )
    parser.add_argument(
        "--ws-ell",
        type=int,
        default=None,
        help="WS orbital angular momentum ell.",
    )
    parser.add_argument(
        "--ws-state-index",
        type=int,
        default=None,
        help="WS bound-state index within given ell (0-based).",
    )
    parser.add_argument(
        "--port-geometry-source",
        type=str,
        choices=["legacy", "ws_sp_gap"],
        default=None,
        help="Source for port geometry: legacy | ws_sp_gap.",
    )
    parser.add_argument(
        "--port-geometry-blend",
        type=str,
        choices=["linear", "log"],
        default=None,
        help="Blend mode for spectral port geometry: linear | log.",
    )
    parser.add_argument(
        "--ws-geom-r-max",
        type=float,
        default=None,
        help="WS geom R_max for spectral s-p gap.",
    )
    parser.add_argument(
        "--ws-geom-r-well",
        type=float,
        default=None,
        help="WS geom R_well (R0) for spectral s-p gap.",
    )
    parser.add_argument(
        "--ws-geom-v0",
        type=float,
        default=None,
        help="WS geom potential depth V0 (>0) for spectral s-p gap.",
    )
    parser.add_argument(
        "--ws-geom-n-grid",
        type=int,
        default=None,
        help="WS geom radial grid size N_grid for spectral s-p gap.",
    )
    parser.add_argument(
        "--ws-geom-gap-scale",
        type=float,
        default=None,
        help="Scale parameter for mapping WS s-p gap to hybrid strength.",
    )
    parser.add_argument(
        "--ws-z-ref",
        type=float,
        default=None,
        help="Reference Z for WS radial scaling (ws_Z_ref).",
    )
    parser.add_argument(
        "--ws-z-alpha",
        type=float,
        default=None,
        help="Exponent alpha for WS radial scaling (ws_Z_alpha).",
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
    if getattr(args, "coupling_density_shape", None) is not None:
        cfg = replace(cfg, coupling_density_shape=float(args.coupling_density_shape))
    if getattr(args, "coupling_port_geometry", None) is not None:
        cfg = replace(
            cfg, coupling_port_geometry=float(args.coupling_port_geometry)
        )
    if getattr(args, "coupling_ws_z", None) is not None:
        cfg = replace(
            cfg, coupling_ws_Z=float(args.coupling_ws_z)
        )
    if getattr(args, "density_model", None):
        cfg = replace(cfg, density_model=str(args.density_model))
    if getattr(args, "density_blend", None):
        cfg = replace(cfg, density_blend=str(args.density_blend))
    if getattr(args, "density_z_ref", None) is not None:
        cfg = replace(cfg, density_Z_ref=float(args.density_z_ref))
    if getattr(args, "density_source", None):
        cfg = replace(cfg, density_source=str(args.density_source))
    if getattr(args, "ws_r_max", None) is not None:
        cfg = replace(cfg, ws_R_max=float(args.ws_r_max))
    if getattr(args, "ws_r_well", None) is not None:
        cfg = replace(cfg, ws_R_well=float(args.ws_r_well))
    if getattr(args, "ws_v0", None) is not None:
        cfg = replace(cfg, ws_V0=float(args.ws_v0))
    if getattr(args, "ws_n_grid", None) is not None:
        cfg = replace(cfg, ws_N_grid=int(args.ws_n_grid))
    if getattr(args, "ws_ell", None) is not None:
        cfg = replace(cfg, ws_ell=int(args.ws_ell))
    if getattr(args, "ws_state_index", None) is not None:
        cfg = replace(cfg, ws_state_index=int(args.ws_state_index))
    if getattr(args, "port_geometry_source", None):
        cfg = replace(cfg, port_geometry_source=str(args.port_geometry_source))
    if getattr(args, "port_geometry_blend", None):
        cfg = replace(cfg, port_geometry_blend=str(args.port_geometry_blend))
    if getattr(args, "ws_geom_r_max", None) is not None:
        cfg = replace(cfg, ws_geom_R_max=float(args.ws_geom_r_max))
    if getattr(args, "ws_geom_r_well", None) is not None:
        cfg = replace(cfg, ws_geom_R_well=float(args.ws_geom_r_well))
    if getattr(args, "ws_geom_v0", None) is not None:
        cfg = replace(cfg, ws_geom_V0=float(args.ws_geom_v0))
    if getattr(args, "ws_geom_n_grid", None) is not None:
        cfg = replace(cfg, ws_geom_N_grid=int(args.ws_geom_n_grid))
    if getattr(args, "ws_geom_gap_scale", None) is not None:
        cfg = replace(cfg, ws_geom_gap_scale=float(args.ws_geom_gap_scale))
    if getattr(args, "ws_z_ref", None) is not None:
        cfg = replace(cfg, ws_Z_ref=float(args.ws_z_ref))
    if getattr(args, "ws_z_alpha", None) is not None:
        cfg = replace(cfg, ws_Z_alpha=float(args.ws_z_alpha))

    set_current_thermo_config(cfg)
