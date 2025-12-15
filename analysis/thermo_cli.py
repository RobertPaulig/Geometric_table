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
        "--coupling-shape-softness",
        type=float,
        default=None,
        help="Override coupling_shape_softness (shape-driven softness).",
    )
    parser.add_argument(
        "--coupling-shape-chi",
        type=float,
        default=None,
        help="Override coupling_shape_chi (shape-driven chi_spec).",
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
    parser.add_argument(
        "--ws-integrator",
        type=str,
        choices=["trapz", "fdm"],
        default=None,
        help="Integrator for WS/shape observables: trapz | fdm.",
    )
    parser.add_argument(
        "--ws-fdm-depth",
        type=int,
        default=None,
        help="Depth for FDM integrator (controls number of points).",
    )
    parser.add_argument(
        "--ws-fdm-base",
        type=int,
        default=None,
        help="Base for FDM tensor grid (default=2).",
    )
    parser.add_argument(
        "--shape-kurt-scale",
        type=float,
        default=None,
        help="Scale for |delta_kurtosis| in shape activity.",
    )
    parser.add_argument(
        "--shape-rrms-scale",
        type=float,
        default=None,
        help="Scale for r_rms_ws in shape activity.",
    )
    parser.add_argument(
        "--shape-softness-gain",
        type=float,
        default=None,
        help="Gain for shape-driven softness amplification.",
    )
    parser.add_argument(
        "--shape-chi-gain",
        type=float,
        default=None,
        help="Gain for shape-driven chi_spec amplification.",
    )
    parser.add_argument(
        "--coupling-topo-3d",
        type=float,
        default=None,
        help="Override coupling_topo_3d (3D entanglement backend gain).",
    )
    parser.add_argument(
        "--topo-3d-beta",
        type=float,
        default=None,
        help="Beta factor for 3D entanglement backend.",
    )
    parser.add_argument(
        "--topo3d-prefilter-tree",
        action="store_true",
        help="Enable 3D entanglement prefilter for trees (cyclomatic=0).",
    )
    parser.add_argument(
        "--grower-use-mh",
        action="store_true",
        help="Enable Metropolis–Hastings layer in grower.",
    )
    parser.add_argument(
        "--coupling-delta-G",
        type=float,
        default=None,
        help="Override coupling_delta_G for MH grower energy scale.",
    )
    parser.add_argument(
        "--temperature-T",
        type=float,
        default=None,
        help="Override temperature_T for MH grower.",
    )
    parser.add_argument(
        "--deltaG-backend",
        type=str,
        default=None,
        help="Backend for MH grower energy: fdm | fdm_entanglement.",
    )
    parser.add_argument(
        "--grower-proposal-policy",
        type=str,
        choices=["uniform", "ctt_biased"],
        default=None,
        help="Grower proposal policy: uniform | ctt_biased.",
    )
    parser.add_argument(
        "--proposal-beta",
        type=float,
        default=None,
        help="Bias strength for proposal softness (beta).",
    )
    parser.add_argument(
        "--proposal-ports-gamma",
        type=float,
        default=None,
        help="Ports-based weight exponent for proposals (gamma).",
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
    if getattr(args, "coupling_shape_softness", None) is not None:
        cfg = replace(
            cfg, coupling_shape_softness=float(args.coupling_shape_softness)
        )
    if getattr(args, "coupling_shape_chi", None) is not None:
        cfg = replace(
            cfg, coupling_shape_chi=float(args.coupling_shape_chi)
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
    if getattr(args, "shape_kurt_scale", None) is not None:
        cfg = replace(cfg, shape_kurt_scale=float(args.shape_kurt_scale))
    if getattr(args, "shape_rrms_scale", None) is not None:
        cfg = replace(cfg, shape_rrms_scale=float(args.shape_rrms_scale))
    if getattr(args, "shape_softness_gain", None) is not None:
        cfg = replace(cfg, shape_softness_gain=float(args.shape_softness_gain))
    if getattr(args, "shape_chi_gain", None) is not None:
        cfg = replace(cfg, shape_chi_gain=float(args.shape_chi_gain))
    if getattr(args, "coupling_topo_3d", None) is not None:
        cfg = replace(cfg, coupling_topo_3d=float(args.coupling_topo_3d))
    if getattr(args, "topo_3d_beta", None) is not None:
        cfg = replace(cfg, topo_3d_beta=float(args.topo_3d_beta))
    if getattr(args, "grower_use_mh", None):
        cfg = replace(cfg, grower_use_mh=True)
    if getattr(args, "coupling_delta_G", None) is not None:
        cfg = replace(cfg, coupling_delta_G=float(args.coupling_delta_G))
    if getattr(args, "temperature_T", None) is not None:
        cfg = replace(cfg, temperature_T=float(args.temperature_T))
    if getattr(args, "deltaG_backend", None):
        cfg = replace(cfg, deltaG_backend=str(args.deltaG_backend))
    if getattr(args, "ws_geom_gap_scale", None) is not None:
        cfg = replace(cfg, ws_geom_gap_scale=float(args.ws_geom_gap_scale))
    if getattr(args, "ws_z_ref", None) is not None:
        cfg = replace(cfg, ws_Z_ref=float(args.ws_z_ref))
    if getattr(args, "ws_z_alpha", None) is not None:
        cfg = replace(cfg, ws_Z_alpha=float(args.ws_z_alpha))
    if getattr(args, "topo3d_prefilter_tree", None):
        cfg = replace(cfg, topo3d_prefilter_tree=True)
    if getattr(args, "ws_integrator", None):
        cfg = replace(cfg, ws_integrator=str(args.ws_integrator))
    if getattr(args, "ws_fdm_depth", None) is not None:
        cfg = replace(cfg, ws_fdm_depth=int(args.ws_fdm_depth))
    if getattr(args, "ws_fdm_base", None) is not None:
        cfg = replace(cfg, ws_fdm_base=int(args.ws_fdm_base))
    if getattr(args, "grower_proposal_policy", None):
        cfg = replace(cfg, grower_proposal_policy=str(args.grower_proposal_policy))
    if getattr(args, "proposal_beta", None) is not None:
        cfg = replace(cfg, proposal_beta=float(args.proposal_beta))
    if getattr(args, "proposal_ports_gamma", None) is not None:
        cfg = replace(cfg, proposal_ports_gamma=float(args.proposal_ports_gamma))

    set_current_thermo_config(cfg)
