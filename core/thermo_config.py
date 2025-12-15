from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Dict, Mapping
from contextlib import contextmanager
import copy
import json

try:
    import yaml  # type: ignore[import]
except Exception:  # pragma: no cover
    yaml = None


@dataclass(frozen=True)
class ThermoConfig:
    """
    Глобальная термодинамическая конфигурация Среды (QSG v7.x).
    Определяет температуру и степень включения физических связей (couplings).
    """

    temperature: float = 1.0

    coupling_delta_F: float = 0.0
    coupling_complexity: float = 0.0
    coupling_softness: float = 0.0
    coupling_density: float = 0.0
    coupling_density_shape: float = 0.0
    coupling_port_geometry: float = 0.0
    coupling_ws_Z: float = 0.0
    coupling_shape_softness: float = 0.0
    coupling_shape_chi: float = 0.0
    coupling_topo_3d: float = 0.0
    coupling_delta_G: float = 1.0

    density_model: str = "tf_radius"
    density_blend: str = "linear"
    density_Z_ref: float = 10.0
    density_source: str = "gaussian"  # gaussian | ws_radial

    # WS radial density params (R&D defaults)
    ws_R_max: float = 12.0
    ws_R_well: float = 5.0
    ws_V0: float = 40.0
    ws_N_grid: int = 220
    ws_ell: int = 0
    ws_state_index: int = 0

    # Port geometry controls (R&D)
    port_geometry_source: str = "legacy"  # legacy | ws_sp_gap
    port_geometry_blend: str = "linear"  # linear | log
    ws_geom_R_max: float = 25.0
    ws_geom_R_well: float = 6.0
    ws_geom_V0: float = 45.0
    ws_geom_N_grid: int = 800
    ws_geom_gap_scale: float = 1.0
    ws_geom_gap_ref: float = 1.0
    ws_Z_ref: float = 10.0
    ws_Z_alpha: float = 1.0 / 3.0

    # Shape-driven chemistry (R&D, disabled by default)
    shape_kurt_scale: float = 0.30
    shape_rrms_scale: float = 1.50
    shape_softness_gain: float = 0.35
    shape_chi_gain: float = 0.30
    topo_3d_beta: float = 1.0

    # WS / shape integrator controls
    ws_integrator: str = "trapz"  # trapz | fdm
    ws_fdm_depth: int = 9
    ws_fdm_base: int = 2

    # MH-grower controls (R&D)
    temperature_T: float = 1.0
    grower_use_mh: bool = False
    deltaG_backend: str = "fdm_entanglement"
    consume_port_on_reject: bool = True
    max_attempts_per_port: int = 1
    grower_proposal_policy: str = "uniform"  # uniform | ctt_biased
    proposal_beta: float = 0.0
    proposal_ports_gamma: float = 0.0

    experiment_name: str = "default_thermo"


_CURRENT_THERMO_CONFIG = ThermoConfig()


def get_current_thermo_config() -> ThermoConfig:
    return _CURRENT_THERMO_CONFIG


def set_current_thermo_config(cfg: ThermoConfig) -> None:
    global _CURRENT_THERMO_CONFIG
    _CURRENT_THERMO_CONFIG = cfg


def _load_dict(path: Path) -> Dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() in {".yaml", ".yml"}:
        if yaml is None:
            raise RuntimeError(
                f"PyYAML is not installed, cannot read YAML thermo config {path}"
            )
        data = yaml.safe_load(text)
    else:
        data = json.loads(text)

    if not isinstance(data, dict):
        raise ValueError(f"Thermo config {path} must contain a mapping at top level")
    return data


def _strict_from_mapping(d: Mapping[str, Any]) -> ThermoConfig:
    allowed = {f.name for f in fields(ThermoConfig)}
    unknown = [k for k in d.keys() if k not in allowed]
    if unknown:
        raise ValueError(f"Unknown thermo config keys: {unknown}")

    base = ThermoConfig()
    return ThermoConfig(
        temperature=float(d.get("temperature", base.temperature)),
        coupling_delta_F=float(d.get("coupling_delta_F", base.coupling_delta_F)),
        coupling_complexity=float(
            d.get("coupling_complexity", base.coupling_complexity)
        ),
        coupling_softness=float(d.get("coupling_softness", base.coupling_softness)),
        coupling_density=float(d.get("coupling_density", base.coupling_density)),
        coupling_density_shape=float(
            d.get("coupling_density_shape", base.coupling_density_shape)
        ),
        coupling_port_geometry=float(
            d.get("coupling_port_geometry", base.coupling_port_geometry)
        ),
        coupling_ws_Z=float(
            d.get("coupling_ws_Z", base.coupling_ws_Z)
        ),
        coupling_shape_softness=float(
            d.get("coupling_shape_softness", base.coupling_shape_softness)
        ),
        coupling_shape_chi=float(
            d.get("coupling_shape_chi", base.coupling_shape_chi)
        ),
        coupling_topo_3d=float(
            d.get("coupling_topo_3d", base.coupling_topo_3d)
        ),
        coupling_delta_G=float(
            d.get("coupling_delta_G", base.coupling_delta_G)
        ),
        density_model=str(d.get("density_model", base.density_model)),
        density_blend=str(d.get("density_blend", base.density_blend)),
        density_Z_ref=float(d.get("density_Z_ref", base.density_Z_ref)),
        density_source=str(d.get("density_source", base.density_source)),
        ws_R_max=float(d.get("ws_R_max", base.ws_R_max)),
        ws_R_well=float(d.get("ws_R_well", base.ws_R_well)),
        ws_V0=float(d.get("ws_V0", base.ws_V0)),
        ws_N_grid=int(d.get("ws_N_grid", base.ws_N_grid)),
        ws_ell=int(d.get("ws_ell", base.ws_ell)),
        ws_state_index=int(d.get("ws_state_index", base.ws_state_index)),
        port_geometry_source=str(
            d.get("port_geometry_source", base.port_geometry_source)
        ),
        port_geometry_blend=str(
            d.get("port_geometry_blend", base.port_geometry_blend)
        ),
        ws_geom_R_max=float(d.get("ws_geom_R_max", base.ws_geom_R_max)),
        ws_geom_R_well=float(d.get("ws_geom_R_well", base.ws_geom_R_well)),
        ws_geom_V0=float(d.get("ws_geom_V0", base.ws_geom_V0)),
        ws_geom_N_grid=int(d.get("ws_geom_N_grid", base.ws_geom_N_grid)),
        ws_geom_gap_scale=float(
            d.get("ws_geom_gap_scale", base.ws_geom_gap_scale)
        ),
        ws_geom_gap_ref=float(d.get("ws_geom_gap_ref", base.ws_geom_gap_ref)),
        ws_Z_ref=float(d.get("ws_Z_ref", base.ws_Z_ref)),
        ws_Z_alpha=float(d.get("ws_Z_alpha", base.ws_Z_alpha)),
        shape_kurt_scale=float(d.get("shape_kurt_scale", base.shape_kurt_scale)),
        shape_rrms_scale=float(d.get("shape_rrms_scale", base.shape_rrms_scale)),
        shape_softness_gain=float(d.get("shape_softness_gain", base.shape_softness_gain)),
        shape_chi_gain=float(d.get("shape_chi_gain", base.shape_chi_gain)),
        topo_3d_beta=float(d.get("topo_3d_beta", base.topo_3d_beta)),
        ws_integrator=str(d.get("ws_integrator", base.ws_integrator)),
        ws_fdm_depth=int(d.get("ws_fdm_depth", base.ws_fdm_depth)),
        ws_fdm_base=int(d.get("ws_fdm_base", base.ws_fdm_base)),
        temperature_T=float(d.get("temperature_T", base.temperature_T)),
        grower_use_mh=bool(d.get("grower_use_mh", base.grower_use_mh)),
        deltaG_backend=str(d.get("deltaG_backend", base.deltaG_backend)),
        consume_port_on_reject=bool(
            d.get("consume_port_on_reject", base.consume_port_on_reject)
        ),
        max_attempts_per_port=int(
            d.get("max_attempts_per_port", base.max_attempts_per_port)
        ),
        grower_proposal_policy=str(
            d.get("grower_proposal_policy", base.grower_proposal_policy)
        ),
        proposal_beta=float(d.get("proposal_beta", base.proposal_beta)),
        proposal_ports_gamma=float(
            d.get("proposal_ports_gamma", base.proposal_ports_gamma)
        ),
        experiment_name=str(d.get("experiment_name", base.experiment_name)),
    )


def load_thermo_config(path_str: str) -> ThermoConfig:
    """
    Поддерживает два формата:
    1) отдельный thermo-файл: {temperature: ..., coupling_delta_F: ...}
    2) общий экспериментальный YAML с секцией thermo: {thermo: {...}, growth: {...}, ...}
    """
    path = Path(path_str)
    data = _load_dict(path)

    section = data.get("thermo", None)
    if isinstance(section, dict):
        thermo_raw: Dict[str, Any] = section
    else:
        thermo_raw = dict(data)

    # Backward-friendly alias из physics.temperature (если thermo.temperature не задан)
    if "temperature" not in thermo_raw:
        phys = data.get("physics", None)
        if isinstance(phys, dict) and "temperature" in phys:
            thermo_raw["temperature"] = phys.get("temperature")

    return _strict_from_mapping(thermo_raw)


@contextmanager
def override_thermo_config(tmp_cfg: ThermoConfig):
    old_cfg = copy.deepcopy(get_current_thermo_config())
    try:
        set_current_thermo_config(tmp_cfg)
        yield
    finally:
        set_current_thermo_config(old_cfg)
