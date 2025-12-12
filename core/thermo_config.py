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
