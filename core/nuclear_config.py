from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict
from contextlib import contextmanager
import copy
import json

try:
    import yaml  # type: ignore[import]
except Exception:  # pragma: no cover
    yaml = None


@dataclass
class NuclearShellConfig:
    lambda_shell: float = 30.0
    sigma_p: float = 6.0
    sigma_n: float = 8.0
    a_p: float = 12.0
    # Базовая ширина изотопной полосы delta_F для scan_isotope_band (legacy 5.0).
    delta_F_base: float = 5.0


@dataclass
class NuclearConfig:
    shell: NuclearShellConfig


_CURRENT_NUCLEAR_CONFIG = NuclearConfig(shell=NuclearShellConfig())


def get_current_nuclear_config() -> NuclearConfig:
    return _CURRENT_NUCLEAR_CONFIG


def set_current_nuclear_config(cfg: NuclearConfig) -> None:
    global _CURRENT_NUCLEAR_CONFIG
    _CURRENT_NUCLEAR_CONFIG = cfg


def _load_dict(path: Path) -> Dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() in {".yaml", ".yml"}:
        if yaml is None:
            raise RuntimeError(
                f"PyYAML is not installed, cannot read YAML nuclear config {path}"
            )
        data = yaml.safe_load(text)
    else:
        data = json.loads(text)

    if not isinstance(data, dict):
        raise ValueError(f"Nuclear config {path} must contain a mapping at top level")
    return data


def load_nuclear_config(path_str: str) -> NuclearConfig:
    path = Path(path_str)
    data = _load_dict(path)

    shell_raw = data.get("shell", {})
    if not isinstance(shell_raw, dict):
        shell_raw = {}

    defaults = NuclearShellConfig()
    shell = NuclearShellConfig(
        lambda_shell=float(shell_raw.get("lambda_shell", defaults.lambda_shell)),
        sigma_p=float(shell_raw.get("sigma_p", defaults.sigma_p)),
        sigma_n=float(shell_raw.get("sigma_n", defaults.sigma_n)),
        a_p=float(shell_raw.get("a_p", defaults.a_p)),
        delta_F_base=float(shell_raw.get("delta_F_base", defaults.delta_F_base)),
    )

    return NuclearConfig(shell=shell)


@contextmanager
def override_nuclear_config(tmp_cfg: NuclearConfig):
    """
    Временная подмена глобального NuclearConfig.

    Пример использования:
        base = get_current_nuclear_config()
        trial = replace(base, shell=replace(base.shell, lambda_shell=10.0))
        with override_nuclear_config(trial):
            F = nuclear_functional(Z, N)
    """
    old_cfg = copy.deepcopy(get_current_nuclear_config())
    try:
        set_current_nuclear_config(tmp_cfg)
        yield
    finally:
        set_current_nuclear_config(old_cfg)
