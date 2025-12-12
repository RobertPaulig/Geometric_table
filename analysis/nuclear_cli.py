from __future__ import annotations

from typing import Optional

from core.nuclear_config import load_nuclear_config, set_current_nuclear_config


def apply_nuclear_config_if_provided(config_path: Optional[str]) -> None:
    """
    Общий helper для ядерных R&D-скриптов.

    Если config_path задан, загружает NuclearConfig и применяет его
    как текущее глобальное состояние.
    """
    if not config_path:
        return
    cfg = load_nuclear_config(config_path)
    set_current_nuclear_config(cfg)

