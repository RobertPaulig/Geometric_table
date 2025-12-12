from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Dict

import json

try:
    import yaml  # type: ignore[import]
except ImportError:
    yaml = None


@dataclass
class GrowthConfig:
    max_atoms: int = 25
    max_depth: int = 4
    p_continue_base: float = 0.8

    role_bonus_hub: float = 0.0
    role_penalty_terminator: float = 0.0

    temperature: float = 1.0

    allow_cycles: bool = False
    max_extra_bonds: int = 0
    p_extra_bond: float = 0.0

    experiment_name: str = "unnamed"
    version: str = "dev"

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "GrowthConfig":
        """
        Собирает GrowthConfig из словаря с возможными секциями:
        - growth: {...}
        - roles: {...}
        - physics: {...}
        - loops: {...}
        Плоские ключи приоритетнее секций.
        """
        merged: Dict[str, Any] = {}

        for section in ("growth", "roles", "physics", "loops"):
            section_dict = data.get(section, {})
            if isinstance(section_dict, Mapping):
                merged.update(section_dict)

        for key, value in data.items():
            if key not in ("growth", "roles", "physics", "loops", "thermo"):
                merged[key] = value

        return cls(**merged)

    def to_growth_params(self) -> "GrowthParams":
        """
        Преобразует конфиг в GrowthParams.

        Импорт сделан внутри, чтобы избежать циклических зависимостей.
        """
        from core.grower import GrowthParams

        return GrowthParams(
            max_depth=self.max_depth,
            max_atoms=self.max_atoms,
            p_continue_base=self.p_continue_base,
            role_bonus_hub=self.role_bonus_hub,
            role_penalty_terminator=self.role_penalty_terminator,
            temperature=self.temperature,
            allow_cycles=self.allow_cycles,
            max_extra_bonds=self.max_extra_bonds,
            p_extra_bond=self.p_extra_bond,
        )


def load_growth_config(path: str | Path) -> GrowthConfig:
    path = Path(path)
    text = path.read_text(encoding="utf-8")

    if path.suffix in {".yaml", ".yml"}:
        if yaml is None:
            raise RuntimeError(
                f"PyYAML не установлен, а конфиг {path} в формате YAML. "
                "Установи pyyaml или используй JSON."
            )
        data = yaml.safe_load(text) or {}
    else:
        data = json.loads(text)

    if not isinstance(data, Mapping):
        raise ValueError(f"Конфиг {path} должен содержать объект верхнего уровня")

    return GrowthConfig.from_dict(data)
