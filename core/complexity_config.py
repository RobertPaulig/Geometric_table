from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import json

try:
    import yaml  # type: ignore[import]
except Exception:  # pragma: no cover
    yaml = None


@dataclass
class LoopyPenaltyConfig:
    alpha_cycle: float = 0.3
    alpha_load: float = 1.0


@dataclass
class CrossingPenaltyConfig:
    beta_cross: float = 1.0
    max_cross_n: int = 8


@dataclass
class ComplexityPenaltyConfig:
    loopy: LoopyPenaltyConfig
    crossing: CrossingPenaltyConfig


_CURRENT_PENALTIES = ComplexityPenaltyConfig(
    loopy=LoopyPenaltyConfig(),
    crossing=CrossingPenaltyConfig(),
)


def get_current_penalties() -> ComplexityPenaltyConfig:
    return _CURRENT_PENALTIES


def set_current_penalties(cfg: ComplexityPenaltyConfig) -> None:
    global _CURRENT_PENALTIES
    _CURRENT_PENALTIES = cfg


def _from_dict(d: Dict[str, Any]) -> ComplexityPenaltyConfig:
    loopy = d.get("loopy", {}) or {}
    crossing = d.get("crossing", {}) or {}

    loopy_cfg = LoopyPenaltyConfig(
        alpha_cycle=float(loopy.get("alpha_cycle", 0.3)),
        alpha_load=float(loopy.get("alpha_load", 1.0)),
    )
    crossing_cfg = CrossingPenaltyConfig(
        beta_cross=float(crossing.get("beta_cross", 1.0)),
        max_cross_n=int(crossing.get("max_cross_n", 8)),
    )
    return ComplexityPenaltyConfig(loopy=loopy_cfg, crossing=crossing_cfg)


def load_complexity_penalties(path: str | Path) -> ComplexityPenaltyConfig:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(p)
    text = p.read_text(encoding="utf-8")

    if p.suffix.lower() in {".yaml", ".yml"}:
        if yaml is None:
            raise RuntimeError(
                f"PyYAML не установлен, а конфиг {p} в формате YAML. "
                "Установи pyyaml или используй JSON."
            )
        data = yaml.safe_load(text)
    else:
        data = json.loads(text)

    if not isinstance(data, dict):
        raise ValueError(f"Config {p} must contain a mapping at top level")

    return _from_dict(data)
