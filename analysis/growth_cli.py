from __future__ import annotations

from typing import Optional

from core.grower import GrowthParams
from core.growth_config import load_growth_config


def make_growth_params_from_config_path(
    config_path: Optional[str],
    *,
    default_max_depth: int = 4,
    default_max_atoms: int = 25,
) -> GrowthParams:
    """
    Унифицированный helper для анализа роста.

    - Если config_path задан: читаем GrowthConfig и строим GrowthParams.
    - Если None: возвращаем v5.0-baseline (деревья) с указанными max_depth/max_atoms.
    """
    if config_path:
        cfg = load_growth_config(config_path)
        return cfg.to_growth_params()

    return GrowthParams(
        max_depth=default_max_depth,
        max_atoms=default_max_atoms,
    )

