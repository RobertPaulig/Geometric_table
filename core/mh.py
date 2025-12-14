from __future__ import annotations

from typing import Any

import math

from core.thermo_config import ThermoConfig, get_current_thermo_config


def mh_accept(delta_G: float, thermo: ThermoConfig | None, rng: Any) -> bool:
    """
    Metropolis–Hastings критерий для ΔG.

    Важно: в режимах "всегда accept" RNG не расходуется (rng.random не вызывается),
    чтобы не ломать детерминизм legacy-grower'а при вырождении.
    """
    if thermo is None:
        thermo = get_current_thermo_config()

    coupling = float(getattr(thermo, "coupling_delta_G", 1.0))
    T = float(getattr(thermo, "temperature_T", 1.0))

    # Вырождение: coupling_delta_G == 0 -> всегда accept без RNG
    if coupling == 0.0:
        return True

    # ΔG <= 0 -> принимаем всегда (улучшение) без RNG
    if delta_G <= 0.0:
        return True

    # Температура -> ∞ (T >= 1e8) -> всегда accept без RNG
    if T >= 1e8:
        return True

    # T <= 0 и ΔG>0 -> всегда reject
    if T <= 0.0:
        return False

    # Обычный MH
    arg = -delta_G / T
    # защита от переполнения
    if arg < -700.0:
        p_accept = 0.0
    else:
        p_accept = math.exp(arg)
    p_accept = min(1.0, max(0.0, p_accept))

    r = rng.random()
    return r < p_accept

