from __future__ import annotations

from typing import Optional


def _clamp_0_1(x: float) -> float:
    return max(0.0, min(float(x), 1.0))


def beta_legacy(Z: int) -> float:
    """
    Legacy toy-модель beta(Z) для FDM-плотности.
    """
    return 0.5 + 0.05 * float(Z)


def beta_hydrogenic(Z: int, Z_eff: Optional[float] = None) -> float:
    """
    Грубая гидрогеноподобная шкала:
    характерный радиус ~ 1/Z_eff => beta ~ Z_eff^2.
    """
    z = float(Z)
    ze = float(Z_eff) if Z_eff is not None else z
    return max(1e-9, ze * ze)


def beta_tf(Z: int) -> float:
    """
    Томас–Ферми-подобная шкала: beta ~ Z^(4/3).
    Монотонная по Z, растёт мягче, чем Z^2.
    """
    z = max(1.0, float(Z))
    return z ** (4.0 / 3.0)


def beta_effective(Z: int, coupling: float, *, model: str = "tf") -> float:
    """
    Смешивание legacy beta(Z) и физически мотивированной шкалы по coupling∈[0,1].
    """
    c = _clamp_0_1(coupling)
    b0 = beta_legacy(Z)
    if model == "hydrogenic":
        b1 = beta_hydrogenic(Z)
    else:
        b1 = beta_tf(Z)
    return b0 * (1.0 - c) + b1 * c

