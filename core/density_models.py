from __future__ import annotations

from typing import Optional
import math


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


def beta_tf_radius(Z: float) -> float:
    """
    TF-radius proxy для Gaussian: r ~ Z^(-1/3) => beta ~ Z^(2/3).
    """
    z = max(float(Z), 1e-9)
    return z ** (2.0 / 3.0)


def rescale_to_match_reference(beta_model_fn, Z_ref: float, beta_ref_value: float):
    """
    Возвращает функцию beta(Z) = s * beta_model_fn(Z),
    где s подобран так, чтобы beta(Z_ref) == beta_ref_value.
    """
    zref = max(float(Z_ref), 1e-9)
    denom = max(float(beta_model_fn(zref)), 1e-12)
    s = float(beta_ref_value) / denom

    def _beta(Z: float) -> float:
        return s * float(beta_model_fn(Z))

    return _beta


def blend_positive(a: float, b: float, c: float, mode: str = "linear") -> float:
    """
    Смешивание двух положительных масштабов с коэффициентом c∈[0,1].
    mode="linear"  — обычная линейная смесь;
    mode="log"     — геометрическое смешивание (по логарифмам).
    """
    c = _clamp_0_1(c)
    a = max(float(a), 1e-12)
    b = max(float(b), 1e-12)
    if mode == "log":
        return math.exp((1.0 - c) * math.log(a) + c * math.log(b))
    return (1.0 - c) * a + c * b


def beta_effective(
    Z: float,
    coupling: float,
    *,
    model: str = "tf_radius",
    blend: str = "linear",
    Z_ref: float = 10.0,
) -> float:
    """
    Смешивание legacy beta(Z) и физически мотивированной шкалы по coupling∈[0,1].
    """
    c = _clamp_0_1(coupling)
    b0 = beta_legacy(int(Z))

    if model == "tf_radius":
        base_fn = beta_tf_radius
    elif model == "tf_energy":
        base_fn = beta_tf
    elif model == "hydrogenic":
        base_fn = beta_hydrogenic
    else:
        base_fn = beta_tf_radius

    beta_phys_fn = rescale_to_match_reference(base_fn, Z_ref=Z_ref, beta_ref_value=beta_legacy(int(Z_ref)))
    b1 = beta_phys_fn(Z)

    return blend_positive(b0, b1, c, mode=blend)
