from __future__ import annotations

import numpy as np

from core.mh import mh_accept
from core.thermo_config import ThermoConfig


class DummyRNG:
    def __init__(self):
        self.calls = 0
        self._rng = np.random.default_rng(123)

    def random(self):
        self.calls += 1
        return float(self._rng.random())


def test_mh_accept_always_true_when_coupling_zero_or_delta_non_positive():
    rng = DummyRNG()
    thermo = ThermoConfig(coupling_delta_G=0.0, temperature_T=1.0)
    assert mh_accept(10.0, thermo, rng) is True
    assert rng.calls == 0

    rng2 = DummyRNG()
    thermo2 = ThermoConfig(coupling_delta_G=1.0, temperature_T=1.0)
    assert mh_accept(0.0, thermo2, rng2) is True
    assert mh_accept(-5.0, thermo2, rng2) is True
    assert rng2.calls == 0


def test_mh_accept_always_true_when_T_infinite_like():
    rng = DummyRNG()
    thermo = ThermoConfig(coupling_delta_G=1.0, temperature_T=1e8)
    assert mh_accept(10.0, thermo, rng) is True
    assert rng.calls == 0


def test_mh_accept_rate_increases_with_temperature():
    deltaG = 1.0
    n = 2000

    def run(T: float) -> float:
        thermo = ThermoConfig(coupling_delta_G=1.0, temperature_T=T)
        rng = DummyRNG()
        acc = 0
        for _ in range(n):
            if mh_accept(deltaG, thermo, rng):
                acc += 1
        return acc / n

    rate_low = run(0.5)
    rate_high = run(2.0)
    assert rate_high > rate_low

