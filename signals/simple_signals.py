from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from signals.base_signal import Signal


@dataclass
class Step(Signal):
    def _signal(self, _: float) -> float:
        return 1.0


@dataclass
class Ramp(Signal):
    def _signal(self, t: float) -> float:
        return t


@dataclass
class Parabolic(Signal):
    def _signal(self, t: float) -> float:
        return t * t / 2.0


@dataclass
class Exponential(Signal):
    alpha: float = 0.0

    def _signal(self, t: float) -> float:
        return np.exp(self.alpha * t)


@dataclass
class Sinusoid(Signal):
    ampl: float = 1.0
    freq: float = 1.0
    phi: float = 0.0

    def _signal(self, t: float) -> float:
        return self.ampl * np.sin(2.0 * np.pi * self.freq * t + self.phi)
