from dataclasses import dataclass
from signals.base_signal import Signal
from signals.simple_signals import Sinusoid, Ramp
import numpy as np


class SeeSaw(Signal):
    """Alternating linear zigzag signal with a constant amplitude and frequency."""

    def __init__(
        self, t_start: float, t_end: float, ampl: float = 1.0, freq: float = 1.0
    ):
        super().__init__(t_start, t_end)
        self.halfperiod = 1 / (2 * freq)
        self.sine = Sinusoid(ampl=ampl, freq=freq)

        # Sample the discrete points from the sinusoid:
        self.sampling_times = np.arange(
            self.t_start - self.halfperiod / 2, t_end + self.halfperiod, self.halfperiod
        )

        # Correct for the half-period shift
        self.sampling_times[0] = self.t_start

        # Discrete samples
        self.samples = np.array([self.sine(t) for t in self.sampling_times])

    def _signal(self, t: float) -> float:
        return np.interp(t, self.sampling_times, self.samples)


@dataclass
class RampSinusoid(Signal):
    ampl_max: float = 1.0
    freq: float = 1.0

    def __post_init__(self):
        rate = self.ampl_max / (self.t_end - self.t_start)
        self.sine = rate * Ramp(t_start=self.t_start) * Sinusoid(freq=self.freq)

    def _signal(self, t: float) -> float:
        return self.sine(t)


class AlternatingRamp(Signal):
    """Alternating linear zigzag signal with a linearly increasing amplitude and frequency."""

    def __init__(
        self, t_start: float, t_end: float, ampl_max: float = 1.0, freq: float = 1.0
    ):
        super().__init__(t_start, t_end)
        self.ampl_max = ampl_max
        self.freq = freq

        self.period = 1 / freq
        self.halfperiod = 1 / (2 * freq)

        rate = self.ampl_max / (self.t_end - self.t_start)
        self.sine = rate * Ramp(t_start=self.t_start) * Sinusoid(freq=self.freq)

        # Sample the discrete points from the sinusoid:
        self.sampling_times = np.arange(
            self.t_start - self.halfperiod / 2, t_end + self.halfperiod, self.halfperiod
        )

        # Correct for the half-period shift
        self.sampling_times[0] = self.t_start

        # Discrete samples
        self.samples = np.array([self.sine(t) for t in self.sampling_times])

    def _signal(self, t: float) -> float:
        return np.interp(t, self.sampling_times, self.samples)


@dataclass
class CosineSmoothedStep(Signal):
    width: float = 1.0

    def _signal(self, t: float) -> float:
        if t < self.width:
            return -(np.cos(np.pi * t / self.width) - 1) / 2
        else:
            return 1.0
