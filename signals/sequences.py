import numpy as np
from typing import Union
from signals.base_signal import Const
from signals.simple_signals import Signal, Step
from signals.complex_signals import CosineSmoothedStep

ArrayLike = Union[list, tuple, np.ndarray]


def StepSequence(times: ArrayLike, amplitudes: ArrayLike) -> Signal:
    """Generates a sequence of Step signals at the elements of t_i in times with height a_i in amplitudes.
    The two arguments must have the same length.
    """
    assert len(times) == len(amplitudes), "sequence arrays must be of equal length"

    signal = Const(0.0)

    # Loop over the provided time locations
    for idx, t0 in enumerate(times):
        # For the first element the step amplitude is relative to 0.0, otherwise its relative to the previous amplitude
        ampl_prev = 0.0 if idx == 0 else amplitudes[idx - 1]
        ampl_i = amplitudes[idx]
        delta_ampl = ampl_i - ampl_prev

        # Add the signal
        signal += delta_ampl * Step(t_start=t0)

    return signal


def SmoothedStepSequence(
    times: ArrayLike, amplitudes: ArrayLike, smooth_width: float
) -> Signal:
    """Generates a sequence of smoothed step signals at the elements of t_i in times with height a_i in amplitudes.
    The two arguments must have the same length.
    """
    assert len(times) == len(amplitudes), "sequence arrays must be of equal length"

    signal = Const(0.0)

    # Loop over the provided time locations
    for idx, t0 in enumerate(times):

        # For the first element the step amplitude is relative to 0.0, otherwise its relative to the previous amplitude
        ampl_prev = 0.0 if idx == 0 else amplitudes[idx - 1]
        ampl_i = amplitudes[idx]
        delta_ampl = ampl_i - ampl_prev

        # Add the signal
        signal += delta_ampl * CosineSmoothedStep(t_start=t0, width=smooth_width)

    return signal
