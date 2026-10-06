import numpy as np
from signals.base_signal import Signal
from signals.sequences import StepSequence, SmoothedStepSequence


def RandomizedStepSequence(
    t_max: float,
    ampl_max: float,
    block_width: float,
    start_with_zero: bool = True,
    n_levels: int = 5,
    vary_timings: float = 0.0,
) -> Signal:

    assert (
        vary_timings < block_width / 2
    ), "vary timings should be smaller than half block width"

    # Starting time of each step block
    t_starts = np.arange(0, t_max, block_width)

    # Possible choices
    ampl_choices = np.linspace(-ampl_max, ampl_max, n_levels)

    # Generate random amplitudes
    amplitudes = np.random.choice(ampl_choices, size=t_starts.size, replace=True)

    if start_with_zero:
        amplitudes[0] = 0.0

    # Vary the timings of the steps
    t_starts = [t_starts[0]] + [
        t + np.random.uniform(-vary_timings, vary_timings) for t in t_starts[1:]
    ]

    return StepSequence(times=t_starts, amplitudes=amplitudes)


def RandomizedCosineStepSequence(
    t_max: float,
    ampl_max: float,
    block_width: float,
    smooth_width: float,
    start_with_zero: bool = True,
    n_levels: int = 10,
    vary_timings: float = 0.0,
) -> Signal:

    assert (
        vary_timings < block_width / 2
    ), "vary timings should be smaller than half block width"

    # Starting time of each step block
    t_starts = np.arange(0, t_max, block_width)

    # Possible choices
    ampl_choices = np.linspace(-ampl_max, ampl_max, n_levels)

    # Generate random amplitudes
    amplitudes = np.random.choice(ampl_choices, size=t_starts.size, replace=True)
    if start_with_zero:
        amplitudes[0] = 0.0

    # Vary the timings of the steps
    t_starts = [t_starts[0]] + [
        t + np.random.uniform(-vary_timings, vary_timings) for t in t_starts[1:]
    ]

    return SmoothedStepSequence(
        times=t_starts, amplitudes=amplitudes, smooth_width=smooth_width
    )
