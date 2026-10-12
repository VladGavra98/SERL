from typing import List
from signals.base_signal import Signal
from signals.simple_signals import Step, Sinusoid
from signals.complex_signals import CosineSmoothedStep


def Doublet(t_start: float, ampl: float, block_width: float) -> Signal:
    w = block_width
    up = ampl * Step(t_start=t_start, t_end=t_start + w)
    down = -ampl * Step(t_start=t_start + w, t_end=t_start + 2 * w)

    return up + down


def ThreeTwoOneOne(
    t_start: float = 1.0, ampl: float = 1.0, block_width: float = 1.0
) -> Signal:
    w = block_width
    up1 = ampl * Step(t_start=t_start, t_end=t_start + 7 * w)
    down1 = -2 * ampl * Step(t_start=t_start + 3 * w, t_end=t_start + 5 * w)
    down2 = -2 * ampl * Step(t_start=t_start + 6 * w, t_end=t_start + 7 * w)

    return up1 + down1 + down2


def CompositeSinusoid(
    t_start: float, t_end: float, frequencies: List[float], amplitudes: List[float]
) -> Signal:
    """Provide a list of frequencies and amplitudes to build a composite sinusoid wave."""
    assert len(frequencies) == len(
        amplitudes
    ), "frequencies and amplitudes arguments must be of the same length"

    s = 0
    for f, a in zip(frequencies, amplitudes):
        s = s + Sinusoid(t_start=t_start, t_end=t_end, ampl=a, freq=f)
    return s


def ThreeTwoOneOneSmoothed(
    t_start: float = 1.0,
    ampl: float = 1.0,
    block_width: float = 1.0,
    smooth_width: float = 0.2,
) -> Signal:
    assert (
        smooth_width < block_width / 2
    ), "smoothing width must smaller than half of the block_width "
    bw = block_width
    sw = smooth_width

    up1 = ampl * CosineSmoothedStep(t_start=t_start, t_end=t_start + 7 * bw, width=sw)
    down1 = (
        -2
        * ampl
        * CosineSmoothedStep(t_start=t_start + 3 * bw, t_end=t_start + 5 * bw, width=sw)
    )
    down2 = (
        -2
        * ampl
        * CosineSmoothedStep(t_start=t_start + 6 * bw, t_end=t_start + 7 * bw, width=sw)
    )

    return up1 + down1 + down2
