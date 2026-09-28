"""Private: the register tier of `numax.signal`, over `Array[T, n]` and
`FloatLike` scalars.

Nothing outside numax imports this package by name. `numax.signal`
exports its names, and each name it shares with the `Tensor` tier is an
overload in the `Tensor`-tier module that forwards here. The algorithms,
their tier and their bounds are documented on the functions in
`signal.mojo`.
"""

from .signal import (
    apply_window,
    blackman,
    convolve,
    correlate,
    firwin,
    MODE_FULL,
    hamming,
    hann,
    lfilter,
    MODE_SAME,
)
