"""numax.signal: convolution, correlation, filtering, and windows.

```mojo
from numax.signal import convolve, correlate, hann, apply_window
```

`convolve` (`mode=full` or `same`), `correlate`, `lfilter` and the
`firwin` design that feeds it, the Hann, Hamming and Blackman windows,
and `apply_window`. Direct sums over comptime-sized `Array`s, tier 1;
`numax.fft.circular_convolve` is the transform-domain route.
"""

from .signal import (
    apply_window,
    blackman,
    convolve,
    full,
    same,
    correlate,
    firwin,
    hamming,
    hann,
    lfilter,
)
