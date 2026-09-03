"""numax.signal: convolution, correlation, and window functions.

```mojo
from numax.signal import convolve, correlate, hann, apply_window
```

`convolve`/`convolve_same`, `correlate`, the Hann, Hamming and Blackman
windows, and `apply_window`. Direct sums over comptime-sized `Array`s,
tier 1; `numax.fft.circular_convolve` is the transform-domain route.
"""

from .signal import (
    apply_window,
    blackman,
    convolve,
    convolve_same,
    correlate,
    hamming,
    hann,
)
