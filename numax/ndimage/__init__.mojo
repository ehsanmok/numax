"""numax.ndimage: N-dimensional image processing, `scipy.ndimage`'s.

```mojo
from numax.ndimage import gaussian_filter, median_filter, convolve
```

| Module | Contents |
|---|---|
| `filters` | `correlate`, `convolve`, `correlate1d`, `convolve1d`, `uniform_filter`, `gaussian_filter`, `gaussian_filter1d`, `median_filter`, `minimum_filter`, `maximum_filter` -- SciPy's five boundary modes and origin convention, rank 1 to 8, one lane per output element on the input's device |

Not re-exported at the root or in the prelude: `convolve` and `correlate`
are `numax.signal`'s names there, with a different meaning (1-D sequence
convolution in `MODE_FULL`/`MODE_SAME`/`MODE_VALID`), so ndimage is one
qualified import away.

Tier 2 over tensors. `filters.mojo` records the MAX gate: MAX's `conv`,
`pool` and `resize` are 2-D NHWC operators with zero padding, which
cannot express ndimage's modes.
"""

from .filters import (
    convolve,
    convolve1d,
    correlate,
    correlate1d,
    gaussian_filter,
    gaussian_filter1d,
    maximum_filter,
    median_filter,
    minimum_filter,
    uniform_filter,
)
