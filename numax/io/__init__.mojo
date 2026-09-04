"""numax.io: tensor I/O and printing.

```mojo
from numax.io import numpy, nmx

numpy.save(a, "grid.npy")     # what numpy.load opens
nmx.save(a, "grid.nmx")       # numax to numax
```

`numpy.save`/`numpy.load` read and write NumPy's `.npy` format directly (no
Python, no NumPy involved -- `.npy` is a documented self-contained
format), so a program ported from NumPy can ingest the files it already
has and hand results back the same way.

`nmx.save`/`nmx.load` use numax's own `NMX1` binary format -- little-endian, with
the dtype, rank and shape in the header, checked on load -- because MAX
ships no array I/O to interchange with. Printing is not here: `Tensor` conforms to `Writable`, so `print(a)` works
on its own and `a.format(precision=8)` is the same output with the
precision and truncation under the caller's control. Tier 2, `Plain`-only, host-side.
"""

from .io import nmx
from .npy import numpy
