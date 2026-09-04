"""numax.io: tensor I/O and printing.

```mojo
from numax.io import numpy, nmx, print_tensor

numpy.save(a, "grid.npy")     # what numpy.load opens
nmx.save(a, "grid.nmx")       # numax to numax
```

`numpy.save`/`numpy.load` read and write NumPy's `.npy` format directly (no
Python, no NumPy involved -- `.npy` is a documented self-contained
format), so a program ported from NumPy can ingest the files it already
has and hand results back the same way.

`nmx.save`/`nmx.load` use numax's own `NMX1` binary format -- little-endian, with
the dtype, rank and shape in the header, checked on load -- because MAX
ships no array I/O to interchange with. `print_tensor` is NumPy-style,
truncating past a threshold. Tier 2, `Plain`-only, host-side.
"""

from .io import nmx, print_tensor
from .npy import numpy
