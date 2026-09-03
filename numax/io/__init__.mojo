"""numax.io: tensor I/O and printing.

```mojo
from numax.io import save, load, print_tensor
```

`save`/`load` use numax's own `NMX1` binary format -- little-endian, with
the dtype, rank and shape in the header, checked on load -- because MAX
ships no array I/O to interchange with. `print_tensor` is NumPy-style,
truncating past a threshold. Tier 2, `Plain`-only, host-side.
"""

from .io import load, print_tensor, save
