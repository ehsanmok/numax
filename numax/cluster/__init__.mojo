"""numax.cluster: vector quantization, k-means and hierarchical clustering,
`scipy.cluster`'s.

```mojo
from numax.cluster import kmeans, vq, whiten, linkage, fcluster
```

| Module | Contents |
|---|---|
| `vq` | `whiten`, `vq` (`VQResult`), `kmeans` (`KMeansResult`), `kmeans2` (`KMeans2Result`) -- SciPy's Lloyd loops, the assignment and centroid passes on the observations' device |
| `hierarchy` | `linkage` (seven methods, from condensed distances or observations), `inconsistent`, `fcluster` (`"inconsistent"`, `"distance"`, `"maxclust"`) -- the merges' distance matrix on the device, SciPy's post-processing and flat-cluster numbering on the host |

Tier 2 over tensors.
"""

from .hierarchy import fcluster, inconsistent, linkage
from .vq import (
    KMeans2Result,
    KMeansResult,
    VQResult,
    kmeans,
    kmeans2,
    vq,
    whiten,
)
