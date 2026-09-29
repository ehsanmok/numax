"""numax.spatial: pairwise distances and nearest-neighbor search,
`scipy.spatial`'s.

```mojo
from numax.spatial import cdist, pdist, squareform, KDTree
```

| Module | Contents |
|---|---|
| `distance` | `cdist`, `pdist`, `squareform` -- euclidean, squared euclidean, cityblock, chebyshev, minkowski, cosine and correlation, one lane per pair on the inputs' device |
| `kdtree` | `KDTree` with `query` (`KDQuery`) and `query_ball_point` (`BallPoints`) -- built on the host, queried on the data's device, one lane per query point |

Tier 2 over tensors. `Delaunay`, `ConvexHull` and `Voronoi` (the Qhull
family) and `Rotation` are out of scope; `docs/parity.md` records why.
"""

from .distance import cdist, pdist, squareform
from .kdtree import BallPoints, KDQuery, KDTree
