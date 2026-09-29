"""`scipy.spatial` and `scipy.cluster` over a `Tensor`: three blobs of
points, found three ways.

```mojo
var km = kmeans2(points, guess)              # Lloyd's, on the device
var z = linkage(points, "ward")              # agglomerative merges
var flat = fcluster(z, 3.0, "maxclust")      # cut into three
```

The points are three well-separated clouds in the plane. `kmeans2`
recovers their centers from a rough guess -- one `cdist` GEMM and an
`argmin` per Lloyd iteration, the same call at `gpu=True` on a device --
and `linkage` plus `fcluster` recovers the same partition with no guess
at all, so the two labelings are checked against each other up to a
renaming. A `KDTree` over the points then answers the two queries a
spatial index exists for: the nearest points to each true center, and how
many lie within a radius of it.

Run: `pixi run example-clustering`
"""

from std.math import cos, sin

from numax.core.tensor import Static
from numax.cluster import fcluster, kmeans2, linkage
from numax.spatial import KDTree, pdist

comptime dtype = DType.float64
comptime per_blob = 40
comptime n = 3 * per_blob


def blobs() raises -> Static[dtype, n, 2]:
    """Three clouds of `per_blob` points around `(0, 0)`, `(6, 0)` and
    `(3, 5)`: a deterministic spiral of offsets rather than an RNG, so the
    printed numbers are stable across machines."""
    var centers: List[Float64] = [0.0, 0.0, 6.0, 0.0, 3.0, 5.0]
    var values = List[Scalar[dtype]](capacity=2 * n)
    for c in range(3):
        for i in range(per_blob):
            var radius = 0.9 * Float64(i + 1) / Float64(per_blob)
            var angle = 2.39996 * Float64(i)
            values.append(Scalar[dtype](centers[2 * c] + radius * cos(angle)))
            values.append(
                Scalar[dtype](centers[2 * c + 1] + radius * sin(angle))
            )
    return Static[dtype, n, 2](values^)


def main() raises:
    var points = blobs()

    # --- k-means from a rough guess.
    var guess = Static[dtype, 3, 2]([1.0, 1.0, 5.0, 1.0, 2.0, 4.0])
    var km = kmeans2(points, guess)
    var centroids = km.centroid.to_host()
    print("kmeans2 centroids:")
    for c in range(3):
        print("  (", centroids[2 * c], ",", centroids[2 * c + 1], ")")

    # --- The same partition from the merge tree, with no guess.
    var z = linkage(points, "ward")
    var flat = fcluster(z, 3.0, "maxclust").to_host()
    var labels = km.label.to_host()
    # Two labelings agree when each k-means label maps to one flat
    # cluster everywhere.
    var mapping = List[Int](length=3, fill=-1)
    var agree = True
    for i in range(n):
        var k = Int(labels[i])
        var f = Int(flat[i])
        if mapping[k] < 0:
            mapping[k] = f
        elif mapping[k] != f:
            agree = False
    print("kmeans2 and ward linkage agree up to renaming:", agree)

    # --- The condensed distance vector the merge tree is built from.
    var d = pdist(points)
    print("pdist length", d.size(), "= n (n - 1) / 2 =", n * (n - 1) // 2)

    # --- A spatial index over the points.
    var tree = KDTree[dtype](points)
    var truth = Static[dtype, 3, 2]([0.0, 0.0, 6.0, 0.0, 3.0, 5.0])
    var nearest = tree.query[count=2](truth)
    var dist = nearest.distances.to_host()
    var idx = nearest.indices.to_host()
    for c in range(3):
        print(
            "center",
            c,
            ": nearest points",
            idx[2 * c],
            idx[2 * c + 1],
            "at",
            dist[2 * c],
            dist[2 * c + 1],
        )
    var ball = tree.query_ball_point(truth, 1.0)
    var offsets = ball.offsets.to_host()
    for c in range(3):
        print(
            "center",
            c,
            ": points within 1.0 =",
            offsets[c + 1] - offsets[c],
            "(all",
            per_blob,
            "of its blob)",
        )
