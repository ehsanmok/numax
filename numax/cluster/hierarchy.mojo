"""Agglomerative clustering over `Tensor`: `linkage`, `inconsistent` and
`fcluster`, SciPy's `scipy.cluster.hierarchy`.

**Tier 2.**

- **`linkage`** merges, at every step, the two closest active clusters,
  and updates the merged cluster's distances by the method's
  Lance-Williams formula (SciPy's `_hierarchy_distance_update.pxi`:
  `single`, `complete`, `average`, `weighted`, `centroid`, `median`,
  `ward`). The distance matrix lives on the input's device; each step is
  a row-minimum launch, a one-lane reduction of the rows, and a row
  update launch, the host reading the `(i, j, d)` of the merge and nothing
  else. This is the primitive algorithm, `O(n^3)`, where SciPy runs the
  nearest-neighbor chain; for distinct distances both merge the same
  pairs. The rows are then what SciPy returns: for the reducible methods
  sorted by distance (stable) and relabeled through a union-find, SciPy's
  `label`; for `centroid` and `median`, in merge order with the new
  cluster numbered `n + k`, as SciPy's generic algorithm leaves them.
- **`inconsistent`** and **`fcluster`** read the linkage matrix on the
  host -- `n - 1` rows of tree structure -- and transcribe SciPy's
  traversals: the per-node statistics over `depth` levels, and the flat
  clusters under the `"distance"`, `"maxclust"` and `"inconsistent"`
  criteria, numbered in SciPy's depth-first order.

## The MAX gate

Nothing: MAX has no clustering. **Extend.**
"""

from std.builtin.sort import sort as _std_sort
from std.math import sqrt as _sqrt

from layout import Coord, coord_to_index_list
from layout.tile_layout import row_major
from max.algorithm.functional import elementwise

from ..core.tensorlike import TensorLike
from ..core.tensor import Dynamic, _dyn_shape, _same_order, asarray
from ..spatial.distance import pdist, squareform

comptime _SINGLE = 0
comptime _COMPLETE = 1
comptime _AVERAGE = 2
comptime _WEIGHTED = 3
comptime _CENTROID = 4
comptime _MEDIAN = 5
comptime _WARD = 6


def _target[gpu: Bool]() -> StaticString:
    comptime if gpu:
        return "gpu"
    else:
        return "cpu"


def _method_code(method: StaticString) raises -> Int:
    if method == "single":
        return _SINGLE
    if method == "complete":
        return _COMPLETE
    if method == "average":
        return _AVERAGE
    if method == "weighted":
        return _WEIGHTED
    if method == "centroid":
        return _CENTROID
    if method == "median":
        return _MEDIAN
    if method == "ward":
        return _WARD
    raise Error(
        (
            "linkage: method must be 'single', 'complete', 'average',"
            " 'weighted', 'centroid', 'median' or 'ward', got '"
        ),
        method,
        "'",
    )


@always_inline
def _update_distance[
    dtype: DType
](
    d_xi: Scalar[dtype],
    d_yi: Scalar[dtype],
    d_xy: Scalar[dtype],
    sx: Scalar[dtype],
    sy: Scalar[dtype],
    si: Scalar[dtype],
    code: Int,
) -> Scalar[dtype] where dtype.is_floating_point():
    """SciPy's Lance-Williams updates, `d(i, x u y)`."""
    if code == _SINGLE:
        return min(d_xi, d_yi)
    if code == _COMPLETE:
        return max(d_xi, d_yi)
    if code == _AVERAGE:
        return (sx * d_xi + sy * d_yi) / (sx + sy)
    if code == _WEIGHTED:
        return Scalar[dtype](0.5) * (d_xi + d_yi)
    if code == _CENTROID:
        return _sqrt(
            (
                (sx * d_xi * d_xi + sy * d_yi * d_yi)
                - (sx * sy * d_xy * d_xy) / (sx + sy)
            )
            / (sx + sy)
        )
    if code == _MEDIAN:
        return _sqrt(
            Scalar[dtype](0.5) * (d_xi * d_xi + d_yi * d_yi)
            - Scalar[dtype](0.25) * d_xy * d_xy
        )
    var t = Scalar[dtype](1) / (sx + sy + si)
    return _sqrt(
        (si + sx) * t * d_xi * d_xi
        + (si + sy) * t * d_yi * d_yi
        - si * t * d_xy * d_xy
    )


def _merges[
    dtype: DType, gpu: Bool
](var dist: Dynamic[dtype, 2], code: Int) raises -> List[
    Float64
] where dtype.is_floating_point():
    """The `n - 1` merges of the primitive algorithm on the square distance
    matrix, flattened `(kept slot, dropped slot, distance)` per step; the
    matrix is updated in place on its device."""
    var ctx = dist.context()
    var n = dist.dim[0]()
    var active = Dynamic[DType.int32, 1](row_major(_dyn_shape[1](n)), ctx)
    var sizes = Dynamic[dtype, 1](row_major(_dyn_shape[1](n)), ctx)
    var rowval = Dynamic[dtype, 1](row_major(_dyn_shape[1](n)), ctx)
    var rowarg = Dynamic[DType.int32, 1](row_major(_dyn_shape[1](n)), ctx)
    var pick = Dynamic[dtype, 1](row_major(_dyn_shape[1](3)), ctx)
    var dp = dist.tile()
    var ap = active.tile()
    var sp = sizes.tile()
    var vp = rowval.tile()
    var rp = rowarg.tile()
    var pp = pick.tile()

    @always_inline
    def start[w: Int, alignment: Int = 1](coord: Coord) {var ap, var sp}:
        var i = coord_to_index_list(coord)[0]
        ap.ptr[unsafe_offset=i] = Int32(1)
        sp.ptr[unsafe_offset=i] = Scalar[dtype](1)

    elementwise[simd_width=1, target=_target[gpu]()](start, Coord(n), ctx)
    ctx.synchronize()
    var out = List[Float64](capacity=3 * (n - 1))
    for _ in range(n - 1):

        @always_inline
        def rows[
            w: Int, alignment: Int = 1
        ](coord: Coord) {var dp, var ap, var vp, var rp, var n}:
            var i = coord_to_index_list(coord)[0]
            var best = Scalar[dtype].MAX
            var arg = -1
            if ap.ptr[unsafe_offset=i] != 0:
                for j in range(i + 1, n):
                    if ap.ptr[unsafe_offset=j] != 0:
                        var d = dp.ptr[unsafe_offset=i * n + j]
                        if d < best:
                            best = d
                            arg = j
            vp.ptr[unsafe_offset=i] = best
            rp.ptr[unsafe_offset=i] = Int32(arg)

        elementwise[simd_width=1, target=_target[gpu]()](rows, Coord(n), ctx)
        ctx.synchronize()

        @always_inline
        def least[
            w: Int, alignment: Int = 1
        ](coord: Coord) {var vp, var rp, var pp, var n}:
            var best = Scalar[dtype].MAX
            var bi = 0
            var bj = 0
            for i in range(n):
                if (
                    rp.ptr[unsafe_offset=i] >= 0
                    and vp.ptr[unsafe_offset=i] < best
                ):
                    best = vp.ptr[unsafe_offset=i]
                    bi = i
                    bj = Int(rp.ptr[unsafe_offset=i])
            pp.ptr[unsafe_offset=0] = Scalar[dtype](bi)
            pp.ptr[unsafe_offset=1] = Scalar[dtype](bj)
            pp.ptr[unsafe_offset=2] = best

        elementwise[simd_width=1, target=_target[gpu]()](least, Coord(1), ctx)
        ctx.synchronize()
        var chosen = pick.to_host()
        var i = Int(chosen[0])
        var j = Int(chosen[1])
        var dij = chosen[2]
        out.append(Float64(i))
        out.append(Float64(j))
        out.append(Float64(dij))

        @always_inline
        def update[
            w: Int, alignment: Int = 1
        ](coord: Coord) {
            var dp, var ap, var sp, var n, var i, var j, var dij, var code
        }:
            var k = coord_to_index_list(coord)[0]
            if ap.ptr[unsafe_offset=k] == 0 or k == i or k == j:
                return
            var d = _update_distance[dtype](
                dp.ptr[unsafe_offset=i * n + k],
                dp.ptr[unsafe_offset=j * n + k],
                dij,
                sp.ptr[unsafe_offset=i],
                sp.ptr[unsafe_offset=j],
                sp.ptr[unsafe_offset=k],
                code,
            )
            dp.ptr[unsafe_offset=i * n + k] = d
            dp.ptr[unsafe_offset=k * n + i] = d

        elementwise[simd_width=1, target=_target[gpu]()](update, Coord(n), ctx)
        ctx.synchronize()

        @always_inline
        def retire[
            w: Int, alignment: Int = 1
        ](coord: Coord) {var ap, var sp, var i, var j}:
            sp.ptr[unsafe_offset=i] = (
                sp.ptr[unsafe_offset=i] + sp.ptr[unsafe_offset=j]
            )
            ap.ptr[unsafe_offset=j] = Int32(0)

        elementwise[simd_width=1, target=_target[gpu]()](retire, Coord(1), ctx)
        ctx.synchronize()
    _ = active^
    _ = sizes^
    _ = rowval^
    _ = rowarg^
    _ = pick^
    _ = dist^
    return out^


def _find(mut parent: List[Int], x: Int) -> Int:
    var root = x
    while parent[root] != root:
        root = parent[root]
    var p = x
    while parent[p] != root:
        var nxt = parent[p]
        parent[p] = root
        p = nxt
    return root


def _linkage_rows(merges: List[Float64], n: Int, code: Int) -> List[Float64]:
    """SciPy's output rows from the merge list: for the reducible methods a
    stable sort by distance and the union-find relabeling of `label`; for
    `centroid`/`median` merge order with `n + k` numbering."""
    var z = List[Float64](capacity=4 * (n - 1))
    if code == _CENTROID or code == _MEDIAN:
        var ids = List[Int](capacity=n)
        var size = List[Int](capacity=n)
        for s in range(n):
            ids.append(s)
            size.append(1)
        for k in range(n - 1):
            var i = Int(merges[3 * k])
            var j = Int(merges[3 * k + 1])
            var a = min(ids[i], ids[j])
            var b = max(ids[i], ids[j])
            z.append(Float64(a))
            z.append(Float64(b))
            z.append(merges[3 * k + 2])
            size[i] = size[i] + size[j]
            z.append(Float64(size[i]))
            ids[i] = n + k
        return z^
    var order = List[Int](capacity=n - 1)
    for k in range(n - 1):
        order.append(k)

    def by_distance(a: Int, b: Int) {imm} -> Bool:
        var da = merges[3 * a + 2]
        var db = merges[3 * b + 2]
        return da < db or (da == db and a < b)

    _std_sort(order, by_distance)
    var parent = List[Int](capacity=2 * n - 1)
    var size = List[Int](capacity=2 * n - 1)
    for s in range(2 * n - 1):
        parent.append(s)
        size.append(1)
    var next_label = n
    for r in range(n - 1):
        var k = order[r]
        var x = _find(parent, Int(merges[3 * k]))
        var y = _find(parent, Int(merges[3 * k + 1]))
        z.append(Float64(min(x, y)))
        z.append(Float64(max(x, y)))
        z.append(merges[3 * k + 2])
        parent[x] = next_label
        parent[y] = next_label
        size[next_label] = size[x] + size[y]
        z.append(Float64(size[next_label]))
        next_label += 1
    return z^


def _linkage_from_square[
    dtype: DType, gpu: Bool
](var square: Dynamic[dtype, 2], code: Int) raises -> Dynamic[
    dtype, 2
] where dtype.is_floating_point():
    var n = square.dim[0]()
    var ctx = square.context()
    var merges = _merges[dtype, gpu](square^, code)
    var rows = _linkage_rows(merges, n, code)
    var values = List[Scalar[dtype]](capacity=len(rows))
    for v in rows:
        values.append(Scalar[dtype](v))
    var flat = asarray(values^, ctx)
    return _same_order(flat, row_major(_dyn_shape[2](n - 1, 4)))


def linkage[
    T: TensorLike, gpu: Bool = False
](y: T, method: StaticString = "single") raises -> Dynamic[T.dtype, 2] where (
    T.dtype.is_floating_point() and T.LayoutType.rank == 1
):
    """Hierarchical clustering of a condensed distance matrix.
    `scipy.cluster.hierarchy.linkage(y, method)`; per this module's
    docstring.

    Parameters:
        T: The tensor type of `y`, a condensed distance vector.
        gpu: Whether the distance matrix and its updates live on the
            device.

    Args:
        y: The condensed distances, `n (n - 1) / 2` of them, as `pdist`
            returns.
        method: The linkage, SciPy's seven; `"centroid"`, `"median"` and
            `"ward"` assume Euclidean distances, as SciPy's do.

    Returns:
        SciPy's `(n - 1) x 4` linkage matrix: the two clusters merged, the
        distance between them, and the new cluster's size.

    Raises:
        On an unknown method, a length that is not a condensed vector's, or
        a device failure.
    """
    var code = _method_code(method)
    var condensed = _same_order(y, row_major(_dyn_shape[1](y.size())))
    var square = squareform[gpu=gpu](condensed)
    if square.dim[0]() < 2:
        raise Error("linkage: need at least two observations")
    return _linkage_from_square[T.dtype, gpu](square^, code)


def linkage[
    T: TensorLike, gpu: Bool = False
](
    x: T, method: StaticString = "single", metric: StaticString = "euclidean"
) raises -> Dynamic[T.dtype, 2] where (
    T.dtype.is_floating_point() and T.LayoutType.rank == 2
):
    """Hierarchical clustering of observations, `pdist` under `metric`
    first. `scipy.cluster.hierarchy.linkage(x, method, metric)`.

    Parameters:
        T: The tensor type of `x`, `n x k`.
        gpu: Whether the distances and the merges run on the device.

    Args:
        x: The observations, one per row.
        method: The linkage, SciPy's seven.
        metric: The `pdist` metric; `"centroid"`, `"median"` and `"ward"`
            require `"euclidean"`, as SciPy's do.

    Returns:
        SciPy's `(n - 1) x 4` linkage matrix.

    Raises:
        On an unknown method or metric, a non-Euclidean metric for the
        centroid-based methods, or a device failure.
    """
    var code = _method_code(method)
    if (
        code == _CENTROID or code == _MEDIAN or code == _WARD
    ) and metric != "euclidean":
        raise Error(
            "linkage: method '", method, "' requires the euclidean metric"
        )
    var condensed = pdist[gpu=gpu](x, metric)
    var square = squareform[gpu=gpu](condensed)
    if square.dim[0]() < 2:
        raise Error("linkage: need at least two observations")
    return _linkage_from_square[T.dtype, gpu](square^, code)


def _host_z[
    T: TensorLike
](z: T) raises -> List[Float64] where T.LayoutType.rank == 2:
    var values = z.to_host()
    var out = List[Float64](capacity=len(values))
    for i in range(len(values)):
        out.append(Float64(values[i]))
    return out^


def _inconsistent_rows(z: List[Float64], n: Int, depth: Int) -> List[Float64]:
    """SciPy's `inconsistent`: per node, the mean, standard deviation and
    count of the link heights within `depth` levels, and the coefficient."""
    var r = List[Float64](length=4 * (n - 1), fill=0.0)
    for i in range(n - 1):
        var stack = List[Int]()
        stack.append(i)
        var visited = List[Bool](length=2 * n - 1, fill=False)
        var count = 0
        var total = 0.0
        var squares = 0.0
        while len(stack) > 0:
            var k = len(stack) - 1
            var root = stack[k]
            if k < depth - 1:
                var lc = Int(z[4 * root])
                if lc >= n and not visited[lc]:
                    visited[lc] = True
                    stack.append(lc - n)
                    continue
                var rc = Int(z[4 * root + 1])
                if rc >= n and not visited[rc]:
                    visited[rc] = True
                    stack.append(rc - n)
                    continue
            var d = z[4 * root + 2]
            count += 1
            total += d
            squares += d * d
            _ = stack.pop()
        r[4 * i] = total / Float64(count)
        r[4 * i + 2] = Float64(count)
        var var_: Float64
        if count < 2:
            var_ = (squares - total * total) / Float64(count)
        else:
            var_ = (squares - total * total / Float64(count)) / Float64(
                count - 1
            )
        if var_ > 0:
            var s = _sqrt(var_)
            r[4 * i + 1] = s
            r[4 * i + 3] = (z[4 * i + 2] - r[4 * i]) / s
    return r^


def inconsistent[
    T: TensorLike
](z: T, d: Int = 2) raises -> Dynamic[T.dtype, 2] where (
    T.dtype.is_floating_point() and T.LayoutType.rank == 2
):
    """The inconsistency statistics of a linkage matrix.
    `scipy.cluster.hierarchy.inconsistent(Z, d)`: per row, the mean,
    standard deviation and count of the link heights within `d` levels
    below it, and `(height - mean) / std`.

    Parameters:
        T: The tensor type of `z`, `(n - 1) x 4`.

    Args:
        z: The linkage matrix.
        d: The number of levels, at least 1.

    Returns:
        The `(n - 1) x 4` statistics, on `z`'s device.

    Raises:
        If `d < 1`, `z` is not `(n - 1) x 4`, or a copy fails.
    """
    if d < 1:
        raise Error("inconsistent: d must be at least 1")
    if z.dim_at(1) != 4:
        raise Error("inconsistent: a linkage matrix has 4 columns")
    var n = z.dim_at(0) + 1
    var rows = _inconsistent_rows(_host_z(z), n, d)
    var values = List[Scalar[T.dtype]](capacity=len(rows))
    for v in rows:
        values.append(Scalar[T.dtype](v))
    return _same_order(
        asarray(values^, z.context()), row_major(_dyn_shape[2](n - 1, 4))
    )


def _max_per_node(
    z: List[Float64], values: List[Float64], stride: Int, field: Int, n: Int
) -> List[Float64]:
    """SciPy's `get_max_dist_for_each_cluster` / `get_max_Rfield`: each
    node's field maximized over its subtree."""
    var out = List[Float64](length=n - 1, fill=0.0)
    var visited = List[Bool](length=2 * n - 1, fill=False)
    var stack = List[Int]()
    stack.append(2 * n - 2)
    while len(stack) > 0:
        var root = stack[len(stack) - 1] - n
        var lc = Int(z[4 * root])
        var rc = Int(z[4 * root + 1])
        if lc >= n and not visited[lc]:
            visited[lc] = True
            stack.append(lc)
            continue
        if rc >= n and not visited[rc]:
            visited[rc] = True
            stack.append(rc)
            continue
        var best = values[stride * root + field]
        if lc >= n:
            best = max(best, out[lc - n])
        if rc >= n:
            best = max(best, out[rc - n])
        out[root] = best
        _ = stack.pop()
    return out^


def _monocrit(
    z: List[Float64], mc: List[Float64], cutoff: Float64, n: Int
) -> List[Int]:
    """SciPy's `cluster_monocrit`: flat clusters numbered in depth-first
    order from the root."""
    var t = List[Int](length=n, fill=0)
    var visited = List[Bool](length=2 * n - 1, fill=False)
    var stack = List[Int]()
    stack.append(2 * n - 2)
    var n_cluster = 0
    var leader = -1
    while len(stack) > 0:
        var root = stack[len(stack) - 1] - n
        var lc = Int(z[4 * root])
        var rc = Int(z[4 * root + 1])
        if leader == -1 and mc[root] <= cutoff:
            leader = root
            n_cluster += 1
        if lc >= n and not visited[lc]:
            visited[lc] = True
            stack.append(lc)
            continue
        if rc >= n and not visited[rc]:
            visited[rc] = True
            stack.append(rc)
            continue
        if lc < n:
            if leader == -1:
                n_cluster += 1
            t[lc] = n_cluster
        if rc < n:
            if leader == -1:
                n_cluster += 1
            t[rc] = n_cluster
        if leader == root:
            leader = -1
        _ = stack.pop()
    return t^


def _maxclust(
    z: List[Float64], mc: List[Float64], n: Int, max_nc: Int
) -> List[Int]:
    """SciPy's `cluster_maxclust_monocrit`: the smallest threshold, found by
    bisection over the nodes, whose flat clusters number at most `max_nc`."""
    if max_nc >= n:
        var t = List[Int](capacity=n)
        for i in range(n):
            t.append(i + 1)
        return t^
    var lower = -1
    var upper = n - 1
    while upper - lower > 1:
        var i = (lower + upper) >> 1
        var thresh = mc[i]
        var visited = List[Bool](length=2 * n - 1, fill=False)
        var nc = 0
        var stack = List[Int]()
        stack.append(2 * n - 2)
        while len(stack) > 0:
            var root = stack[len(stack) - 1] - n
            var lc = Int(z[4 * root])
            var rc = Int(z[4 * root + 1])
            if mc[root] <= thresh:
                nc += 1
                if nc > max_nc:
                    break
                _ = stack.pop()
                visited[lc] = True
                visited[rc] = True
                continue
            if not visited[lc]:
                visited[lc] = True
                if lc >= n:
                    stack.append(lc)
                    continue
                else:
                    nc += 1
                    if nc > max_nc:
                        break
            if not visited[rc]:
                visited[rc] = True
                if rc >= n:
                    stack.append(rc)
                    continue
                else:
                    nc += 1
                    if nc > max_nc:
                        break
            _ = stack.pop()
        if nc > max_nc:
            lower = i
        else:
            upper = i
    return _monocrit(z, mc, mc[upper], n)


def fcluster[
    T: TensorLike
](
    z: T, t: Float64, criterion: StaticString = "inconsistent", depth: Int = 2
) raises -> Dynamic[DType.int32, 1] where (
    T.dtype.is_floating_point() and T.LayoutType.rank == 2
):
    """Flat clusters from a linkage matrix. `scipy.cluster.hierarchy.fcluster(Z,
    t, criterion, depth)`, numbered as SciPy numbers them.

    Parameters:
        T: The tensor type of `z`, `(n - 1) x 4`.

    Args:
        z: The linkage matrix.
        t: The threshold: the largest cophenetic distance within a cluster
            (`"distance"`), the most clusters (`"maxclust"`), or the largest
            inconsistency coefficient (`"inconsistent"`, SciPy's default).
        criterion: `"inconsistent"`, `"distance"` or `"maxclust"`.
        depth: The levels `"inconsistent"` looks down.

    Returns:
        Each observation's cluster, `1 ..`, on `z`'s device.

    Raises:
        On an unknown criterion, a malformed `z`, or a copy failure.
    """
    if z.dim_at(1) != 4:
        raise Error("fcluster: a linkage matrix has 4 columns")
    var n = z.dim_at(0) + 1
    var host = _host_z(z)
    var labels: List[Int]
    if criterion == "distance":
        labels = _monocrit(host, _max_per_node(host, host, 4, 2, n), t, n)
    elif criterion == "maxclust":
        labels = _maxclust(host, _max_per_node(host, host, 4, 2, n), n, Int(t))
    elif criterion == "inconsistent":
        var r = _inconsistent_rows(host, n, depth)
        labels = _monocrit(host, _max_per_node(host, r, 4, 3, n), t, n)
    else:
        raise Error(
            (
                "fcluster: criterion must be 'inconsistent', 'distance' or"
                " 'maxclust', got '"
            ),
            criterion,
            "'",
        )
    var values = List[Scalar[DType.int32]](capacity=n)
    for v in labels:
        values.append(Int32(v))
    return asarray(values^, z.context())
