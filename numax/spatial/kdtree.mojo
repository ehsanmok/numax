"""A k-d tree for nearest-neighbor search: `KDTree`, SciPy's
`scipy.spatial.KDTree`, with `query` and `query_ball_point`.

**Tier 2.** The tree is built on the host -- median splits on the widest
dimension, down to `leafsize` points a leaf, an `O(n log^2 n)` sort-based
build over a host copy of the data -- and then lives on the data's
device: its nodes as five arrays (split dimension and value, the two
children, and each leaf's range) and the points permuted into leaf order.
The queries are device kernels, one lane per query point, each walking
the tree depth-first with an explicit fixed-size stack and pruning every
subtree whose splitting plane is farther than the current bound. The
distance is Euclidean.

- `query[k]` keeps the `k` best candidates in a sorted array in the lane
  and returns the `q x k` distances and original indices, nearest first.
- `query_ball_point` counts each query's points within `r` in one pass,
  lays the counts out as offsets on the host (`q + 1` integers), fills the
  indices in a second pass, and sorts each query's run, SciPy's
  multi-point `return_sorted` behavior. The result is compressed rows,
  `BallPoints`.

## The MAX gate

Nothing: MAX has no spatial index. **Extend.**
"""

from std.builtin.sort import sort as _std_sort
from std.collections import Array

from layout import Coord, coord_to_index_list
from layout.tile_layout import row_major
from max.algorithm.functional import elementwise
from max.gpu.host import DeviceContext

from ..core.tensorlike import TensorLike
from ..core.tensor import Dynamic, _dyn_shape, _same_order, asarray

comptime _STACK = 64
"""The traversal stack's depth. A median-split tree over `n` points is
`log2(n / leafsize)` deep, and the walk pushes at most one sibling per
level, so 64 covers any `n` that fits in memory."""


def _target[gpu: Bool]() -> StaticString:
    comptime if gpu:
        return "gpu"
    else:
        return "cpu"


struct KDQuery[dtype: DType](Movable):
    """What `KDTree.query` returns: SciPy's `(d, i)`, both `q x k`,
    nearest first."""

    var distances: Dynamic[Self.dtype, 2]
    """The distances to the `k` nearest points of each query."""
    var indices: Dynamic[DType.int64, 2]
    """Their rows in the tree's data."""

    def __init__(
        out self,
        var distances: Dynamic[Self.dtype, 2],
        var indices: Dynamic[DType.int64, 2],
    ):
        """Build from the two arrays.

        Args:
            distances: The distances, `q x k`.
            indices: The data rows, `q x k`.
        """
        self.distances = distances^
        self.indices = indices^


struct BallPoints(Movable):
    """What `KDTree.query_ball_point` returns: compressed rows, query `j`'s
    neighbors being `indices[offsets[j] : offsets[j + 1]]`, ascending."""

    var offsets: Dynamic[DType.int64, 1]
    """`q + 1` offsets into `indices`."""
    var indices: Dynamic[DType.int64, 1]
    """Every query's neighbors, run after run."""

    def __init__(
        out self,
        var offsets: Dynamic[DType.int64, 1],
        var indices: Dynamic[DType.int64, 1],
    ):
        """Build from the two arrays.

        Args:
            offsets: The run boundaries.
            indices: The neighbors.
        """
        self.offsets = offsets^
        self.indices = indices^


struct KDTree[dtype: DType, gpu: Bool = False](Movable):
    """A k-d tree over `n` points of `k` coordinates. `scipy.spatial.KDTree`;
    the module docstring has the layout and the algorithms."""

    var n: Int
    var k: Int
    var nodes: Int
    var points: Dynamic[Self.dtype, 2]
    """The data, rows permuted into leaf order."""
    var index: Dynamic[DType.int64, 1]
    """Each permuted row's original row."""
    var split_dim: Dynamic[DType.int32, 1]
    """A node's splitting coordinate; `-1` for a leaf."""
    var split_val: Dynamic[Self.dtype, 1]
    var left: Dynamic[DType.int32, 1]
    var right: Dynamic[DType.int32, 1]
    var start: Dynamic[DType.int32, 1]
    """A leaf's first permuted row."""
    var stop: Dynamic[DType.int32, 1]
    """One past a leaf's last permuted row."""

    def __init__[
        T: TensorLike
    ](out self, data: T, leafsize: Int = 10) raises where (
        T.dtype == Self.dtype
        and T.LayoutType.rank == 2
        and Self.dtype.is_floating_point()
    ):
        """Build the tree over the rows of `data`.

        Parameters:
            T: The tensor type of `data`, `n x k`, of the tree's `dtype`.

        Args:
            data: The points, one per row.
            leafsize: The most points a leaf holds, SciPy's default `10`.

        Raises:
            If `data` is empty, `leafsize < 1`, or a device copy fails.
        """
        var n = data.dim_at(0)
        var k = data.dim_at(1)
        if n == 0 or k == 0:
            raise Error("KDTree: the data is empty")
        if leafsize < 1:
            raise Error("KDTree: leafsize must be at least 1")
        var host = data.to_host()
        var values = List[Float64](capacity=n * k)
        for i in range(n * k):
            values.append(Float64(host[i]))
        var order = List[Int](capacity=n)
        for i in range(n):
            order.append(i)
        var sdim = List[Int32]()
        var sval = List[Float64]()
        var lefts = List[Int32]()
        var rights = List[Int32]()
        var starts = List[Int32]()
        var stops = List[Int32]()
        # Node `j` covers `order[lo[j]:hi[j]]`; built breadth-first so a
        # node's children are appended after it.
        var los = List[Int]()
        var his = List[Int]()
        los.append(0)
        his.append(n)
        var cursor = 0
        while cursor < len(los):
            var lo = los[cursor]
            var hi = his[cursor]
            sdim.append(-1)
            sval.append(0.0)
            lefts.append(-1)
            rights.append(-1)
            starts.append(Int32(lo))
            stops.append(Int32(hi))
            if hi - lo > leafsize:
                # The widest coordinate over this node's points.
                var best = 0
                var spread = -1.0
                for d in range(k):
                    var mn = values[order[lo] * k + d]
                    var mx = mn
                    for p in range(lo + 1, hi):
                        var v = values[order[p] * k + d]
                        mn = min(mn, v)
                        mx = max(mx, v)
                    if mx - mn > spread:
                        spread = mx - mn
                        best = d
                if spread > 0.0:
                    var seg = List[Int](capacity=hi - lo)
                    for p in range(lo, hi):
                        seg.append(order[p])
                    var axis = best

                    def by_value(a: Int, b: Int) {imm} -> Bool:
                        var va = values[a * k + axis]
                        var vb = values[b * k + axis]
                        return va < vb or (va == vb and a < b)

                    _std_sort(seg, by_value)
                    for p in range(lo, hi):
                        order[p] = seg[p - lo]
                    var mid = (lo + hi) // 2
                    sdim[cursor] = Int32(best)
                    sval[cursor] = values[order[mid] * k + best]
                    lefts[cursor] = Int32(len(los))
                    los.append(lo)
                    his.append(mid)
                    rights[cursor] = Int32(len(los))
                    los.append(mid)
                    his.append(hi)
            cursor += 1
        var ctx = data.context()
        var permuted = List[Scalar[Self.dtype]](capacity=n * k)
        var original = List[Scalar[DType.int64]](capacity=n)
        for p in range(n):
            original.append(Int64(order[p]))
            for d in range(k):
                permuted.append(Scalar[Self.dtype](values[order[p] * k + d]))
        self.n = n
        self.k = k
        self.nodes = len(sdim)
        self.points = _same_order(
            asarray(permuted^, ctx), row_major(_dyn_shape[2](n, k))
        )
        self.index = asarray(original^, ctx)
        self.split_dim = asarray(sdim^, ctx)
        var svals = List[Scalar[Self.dtype]](capacity=len(sval))
        for v in sval:
            svals.append(Scalar[Self.dtype](v))
        self.split_val = asarray(svals^, ctx)
        self.left = asarray(lefts^, ctx)
        self.right = asarray(rights^, ctx)
        self.start = asarray(starts^, ctx)
        self.stop = asarray(stops^, ctx)

    def query[
        count: Int = 1, Q: TensorLike = Dynamic[Self.dtype, 2]
    ](mut self, x: Q) raises -> KDQuery[Self.dtype] where (
        Q.dtype == Self.dtype
        and Q.LayoutType.rank == 2
        and Self.dtype.is_floating_point()
    ):
        """The `count` nearest data points to each row of `x`.
        `scipy.spatial.KDTree.query(x, k=count)`.

        Parameters:
            count: How many neighbors, SciPy's `k`; at most 64, which is
                the in-lane candidate array's size.
            Q: The tensor type of `x`, `q x k`.

        Args:
            x: The query points, one per row, on the tree's device.

        Returns:
            A `KDQuery` of `q x count` distances and indices, nearest
            first; with fewer than `count` data points the tail holds
            `inf` and `-1`.

        Raises:
            If `x`'s rows are not `k` long, or a device operation fails.
        """
        comptime assert count >= 1 and count <= 64, (
            "KDTree.query: count must be in [1, 64], the in-lane candidate"
            " array's size"
        )
        if x.dim_at(1) != self.k:
            raise Error(
                "KDTree.query: the points have ",
                x.dim_at(1),
                " coordinates, the tree ",
                self.k,
            )
        var q = x.dim_at(0)
        var ctx = self.points.context()
        var xs = rebind_var[Dynamic[Self.dtype, 2]](
            _same_order(x, row_major(_dyn_shape[2](q, self.k)))
        )
        var dist = Dynamic[Self.dtype, 2](
            row_major(_dyn_shape[2](q, count)), ctx
        )
        var idx = Dynamic[DType.int64, 2](
            row_major(_dyn_shape[2](q, count)), ctx
        )
        var xp = xs.tile()
        var pp = self.points.tile()
        var ip = self.index.tile()
        var dp = self.split_dim.tile()
        var vp = self.split_val.tile()
        var lp = self.left.tile()
        var rp = self.right.tile()
        var sp = self.start.tile()
        var ep = self.stop.tile()
        var op = dist.tile()
        var np = idx.tile()
        var kk = self.k

        @always_inline
        def body[
            w: Int, alignment: Int = 1
        ](coord: Coord) {
            var xp,
            var pp,
            var ip,
            var dp,
            var vp,
            var lp,
            var rp,
            var sp,
            var ep,
            var op,
            var np,
            var kk,
        }:
            var qi = coord_to_index_list(coord)[0]
            var inf = Scalar[Self.dtype].MAX
            var best = Array[Scalar[Self.dtype], count](fill=inf)
            var who = Array[Int32, count](fill=-1)
            var stack = Array[Int32, _STACK](fill=0)
            var bound = Array[Scalar[Self.dtype], _STACK](fill=0)
            var top = 1
            while top > 0:
                top -= 1
                var node = Int(stack[top])
                if bound[top] > best[count - 1]:
                    continue
                var d = Int(dp.ptr[unsafe_offset=node])
                if d < 0:
                    for p in range(
                        Int(sp.ptr[unsafe_offset=node]),
                        Int(ep.ptr[unsafe_offset=node]),
                    ):
                        var acc = Scalar[Self.dtype](0)
                        for c in range(kk):
                            var diff = (
                                xp.ptr[unsafe_offset=qi * kk + c]
                                - pp.ptr[unsafe_offset=p * kk + c]
                            )
                            acc += diff * diff
                        if acc < best[count - 1]:
                            # Insert, keeping `best` ascending.
                            var slot = count - 1
                            while slot > 0 and best[slot - 1] > acc:
                                best[slot] = best[slot - 1]
                                who[slot] = who[slot - 1]
                                slot -= 1
                            best[slot] = acc
                            who[slot] = Int32(p)
                else:
                    var diff = (
                        xp.ptr[unsafe_offset=qi * kk + d]
                        - vp.ptr[unsafe_offset=node]
                    )
                    var near = (
                        lp.ptr[unsafe_offset=node] if diff
                        < 0 else rp.ptr[unsafe_offset=node]
                    )
                    var far = (
                        rp.ptr[unsafe_offset=node] if diff
                        < 0 else lp.ptr[unsafe_offset=node]
                    )
                    var here = bound[top]
                    var plane = diff * diff
                    if top + 2 <= _STACK:
                        stack[top] = far
                        bound[top] = max(here, plane)
                        stack[top + 1] = near
                        bound[top + 1] = here
                        top += 2
            for j in range(count):
                var p = Int(who[j])
                op.ptr[unsafe_offset=qi * count + j] = (
                    best[j].__pow__(Scalar[Self.dtype](0.5)) if p >= 0 else inf
                )
                np.ptr[unsafe_offset=qi * count + j] = ip.ptr[
                    unsafe_offset=p
                ] if p >= 0 else Int64(-1)

        elementwise[simd_width=1, target=_target[Self.gpu]()](
            body, Coord(q), ctx
        )
        ctx.synchronize()
        _ = xs^
        return KDQuery[Self.dtype](dist^, idx^)

    def query_ball_point[
        Q: TensorLike
    ](mut self, x: Q, r: Float64) raises -> BallPoints where (
        Q.dtype == Self.dtype
        and Q.LayoutType.rank == 2
        and Self.dtype.is_floating_point()
    ):
        """Every data point within `r` of each row of `x`, as compressed
        rows with each run ascending.
        `scipy.spatial.KDTree.query_ball_point(x, r, return_sorted=True)`.

        Parameters:
            Q: The tensor type of `x`, `q x k`.

        Args:
            x: The query points, one per row, on the tree's device.
            r: The radius; a point at exactly `r` is included, as SciPy's
                is.

        Returns:
            A `BallPoints` with `q + 1` offsets and the neighbors.

        Raises:
            If `x`'s rows are not `k` long, or a device operation fails.
        """
        if x.dim_at(1) != self.k:
            raise Error(
                "KDTree.query_ball_point: the points have ",
                x.dim_at(1),
                " coordinates, the tree ",
                self.k,
            )
        var q = x.dim_at(0)
        var ctx = self.points.context()
        var xs = rebind_var[Dynamic[Self.dtype, 2]](
            _same_order(x, row_major(_dyn_shape[2](q, self.k)))
        )
        var counts = Dynamic[DType.int64, 1](row_major(_dyn_shape[1](q)), ctx)
        var r2 = Scalar[Self.dtype](r * r)
        var unused = Dynamic[DType.int64, 1](row_major(_dyn_shape[1](1)), ctx)
        self._ball[False](xs, unused, counts, r2)
        var host = counts.to_host()
        var offsets_host = List[Scalar[DType.int64]](capacity=q + 1)
        var running = Int64(0)
        offsets_host.append(running)
        for j in range(q):
            running += host[j]
            offsets_host.append(running)
        var offsets = asarray(offsets_host^, ctx)
        var indices = Dynamic[DType.int64, 1](
            row_major(_dyn_shape[1](max(Int(running), 1))), ctx
        )
        if running > 0:
            self._ball[True](xs, offsets, indices, r2)
        return BallPoints(offsets^, indices^)

    def _ball[
        fill: Bool
    ](
        mut self,
        mut xs: Dynamic[Self.dtype, 2],
        mut offsets: Dynamic[DType.int64, 1],
        mut out: Dynamic[DType.int64, 1],
        r2: Scalar[Self.dtype],
    ) raises where Self.dtype.is_floating_point():
        """One traversal per query: count the points within `r` (`fill`
        false, into `out`), or write them from `offsets` and sort the run
        (`fill` true)."""
        var q = xs.dim[0]()
        var ctx = self.points.context()
        var xp = xs.tile()
        var pp = self.points.tile()
        var ip = self.index.tile()
        var dp = self.split_dim.tile()
        var vp = self.split_val.tile()
        var lp = self.left.tile()
        var rp = self.right.tile()
        var sp = self.start.tile()
        var ep = self.stop.tile()
        var fp = offsets.tile()
        var op = out.tile()
        var kk = self.k

        @always_inline
        def body[
            w: Int, alignment: Int = 1
        ](coord: Coord) {
            var xp,
            var pp,
            var ip,
            var dp,
            var vp,
            var lp,
            var rp,
            var sp,
            var ep,
            var fp,
            var op,
            var kk,
            var r2,
        }:
            var qi = coord_to_index_list(coord)[0]
            var stack = Array[Int32, _STACK](fill=0)
            var bound = Array[Scalar[Self.dtype], _STACK](fill=0)
            var top = 1
            var found = 0
            var base = 0
            comptime if fill:
                base = Int(fp.ptr[unsafe_offset=qi])
            while top > 0:
                top -= 1
                var node = Int(stack[top])
                if bound[top] > r2:
                    continue
                var d = Int(dp.ptr[unsafe_offset=node])
                if d < 0:
                    for p in range(
                        Int(sp.ptr[unsafe_offset=node]),
                        Int(ep.ptr[unsafe_offset=node]),
                    ):
                        var acc = Scalar[Self.dtype](0)
                        for c in range(kk):
                            var diff = (
                                xp.ptr[unsafe_offset=qi * kk + c]
                                - pp.ptr[unsafe_offset=p * kk + c]
                            )
                            acc += diff * diff
                        if acc <= r2:
                            comptime if fill:
                                op.ptr[unsafe_offset=base + found] = ip.ptr[
                                    unsafe_offset=p
                                ]
                            found += 1
                else:
                    var diff = (
                        xp.ptr[unsafe_offset=qi * kk + d]
                        - vp.ptr[unsafe_offset=node]
                    )
                    var here = bound[top]
                    var plane = diff * diff
                    if top + 2 <= _STACK:
                        stack[top] = (
                            rp.ptr[unsafe_offset=node] if diff
                            < 0 else lp.ptr[unsafe_offset=node]
                        )
                        bound[top] = max(here, plane)
                        stack[top + 1] = (
                            lp.ptr[unsafe_offset=node] if diff
                            < 0 else rp.ptr[unsafe_offset=node]
                        )
                        bound[top + 1] = here
                        top += 2
            comptime if fill:
                # Insertion sort of this query's run.
                for a in range(1, found):
                    var v = op.ptr[unsafe_offset=base + a]
                    var b = a
                    while b > 0 and op.ptr[unsafe_offset=base + b - 1] > v:
                        op.ptr[unsafe_offset=base + b] = op.ptr[
                            unsafe_offset=base + b - 1
                        ]
                        b -= 1
                    op.ptr[unsafe_offset=base + b] = v
            else:
                op.ptr[unsafe_offset=qi] = Int64(found)

        elementwise[simd_width=1, target=_target[Self.gpu]()](
            body, Coord(q), ctx
        )
        ctx.synchronize()
