"""Pairwise distances over `Tensor`: `cdist`, `pdist` and `squareform`,
SciPy's `scipy.spatial.distance`.

**Tier 2.** One lane per pair, each summing over the features, on the
inputs' device at `gpu=True`. The metric is a run-time string, the same
for every lane, so the branch on it is uniform: `"euclidean"`,
`"sqeuclidean"`, `"cityblock"`, `"chebyshev"`, `"minkowski"` (with `p`),
`"cosine"` and `"correlation"`, each defined as SciPy defines it.

## The MAX gate

MAX has no pairwise-distance kernel. The squared Euclidean distance has a
GEMM form, `|a|^2 + |b|^2 - 2 a . b`, which would put the work in
`linalg.matmul`; it is not taken here, because the expansion cancels for
nearby points -- at distance `1e-4` between unit-norm points it leaves
about `5e-9` of relative error, where the direct difference SciPy takes
is exact to rounding. The GEMM form is the upgrade for the throughput
case (`cluster.kmeans` over many centroids) where that accuracy is not
the point. **Extend.**
"""

from layout import Coord, coord_to_index_list
from layout.tile_layout import row_major
from max.algorithm.functional import elementwise

from ..core._drive import _check_device, _notice
from ..core.tensorlike import TensorLike
from ..core.tensor import Dynamic, _dyn_shape, _same_order, asarray

comptime _EUCLIDEAN = 0
comptime _SQEUCLIDEAN = 1
comptime _CITYBLOCK = 2
comptime _CHEBYSHEV = 3
comptime _MINKOWSKI = 4
comptime _COSINE = 5
comptime _CORRELATION = 6


def _target[gpu: Bool]() -> StaticString:
    comptime if gpu:
        return "gpu"
    else:
        return "cpu"


def _metric_code(metric: StaticString) raises -> Int:
    if metric == "euclidean":
        return _EUCLIDEAN
    if metric == "sqeuclidean":
        return _SQEUCLIDEAN
    if metric == "cityblock":
        return _CITYBLOCK
    if metric == "chebyshev":
        return _CHEBYSHEV
    if metric == "minkowski":
        return _MINKOWSKI
    if metric == "cosine":
        return _COSINE
    if metric == "correlation":
        return _CORRELATION
    raise Error(
        (
            "distance: metric must be 'euclidean', 'sqeuclidean', 'cityblock',"
            " 'chebyshev', 'minkowski', 'cosine' or 'correlation', got '"
        ),
        metric,
        "'",
    )


@always_inline
def _pair_distance[
    dtype: DType
](
    a: Pointer[Scalar[dtype], ImmutAnyOrigin],
    b: Pointer[Scalar[dtype], ImmutAnyOrigin],
    k: Int,
    code: Int,
    p: Scalar[dtype],
) -> Scalar[dtype] where dtype.is_floating_point():
    """The distance between two `k`-vectors under metric `code`."""
    var zero = Scalar[dtype](0)
    var one = Scalar[dtype](1)
    if code == _COSINE or code == _CORRELATION:
        var ma = zero
        var mb = zero
        if code == _CORRELATION:
            for f in range(k):
                ma += a[unsafe_offset=f]
                mb += b[unsafe_offset=f]
            ma /= Scalar[dtype](k)
            mb /= Scalar[dtype](k)
        var dot = zero
        var na = zero
        var nb = zero
        for f in range(k):
            var u = a[unsafe_offset=f] - ma
            var v = b[unsafe_offset=f] - mb
            dot += u * v
            na += u * u
            nb += v * v
        return one - dot / (na * nb).__pow__(Scalar[dtype](0.5))
    var acc = zero
    for f in range(k):
        var d = abs(a[unsafe_offset=f] - b[unsafe_offset=f])
        if code == _EUCLIDEAN or code == _SQEUCLIDEAN:
            acc += d * d
        elif code == _CITYBLOCK:
            acc += d
        elif code == _CHEBYSHEV:
            acc = max(acc, d)
        else:
            acc += d.__pow__(p)
    if code == _EUCLIDEAN:
        return acc.__pow__(Scalar[dtype](0.5))
    if code == _MINKOWSKI:
        return acc.__pow__(one / p)
    return acc


def _as_rows[
    T: TensorLike
](a: T) raises -> Dynamic[T.dtype, 2] where T.LayoutType.rank == 2:
    return _same_order(a, row_major(_dyn_shape[2](a.dim_at(0), a.dim_at(1))))


def _cdist[
    dtype: DType, gpu: Bool
](
    xa: Dynamic[dtype, 2], xb: Dynamic[dtype, 2], code: Int, p: Float64
) raises -> Dynamic[dtype, 2] where dtype.is_floating_point():
    var ctx = xa.context()
    var ma = xa.dim[0]()
    var mb = xb.dim[0]()
    var k = xa.dim[1]()
    var out = Dynamic[dtype, 2](row_major(_dyn_shape[2](ma, mb)), ctx)
    var ap = xa.tile().ptr.unsafe_origin_cast[ImmutAnyOrigin]()
    var bp = xb.tile().ptr.unsafe_origin_cast[ImmutAnyOrigin]()
    var op = out.tile()
    var pp = Scalar[dtype](p)

    @always_inline
    def body[
        w: Int, alignment: Int = 1
    ](coord: Coord) {var ap, var bp, var op, var mb, var k, var code, var pp}:
        var e = coord_to_index_list(coord)[0]
        var i = e // mb
        var j = e % mb
        op.ptr[unsafe_offset=e] = _pair_distance[dtype](
            ap.unsafe_offset(i * k), bp.unsafe_offset(j * k), k, code, pp
        )

    elementwise[simd_width=1, target=_target[gpu]()](body, Coord(ma * mb), ctx)
    ctx.synchronize()
    return out^


def cdist[
    A: TensorLike, B: TensorLike, gpu: Bool = False
](
    xa: A, xb: B, metric: StaticString = "euclidean", p: Float64 = 2.0
) raises -> Dynamic[A.dtype, 2] where (
    A.dtype.is_floating_point()
    and A.LayoutType.rank == 2
    and B.LayoutType.rank == 2
    and B.dtype == A.dtype
):
    """The distance between every row of `xa` and every row of `xb`.
    `scipy.spatial.distance.cdist(xa, xb, metric, p=p)`.

    Parameters:
        A: The tensor type of `xa`, `ma x k`.
        B: The tensor type of `xb`, `mb x k`, the same `dtype`.
        gpu: Whether to compute on the inputs' device; a residency
            mismatch falls back to the host with a notice.

    Args:
        xa: The first set of points, one per row.
        xb: The second set of points, one per row.
        metric: The distance, per this module's docstring.
        p: The Minkowski order, read only for `"minkowski"`.

    Returns:
        The `ma x mb` distances.

    Raises:
        On an unknown metric, rows of different lengths, or a device
        failure.
    """
    var code = _metric_code(metric)
    if xa.dim_at(1) != xb.dim_at(1):
        raise Error(
            "cdist: the rows have ",
            xa.dim_at(1),
            " and ",
            xb.dim_at(1),
            " features",
        )
    var a = _as_rows(xa)
    var b = rebind_var[Dynamic[A.dtype, 2]](_as_rows(xb))
    if _check_device[A, gpu](xa):
        return _cdist[A.dtype, gpu](a, b, code, p)
    _notice[gpu]("cdist")
    var ha = asarray(a.to_host())
    var hb = asarray(b.to_host())
    return _cdist[A.dtype, False](
        _same_order(ha, row_major(_dyn_shape[2](a.dim[0](), a.dim[1]()))),
        _same_order(hb, row_major(_dyn_shape[2](b.dim[0](), b.dim[1]()))),
        code,
        p,
    )


def _pdist[
    dtype: DType, gpu: Bool
](x: Dynamic[dtype, 2], code: Int, p: Float64) raises -> Dynamic[
    dtype, 1
] where dtype.is_floating_point():
    var ctx = x.context()
    var m = x.dim[0]()
    var k = x.dim[1]()
    var count = m * (m - 1) // 2
    var out = Dynamic[dtype, 1](row_major(_dyn_shape[1](max(count, 0))), ctx)
    if count <= 0:
        return out^
    var xp = x.tile().ptr.unsafe_origin_cast[ImmutAnyOrigin]()
    var op = out.tile()
    var pp = Scalar[dtype](p)

    @always_inline
    def body[
        w: Int, alignment: Int = 1
    ](coord: Coord) {var xp, var op, var m, var k, var code, var pp}:
        # Entry `e` of the condensed form is the pair `(i, j)`, `i < j`,
        # in row-major order of the upper triangle: row `i` starts at
        # `i m - i (i + 1) / 2`.
        var e = coord_to_index_list(coord)[0]
        var i = 0
        while (i + 1) * m - (i + 1) * (i + 2) // 2 <= e:
            i += 1
        var j = e - (i * m - i * (i + 1) // 2) + i + 1
        op.ptr[unsafe_offset=e] = _pair_distance[dtype](
            xp.unsafe_offset(i * k), xp.unsafe_offset(j * k), k, code, pp
        )

    elementwise[simd_width=1, target=_target[gpu]()](body, Coord(count), ctx)
    ctx.synchronize()
    return out^


def pdist[
    T: TensorLike, gpu: Bool = False
](x: T, metric: StaticString = "euclidean", p: Float64 = 2.0) raises -> Dynamic[
    T.dtype, 1
] where (T.dtype.is_floating_point() and T.LayoutType.rank == 2):
    """The distance between every pair of rows of `x`, condensed: the
    upper triangle of the distance matrix, row by row, `m (m - 1) / 2`
    values. `scipy.spatial.distance.pdist(x, metric, p=p)`.

    Parameters:
        T: The tensor type of `x`, `m x k`.
        gpu: Whether to compute on `x`'s device; a residency mismatch
            falls back to the host with a notice.

    Args:
        x: The points, one per row.
        metric: The distance, per this module's docstring.
        p: The Minkowski order, read only for `"minkowski"`.

    Returns:
        The condensed distances; `squareform` makes the matrix.

    Raises:
        On an unknown metric or a device failure.
    """
    var code = _metric_code(metric)
    var a = _as_rows(x)
    if _check_device[T, gpu](x):
        return _pdist[T.dtype, gpu](a, code, p)
    _notice[gpu]("pdist")
    var h = asarray(a.to_host())
    return _pdist[T.dtype, False](
        _same_order(h, row_major(_dyn_shape[2](a.dim[0](), a.dim[1]()))),
        code,
        p,
    )


def squareform[
    T: TensorLike, gpu: Bool = False
](v: T) raises -> Dynamic[T.dtype, 2] where T.LayoutType.rank == 1:
    """The symmetric distance matrix of a condensed vector, with a zero
    diagonal. `scipy.spatial.distance.squareform(v)`.

    Parameters:
        T: The tensor type of `v`, rank 1.
        gpu: Whether to expand on `v`'s device.

    Args:
        v: A condensed distance vector, `m (m - 1) / 2` long.

    Returns:
        The `m x m` matrix.

    Raises:
        If `v`'s length is not a triangular number, or a device failure.
    """
    comptime dtype = T.dtype
    var count = v.size()
    var m = 1
    while m * (m - 1) // 2 < count:
        m += 1
    if m * (m - 1) // 2 != count:
        raise Error(
            "squareform: ",
            count,
            " is not the length of a condensed distance vector",
        )
    var ctx = v.context()
    var src = _same_order(v, row_major(_dyn_shape[1](count)))
    var out = Dynamic[dtype, 2](row_major(_dyn_shape[2](m, m)), ctx)
    var sp = src.tile().ptr.unsafe_origin_cast[MutAnyOrigin]()
    var op = out.tile()

    @always_inline
    def body[w: Int, alignment: Int = 1](coord: Coord) {var sp, var op, var m}:
        var e = coord_to_index_list(coord)[0]
        var r = e // m
        var c = e % m
        if r == c:
            op.ptr[unsafe_offset=e] = Scalar[dtype](0)
        else:
            var i = min(r, c)
            var j = max(r, c)
            op.ptr[unsafe_offset=e] = sp[
                unsafe_offset=i * m - i * (i + 1) // 2 + j - i - 1
            ]

    if _check_device[T, gpu](v):
        elementwise[simd_width=1, target=_target[gpu]()](
            body, Coord(m * m), ctx
        )
    else:
        _notice[gpu]("squareform")
        return squareform(asarray(src.to_host()))
    ctx.synchronize()
    _ = src^
    return out^


def squareform[
    T: TensorLike, gpu: Bool = False
](d: T) raises -> Dynamic[T.dtype, 1] where T.LayoutType.rank == 2:
    """The condensed vector of a square distance matrix: its upper
    triangle, row by row. `scipy.spatial.distance.squareform(d)`; the
    diagonal and the lower triangle are not read, as SciPy's
    `checks=False` does not read them.

    Parameters:
        T: The tensor type of `d`, square.
        gpu: Whether to condense on `d`'s device.

    Args:
        d: The `m x m` distance matrix.

    Returns:
        The `m (m - 1) / 2` condensed distances.

    Raises:
        If `d` is not square, or a device failure.
    """
    comptime dtype = T.dtype
    var m = d.dim_at(0)
    if d.dim_at(1) != m:
        raise Error("squareform: the matrix is ", m, " x ", d.dim_at(1))
    var count = m * (m - 1) // 2
    var ctx = d.context()
    var src = _same_order(d, row_major(_dyn_shape[2](m, m)))
    var out = Dynamic[dtype, 1](row_major(_dyn_shape[1](max(count, 0))), ctx)
    if count <= 0:
        return out^
    var sp = src.tile().ptr.unsafe_origin_cast[MutAnyOrigin]()
    var op = out.tile()

    @always_inline
    def body[w: Int, alignment: Int = 1](coord: Coord) {var sp, var op, var m}:
        var e = coord_to_index_list(coord)[0]
        var i = 0
        while (i + 1) * m - (i + 1) * (i + 2) // 2 <= e:
            i += 1
        var j = e - (i * m - i * (i + 1) // 2) + i + 1
        op.ptr[unsafe_offset=e] = sp[unsafe_offset=i * m + j]

    if _check_device[T, gpu](d):
        elementwise[simd_width=1, target=_target[gpu]()](
            body, Coord(count), ctx
        )
    else:
        _notice[gpu]("squareform")
        var h = asarray(src.to_host())
        return squareform(_same_order(h, row_major(_dyn_shape[2](m, m))))
    ctx.synchronize()
    _ = src^
    return out^
