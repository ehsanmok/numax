"""Interpolation over `numax.core.array.Tensor`: NumPy's `interp` and
polynomial evaluation over a tensor of query points.

**This module is tier 2.** Each routine is one `elementwise` launch over
the query points -- host-driven, device-resident, `Plain`-only -- and
inside that launch a lane may branch on data: the interval search is a
bisection whose trip count is `log2(n)` and whose path depends on the
query. That is exactly what the `FloatLike` tier cannot do, and why
`numax.interpolate.array`'s spline scans every interval and blends instead.
The two tiers are cross-referenced rather than ranked: that one is for a
handful of knots inside a per-lane kernel, this one for a device buffer of
samples.

## The MAX gate

MAX ships no interpolation at arbitrary query points. What it has is image
resampling: `nn.resize_linear` and `nn.resize_nearest_neighbor` (host-only)
and `nn.resize_bicubic` (`target` and a `DeviceContext`) each take an
NCHW image and a *scale factor* and produce the whole resampled image on a
fixed output grid. `numpy.interp(x, xp, fp)` and the `scipy.interpolate`
objects answer a different question -- the value at *these* points, on a
grid that need not be uniform -- and no scale factor expresses it. So this
module is an **extend**: numax's kernels, in MAX's idiom, with nothing to
delegate to.

## Borrowed, not consumed

Unlike the transforms in `numax.fft`, these take their tensors `mut` and
leave them alone: a grid is queried many times, and consuming it on the
first call would make the second one a copy. `mut` because `view()` needs
it, so a caller holds the tensors in `var` bindings -- which is how a grid
that is reused is held anyway.
"""

from layout import Coord, TileTensor, coord_to_index_list
from layout.tile_layout import TensorLayout
from max.algorithm.functional import elementwise

from ..core.array import Static

comptime _View[dtype: DType, LayoutType: TensorLayout] = TileTensor[
    dtype, LayoutType, MutAnyOrigin
]


@always_inline
def _interval[
    dtype: DType, LayoutType: TensorLayout
](knots: _View[dtype, LayoutType], n: Int, x: Scalar[dtype]) -> Int:
    """The `i` in `[0, n - 2]` with `knots[i] <= x < knots[i + 1]`, clamped
    to the end intervals for an `x` outside the knots.

    Bisection, `log2(n)` steps, on ascending knots. Runs inside a kernel
    body, one lane per query; the branch on `knots[mid] <= x` is the
    data-dependent step that makes this tier 2.
    """
    var lo = 0
    var hi = n - 1
    while hi - lo > 1:
        var mid = (lo + hi) // 2
        if knots[Coord(mid)] <= x:
            lo = mid
        else:
            hi = mid
    return lo


def interp[
    dtype: DType, m: Int, n: Int, gpu: Bool = False
](
    mut x: Static[dtype, m],
    mut xp: Static[dtype, n],
    mut fp: Static[dtype, n],
    left: Optional[Scalar[dtype]] = None,
    right: Optional[Scalar[dtype]] = None,
) raises -> Static[dtype, m] where (
    dtype.is_floating_point() and m > 0 and n > 0
):
    """One-dimensional linear interpolation of the samples `(xp, fp)` at the
    points `x`. `numpy.interp(x, xp, fp, left, right)`.

    `xp` must be ascending, which is not checked -- NumPy does not check it
    either, and the bisection returns *an* interval on unsorted knots
    rather than raising. Outside `[xp[0], xp[n-1]]` the value is `left` and
    `right`, defaulting to `fp[0]` and `fp[n-1]` as NumPy's do; a point
    exactly on `xp[0]` is inside, so it takes `fp[0]` rather than `left`.
    `period` is not provided.

    One launch over the `m` queries: bisection to the interval, then the
    linear blend, so a query costs `O(log n)` and the grid is read in
    place. `numax.interpolate.array` has no `interp` -- an `Array` of knots
    small enough for registers is a spline's or a polynomial's, not a
    lookup table's.
    """
    var ctx = x.context()
    var out = Static[dtype, m]._uninitialized(ctx)
    var xs = x.view()
    var knots = xp.view()
    var values = fp.view()
    var ys = out.view()
    var has_left = Bool(left)
    var left_value = left.value() if left else Scalar[dtype](0)
    var has_right = Bool(right)
    var right_value = right.value() if right else Scalar[dtype](0)

    @always_inline
    def lookup[
        w: Int, alignment: Int = 1
    ](coord: Coord) {
        var xs,
        var knots,
        var values,
        var ys,
        var has_left,
        var left_value,
        var has_right,
        var right_value,
    }:
        var q = coord_to_index_list(coord)[0]
        var at = xs[Coord(q)]
        var y: Scalar[dtype]
        if at < knots[Coord(0)]:
            y = left_value if has_left else values[Coord(0)]
        elif at > knots[Coord(n - 1)]:
            y = right_value if has_right else values[Coord(n - 1)]
        else:
            var i = _interval(knots, n, at)
            var x0 = knots[Coord(i)]
            var x1 = knots[Coord(i + 1)]
            var y0 = values[Coord(i)]
            var y1 = values[Coord(i + 1)]
            # `x1 == x0` only at `n == 1`, where the clamps above have
            # already answered; the guard keeps the lane finite regardless.
            var span = x1 - x0
            var t = (at - x0) / span if span != 0 else Scalar[dtype](0)
            y = y0 + t * (y1 - y0)
        ys.store[1](Coord(q), y)

    elementwise[simd_width=1, target="gpu" if gpu else "cpu"](
        lookup, Coord(m), ctx
    )
    ctx.synchronize()
    return out^


def horner[
    dtype: DType, k: Int, m: Int, gpu: Bool = False
](mut coefficients: Static[dtype, k], mut x: Static[dtype, m]) raises -> Static[
    dtype, m
] where (dtype.is_floating_point() and k > 0 and m > 0):
    """The polynomial with ascending `coefficients` (`coefficients[i]`
    multiplies `x^i`) evaluated at every point of `x`.
    `numpy.polynomial.polynomial.polyval(x, c)`.

    Horner's rule per lane, `k - 1` multiply-adds, the coefficients read in
    place from the device buffer. The same rule as
    `numax.interpolate.array.horner`, over a tensor of points rather than
    one `FloatLike` value -- and the ascending order is NumPy's
    `polynomial` package's, not the descending order of the legacy
    `numpy.polyval`.
    """
    var ctx = x.context()
    var out = Static[dtype, m]._uninitialized(ctx)
    var cs = coefficients.view()
    var xs = x.view()
    var ys = out.view()

    @always_inline
    def evaluate[
        w: Int, alignment: Int = 1
    ](coord: Coord) {var cs, var xs, var ys}:
        var q = coord_to_index_list(coord)[0]
        var at = xs[Coord(q)]
        var total = cs[Coord(k - 1)]
        for step in range(1, k):
            total = total * at + cs[Coord(k - 1 - step)]
        ys.store[1](Coord(q), total)

    elementwise[simd_width=1, target="gpu" if gpu else "cpu"](
        evaluate, Coord(m), ctx
    )
    ctx.synchronize()
    return out^
