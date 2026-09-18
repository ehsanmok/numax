"""Interpolation over `numax.core.array.Tensor`: NumPy's `interp`, the
legacy `numpy.poly*` family, and polynomial evaluation over a tensor of
query points.

## Two coefficient orders, both NumPy's

`horner` takes coefficients **ascending** (`c[i]` multiplies `x^i`), which
is `numpy.polynomial.polynomial.polyval`'s order and the modern one.
`polyval`, `polyder`, `polyint`, `roots` and `polyfit` take and return
them **descending** (`p[0]` is the highest power), which is the legacy
`numpy.poly*` order. Both ship because both are NumPy, and neither order
is a wrapper's arbitrary choice: a caller porting `numpy.polyfit` output
into `numpy.polyval` needs the descending pair to agree with each other,
and `scipy.linalg.companion` -- which `roots` is built on -- is descending
too. Each docstring names its order, and `polyval` says how to get the
other one.

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

from ..core.tensorlike import TensorLike, View, dim, is_row_major
from ..core.array import Static, vander
from ..linalg.eigen import Eigenvalues, eigvals
from ..linalg.qr import lstsq
from ..linalg.special_matrices import companion

comptime _View[dtype: DType, LayoutType: TensorLayout] = TileTensor[
    dtype, LayoutType, MutAnyOrigin
]


@always_inline
def _interval[
    dtype: DType, LayoutType: TensorLayout
](knots: TileTensor[dtype, LayoutType, _], n: Int, x: Scalar[dtype]) -> Int:
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
    A: TensorLike,
    B: TensorLike,
    C: TensorLike,
    gpu: Bool = False,
](
    x: A,
    xp: B,
    fp: C,
    left: Optional[Scalar[A.dtype]] = None,
    right: Optional[Scalar[A.dtype]] = None,
) raises -> Static[A.dtype, dim[A, 0]] where (
    (A.dtype.is_floating_point() and dim[A, 0] > 0 and dim[B, 0] > 0)
    and A.LayoutType.rank == 1
    and A.LayoutType.all_dims_known
    and B.dtype == A.dtype
    and B.LayoutType.rank == 1
    and B.LayoutType.all_dims_known
    and C.dtype == A.dtype
    and C.LayoutType.rank == 1
    and C.LayoutType.all_dims_known
    and dim[C, 0] == dim[B, 0]
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
    comptime m = dim[A, 0]
    comptime n = dim[B, 0]
    var ctx = x.context()
    var out = Static[A.dtype, m]._uninitialized(ctx)
    var xs = x.view()
    var knots = xp.view_as[A.dtype]()
    var values = fp.view_as[A.dtype]()
    var ys = out.view()
    var has_left = Bool(left)
    var left_value = left.value() if left else Scalar[A.dtype](0)
    var has_right = Bool(right)
    var right_value = right.value() if right else Scalar[A.dtype](0)

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
        var y: Scalar[A.dtype]
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
            var t = (at - x0) / span if span != 0 else Scalar[A.dtype](0)
            y = y0 + t * (y1 - y0)
        ys.store[1](Coord(q), y)

    elementwise[simd_width=1, target="gpu" if gpu else "cpu"](
        lookup, Coord(m), ctx
    )
    ctx.synchronize()
    return out^


def horner[
    A: TensorLike,
    B: TensorLike,
    gpu: Bool = False,
](coefficients: A, x: B) raises -> Static[A.dtype, dim[B, 0]] where (
    (A.dtype.is_floating_point() and dim[A, 0] > 0 and dim[B, 0] > 0)
    and A.LayoutType.rank == 1
    and A.LayoutType.all_dims_known
    and B.dtype == A.dtype
    and B.LayoutType.rank == 1
    and B.LayoutType.all_dims_known
):
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
    comptime k = dim[A, 0]
    comptime m = dim[B, 0]
    var ctx = x.context()
    var out = Static[A.dtype, m]._uninitialized(ctx)
    var cs = coefficients.view()
    var xs = x.view_as[A.dtype]()
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


def polyval[
    A: TensorLike,
    B: TensorLike,
    gpu: Bool = False,
](p: A, x: B) raises -> Static[A.dtype, dim[B, 0]] where (
    (A.dtype.is_floating_point() and dim[A, 0] > 0 and dim[B, 0] > 0)
    and A.LayoutType.rank == 1
    and A.LayoutType.all_dims_known
    and B.dtype == A.dtype
    and B.LayoutType.rank == 1
    and B.LayoutType.all_dims_known
):
    """The polynomial with **descending** coefficients `p` evaluated at
    every point of `x`. `numpy.polyval`.

    `p[0]` multiplies `x ** (k - 1)` and `p[k - 1]` is the constant term,
    which is the legacy `numpy.polyval` order and the order `polyfit`
    returns and `roots` consumes. `horner` above is the same evaluation
    with the coefficients ascending; `polyval(p, x)` is
    `horner(flip(p), x)`, and either spelling works if the order is the
    one being converted.

    One `elementwise` launch of Horner's rule per lane, as `horner`'s is.
    """
    comptime k = dim[A, 0]
    comptime m = dim[B, 0]
    var ctx = x.context()
    var out = Static[A.dtype, m]._uninitialized(ctx)
    var cs = p.view()
    var xs = x.view_as[A.dtype]()
    var ys = out.view()

    @always_inline
    def evaluate[
        w: Int, alignment: Int = 1
    ](coord: Coord) {var cs, var xs, var ys}:
        var q = coord_to_index_list(coord)[0]
        var at = xs[Coord(q)]
        var total = cs[Coord(0)]
        for step in range(1, k):
            total = total * at + cs[Coord(step)]
        ys.store[1](Coord(q), total)

    elementwise[simd_width=1, target="gpu" if gpu else "cpu"](
        evaluate, Coord(m), ctx
    )
    ctx.synchronize()
    return out^


def polyder[
    T: TensorLike,
](p: T) raises -> Static[T.dtype, dim[T, 0] - 1] where (
    (T.dtype.is_floating_point() and dim[T, 0] >= 2)
    and T.LayoutType.rank == 1
    and T.LayoutType.all_dims_known
):
    """The derivative of the polynomial with descending coefficients `p`.
    `numpy.polyder`.

    A degree-`k - 1` polynomial differentiates to a degree-`k - 2` one, so
    the result is one shorter and the length change is in the type. A
    constant (`k == 1`) would differentiate to the empty polynomial rather
    than to zero, which numax has no tensor for, so `k >= 2` is a `where`
    clause: differentiate a constant and it fails to compile rather than
    returning something of a length nobody asked for.

    Higher orders compose -- `polyder(polyder(p))` is the second
    derivative -- rather than taking an `m` parameter, because each
    application changes the return type and a parameterized `m` would have
    to spell `k - m` with `m` proven below `k`.

    Host-side: `k` coefficient multiplies, which is not work worth a launch.
    """
    comptime k = dim[T, 0]
    var source = p.to_host()
    var values = List[Scalar[T.dtype]](capacity=k - 1)
    for i in range(k - 1):
        # Descending: p[i] multiplies x ** (k - 1 - i), whose derivative
        # is (k - 1 - i) * x ** (k - 2 - i).
        values.append(source[i] * Scalar[T.dtype](k - 1 - i))
    return Static[T.dtype, k - 1](p.context(), values^)


def polyint[
    T: TensorLike,
](p: T, constant: Scalar[T.dtype] = 0) raises -> Static[
    T.dtype, dim[T, 0] + 1
] where (
    (T.dtype.is_floating_point() and dim[T, 0] >= 1)
    and T.LayoutType.rank == 1
    and T.LayoutType.all_dims_known
):
    """The antiderivative of the polynomial with descending coefficients
    `p`, with integration constant `constant`. `numpy.polyint`.

    One longer than its input, and the constant lands in the last slot
    because that is the `x ** 0` position in descending order. The inverse
    of `polyder` up to that constant: `polyder(polyint(p))` is `p`.
    """
    comptime k = dim[T, 0]
    var source = p.to_host()
    var values = List[Scalar[T.dtype]](capacity=k + 1)
    for i in range(k):
        values.append(source[i] / Scalar[T.dtype](k - i))
    values.append(constant)
    return Static[T.dtype, k + 1](p.context(), values^)


def roots[
    T: TensorLike,
    gpu: Bool = False,
](p: T) raises -> Eigenvalues[T.dtype, dim[T, 0] - 1] where (
    (T.dtype.is_floating_point() and dim[T, 0] >= 2)
    and T.LayoutType.rank == 1
    and T.LayoutType.all_dims_known
):
    """The roots of the polynomial with descending coefficients `p`, real
    and complex. `numpy.roots`.

    The companion matrix's eigenvalues, which is how `numpy.roots` is
    implemented and what `numax.linalg.companion`'s docstring already
    pointed at: build the companion, take its spectrum, and those are the
    roots. So this is two existing calls under the name a caller looks
    for, not a new algorithm.

    The result is an `Eigenvalues` -- a real tensor and an imaginary one --
    for the reason that struct records: a `T.dtype`-monomorphic `Tensor`
    cannot hold a complex value. A real root has `im == 0` exactly.

    `p[0]` must be nonzero; `companion` divides by it and its docstring
    explains why trimming a leading zero is the caller's decision. The
    order is NumPy's `roots`, not its `polynomial` package's, matching
    `polyfit` and `polyval`.

    **Tier 2, and `gpu=True` does not compile**, because `eigvals` refuses
    it -- see `numax.linalg.eigvals`.
    """
    var c = companion[gpu=gpu](p)
    return eigvals[gpu=gpu](c)


def polyfit[
    A: TensorLike,
    B: TensorLike,
    deg: Int,
    gpu: Bool = False,
](x: A, y: B) raises -> Static[A.dtype, deg + 1] where (
    (A.dtype.is_floating_point() and dim[A, 0] >= deg + 1 and deg + 1 >= 1)
    and A.LayoutType.rank == 1
    and A.LayoutType.all_dims_known
    and B.dtype == A.dtype
    and B.LayoutType.rank == 1
    and B.LayoutType.all_dims_known
    and dim[B, 0] == dim[A, 0]
):
    """The degree-`deg` least-squares polynomial fit of `y` against `x`, as
    **descending** coefficients. `numpy.polyfit`.

    The Vandermonde design matrix fed to `lstsq`, which is NumPy's own
    route: `vander` with its default descending powers gives a column per
    power, and the least-squares solution of `V c = y` is the fit. Both
    pieces already existed; this is the name that joins them, and its
    output feeds `polyval` and `roots` without a reversal.

    `n >= deg + 1` is a `where` clause rather than a run-time check: fewer
    points than coefficients is an underdetermined system with a solution
    space rather than a solution, and `lstsq` requires the overdetermined
    shape anyway.

    No weights and no conditioning report. `numpy.polyfit`'s `w`, `cov`
    and its rank warning are absent; a caller wanting the conditioning
    takes `cond` of the `vander` matrix, and one wanting the
    minimum-norm answer for a rank-deficient fit spells
    `lstsq[method="svd"]` on `vander(x, deg + 1)` directly.
    """
    comptime n = dim[A, 0]
    var design = vander[cols=deg + 1](x)
    return lstsq[gpu=gpu](design, y)
