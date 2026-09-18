"""Chebyshev series over `numax.core.array.Tensor`: a least-squares fit to
*data* and Clenshaw evaluation at a tensor of points.
`numpy.polynomial.chebyshev.Chebyshev.fit` and `chebval`.

**Tier 2**, like the rest of `numax.interpolate` over `Tensor`. The fit is
one `numax.linalg.lstsq` -- a QR of the Chebyshev Vandermonde matrix, the
same computation NumPy's `chebfit` does -- and evaluation is one
`elementwise` launch running Clenshaw's recurrence per lane.

## Data, not a function

`numax.interpolate.array.Chebyshev` fits a `FloatLike` *function* by
sampling it at the Chebyshev nodes, which is the near-minimax construction
and the one to use when the function is available. This tier has
*samples* at points the caller did not choose, so it does what NumPy's
`Chebyshev.fit(x, y, deg)` does: map the data's range onto `[-1, 1]`,
build `T_k` at every mapped point, and solve the least-squares problem.
With as many terms as points the fit interpolates; with fewer it is the
best `L2` approximation in the Chebyshev basis, which is what a caller
smoothing noisy samples wants. Same name at both tiers, one import each.

## The MAX gate

Nothing to delegate to beyond what `lstsq` already routes -- the QR is
numax's blocked Householder over `linalg.matmul`, and MAX has no
polynomial fitting or evaluation of its own.
"""

from layout import Coord, coord_to_index_list
from max.algorithm.functional import elementwise

from ..core.tensorlike import TensorLike, View, dim, is_row_major
from ..core.array import _canonical, Static
from ..linalg.qr import lstsq


struct Chebyshev[dtype: DType, n: Int](Movable):
    """A Chebyshev series of `n` terms on `[a, b]`, fitted to data and
    evaluated at tensors of points.
    `numpy.polynomial.chebyshev.Chebyshev`, with its `fit`.

    ```mojo
    var series = Chebyshev[DType.float64, 4].fit(x, y)   # degree 3
    var values = series(points)
    ```

    `coefficients[k]` multiplies `T_k(u)` with `u = (2x - a - b) / (b - a)`,
    NumPy's convention including a full-weight `c_0` -- where
    `numax.interpolate.array`'s nodal fit halves `c_0`, since its `2/N`
    normalization produces that; the two are the same series written two
    ways, and each `__call__` matches its own `fit`.
    """

    var coefficients: Static[Self.dtype, Self.n]
    var a: Scalar[Self.dtype]
    var b: Scalar[Self.dtype]

    def __init__(
        out self,
        var coefficients: Static[Self.dtype, Self.n],
        a: Scalar[Self.dtype],
        b: Scalar[Self.dtype],
    ):
        """A series from its coefficients and domain."""
        self.coefficients = coefficients^
        self.a = a
        self.b = b

    @staticmethod
    def fit[
        A: TensorLike,
        B: TensorLike,
        gpu: Bool = False,
    ](x: A, y: B) raises -> Self where (
        (Self.dtype.is_floating_point() and Self.n >= 1 and dim[A, 0] >= Self.n)
        and A.dtype == Self.dtype
        and A.LayoutType.rank == 1
        and A.LayoutType.all_dims_known
        and B.dtype == Self.dtype
        and B.LayoutType.rank == 1
        and B.LayoutType.all_dims_known
        and dim[B, 0] == dim[A, 0]
    ):
        """The least-squares Chebyshev fit of degree `n - 1` to the samples
        `(x, y)`. `Chebyshev.fit(x, y, deg=n-1)`, with NumPy's default
        domain: `[min(x), max(x)]`, mapped onto `[-1, 1]`.

        The Chebyshev Vandermonde matrix `V[i, k] = T_k(u_i)` is built on
        the host by the three-term recurrence and solved through
        `numax.linalg.lstsq`, so the conditioning is the QR's rather than
        the normal equations'. `m >= n` samples, as any least-squares fit
        needs; `m == n` interpolates.
        """
        comptime m = dim[A, 0]
        var xs = x.to_host[Self.dtype]()
        var lo = Float64(xs[0])
        var hi = Float64(xs[0])
        for i in range(1, m):
            lo = min(lo, Float64(xs[i]))
            hi = max(hi, Float64(xs[i]))
        var half = (hi - lo) / 2.0
        var mid = (hi + lo) / 2.0

        var entries = List[Scalar[Self.dtype]](
            length=m * Self.n, fill=Scalar[Self.dtype](0)
        )
        for i in range(m):
            var u = (Float64(xs[i]) - mid) / half if half != 0 else 0.0
            var previous = 1.0
            var current = u
            entries[i * Self.n] = Scalar[Self.dtype](1)
            if Self.n > 1:
                entries[i * Self.n + 1] = Scalar[Self.dtype](u)
            for k in range(2, Self.n):
                var next = 2.0 * u * current - previous
                entries[i * Self.n + k] = Scalar[Self.dtype](next)
                previous = current
                current = next
        var vandermonde = Static[Self.dtype, m, Self.n](x.context(), entries^)
        var coefficients = lstsq[gpu=gpu](vandermonde, y)
        return Self(
            coefficients^, Scalar[Self.dtype](lo), Scalar[Self.dtype](hi)
        )

    def __call__[
        T: TensorLike,
        gpu: Bool = False,
    ](mut self, points: T) raises -> Static[Self.dtype, dim[T, 0]] where (
        (Self.dtype.is_floating_point() and Self.n >= 1 and dim[T, 0] > 0)
        and T.dtype == Self.dtype
        and T.LayoutType.rank == 1
        and T.LayoutType.all_dims_known
    ):
        """The series at every point, by Clenshaw's recurrence.

        Each lane maps its point onto `[-1, 1]` and folds the recurrence
        from the highest coefficient down, so no `T_k` is ever formed --
        the numerically stable way to evaluate this basis, and `O(n)` per
        point.
        """
        comptime m = dim[T, 0]
        return _clenshaw[gpu=gpu](self.coefficients, points, self.a, self.b)


def chebval[
    A: TensorLike,
    B: TensorLike,
    gpu: Bool = False,
](x: A, c: B) raises -> Static[A.dtype, dim[A, 0]] where (
    (A.dtype.is_floating_point() and dim[B, 0] >= 1 and dim[A, 0] > 0)
    and A.LayoutType.rank == 1
    and A.LayoutType.all_dims_known
    and B.dtype == A.dtype
    and B.LayoutType.rank == 1
    and B.LayoutType.all_dims_known
):
    """The Chebyshev series with coefficients `c` at every point of `x`,
    on the natural domain `[-1, 1]`. `numpy.polynomial.chebyshev.chebval(x,
    c)`, argument order included.

    The raw form: no domain mapping, so this is `Chebyshev.__call__` with
    `a = -1`, `b = 1`, and a point outside `[-1, 1]` evaluates the
    polynomial there rather than being rejected -- NumPy's behaviour too.
    """
    comptime m = dim[A, 0]
    comptime n = dim[B, 0]
    return _clenshaw[gpu=gpu](
        _canonical[n, dtype=A.dtype](c),
        x,
        Scalar[A.dtype](-1),
        Scalar[A.dtype](1),
    )


def _clenshaw[
    A: TensorLike,
    B: TensorLike,
    gpu: Bool,
](c: A, x: B, a: Scalar[A.dtype], b: Scalar[A.dtype]) raises -> Static[
    A.dtype, dim[B, 0]
] where (
    A.LayoutType.rank == 1
    and A.LayoutType.all_dims_known
    and B.dtype == A.dtype
    and B.LayoutType.rank == 1
    and B.LayoutType.all_dims_known
):
    """One launch: `sum_k c[k] T_k(u)` at `u = (2x - a - b) / (b - a)`,
    Clenshaw from the top coefficient down."""
    comptime n = dim[A, 0]
    comptime m = dim[B, 0]
    var ctx = x.context()
    var out = Static[A.dtype, m]._uninitialized(ctx)
    var cs = c.view()
    var xs = x.view_as[A.dtype]()
    var ys = out.view()
    var lo = a
    var hi = b

    @always_inline
    def evaluate[
        w: Int, alignment: Int = 1
    ](coord: Coord) {var cs, var xs, var ys, var lo, var hi}:
        var q = coord_to_index_list(coord)[0]
        var u = (2 * xs[Coord(q)] - lo - hi) / (hi - lo)
        var two_u = 2 * u
        var d = Scalar[A.dtype](0)
        var dd = Scalar[A.dtype](0)
        for step in range(1, n):
            var k = n - step
            var saved = d
            d = two_u * d - dd + cs[Coord(k)]
            dd = saved
        ys.store[1](Coord(q), u * d - dd + cs[Coord(0)])

    elementwise[simd_width=1, target="gpu" if gpu else "cpu"](
        evaluate, Coord(m), ctx
    )
    ctx.synchronize()
    return out^
