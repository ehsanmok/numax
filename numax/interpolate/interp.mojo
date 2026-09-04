"""Polynomial evaluation, cubic splines, and Chebyshev approximation.

Three ways to turn a set of samples (or a function you'd rather not call
repeatedly) into something cheap to evaluate, all `FloatLike`-generic so
the result differentiates and runs on GPU like everything else here.

## The uniform-grid restriction, and why it isn't a shortcut

`cubic_spline_eval` takes a grid origin and spacing rather than an array of
`x` values, and there's a reason it can't take arbitrary knots. Evaluating
a spline means first finding *which* interval `x` falls in -- a binary
search, whose trip count depends on the data. Inside a SIMD vector, lanes
would land in different intervals and want different numbers of search
steps, which `FloatLike` has no way to express, and on GPU it's a
divergent branch. So non-uniform knots are genuinely out of scope here, not
merely unimplemented.

A uniform grid replaces the search with arithmetic (`(x - x0)/h`), which is
lane-independent. Even then, *indexing* an `Array` at a per-lane-varying
position isn't possible either, so `cubic_spline_eval` scans all `n-1`
intervals and blends -- `O(n)` per point instead of `O(log n)`. That's the
real cost of lane independence, and it's fine for the small `n` this module
targets (the coefficients live in registers) while being the wrong choice
for a spline over thousands of knots.

Outside the grid, evaluation clamps to the nearest endpoint rather than
extrapolating the end cubic. Cubic extrapolation diverges fast and is
almost never what a caller wants; clamping is also what keeps the discarded
intervals' arithmetic bounded, since a blend evaluates every branch.

## Chebyshev fits are the alternative when you have a function, not samples

`chebyshev_fit` samples `f` at Chebyshev nodes and returns coefficients
whose `chebyshev_eval` is near-minimax on the interval -- for a smooth `f`,
far more accurate per coefficient than an equally-spaced polynomial fit,
and free of the Runge oscillation those suffer from. This is the tool for
replacing an expensive kernel with a cheap one inside a hot loop.
"""

from std.collections import Array
from std.math import cos as _cos_f64

from ..linalg.linalg import tridiagonal_solve
from ..core.numeric import FloatLike, blend, ge_indicator, max_of, min_of

comptime _PI = 3.14159265358979323846


def horner[T: FloatLike, n: Int](coefficients: Array[T, n], x: T) -> T:
    """Evaluate a polynomial at `x`, coefficients in ascending order
    (`coefficients[i]` multiplies `x^i`).

    Horner's rule rather than summing `c_i * x^i`: `n-1` multiply-adds
    instead of `n` powers, and better conditioned, since it never forms the
    large intermediate `x^i` that then has to cancel against its neighbours.
    """
    var total = coefficients[n - 1].copy()
    for step in range(1, n):
        total = total * x + coefficients[n - 1 - step]
    return total^


def cubic_spline_moments[
    T: FloatLike, n: Int
](y: Array[T, n], h: T) -> Array[T, n]:
    """The natural cubic spline's second derivatives at `n` uniformly
    spaced knots (`n >= 3`).

    "Natural" fixes the second derivative to zero at both ends, which
    leaves an `(n-2)`-unknown tridiagonal system with the constant pattern
    `M[i-1] + 4*M[i] + M[i+1] = 6*(y[i+1] - 2*y[i] + y[i-1])/h^2`. That
    goes straight to `numax.linalg.tridiagonal_solve` -- `O(n)`, no
    pivoting needed (the system is diagonally dominant), and fixed work.

    Returned separately from evaluation so a spline built once can be
    evaluated many times; pass the result to `cubic_spline_eval`.
    """
    comptime interior = n - 2

    var sub = Array[T, interior](fill=T.one())
    var diag = Array[T, interior](fill=T.constant(4.0))
    var sup = Array[T, interior](fill=T.one())
    var rhs = Array[T, interior](fill=T.constant(0.0))

    var six_over_h2 = T.constant(6.0) / (h * h)
    for i in range(interior):
        var second_difference = y[i + 2] - (T.constant(2.0) * y[i + 1]) + y[i]
        rhs[i] = six_over_h2 * second_difference

    var interior_moments = tridiagonal_solve[T, interior](sub, diag, sup, rhs)

    var moments = Array[T, n](fill=T.constant(0.0))
    for i in range(interior):
        moments[i + 1] = interior_moments[i].copy()
    return moments^


def cubic_spline_eval[
    T: FloatLike, n: Int
](y: Array[T, n], moments: Array[T, n], x0: T, h: T, x: T,) -> T:
    """Evaluate the spline through `y` (with `moments` from
    `cubic_spline_moments`) at `x`.

    See this module's docstring for the `O(n)` scan and the clamping
    behaviour outside `[x0, x0 + (n-1)*h]`.
    """
    var last = T.constant(Float64(n - 1))
    var clamped = min_of(max_of((x - x0) / h, T.constant(0.0)), last)

    var result = T.constant(0.0)
    var interval_start = T.constant(0.0)

    for i in range(n - 1):
        var t = (clamped - interval_start) * h

        # The cubic on interval `i`, in the standard moment form.
        var slope = (y[i + 1] - y[i]) / h + (
            -(
                h
                * (T.constant(2.0) * moments[i] + moments[i + 1])
                / T.constant(6.0)
            )
        )
        var curvature = moments[i] / T.constant(2.0)
        var jerk = (moments[i + 1] - moments[i]) / (T.constant(6.0) * h)
        var value = y[i] + t * (slope + t * (curvature + t * jerk))

        # `1` exactly on the interval containing `clamped`. The last
        # interval takes the right endpoint too, which is why its upper
        # test is dropped -- otherwise `x = x0 + (n-1)*h` would select no
        # interval at all and evaluate to zero.
        var above = ge_indicator(clamped, interval_start)
        var selected: T
        if i == n - 2:
            selected = above.copy()
        else:
            selected = above * (
                T.one() - ge_indicator(clamped, interval_start + T.one())
            )

        result = result + selected * value
        interval_start = interval_start + T.one()

    return result^


def chebyshev_fit[
    T: FloatLike,
    f: def[U: FloatLike](U) thin -> U,
    n_terms: Int = 16,
](a: T, b: T) -> Array[T, n_terms]:
    """Chebyshev coefficients approximating `f` on `[a, b]`.

    `c[k] = (2/N) * sum_j f(t_j) * cos(k*pi*(j+0.5)/N)` at the `N`
    Chebyshev nodes `t_j`, the standard discrete construction. The cosine
    table depends only on `n_terms`, so it's built at compile time and the
    run-time cost is exactly `n_terms` evaluations of `f` plus the sums.

    `f` is evaluated at `T`, so fitting at `Dual` gives coefficients that
    carry derivatives with respect to whatever the endpoints depend on.
    """
    comptime nodes = _chebyshev_nodes[n_terms]()
    comptime table = _chebyshev_cosine_table[n_terms]()

    var half_width = (b - a) / T.constant(2.0)
    var midpoint = (a + b) / T.constant(2.0)

    var samples = Array[T, n_terms](fill=T.constant(0.0))
    comptime for j in range(n_terms):
        comptime node = nodes[j]
        samples[j] = f[T](midpoint + half_width * T.constant(node))

    var out = Array[T, n_terms](fill=T.constant(0.0))
    comptime two_over_n = 2.0 / Float64(n_terms)

    # Both loops are `comptime for` so each table entry folds into a
    # `dtype`-native literal. Materializing the table into a runtime array
    # instead would keep it in float64 and make this CPU-only, the same
    # trap `numax.special.orthopoly`'s module docstring describes. The cost is an
    # `n_terms^2` unroll, which is why `n_terms` is meant to stay modest.
    comptime for k in range(n_terms):
        var total = T.constant(0.0)
        comptime for j in range(n_terms):
            comptime entry = table[k * n_terms + j]
            total = total + samples[j] * T.constant(entry)
        out[k] = total * T.constant(two_over_n)

    return out^


def chebyshev_eval[
    T: FloatLike, n: Int
](coefficients: Array[T, n], a: T, b: T, x: T) -> T:
    """Evaluate a `chebyshev_fit` result at `x`, by Clenshaw recurrence.

    Clenshaw rather than summing `c[k]*T_k(x)` term by term: it folds the
    Chebyshev recurrence into the summation, so no `T_k` is ever formed,
    and it's the numerically stable way to evaluate this basis.

    `c[0]` enters at half weight, which is the convention `chebyshev_fit`'s
    `2/N` normalization produces.
    """
    var y = (T.constant(2.0) * x - a - b) / (b - a)
    var two_y = T.constant(2.0) * y

    var d = T.constant(0.0)
    var dd = T.constant(0.0)
    for step in range(1, n):
        var k = n - step
        var saved = d.copy()
        d = two_y * d - dd + coefficients[k]
        dd = saved^

    return y * d - dd + T.constant(0.5) * coefficients[0]


def _chebyshev_nodes[n: Int]() -> Array[Float64, n]:
    """`cos(pi*(j+0.5)/n)` -- the Chebyshev points of the first kind on
    `[-1, 1]`, which cluster near the endpoints and are what make the fit
    near-minimax rather than merely least-squares."""
    var out = Array[Float64, n](fill=0.0)
    for j in range(n):
        out[j] = _cos_f64(_PI * (Float64(j) + 0.5) / Float64(n))
    return out^


def _chebyshev_cosine_table[n: Int]() -> Array[Float64, n * n]:
    """`cos(k*pi*(j+0.5)/n)` for every `(k, j)`, row-major in `k`."""
    var out = Array[Float64, n * n](fill=0.0)
    for k in range(n):
        for j in range(n):
            out[k * n + j] = _cos_f64(
                Float64(k) * _PI * (Float64(j) + 0.5) / Float64(n)
            )
    return out^


# --------------------------------------------------------------------------
# The scipy.interpolate-shaped objects over the two-call protocols above
# --------------------------------------------------------------------------


@fieldwise_init
struct CubicSpline[T: FloatLike, n: Int](Copyable, Movable):
    """A natural cubic spline through `n` uniformly spaced knots, built
    once and called many times. `scipy.interpolate.CubicSpline`.

    ```mojo
    var spline = CubicSpline[Plain[f64], 5](y, x0, h)
    var value = spline(x)
    ```

    The moments are solved in the constructor and kept, which is the whole
    point of the split `cubic_spline_moments`/`cubic_spline_eval` protocol
    this wraps -- those stay public and stay tier 1, since they are what a
    GPU-launchable kernel calls. This is the convenience over them, not a
    replacement.
    """

    var y: Array[Self.T, Self.n]
    var moments: Array[Self.T, Self.n]
    var x0: Self.T
    var h: Self.T

    def __init__(out self, y: Array[Self.T, Self.n], x0: Self.T, h: Self.T):
        """Solve for the spline's second derivatives at the knots."""
        self.moments = cubic_spline_moments[Self.T, Self.n](y, h)
        self.y = y.copy()
        self.x0 = x0.copy()
        self.h = h.copy()

    def __call__(self, x: Self.T) -> Self.T:
        """The spline's value at `x`, clamped to the knot range."""
        return cubic_spline_eval[Self.T, Self.n](
            self.y, self.moments, self.x0, self.h, x
        )


@fieldwise_init
struct Chebyshev[T: FloatLike, n: Int](Copyable, Movable):
    """A Chebyshev series on `[a, b]`, built once and called many times.
    `numpy.polynomial.chebyshev.Chebyshev`.

    ```mojo
    var series = Chebyshev[Plain[f64], 16](
        chebyshev_fit[Plain[f64], g, 16](a, b), a, b
    )
    var value = series(x)
    ```

    Wraps the `chebyshev_fit`/`chebyshev_eval` pair, which stay public and
    tier 1. `fit` below is the shorthand that does both.
    """

    var coefficients: Array[Self.T, Self.n]
    var a: Self.T
    var b: Self.T

    @staticmethod
    def fit[f: def[U: FloatLike](U) thin -> U](a: Self.T, b: Self.T) -> Self:
        """Fit `f` on `[a, b]` and keep the coefficients.
        `Chebyshev[Plain[f64], 16].fit[g](a, b)`."""
        return Self(chebyshev_fit[Self.T, f, Self.n](a, b), a.copy(), b.copy())

    def __call__(self, x: Self.T) -> Self.T:
        """The series' value at `x`, by Clenshaw recurrence."""
        return chebyshev_eval[Self.T, Self.n](
            self.coefficients, self.a, self.b, x
        )
