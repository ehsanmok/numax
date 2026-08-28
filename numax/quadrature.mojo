"""Numerical integration on a fixed grid: Gauss-Legendre, Simpson, and the
trapezoid rule.

Adaptive quadrature -- subdividing wherever the integrand misbehaves -- is
out of scope by the same rule as everywhere else here: the subdivision
pattern is data-dependent, so two SIMD lanes integrating different
functions would want different grids. Fixed-node quadrature gives that up
willingly, because it was never really a compromise for this family:
Gauss-Legendre does a fixed amount of work by construction, and it is
*exact* for polynomials up to degree `2n-1` with `n` nodes, so the
"adaptive" question mostly doesn't arise for the smooth integrands it's
aimed at.

Two things fall out of writing this against `FloatLike` rather than a
concrete float:

- Integrating at `Dual` differentiates the integral -- with respect to
  either limit (recovering the fundamental theorem of calculus) or with
  respect to a parameter baked into the integrand. No separate
  "differentiate under the integral sign" machinery.
- The Gauss-Legendre nodes are the roots of `P_n`, and the weights involve
  `P_n'`. Both come from inside `numax`: `numax.legendre`'s recurrence for
  `P_n`, `numax.solve`'s Newton for the roots, and `numax.dual` for the
  derivative in the weight formula. Nothing here is a transcribed table.

That last point has a payoff beyond self-containment. Because the node
solve is written over `FloatLike`, it can be run at `Plain[float64, 1]`
*during compilation* -- `comptime nodes = _gauss_legendre_nodes[n]()`
evaluates the entire Newton iteration in the compiler, so the generated
code sees a plain list of constants. The nodes cost nothing at runtime and
still aren't a table anyone had to type in. (Verified directly: this is the
same mechanism `numax.compensated`'s `_split_f64` uses to keep float64
arithmetic out of device code.)
"""

from std.collections import Array
from std.math import cos as _cos_f64

from .dual import Dual
from .legendre import legendre_p
from .numeric import FloatLike
from .plain import Plain
from .solve import newton

comptime _PI = 3.14159265358979323846

# The type the compile-time node solve runs at.
comptime _CT = Plain[DType.float64, 1]


def _legendre_p_n[n: Int, U: FloatLike](x: U) -> U:
    """`P_n` with its degree bound, leaving the `FloatLike` parameter open
    -- the shape `numax.solve`'s `f` parameter needs (`_legendre_p_n[n, _]`
    at the call site)."""
    return legendre_p(n, x)


def _gauss_legendre_nodes[n: Int]() -> Array[Float64, n]:
    """The `n` roots of `P_n`, by Newton from the standard Chebyshev-like
    seed `cos(pi*(i+0.75)/(n+0.5))`, which is close enough to every root
    that a fixed 10 iterations converges to full `float64` precision.

    Intended to be called in a `comptime` binding, so the whole solve
    happens in the compiler.
    """
    var out = Array[Float64, n](fill=0.0)
    for i in range(n):
        var seed = _cos_f64(_PI * (Float64(i) + 0.75) / (Float64(n) + 0.5))
        out[i] = Float64(
            newton[f=_legendre_p_n[n, _], num_iters=10](_CT.constant(seed)).v
        )
    return out^


def _gauss_legendre_weights[n: Int]() -> Array[Float64, n]:
    """`w_i = 2 / ((1 - x_i^2) * P_n'(x_i)^2)` at each node.

    `P_n'` comes from evaluating `legendre_p` at a `Dual` -- there's no
    derivative recurrence written out anywhere for this.
    """
    var nodes = _gauss_legendre_nodes[n]()
    var out = Array[Float64, n](fill=0.0)
    for i in range(n):
        var x = nodes[i]
        var seeded = legendre_p(n, Dual[_CT](_CT.constant(x), _CT.one()))
        var d = Float64(seeded.deriv.v)
        out[i] = 2.0 / ((1.0 - x * x) * d * d)
    return out^


def gauss_legendre[
    T: FloatLike,
    f: def[U: FloatLike](U) thin -> U,
    n: Int = 8,
](a: T, b: T) -> T:
    """Integrate `f` over `[a, b]` with `n`-point Gauss-Legendre quadrature.

    Exact (to rounding) for any polynomial integrand of degree `2n-1` or
    less, and extremely accurate for smooth non-polynomial integrands --
    which is why the default `n` is small. `n=8` already beats a
    thousand-point trapezoid rule on a smooth integrand, at a hundredth the
    number of evaluations.

    Nodes and weights are compile-time constants (see this module's
    docstring), so the runtime cost is exactly `n` evaluations of `f`, `n`
    multiply-adds, and one affine map from `[-1, 1]` to `[a, b]`.

    Accuracy depends on the integrand being smooth on `[a, b]`. A
    discontinuity, a kink, or an endpoint singularity is where an adaptive
    rule would earn its keep and this won't -- integrate up to the trouble
    spot and past it as two calls instead.
    """
    comptime nodes = _gauss_legendre_nodes[n]()
    comptime weights = _gauss_legendre_weights[n]()

    var half_width = (b + (-a)) / T.constant(2.0)
    var midpoint = (a + b) / T.constant(2.0)

    var total = T.constant(0.0)
    comptime for i in range(n):
        comptime node = nodes[i]
        comptime weight = weights[i]
        var x = midpoint + half_width * T.constant(node)
        total = total + T.constant(weight) * f[T](x^)

    return total * half_width


def trapezoid[
    T: FloatLike,
    f: def[U: FloatLike](U) thin -> U,
    num_intervals: Int = 128,
](a: T, b: T) -> T:
    """Integrate `f` over `[a, b]` with the composite trapezoid rule.

    Second-order accurate (error `O(h^2)`), so it needs far more
    evaluations than `gauss_legendre` for the same accuracy on a smooth
    integrand. It earns its place on integrands Gauss-Legendre struggles
    with: a uniform grid doesn't concentrate its points near the endpoints
    the way Gauss nodes do, which matters when the integrand is only
    piecewise smooth.
    """
    var h = (b + (-a)) / T.constant(Float64(num_intervals))
    var total = (f[T](a.copy()) + f[T](b.copy())) / T.constant(2.0)

    # `a + k*h` rather than a running `x += h`: the multiply reintroduces
    # no drift, where repeated addition accumulates it across the grid.
    # `k` itself is carried as a `T` for the GPU reason `numax.orthopoly`'s
    # module docstring records.
    var kf = T.one()

    for _ in range(1, num_intervals):
        var x = a + kf * h
        total = total + f[T](x^)
        kf = kf + T.one()

    return total * h


def simpson[
    T: FloatLike,
    f: def[U: FloatLike](U) thin -> U,
    num_panels: Int = 64,
](a: T, b: T) -> T:
    """Integrate `f` over `[a, b]` with composite Simpson's rule, over
    `num_panels` panels of two subintervals each.

    Fourth-order accurate (`O(h^4)`) on the same uniform grid the trapezoid
    rule walks, which makes it the better default of the two whenever the
    integrand has a few continuous derivatives.

    Simpson needs an even number of subintervals. That's expressed here by
    counting *panels* rather than subintervals, so the requirement is
    structural and can't be violated -- an attempt to say it as
    `where num_intervals % 2 == 0` instead doesn't compile, since Mojo's
    constraint solver can't evaluate `Int.__mod__` (it reports "cannot
    evaluate call to non-builtin function"), and silently rounding an odd
    count would be worse than either.

    The alternating 4/2 coefficient pattern is a branch on the loop index,
    a scalar identical in every lane -- not a per-lane branch on the data,
    so it doesn't run into the invariant the rest of this library observes.
    """
    comptime num_intervals = 2 * num_panels
    var h = (b + (-a)) / T.constant(Float64(num_intervals))
    var total = f[T](a.copy()) + f[T](b.copy())
    var kf = T.one()

    for k in range(1, num_intervals):
        var x = a + kf * h
        var coefficient = 4.0 if k % 2 == 1 else 2.0
        total = total + T.constant(coefficient) * f[T](x^)
        kf = kf + T.one()

    return total * h / T.constant(3.0)
