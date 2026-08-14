"""The gamma family: `gamma`, `lgamma`, and the incomplete gamma functions.

Every kernel here uses a **fixed** number of terms or iterations rather than
a data-dependent convergence check: a GPU thread can't branch per-lane on
"has this series converged yet" the way a scalar loop could (every lane in
a SIMD/SIMT group has to do the same amount of work), so accuracy here is
traded for a bounded, uniform amount of work instead of adaptive precision.
"""

from std.collections import Array

from .numeric import FloatLike


def lgamma[T: FloatLike](x: T) -> T:
    """`ln(Gamma(x))`, via the Lanczos approximation (`g=7`, 9 terms).

    Scoped to `x > 0` -- the usual reflection formula for `x <= 0.5` needs
    `sin`, which isn't in `FloatLike`, and `Gamma` itself isn't defined at
    the non-positive integers regardless. Positive reals cover the common
    downstream uses this is built for: factorials (`gamma(n+1) = n!`),
    Beta-function normalizing constants, and `gammainc` below.
    """
    comptime coef: Array[Float64, 9] = [
        0.99999999999980993,
        676.5203681218851,
        -1259.1392167224028,
        771.32342877765313,
        -176.61502916214059,
        12.507343278686905,
        -0.1385710952657201,
        9.984369578019572e-06,
        1.5056327351493116e-07,
    ]
    comptime g = 7.0

    comptime c0 = coef[0]
    var z = x + (-T.one())
    var a = T.constant(c0)

    comptime for i in range(1, 9):
        comptime ci = coef[i]
        a = a + T.constant(ci) / (z + T.constant(Float64(i)))

    var t = z + T.constant(g + 0.5)

    # ln(sqrt(2*pi) * t^(z+0.5) * exp(-t) * a)
    #   = 0.5*ln(2*pi) + (z+0.5)*ln(t) - t + ln(a)
    var half_ln_2pi = T.constant(0.9189385332046727)
    return half_ln_2pi + (z + T.constant(0.5)) * t.ln() + (-t) + a.ln()


def gamma[T: FloatLike](x: T) -> T:
    """`Gamma(x)`, via `exp(lgamma(x))`. `x > 0` -- see `lgamma`."""
    return lgamma(x).exp()


def gammainc[T: FloatLike](a: T, x: T) -> T:
    """The regularized lower incomplete gamma function, `P(a,x) = γ(a,x)/Γ(a)`.

    `x^a * exp(-x) / Gamma(a)` times a fixed 100-term series (`sum_{n=0}^{99}
    x^n / (a*(a+1)*...*(a+n))`, accumulated term-by-term rather than as a
    ratio of factorials to avoid overflow) -- accurate to close to `dtype`
    precision when `x` isn't much larger than `a`, since that's when the
    series converges fastest; the further `x` exceeds `a`, the more of the
    fixed 100 terms it takes to reach the same accuracy; there's no
    continued-fraction branch here to compensate for that (see this
    module's docstring on why a data-dependent branch is avoided).
    """
    comptime num_terms = 100

    var log_prefactor = a * x.ln() + (-x) + (-lgamma(a))
    var prefactor = log_prefactor.exp()

    var term = T.one() / a
    var total = term.copy()
    for n in range(1, num_terms):
        term = term * x / (a + T.constant(Float64(n)))
        total = total + term

    return prefactor * total


def gammaincc[T: FloatLike](a: T, x: T) -> T:
    """The regularized upper incomplete gamma function, `Q(a,x) = 1 - P(a,x)`.
    """
    return T.one() + (-gammainc(a, x))
