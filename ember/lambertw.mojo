"""The Lambert W function, principal branch."""

from .numeric import FloatLike


def lambertw[T: FloatLike](x: T) -> T:
    """The principal branch of the Lambert W function: solves `w*exp(w) = x`.

    Halley's method on `f(w) = w*exp(w) - x`, seeded from `ln(1 + x)` and
    iterated a **fixed** number of times rather than to a data-dependent
    tolerance (see `ember.gamma`'s docstring for why: a GPU thread can't
    branch per-lane on "has this converged yet"). Scoped to `x >= 0`, where
    `ln(1 + x)` is always defined and Halley's method converges to it
    monotonically for this particular `f`.
    """
    comptime num_iters = 20

    var w = (T.one() + x).ln()

    for _ in range(num_iters):
        var ew = w.exp()
        var f = w * ew + (-x)
        var fprime = ew * (w + T.one())
        var fprime2 = ew * (w + T.constant(2.0))
        var numerator = T.constant(2.0) * f * fprime
        var denominator = T.constant(2.0) * (fprime * fprime) + (-(f * fprime2))
        w = w + (-(numerator / denominator))

    return w^
