"""Legendre polynomials, `P_n(x)`, via their three-term recurrence.

Unlike every other special function in `numax`, there's no approximation
here to trade accuracy against: `P_n` *is* a polynomial, and Bonnet's
recurrence

    (k+1) * P_{k+1}(x) = (2k+1) * x * P_k(x) - k * P_{k-1}(x)

evaluates it exactly (up to rounding) from `P_0 = 1` and `P_1 = x`. So
there's no domain restriction and no error bound to document -- the only
cost is `n` iterations of the recurrence.

That loop bound deserves a note against this library's "no data-dependent
iteration counts" rule: `n` is a degree chosen by the caller, a scalar
that's identical across every SIMD lane and every GPU thread in a launch,
not a per-lane function of the data. Looping
`n` times is the same shape as `gammainc`'s fixed 100 terms, just with the
count named at the call site instead of baked in. Nothing here branches on
`x`.

There's deliberately no `legendre_p_deriv` alongside this: `P_n'(x)` is
`legendre_p(n, Dual(x, 1)).deriv`, which is where `numax.integrate` gets
the derivative it needs for Gauss-Legendre weights.
"""

from ..core.numeric import FloatLike


def legendre_p[T: FloatLike](n: Int, x: T) -> T:
    """`P_n(x)`, the Legendre polynomial of degree `n`, for any real `x`.

    `n <= 0` returns `P_0 = 1`. Evaluated by upward recurrence rather than
    by expanding the explicit coefficient formula, which loses precision
    badly for larger `n` (the coefficients alternate in sign and grow).
    """
    if n <= 0:
        return T.one()

    var p_prev = T.one()
    var p_curr = x.copy()

    # The loop counter is carried as a `T` rather than converted per
    # iteration from the `Int` index. `T.constant` takes a `Float64`, and
    # converting a *runtime* `Int` into one emits an int64-to-double
    # instruction that Metal rejects outright ("air.convert.f.f64.s.i64 has
    # Metal-unsupported instructions"), which would make this function
    # CPU-only. Counting in `T` keeps every value `dtype`-native.
    var kf = T.one()

    for _ in range(1, n):
        var p_next = (
            (T.constant(2.0) * kf + T.one()) * x * p_curr + (-(kf * p_prev))
        ) / (kf + T.one())
        p_prev = p_curr.copy()
        p_curr = p_next.copy()
        kf = kf + T.one()

    return p_curr^
