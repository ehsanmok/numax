"""`numax.gamma`, `numax.bessel`, `numax.lambertw`, and `numax.elliptic`,
differentiated for free.

Same pattern as `examples/activations.mojo`: each kernel is written once
against `FloatLike`, so calling it with `Dual` gets the value and its
derivative together, with no second formula to keep in sync -- here that
means `Gamma'(x) = Gamma(x) * digamma(x)` and `dW/dx = W(x) / (x*(1+W(x)))`
fall out for free, without either derivative being spelled out anywhere in
`numax.gamma` or `numax.lambertw` -- the latter identity holds for
`lambertw_m1` too, since it's the same implicit-differentiation result
regardless of which real branch `W` came from. `elliptic_k`/`elliptic_e`
get the same treatment via their own standard derivative identities.
"""

from std.math import exp

from numax import (
    Dual,
    Plain,
    bessel_j0,
    bessel_j1,
    bessel_y0,
    bessel_y1,
    elliptic_e,
    elliptic_k,
    gamma,
    gammainc,
    lambertw,
    lambertw_m1,
    lgamma,
)

comptime dtype = DType.float64
comptime width = 1


def main():
    print("--- gamma / lgamma (now valid for x <= 0 too, via reflection) ---")
    for x_raw in [0.5, 1.0, 2.0, 5.0, -0.5, -2.5, -4.3, -10.5]:
        var x = Dual[Plain[dtype, width]](
            Plain[dtype, width](SIMD[dtype, width](x_raw)),
            Plain[dtype, width](SIMD[dtype, width](1)),
        )
        var g = gamma(x)
        var lg = lgamma(x)
        print(
            "x=",
            x_raw,
            " gamma=",
            g.value.v,
            " gamma'=",
            g.deriv.v,
            " lgamma=",
            lg.value.v,
        )

    print("--- gammainc(a, x): regularized lower incomplete gamma ---")
    var a = Plain[dtype, width](SIMD[dtype, width](2))
    for x_raw in [0.5, 1.0, 3.0, 10.0]:
        var x = Plain[dtype, width](SIMD[dtype, width](x_raw))
        print("a=2 x=", x_raw, " P(a,x)=", gammainc(a, x).v)

    var neg_a = Plain[dtype, width](SIMD[dtype, width](-0.5))
    var x_for_neg_a = Plain[dtype, width](SIMD[dtype, width](2.0))
    print(
        "a=-0.5 (now valid too) x=2.0  P(a,x)=",
        gammainc(neg_a, x_for_neg_a).v,
    )

    print(
        "--- bessel J0/J1/Y0/Y1 (now valid for |x| > 3 too, via an"
        " asymptotic far branch) ---"
    )
    for x_raw in [0.0, 1.0, 2.0, 2.4048, 3.0, 5.0, 10.0, 20.0]:
        var x = Plain[dtype, width](SIMD[dtype, width](x_raw))
        print("x=", x_raw, " J0(x)=", bessel_j0(x).v, " J1(x)=", bessel_j1(x).v)

    print("--- bessel Y0/Y1 (x > 0) ---")
    for x_raw in [0.5, 1.0, 2.0, 5.0, 10.0, 20.0]:
        var x = Plain[dtype, width](SIMD[dtype, width](x_raw))
        print("x=", x_raw, " Y0(x)=", bessel_y0(x).v, " Y1(x)=", bessel_y1(x).v)

    print(
        "--- lambertw, W0 branch (now valid down to the branch point"
        " x=-1/e, not just x >= 0) ---"
    )
    for x_raw in [-0.36787944, -0.3, -0.1, 0.0, 1.0, 2.718281828459045, 10.0]:
        var x = Dual[Plain[dtype, width]](
            Plain[dtype, width](SIMD[dtype, width](x_raw)),
            Plain[dtype, width](SIMD[dtype, width](1)),
        )
        var w = lambertw(x)
        print(
            "x=",
            x_raw,
            " W(x)=",
            w.value.v,
            " W'(x)=",
            w.deriv.v,
            " check w*exp(w)=",
            w.value.v * exp(w.value.v),
        )

    print("--- lambertw_m1, the other real branch (-1/e <= x < 0, w <= -1) ---")
    for x_raw in [-0.36787944, -0.3, -0.1, -0.01, -1e-6]:
        var x = Dual[Plain[dtype, width]](
            Plain[dtype, width](SIMD[dtype, width](x_raw)),
            Plain[dtype, width](SIMD[dtype, width](1)),
        )
        var w = lambertw_m1(x)
        print(
            "x=",
            x_raw,
            " W_-1(x)=",
            w.value.v,
            " check w*exp(w)=",
            w.value.v * exp(w.value.v),
        )

    print("--- elliptic_k / elliptic_e (parameter m = k^2, 0 <= m <= 1) ---")
    for m_raw in [0.0, 0.3, 0.5, 0.9, 0.99, 1.0]:
        var m = Dual[Plain[dtype, width]](
            Plain[dtype, width](SIMD[dtype, width](m_raw)),
            Plain[dtype, width](SIMD[dtype, width](1)),
        )
        var k = elliptic_k(m)
        var e = elliptic_e(m)
        print(
            "m=",
            m_raw,
            " K(m)=",
            k.value.v,
            " E(m)=",
            e.value.v,
            " E'(m)=",
            e.deriv.v,
        )
