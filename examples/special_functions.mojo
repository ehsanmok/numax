"""`ember.gamma`, `ember.bessel`, and `ember.lambertw`, differentiated for free.

Same pattern as `examples/activations.mojo`: each kernel is written once
against `FloatLike`, so calling it with `Dual` gets the value and its
derivative together, with no second formula to keep in sync -- here that
means `Gamma'(x) = Gamma(x) * digamma(x)` and `dW/dx = W(x) / (x*(1+W(x)))`
fall out for free, without either derivative being spelled out anywhere in
`ember.gamma` or `ember.lambertw`.
"""

from std.math import exp

from ember import Dual, Plain, bessel_j0, gamma, gammainc, lambertw, lgamma

comptime dtype = DType.float64
comptime width = 1


def main():
    print("--- gamma / lgamma ---")
    for x_raw in [0.5, 1.0, 2.0, 5.0]:
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

    print("--- bessel_j0 (valid for |x| <= 3) ---")
    for x_raw in [0.0, 1.0, 2.0, 2.4048, 3.0]:
        var x = Plain[dtype, width](SIMD[dtype, width](x_raw))
        print("x=", x_raw, " J0(x)=", bessel_j0(x).v)

    print("--- lambertw (principal branch, x >= 0) ---")
    for x_raw in [0.0, 1.0, 2.718281828459045, 10.0]:
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
