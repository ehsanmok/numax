"""`numax.core.complex.Complex[Inner]`: a `FloatLike` conformer that nests over
any other one, the same trick `Dual` uses for higher-order derivatives.

Two things this demonstrates:

1. `Complex[Plain[...]]` gets the usual arithmetic plus the
   complex-analytic extensions of `exp`/`ln`/`sin`/`cos`/`erf` for free,
   since every one of them is written against `Inner: FloatLike` rather
   than a hardcoded real number.
2. `Complex[Dual[Plain[...]]]` differentiates a complex-valued kernel
   *holomorphically* -- no separate "complex autodiff" implementation
   anywhere, just `Dual` and `Complex` nesting into each other the way
   they were each written to.
"""

from numax import Complex, Dual, Plain

comptime dtype = DType.float64
comptime width = 1
comptime CT = Complex[Plain[dtype, width]]
comptime DCT = Complex[Dual[Plain[dtype, width]]]


def c(re: Float64, im: Float64) -> CT:
    return CT(
        Plain[dtype, width].constant(re),
        Plain[dtype, width].constant(im),
    )


def main():
    print("--- Complex[Plain]: ordinary complex arithmetic ---")
    var a = c(3.0, 4.0)
    var b = c(1.0, -2.0)
    print("a =", a.re.v, "+", a.im.v, "i")
    print("b =", b.re.v, "+", b.im.v, "i")
    print("a+b =", (a + b).re.v, "+", (a + b).im.v, "i")
    print("a*b =", (a * b).re.v, "+", (a * b).im.v, "i")
    print("a/b =", (a / b).re.v, "+", (a / b).im.v, "i")
    print("|a| =", a.abs().re.v, " (a real number, embedded as Complex)")

    print("--- exp/ln/sin/cos, the complex-analytic extensions ---")
    print("exp(a) =", a.exp().re.v, "+", a.exp().im.v, "i")
    print("ln(a)  =", a.ln().re.v, "+", a.ln().im.v, "i")
    print("sin(a) =", a.sin().re.v, "+", a.sin().im.v, "i")
    print("cos(a) =", a.cos().re.v, "+", a.cos().im.v, "i")
    var round_trip = a.ln().exp()
    print(
        "ln(a).exp() =",
        round_trip.re.v,
        "+",
        round_trip.im.v,
        "i  (check: recovers a)",
    )

    print("--- Complex[Dual[Plain]]: holomorphic derivatives for free ---")
    print("d/dz[z^2] at z = 3+4i should be 2z = 6+8i")
    var re = Dual[Plain[dtype, width]](
        Plain[dtype, width].constant(3.0),
        Plain[dtype, width].constant(1.0),
    )
    var im = Dual[Plain[dtype, width]](
        Plain[dtype, width].constant(4.0),
        Plain[dtype, width].constant(0.0),
    )
    var z = DCT(re^, im^)
    var w = z * z
    print(
        "z^2 =",
        w.re.value.v,
        "+",
        w.im.value.v,
        "i   d(z^2)/dz =",
        w.re.deriv.v,
        "+",
        w.im.deriv.v,
        "i",
    )
