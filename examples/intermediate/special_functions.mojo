"""`numax.special`'s `gamma`, `bessel`, `lambertw`, and `elliptic`,
differentiated for free.

Same pattern as `examples/activations.mojo`: each kernel is written once
against `FloatLike`, so calling it with `Dual` gets the value and its
derivative together, with no second formula to keep in sync -- here that
means `Gamma'(x) = Gamma(x) * digamma(x)` and `dW/dx = W(x) / (x*(1+W(x)))`
fall out for free, without either derivative being spelled out anywhere in
`numax.special.gamma` or `numax.special.lambertw` -- the latter identity holds for
`lambertw_m1` too, since it's the same implicit-differentiation result
regardless of which real branch `W` came from. `elliptic_k`/`elliptic_e`
get the same treatment via their own standard derivative identities.

The second half is the rest of `numax.special`'s tier-1 surface -- `jv`/`yv`
at arbitrary real order, `airy`, `zeta`, `erfinv`, `exp1`/`expi`, `sici`,
`hyp2f1`, `owens_t` -- each printed next to an identity that pins it, since
a special function's value carries no evidence on its own. They take
conformers, not floats: `jv(P.constant(0.5), P.constant(2.0))`, never
`jv(0.5, 2.0)`.
"""

from std.math import atan, cos, sin, sqrt

from numax import (
    Dual,
    Plain,
    airy,
    j0,
    j1,
    jv,
    y0,
    y1,
    yv,
    elliptic_e,
    elliptic_k,
    erf,
    erfc,
    erfinv,
    exp1,
    expi,
    gamma,
    gammainc,
    hyp2f1,
    lambertw,
    lambertw_m1,
    lgamma,
    owens_t,
    sici,
    zeta,
)

comptime dtype = DType.float64
comptime width = 1

# The two conformers the second half spells over and over.
comptime P = Plain[dtype, width]
comptime D = Dual[P]

comptime pi = 3.141592653589793


def main():
    print("--- gamma / lgamma (now valid for x <= 0 too, via reflection) ---")
    for x_raw in [0.5, 1.0, 2.0, 5.0, -0.5, -2.5, -4.3, -10.5]:
        var x = Dual[Plain[dtype, width]](
            Plain[dtype, width].constant(x_raw),
            Plain[dtype, width].constant(1),
        )
        var g = gamma(x)
        var lg = lgamma(x)
        print(
            "x=",
            x_raw,
            " gamma=",
            g.value,
            " gamma'=",
            g.deriv,
            " lgamma=",
            lg.value,
        )

    print("--- gammainc(a, x): regularized lower incomplete gamma ---")
    var a = Plain[dtype, width].constant(2)
    for x_raw in [0.5, 1.0, 3.0, 10.0]:
        var x = Plain[dtype, width].constant(x_raw)
        print("a=2 x=", x_raw, " P(a,x)=", gammainc(a, x))

    var neg_a = Plain[dtype, width].constant(-0.5)
    var x_for_neg_a = Plain[dtype, width].constant(2.0)
    print(
        "a=-0.5 (now valid too) x=2.0  P(a,x)=",
        gammainc(neg_a, x_for_neg_a),
    )

    print(
        "--- bessel J0/J1/Y0/Y1 (now valid for |x| > 3 too, via an"
        " asymptotic far branch) ---"
    )
    for x_raw in [0.0, 1.0, 2.0, 2.4048, 3.0, 5.0, 10.0, 20.0]:
        var x = Plain[dtype, width].constant(x_raw)
        print("x=", x_raw, " J0(x)=", j0(x), " J1(x)=", j1(x))

    print("--- bessel Y0/Y1 (x > 0) ---")
    for x_raw in [0.5, 1.0, 2.0, 5.0, 10.0, 20.0]:
        var x = Plain[dtype, width].constant(x_raw)
        print("x=", x_raw, " Y0(x)=", y0(x), " Y1(x)=", y1(x))

    print(
        "--- lambertw, W0 branch (now valid down to the branch point"
        " x=-1/e, not just x >= 0) ---"
    )
    # `w exp(w) == x` is the definition, so it is the check -- and the
    # exponential in it is `Plain`'s own rather than `std.math`'s, for the
    # reason the `erfinv` block below spells out: at float64 `std.math.exp`
    # is ~1e-11 relative off, which is larger than anything `lambertw`
    # contributes, so the check would be reporting the reference's error
    # as the library's.
    for x_raw in [-0.36787944, -0.3, -0.1, 0.0, 1.0, 2.718281828459045, 10.0]:
        var x = Dual[Plain[dtype, width]](
            Plain[dtype, width].constant(x_raw),
            Plain[dtype, width].constant(1),
        )
        var w = lambertw(x)
        print(
            "x=",
            x_raw,
            " W(x)=",
            w.value,
            " W'(x)=",
            w.deriv,
            " check w*exp(w)=",
            (w.value * w.value.exp()).v,
        )

    print("--- lambertw_m1, the other real branch (-1/e <= x < 0, w <= -1) ---")
    for x_raw in [-0.36787944, -0.3, -0.1, -0.01, -1e-6]:
        var x = Dual[Plain[dtype, width]](
            Plain[dtype, width].constant(x_raw),
            Plain[dtype, width].constant(1),
        )
        var w = lambertw_m1(x)
        print(
            "x=",
            x_raw,
            " W_-1(x)=",
            w.value,
            " check w*exp(w)=",
            (w.value * w.value.exp()).v,
        )

    print("--- elliptic_k / elliptic_e (parameter m = k^2, 0 <= m <= 1) ---")
    for m_raw in [0.0, 0.3, 0.5, 0.9, 0.99, 1.0]:
        var m = Dual[Plain[dtype, width]](
            Plain[dtype, width].constant(m_raw),
            Plain[dtype, width].constant(1),
        )
        var k = elliptic_k(m)
        var e = elliptic_e(m)
        print(
            "m=",
            m_raw,
            " K(m)=",
            k.value,
            " E(m)=",
            e.value,
            " E'(m)=",
            e.deriv,
        )

    print("--- jv / yv: real order, not just the integer j0/j1/y0/y1 ---")
    # Half-integer order is where the general kernel meets elementary
    # functions: J_{1/2}(x) = sqrt(2/(pi x)) sin(x) and
    # Y_{1/2}(x) = -sqrt(2/(pi x)) cos(x), so the closed form beside each
    # value is a check, not a restatement.
    var half = P.constant(0.5)
    for x_raw in [1.0, 2.0, 5.0, 10.0]:
        var x = P.constant(x_raw)
        var scale = sqrt(2.0 / (pi * x_raw))
        print(
            "x=",
            x_raw,
            " J_1/2=",
            jv(half, x),
            " sqrt(2/pi x) sin x=",
            scale * sin(x_raw),
            " Y_1/2=",
            yv(half, x),
            " -sqrt(2/pi x) cos x=",
            -scale * cos(x_raw),
        )

    # At integer order the two halves of numax.special.bessel overlap, and
    # they part company around the eighth digit: j0/j1 are Abramowitz &
    # Stegun polynomial fits carrying a ~1e-8 error bound, while jv runs
    # Temme's recurrence to near machine precision. jv is the accurate one;
    # j0/j1 are the cheap one.
    for x_raw in [1.0, 4.0, 12.0]:
        var x = P.constant(x_raw)
        print(
            "x=",
            x_raw,
            " jv(0,x)=",
            jv(P.constant(0), x),
            " j0(x)=",
            j0(x),
            " jv(1,x)=",
            jv(P.one(), x),
            " j1(x)=",
            j1(x),
        )

    print("--- jv at Dual: dJ_v/dx = (J_{v-1}(x) - J_{v+1}(x)) / 2 ---")
    # The order is a constant of the differentiation, so it carries a zero
    # derivative while x carries one. Nothing in numax.special.bessel spells
    # the recurrence out; it is the forward-mode chain rule reproducing it.
    var order = D(P.constant(2.5), P.constant(0))
    for x_raw in [1.0, 3.0, 7.5]:
        var x = D(P.constant(x_raw), P.one())
        var j = jv(order, x)
        var xp = P.constant(x_raw)
        var recurrence = (
            jv(P.constant(1.5), xp).v - jv(P.constant(3.5), xp).v
        ) * 0.5
        print(
            "x=",
            x_raw,
            " J_2.5(x)=",
            j.value,
            " J_2.5'(x)=",
            j.deriv,
            " (J_1.5 - J_3.5)/2=",
            recurrence,
        )

    print("--- airy(x) -> (Ai, Ai', Bi, Bi') in one call ---")
    # The Wronskian Ai(x) Bi'(x) - Ai'(x) Bi(x) = 1/pi = 0.3183098861837907
    # holds for every x, so one number checks all four components at once,
    # across the three regions the kernel blends (series, then Bessel I/K
    # above 1.5 and J/Y below -1.5).
    for x_raw in [-8.0, -5.0, -1.0, 0.0, 1.0, 5.0, 10.0]:
        var a4 = airy(P.constant(x_raw))
        print(
            "x=",
            x_raw,
            " Ai=",
            a4[0],
            " Ai'=",
            a4[1],
            " Bi=",
            a4[2],
            " Bi'=",
            a4[3],
            " wronskian=",
            a4[0].v * a4[3].v - a4[1].v * a4[2].v,
        )

    print("--- zeta(s): one Euler-Maclaurin sum on both sides of the pole ---")
    # Above s = 1 the Dirichlet series' own values, below it the analytic
    # continuation, out of the same formula with no reflection: zeta(2) =
    # pi^2/6 = 1.6449340668482264, zeta(4) = pi^4/90 = 1.0823232337111382,
    # zeta(0) = -1/2, zeta(-1) = -1/12, and zeta(-2) = 0, the first trivial
    # zero, which arrives as a few 1e-15 because it is a difference of terms
    # near 1 rather than an exact cancellation. The sweep stops at s = -2.5,
    # which is as far as the fixed eight Bernoulli corrections reach.
    for s_raw in [-2.5, -2.0, -1.0, 0.0, 0.5, 2.0, 4.0, 10.0]:
        print("s=", s_raw, " zeta(s)=", zeta(P.constant(s_raw)))

    # The Hurwitz overload is the same sum from k = 0 over (k + q)^{-s}, so
    # zeta(s, 1) is the Riemann value and zeta(s, 2) is it minus the k = 1
    # term, which is 1.
    var s2 = P.constant(2)
    print(
        "zeta(2, 1)=",
        zeta(s2, P.one()),
        " zeta(2, 2)=",
        zeta(s2, P.constant(2)),
        " zeta(2) - 1=",
        zeta(s2).v - 1.0,
    )

    print("--- erfinv: the inverse of erf, checked by composing them ---")
    for y_raw in [-0.999, -0.9, -0.5, 0.0, 0.5, 0.9, 0.999999]:
        var x = erfinv(P.constant(y_raw))
        print("y=", y_raw, " erfinv(y)=", x, " erf(erfinv(y))=", erf(x))

    print(
        "--- erfinv at Dual: d/dy erfinv(y) = sqrt(pi)/2 exp(erfinv(y)^2) ---"
    )
    # The inverse-function derivative, which forward mode gets by
    # differentiating the three Newton steps rather than by knowing it.
    # The exponential on the right is P's own, not std.math's: at float64
    # numax.core.libm is within one ulp where std.math.exp is 1e-11
    # relative off, which would otherwise show up here as the check
    # disagreeing with a correct derivative.
    for y_raw in [-0.5, 0.25, 0.9]:
        var y = D(P.constant(y_raw), P.one())
        var x = erfinv(y)
        print(
            "y=",
            y_raw,
            " erfinv(y)=",
            x.value,
            " erfinv'(y)=",
            x.deriv,
            " sqrt(pi)/2 exp(x^2)=",
            sqrt(pi) / 2.0 * (x.value * x.value).exp().v,
        )

    print("--- exp1 / expi: the exponential integrals ---")
    # E_1 is defined for x > 0 only; Ei takes either sign, and on the
    # negative axis the two are the same function: Ei(-x) = -E_1(x).
    for x_raw in [0.5, 1.0, 2.0, 10.0]:
        var x = P.constant(x_raw)
        print(
            "x=",
            x_raw,
            " E_1(x)=",
            exp1(x),
            " Ei(x)=",
            expi(x),
            " Ei(-x)=",
            expi(P.constant(-x_raw)),
            " -E_1(x)=",
            -exp1(x).v,
        )

    print("--- sici(x) -> (Si, Ci) ---")
    # Si climbs to pi/2 = 1.5707963267948966 and Ci decays to 0, both as
    # oscillations around the limit rather than monotonically, which is why
    # the tail values straddle it.
    for x_raw in [0.5, 1.0, 2.0, 10.0, 50.0, 200.0]:
        var pair = sici(P.constant(x_raw))
        print("x=", x_raw, " Si=", pair[0], " Ci=", pair[1])

    print("--- hyp2f1(a, b; c; z), inside the unit disc ---")
    # 2F1(1, 1; 2; z) = -ln(1 - z) / z is the closed form these parameters
    # collapse to. |z| < 1 is the domain: the series runs directly for
    # 0 <= z < 1 and through Pfaff's transformation for z < 0, and neither
    # covers z >= 1, so the sweep stays inside. The logarithm is P's, for
    # the same reason the erfinv check's exponential was.
    for z_raw in [-0.9, -0.5, 0.25, 0.5, 0.9]:
        var z = P.constant(z_raw)
        print(
            "z=",
            z_raw,
            " 2F1(1,1;2;z)=",
            hyp2f1(P.one(), P.one(), P.constant(2), z),
            " -ln(1-z)/z=",
            -(P.one() - z).ln().v / z_raw,
        )

    print("--- owens_t(h, a): the bivariate-normal orthant integral ---")
    # T(0, a) = atan(a) / (2 pi) exactly, which pins the a direction.
    for a_raw in [0.25, 1.0, 4.0, 100.0]:
        print(
            "h=0 a=",
            a_raw,
            " T=",
            owens_t(P.constant(0), P.constant(a_raw)),
            " atan(a)/(2 pi)=",
            atan(a_raw) / (2.0 * pi),
        )

    # T(h, 1) = Phi(h) (1 - Phi(h)) / 2 pins the h direction, with
    # Phi(h) = erfc(-h / sqrt(2)) / 2 the standard normal CDF.
    for h_raw in [0.0, 0.5, 1.0, 2.0, 4.0]:
        var phi = 0.5 * erfc(P.constant(-h_raw / sqrt(2.0))).v
        print(
            "h=",
            h_raw,
            " a=1 T=",
            owens_t(P.constant(h_raw), P.one()),
            " Phi(h)(1-Phi(h))/2=",
            phi * (1.0 - phi) / 2.0,
        )
