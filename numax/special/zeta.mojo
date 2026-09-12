"""The Riemann and Hurwitz zeta functions, `zeta(s)` and `zeta(s, q)`.
`scipy.special.zeta`.

**This module is tier 1.** One Euler-Maclaurin formula, at a fixed `N =
10` direct terms and `M = 8` Bernoulli corrections, evaluates the analytic
continuation everywhere at once: `sum_{k<N} (k+q)^{-s} + (N+q)^{1-s} /
(s-1) + (N+q)^{-s} / 2 + sum_{j<=M} B_{2j} / (2j)! s (s+1) ... (s+2j-2)
(N+q)^{-s-2j+1}`. The `(N+q)^{1-s} / (s-1)` term is what makes it valid
below `s = 1` as well as above -- no reflection formula, no second
branch, no blend -- and the only data-dependent hazard is the pole at `s =
1`, where `1 / (s - 1)` is guarded away from `0 / 0` and comes out
infinite as it should.

Checked in mpmath before transcription at `N = 10, M = 8`: within `1e-18`
of `zeta(s)` for `s` in `[-2.5, 30]` and of `zeta(s, q)` at `q = 0.5` and
`q = 3` -- far below double precision, so what `pixi run accuracy`
measures is rounding through the `(k+q)^{-s} = exp(-s ln(k+q))` powers,
not the formula's error: `6e-16` relative above the pole, `8e-11` below
it, where `zeta(-2.5) = 0.0085` is a small difference of terms near `300`
and each term's last bit is `1e-13` of the answer. The Riemann overload
takes its logarithms from an exact compile-time table; the Hurwitz one
takes them at run time. Further from the origin than `s =
-2.5` the Bernoulli corrections grow and `M = 8` is no longer enough; the
formula is meant for the half-plane the Dirichlet series and its first
continuation cover, which is what SciPy's `zeta` covers too (it returns
NaN for `s < 1` in the Hurwitz form and the Riemann values below).

## The MAX gate

Nothing: MAX has no zeta function. **Extend.**
"""

from std.collections import Array

from ..core.numeric import FloatLike, guard_nonzero

comptime _N = 10
comptime _M = 8
comptime _TINY = 1e-300

# `ln k` for `k = 1 .. 11`, so the Riemann overload's powers `k^{-s} =
# exp(-s ln k)` take a logarithm correct to the last bit rather than a
# rounded one: at `s = -2.5` each term is near 300 and the answer is
# `0.0085`, so half an ulp of `ln k` is already `4e-14` of the answer,
# and the table costs nothing.
comptime _LN: Array[Float64, 11] = [
    0.0,
    0.6931471805599453,
    1.0986122886681098,
    1.3862943611198906,
    1.6094379124341003,
    1.791759469228055,
    1.9459101490553132,
    2.0794415416798357,
    2.1972245773362196,
    2.302585092994046,
    2.3978952727983707,
]

# `B_{2j} / (2j)!` for `j = 1 .. 8`: 1/12, -1/720, 1/30240, -1/1209600,
# 1/47900160, -691/1307674368000, 1/74724249600, -3617/10670622842880000.
# A compile-time table like `_LN`, not a runtime `List[Float64]`: the list
# put `double` stores in the kernel body and Metal rejected it.
comptime _BERNOULLI: Array[Float64, _M] = [
    0.08333333333333333,
    -0.001388888888888889,
    3.306878306878307e-05,
    -8.267195767195768e-07,
    2.08767569878681e-08,
    -5.284190138687493e-10,
    1.3382536530684679e-11,
    -3.3896802963225829e-13,
]


def _euler_maclaurin[T: FloatLike, riemann: Bool](s: T, q: T) -> T:
    """The Euler-Maclaurin sum for `zeta(s, q)` at fixed `N` and `M`. With
    `riemann`, `q` is `1` and the logarithms of `k + 1` are the exact
    compile-time table above; otherwise they are taken at run time."""
    var total = T.constant(0.0)
    comptime for k in range(_N):
        comptime if riemann:
            # Bound as a `comptime` alias so the table read folds to a
            # literal, the idiom `numax.interpolate.array` uses for its nodes.
            comptime ln_k = _LN[k]
            total = total + (-(s * T.constant(ln_k))).exp()
        else:
            total = total + (-(s * (q + T.constant(Float64(k))).ln())).exp()

    var edge = q + T.constant(Float64(_N))
    var log_edge: T
    comptime if riemann:
        comptime ln_edge = _LN[_N]
        log_edge = T.constant(ln_edge)
    else:
        log_edge = edge.ln()
    var edge_pow = (-(s * log_edge)).exp()
    var denominator = guard_nonzero(s - T.one(), T.constant(_TINY))
    total = total + edge * edge_pow / denominator + edge_pow / T.constant(2.0)

    # `B_{2j} / (2j)! * s (s+1) ... (s+2j-2) * edge^{-s-2j+1}`, the rising
    # product and the power carried along from one correction to the next.
    var rising = s.copy()
    var power = edge_pow / edge
    comptime for j in range(1, _M + 1):
        comptime b_j = _BERNOULLI[j - 1]
        total = total + T.constant(b_j) * rising * power
        rising = (
            rising
            * (s + T.constant(Float64(2 * j - 1)))
            * (s + T.constant(Float64(2 * j)))
        )
        power = power / (edge * edge)
    return total^


def zeta[T: FloatLike](s: T) -> T:
    """The Riemann zeta function `sum_{k>=1} k^{-s}` and its analytic
    continuation, for `s != 1`. `scipy.special.zeta(s)`.

    One Euler-Maclaurin sum, valid on both sides of the pole; `zeta(0) =
    -1/2`, `zeta(-1) = -1/12`, `zeta(2) = pi^2 / 6` all come out of the
    same formula. Meant for `s >= -2.5`; the module docstring has the
    reason and the bound.
    """
    return _euler_maclaurin[T, True](s, T.one())


def zeta[T: FloatLike](s: T, q: T) -> T:
    """The Hurwitz zeta function `sum_{k>=0} (k + q)^{-s}` for `q > 0`,
    continued below `s = 1` as the Riemann form is. `scipy.special.zeta(s,
    q)`. `zeta(s, 1)` is `zeta(s)`."""
    return _euler_maclaurin[T, False](s, q)
