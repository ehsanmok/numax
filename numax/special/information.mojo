"""The information-theoretic elementwise functions: `xlogy`, `xlog1py`,
`entr`, `rel_entr`, `kl_div` and `logit`, with `scipy.special`'s
conventions at the edges.

**This module is tier 1.** Each of these has a removable singularity or a
convention at zero -- `0 log 0 = 0` is the whole reason `xlogy` exists --
and the convention is applied as a `0`/`1` blend rather than a branch, so
every function runs per SIMD lane inside a kernel and differentiates at
`Dual` where the derivative exists. The rule the blends follow is the one
`numax.special.gamma` states: both sides of a blend are always evaluated,
so the side being discarded is fed a guarded argument (`ln(1)` rather than
`ln(0)`) and never produces the NaN that would poison the sum.

Where SciPy returns infinity -- `xlogy(x, 0)` for `x != 0`, `rel_entr(x,
0)` for `x > 0`, `entr(x)` for `x < 0` -- these do too, because `ln(0)`
does, and always as a term *added* to a finite value rather than blended
in: a `blend` multiplies its unselected side by `0`, and `0 * inf` is NaN; where SciPy returns infinity for a *negative* argument to `rel_entr`
or `kl_div`, these return NaN from the logarithm instead, and the
docstrings say so. The functions are defined for the non-negative
arguments they are used at.
"""

from ..core.numeric import FloatLike, blend, ge_indicator, max_of

comptime _TINY = 1e-300


def _is_zero[T: FloatLike](x: T) -> T:
    """`1` exactly where `x == 0`, `0` elsewhere: `0 >= |x|`, one indicator
    on the magnitude. Not `x >= 0` times `-x >= 0` -- negating `0.0` gives
    `-0.0`, whose sign `copysign` reads as negative, and the second factor
    would then miss zero itself."""
    return ge_indicator(T.constant(0.0), x.abs())


def xlogy[T: FloatLike](x: T, y: T) -> T:
    """`x * log(y)`, with `0` where `x == 0` whatever `y` is.
    `scipy.special.xlogy(x, y)`. The term an entropy or a log-likelihood
    is built from, with the `0 log 0` convention applied rather than left
    as NaN. `y == 0` with `x != 0` is `-inf` times the sign of `x`, as
    SciPy's is."""
    var zero = _is_zero(x)
    var guarded = blend(zero, T.one(), y)
    return blend(zero, T.constant(0.0), x * guarded.ln())


def xlog1py[T: FloatLike](x: T, y: T) -> T:
    """`x * log(1 + y)`, `0` where `x == 0`. `scipy.special.xlog1py`.
    `log(1 + y)` is formed as such -- `FloatLike` has no `log1p` -- so a
    `y` near `-1` or within rounding of `0` loses the digits `log1p` would
    keep; the docstring is the bound."""
    var zero = _is_zero(x)
    var guarded = blend(zero, T.constant(0.0), y)
    return blend(zero, T.constant(0.0), x * (T.one() + guarded).ln())


def entr[T: FloatLike](x: T) -> T:
    """The entropy term `-x log(x)`: `0` at `x == 0`, `-inf` for `x < 0`.
    `scipy.special.entr(x)`. Summed over a distribution it is the Shannon
    entropy `numax.stats.entropy` returns."""
    # Finite for every `x` (`-0 * ln(tiny)` is `0` at zero), plus a term
    # that is `ln(1) = 0` for `x >= 0` and `ln(0) = -inf` below -- an
    # infinity *added* rather than blended in, since `blend` would
    # multiply it by `0` and get NaN.
    var guarded = max_of(x, T.constant(_TINY))
    var finite = -(x * guarded.ln())
    return finite + ge_indicator(x, T.constant(0.0)).ln()


def rel_entr[T: FloatLike](x: T, y: T) -> T:
    """The relative entropy term `x log(x / y)`: `0` where `x == 0`, `inf`
    where `x > 0` and `y == 0`. `scipy.special.rel_entr(x, y)`. Summed, it
    is the Kullback-Leibler divergence. Defined for `x, y >= 0`; a
    negative argument gives NaN where SciPy gives `inf`.

    Formed as `x (ln x - ln y)` rather than `x ln(x / y)`: `Plain.ln`
    saturates at the largest finite argument, so `ln(x / 0) = ln(inf)`
    would come back as `709.78` where `-ln(0) = -(-inf)` is the infinity
    SciPy returns.
    """
    var zero = _is_zero(x)
    var guarded_x = max_of(x, T.constant(_TINY))
    var guarded_y = blend(zero, T.one(), y)
    return blend(zero, T.constant(0.0), x * (guarded_x.ln() - guarded_y.ln()))


def kl_div[T: FloatLike](x: T, y: T) -> T:
    """`x log(x / y) - x + y`, the Kullback-Leibler term that is
    non-negative pointwise and zero exactly at `x == y`.
    `scipy.special.kl_div(x, y)`. `rel_entr` plus the linear terms that
    make it a Bregman divergence."""
    return rel_entr(x, y) - x + y


def logit[T: FloatLike](p: T) -> T:
    """The log-odds `log(p / (1 - p))`, the inverse of `sigmoid`
    (`scipy.special.expit`). `scipy.special.logit(p)`. `-inf` at `0`,
    `inf` at `1`, NaN outside `[0, 1]`, as SciPy's."""
    return (p / (T.one() - p)).ln()
