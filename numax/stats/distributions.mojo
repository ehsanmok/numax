"""Probability densities, cumulative distributions, and quantiles.

**This module is tier 1.** The CDFs are changes of variable over
`gammainc`/`betainc`/`erfc`, and each quantile runs its own fixed-iteration
Newton loop against the analytic density rather than iterating to a
tolerance.

Seventeen namespaces, spelled the way `scipy.stats` spells them --
`norm`, `expon`, `gamma`, `chi2`, `beta`, `t`, `f`, `poisson`, `binom`,
and since 0.3 `lognorm`, `weibull_min`, `cauchy`, `laplace`, `rayleigh`,
`logistic`, `pareto` and `uniform_dist` (SciPy's `uniform`, renamed
because `numax.stats.uniform` is NumPy's sampler), and the discrete
`bernoulli`, `geom`, `nbinom` and `hypergeom` -- each
carrying the eight methods a `scipy.stats` distribution carries: `.pdf`
(or `.pmf`) and `.logpdf` (`.logpmf`), `.cdf` and `.logcdf`, `.sf` and
`.logsf`, `.ppf` and `.isf`:

```mojo
from numax.stats import norm, chi2
var p = norm.cdf(x, mu, sigma)
var q = chi2.ppf(P(0.95), P(3.0))
```

Each namespace also carries SciPy's `rvs`, `mean`, `var`, `std`,
`interval` and `entropy`. The moments take `Float64` parameters and
return `Float64`, SciPy's `inf`/NaN included where a moment does not
exist; `rvs[dtype, *dims]` draws from a `Generator` -- through its exact
samplers for the normal, exponential, gamma, chi-squared, beta, Poisson
and binomial, and by inverting `ppf` for `t` and `f` -- on the device at
`gpu=True`.

Every `pdf`/`pmf`, `cdf` and `ppf` also has a `Tensor` overload beside the
`FloatLike` one, taking the distribution's parameters as `Scalar`s:

```mojo
var z = norm.cdf(samples, Scalar[f32](0.0), Scalar[f32](1.0))   # a Tensor
var q = gamma.ppf[gpu=True](probabilities, shape, scale)        # on the device
```

The `Tensor` form is the `FloatLike` kernel driven across the tensor by
`numax.core._drive` -- threaded at native SIMD width on the host, one
thread per element at `gpu=True` -- with the distribution's parameters
captured by the body. One overload per method covers both shapes: a tensor
whose extents are known only at run time launches on the device exactly as
a compile-time-shaped one does, because `max.algorithm.elementwise`
computes its grid from a run-time `Coord`. It is `enqueue_function` that
needs the extent in the type, and this route does not use it. A `gpu` that
disagrees with where the tensor's memory lives falls back to a host walk
and says so on `stderr`. Both forms are one definition: the `Tensor`
overload *is* the `FloatLike` one, evaluated per lane.
`numax.core.functional.map`'s scalar-parameter overloads remain the route for a
caller launching the raw kernel by hand.

`gamma` and `beta` are the *distributions*; `numax.special.gamma` and
`numax.special.beta` are the functions they are named after. That is why
these namespaces are reached through `numax.stats` and are not re-exported
at the root.

Almost none of this is new numerics. A distribution's CDF is nearly always
a special function already in `numax` under a change of variables -- the
gamma family is `gammainc`, the beta family (which includes Student-t, F,
and the binomial) is `betainc`, the normal family is `erfc` -- so the work
here is composition, and the payoff is that every one of them inherits
`Dual`-differentiability, `Compensated` precision, and GPU-launchability
from the function underneath.

`d/dx` of a CDF is the PDF, and evaluating any `.cdf` here at `Dual`
recovers exactly that, which is how `tests/stats/test_distributions.mojo` checks
each pair against the other rather than against a table.

## Conventions

- Shape parameters are `T` values, not `Int`s, so they can vary per SIMD
  lane. That includes the discrete distributions' counts: `poisson.cdf`'s
  `k` and `binom.cdf`'s `n`/`k` are continuous extensions of the usual
  integer-argument definitions, which is what the underlying `gammaincc`
  and `betainc` compute anyway.
- Densities are `0` outside their support rather than undefined, applied by
  a branchless indicator. The log-space interior is always evaluated on a
  clamped argument first, so the discarded side never produces the NaN that
  would survive multiplication by a `0` indicator.
- Everything is scoped to valid parameters (positive shapes, `sigma > 0`,
  `0 < p < 1`); nothing validates, in keeping with the rest of `numax`.
- Log densities are the log-space interior directly, never `ln(pdf)`, so
  they stay finite where the density underflows -- which is what a log
  density is for. Outside the support they return `_LOG_ZERO`, a large
  finite negative, rather than SciPy's `-inf`: finite so a `Dual`
  derivative through it stays finite, and chosen so that `exp` of it is
  exactly the `0` the density returns there.
- Survival functions read the upper tail off the complement special
  function (`erfc`, `betaincc`, `gammainc` for the discrete upper tails)
  where one exists, so they keep their digits where `1 - cdf` would cancel
  them. `gamma.sf` is the exception in effect if not in spelling: numax's
  `gammaincc` is literally `1 - gammainc`, so the name is there and the
  tail accuracy waits on an upper-tail continued fraction.

## Quantiles

Each quantile is a seed plus a fixed number of Newton steps against its own
CDF, using the analytic PDF as the derivative. They do *not* go through
`numax.optimize.newton`, and the reason is a real limitation worth naming:
`solve`'s `f` is `def[U: FloatLike](U) thin -> U`, a non-capturing function
required to be `thin` so a solver can be launched on GPU. A distribution's
parameters have nowhere to live in that signature -- `beta.ppf(p, a,
b)` would need `a` and `b` inside `f`, which a `thin` function cannot
close over. Writing the loop locally also happens to be cheaper here, since
the exact derivative is already in hand and doesn't need a `Dual` pass.

Seeds matter more than iteration counts for a fixed-work solver, so each
one uses the standard published approximation for its family rather than a
constant: Abramowitz & Stegun 26.2.23 for the normal, Wilson-Hilferty for
the gamma, the distribution mean for the beta. `f.ppf` is `beta.ppf` under
the change of variables its CDF already uses, and `expon.ppf` is closed
form. Every `isf` is its `ppf` at `1 - p`, or the symmetric spelling of it.

## Discrete quantiles

`poisson.ppf` and `binom.ppf` return what SciPy returns -- the smallest
integer `k` with `cdf(k) >= p` -- and do it branchlessly: `k` starts at
`0` and, for a compile-time `max_k` steps, adds `1` wherever `cdf(k)` is
still below `p`. Once the CDF has caught up the indicator is `0` and `k`
stops. Fixed work, no per-lane branch, and no `floor`, which `FloatLike`
does not have and which is why a Newton step on the continuous extension
is not the route.

`ponytail:` the cost is `max_k` CDF evaluations per lane -- `max_k`
incomplete-gamma series or incomplete-beta continued fractions -- and a
quantile above `max_k` is silently reported as `max_k`. The default of
`64` covers a Poisson mean up to about `40` at `p = 0.999`; pass a larger
`max_k` for a larger rate. The upgrade is a normal-approximation seed plus
a short scan around it, once `FloatLike` can round a seed to an integer.
"""

from layout import Coord
from layout.tile_layout import TensorLayout

from ..core._drive import (
    _check_device,
    _flat,
    _flat_out,
    _launch,
    _notice,
    _width,
)
from ..core.tensorlike import TensorLike, dim, is_row_major
from ..core.tensor import Tensor
from ..core.plain import Plain
from max.gpu.host import DeviceContext

from ..special.beta import betainc, betaincc, betaln
from ..special.gamma import digamma, gammainc, gammaincc, lgamma
from ..core.tensor import Static, _LayoutOf, _product
from layout.tile_layout import row_major
from .random import Generator
from ..core.numeric import FloatLike, blend, ge_indicator, max_of, min_of

comptime _SQRT_2 = 1.4142135623730951
comptime _SQRT_2PI = 2.5066282746310002
comptime _LN_2PI = 1.8378770664093453
comptime _LN_PI = 1.1447298858494002

# Small enough to be indistinguishable from the boundary at any supported
# `dtype`, large enough that `ln` of it is finite in float32.
comptime _TINY = 1e-30

# What a log density returns outside its support: finite, representable at
# float32, and with `exp` of it exactly `0` -- see the module docstring.
comptime _LOG_ZERO = -1e30


def _safe_ln[T: FloatLike](x: T) -> T:
    """`ln(x)` with the argument floored away from zero.

    Every density below evaluates its log-space interior even on lanes
    outside the support, because a branchless blend evaluates both sides.
    This keeps that evaluation finite so the discarded side contributes a
    large negative number rather than the `inf - inf` NaN that would
    survive multiplication by a `0` indicator. `max_of` is an exact
    selection, so one clamp does it even for an arbitrarily negative `x`.
    """
    return max_of(x, T.constant(_TINY)).ln()


def _log_beta[T: FloatLike](a: T, b: T) -> T:
    return lgamma(a.copy()) + lgamma(b.copy()) - lgamma(a + b)


def _over1[
    T: TensorLike,
    step: def[w: Int](SIMD[T.dtype, w], SIMD[T.dtype, 1]) thin -> SIMD[
        T.dtype, w
    ],
    gpu: Bool,
    name: StaticString,
](x: T, p0: Scalar[T.dtype]) raises -> Tensor[T.dtype, T.LayoutType] where (
    is_row_major[T] and T.dtype.is_floating_point()
):
    """Drive a one-parameter distribution kernel across `x` through
    `numax.core._drive`: threaded at native SIMD width on the host, one
    thread per element on the device, with `p0` captured by the body.

    One signature covers `Static` and `Dynamic` alike. `_flat` builds the
    rank-1 view over the buffer pointer rather than `coalesce()`, and
    `max.algorithm.elementwise` computes its grid from a run-time `Coord`,
    so a tensor whose extents are known only at run time launches on the
    device exactly as a compile-time-shaped one does. `enqueue_function` is
    what needs the extent in the type, and this route does not use it.

    `elementwise` rather than `map[gpu=True]` under `enqueue_function` for
    the reason `findings.mdc` records: a kernel that carries a layout
    `where` clause cannot be named inside `enqueue_function` from a generic
    function. `numax.fft` and `transpose` launch this way for the same
    reason.

    A `gpu` that disagrees with where `x`'s memory lives runs the host walk
    and says so on `stderr` -- the same fallback, through the same
    `_check_device`/`_notice`, that the elementwise surface takes.
    """
    comptime dtype = T.dtype
    comptime LayoutType = T.LayoutType
    if not _check_device[gpu=gpu](x):
        _notice[gpu](name)
        var values = x.to_host()
        var walked = List[Scalar[dtype]](length=len(values), fill=0)
        for i in range(len(values)):
            walked[i] = step[1](values[i], p0)[0]
        return Tensor[dtype, LayoutType](x.tile().layout, walked^, x.context())

    var ctx = x.context()
    var out = Tensor[dtype, LayoutType]._uninitialized(ctx, x.tile().layout)
    var xs = _flat(x)
    var ys = _flat_out(out)

    @always_inline
    def body[
        width: Int, alignment: Int = 1
    ](coord: Coord) {var xs, var ys, var p0}:
        ys.store[width](coord, step[width](xs.load[width](coord), p0))

    _launch[gpu=gpu, lanes=_width[dtype, gpu]()](body, x.size(), ctx)
    return out^


def _over2[
    T: TensorLike,
    step: def[w: Int](
        SIMD[T.dtype, w], SIMD[T.dtype, 1], SIMD[T.dtype, 1]
    ) thin -> SIMD[T.dtype, w],
    gpu: Bool,
    name: StaticString,
](x: T, p0: Scalar[T.dtype], p1: Scalar[T.dtype]) raises -> Tensor[
    T.dtype, T.LayoutType
] where (is_row_major[T] and T.dtype.is_floating_point()):
    """`_over1` for a two-parameter distribution."""
    comptime dtype = T.dtype
    comptime LayoutType = T.LayoutType
    if not _check_device[gpu=gpu](x):
        _notice[gpu](name)
        var values = x.to_host()
        var walked = List[Scalar[dtype]](length=len(values), fill=0)
        for i in range(len(values)):
            walked[i] = step[1](values[i], p0, p1)[0]
        return Tensor[dtype, LayoutType](x.tile().layout, walked^, x.context())

    var ctx = x.context()
    var out = Tensor[dtype, LayoutType]._uninitialized(ctx, x.tile().layout)
    var xs = _flat(x)
    var ys = _flat_out(out)

    @always_inline
    def body[
        width: Int, alignment: Int = 1
    ](coord: Coord) {var xs, var ys, var p0, var p1}:
        ys.store[width](coord, step[width](xs.load[width](coord), p0, p1))

    _launch[gpu=gpu, lanes=_width[dtype, gpu]()](body, x.size(), ctx)
    return out^


def _norm_pdf_step[
    dtype: DType, w: Int
](x: SIMD[dtype, w], mu: SIMD[dtype, 1], sigma: SIMD[dtype, 1]) -> SIMD[
    dtype, w
] where dtype.is_floating_point():
    return norm.pdf(
        Plain[dtype, w](x),
        Plain[dtype, w](SIMD[dtype, w](mu[0])),
        Plain[dtype, w](SIMD[dtype, w](sigma[0])),
    ).v


def _norm_cdf_step[
    dtype: DType, w: Int
](x: SIMD[dtype, w], mu: SIMD[dtype, 1], sigma: SIMD[dtype, 1]) -> SIMD[
    dtype, w
] where dtype.is_floating_point():
    return norm.cdf(
        Plain[dtype, w](x),
        Plain[dtype, w](SIMD[dtype, w](mu[0])),
        Plain[dtype, w](SIMD[dtype, w](sigma[0])),
    ).v


def _norm_ppf_step[
    dtype: DType, w: Int
](x: SIMD[dtype, w], mu: SIMD[dtype, 1], sigma: SIMD[dtype, 1]) -> SIMD[
    dtype, w
] where dtype.is_floating_point():
    return norm.ppf(
        Plain[dtype, w](x),
        Plain[dtype, w](SIMD[dtype, w](mu[0])),
        Plain[dtype, w](SIMD[dtype, w](sigma[0])),
    ).v


def _gamma_pdf_step[
    dtype: DType, w: Int
](x: SIMD[dtype, w], shape: SIMD[dtype, 1], scale: SIMD[dtype, 1]) -> SIMD[
    dtype, w
] where dtype.is_floating_point():
    return gamma.pdf(
        Plain[dtype, w](x),
        Plain[dtype, w](SIMD[dtype, w](shape[0])),
        Plain[dtype, w](SIMD[dtype, w](scale[0])),
    ).v


def _gamma_cdf_step[
    dtype: DType, w: Int
](x: SIMD[dtype, w], shape: SIMD[dtype, 1], scale: SIMD[dtype, 1]) -> SIMD[
    dtype, w
] where dtype.is_floating_point():
    return gamma.cdf(
        Plain[dtype, w](x),
        Plain[dtype, w](SIMD[dtype, w](shape[0])),
        Plain[dtype, w](SIMD[dtype, w](scale[0])),
    ).v


def _gamma_ppf_step[
    dtype: DType, w: Int
](x: SIMD[dtype, w], shape: SIMD[dtype, 1], scale: SIMD[dtype, 1]) -> SIMD[
    dtype, w
] where dtype.is_floating_point():
    return gamma.ppf(
        Plain[dtype, w](x),
        Plain[dtype, w](SIMD[dtype, w](shape[0])),
        Plain[dtype, w](SIMD[dtype, w](scale[0])),
    ).v


def _beta_pdf_step[
    dtype: DType, w: Int
](x: SIMD[dtype, w], a: SIMD[dtype, 1], b: SIMD[dtype, 1]) -> SIMD[
    dtype, w
] where dtype.is_floating_point():
    return beta.pdf(
        Plain[dtype, w](x),
        Plain[dtype, w](SIMD[dtype, w](a[0])),
        Plain[dtype, w](SIMD[dtype, w](b[0])),
    ).v


def _beta_cdf_step[
    dtype: DType, w: Int
](x: SIMD[dtype, w], a: SIMD[dtype, 1], b: SIMD[dtype, 1]) -> SIMD[
    dtype, w
] where dtype.is_floating_point():
    return beta.cdf(
        Plain[dtype, w](x),
        Plain[dtype, w](SIMD[dtype, w](a[0])),
        Plain[dtype, w](SIMD[dtype, w](b[0])),
    ).v


def _beta_ppf_step[
    dtype: DType, w: Int
](x: SIMD[dtype, w], a: SIMD[dtype, 1], b: SIMD[dtype, 1]) -> SIMD[
    dtype, w
] where dtype.is_floating_point():
    return beta.ppf(
        Plain[dtype, w](x),
        Plain[dtype, w](SIMD[dtype, w](a[0])),
        Plain[dtype, w](SIMD[dtype, w](b[0])),
    ).v


def _f_pdf_step[
    dtype: DType, w: Int
](x: SIMD[dtype, w], df1: SIMD[dtype, 1], df2: SIMD[dtype, 1]) -> SIMD[
    dtype, w
] where dtype.is_floating_point():
    return f.pdf(
        Plain[dtype, w](x),
        Plain[dtype, w](SIMD[dtype, w](df1[0])),
        Plain[dtype, w](SIMD[dtype, w](df2[0])),
    ).v


def _f_cdf_step[
    dtype: DType, w: Int
](x: SIMD[dtype, w], df1: SIMD[dtype, 1], df2: SIMD[dtype, 1]) -> SIMD[
    dtype, w
] where dtype.is_floating_point():
    return f.cdf(
        Plain[dtype, w](x),
        Plain[dtype, w](SIMD[dtype, w](df1[0])),
        Plain[dtype, w](SIMD[dtype, w](df2[0])),
    ).v


def _f_ppf_step[
    dtype: DType, w: Int
](x: SIMD[dtype, w], df1: SIMD[dtype, 1], df2: SIMD[dtype, 1]) -> SIMD[
    dtype, w
] where dtype.is_floating_point():
    return f.ppf(
        Plain[dtype, w](x),
        Plain[dtype, w](SIMD[dtype, w](df1[0])),
        Plain[dtype, w](SIMD[dtype, w](df2[0])),
    ).v


def _binom_pmf_step[
    dtype: DType, w: Int
](x: SIMD[dtype, w], n: SIMD[dtype, 1], prob: SIMD[dtype, 1]) -> SIMD[
    dtype, w
] where dtype.is_floating_point():
    return binom.pmf(
        Plain[dtype, w](x),
        Plain[dtype, w](SIMD[dtype, w](n[0])),
        Plain[dtype, w](SIMD[dtype, w](prob[0])),
    ).v


def _binom_cdf_step[
    dtype: DType, w: Int
](x: SIMD[dtype, w], n: SIMD[dtype, 1], prob: SIMD[dtype, 1]) -> SIMD[
    dtype, w
] where dtype.is_floating_point():
    return binom.cdf(
        Plain[dtype, w](x),
        Plain[dtype, w](SIMD[dtype, w](n[0])),
        Plain[dtype, w](SIMD[dtype, w](prob[0])),
    ).v


def _binom_ppf_step[
    dtype: DType, w: Int
](x: SIMD[dtype, w], n: SIMD[dtype, 1], prob: SIMD[dtype, 1]) -> SIMD[
    dtype, w
] where dtype.is_floating_point():
    return binom.ppf(
        Plain[dtype, w](x),
        Plain[dtype, w](SIMD[dtype, w](n[0])),
        Plain[dtype, w](SIMD[dtype, w](prob[0])),
    ).v


def _expon_pdf_step[
    dtype: DType, w: Int
](x: SIMD[dtype, w], rate: SIMD[dtype, 1]) -> SIMD[
    dtype, w
] where dtype.is_floating_point():
    return expon.pdf(
        Plain[dtype, w](x), Plain[dtype, w](SIMD[dtype, w](rate[0]))
    ).v


def _expon_cdf_step[
    dtype: DType, w: Int
](x: SIMD[dtype, w], rate: SIMD[dtype, 1]) -> SIMD[
    dtype, w
] where dtype.is_floating_point():
    return expon.cdf(
        Plain[dtype, w](x), Plain[dtype, w](SIMD[dtype, w](rate[0]))
    ).v


def _expon_ppf_step[
    dtype: DType, w: Int
](x: SIMD[dtype, w], rate: SIMD[dtype, 1]) -> SIMD[
    dtype, w
] where dtype.is_floating_point():
    return expon.ppf(
        Plain[dtype, w](x), Plain[dtype, w](SIMD[dtype, w](rate[0]))
    ).v


def _chi2_pdf_step[
    dtype: DType, w: Int
](x: SIMD[dtype, w], df: SIMD[dtype, 1]) -> SIMD[
    dtype, w
] where dtype.is_floating_point():
    return chi2.pdf(
        Plain[dtype, w](x), Plain[dtype, w](SIMD[dtype, w](df[0]))
    ).v


def _chi2_cdf_step[
    dtype: DType, w: Int
](x: SIMD[dtype, w], df: SIMD[dtype, 1]) -> SIMD[
    dtype, w
] where dtype.is_floating_point():
    return chi2.cdf(
        Plain[dtype, w](x), Plain[dtype, w](SIMD[dtype, w](df[0]))
    ).v


def _chi2_ppf_step[
    dtype: DType, w: Int
](x: SIMD[dtype, w], df: SIMD[dtype, 1]) -> SIMD[
    dtype, w
] where dtype.is_floating_point():
    return chi2.ppf(
        Plain[dtype, w](x), Plain[dtype, w](SIMD[dtype, w](df[0]))
    ).v


def _t_pdf_step[
    dtype: DType, w: Int
](x: SIMD[dtype, w], df: SIMD[dtype, 1]) -> SIMD[
    dtype, w
] where dtype.is_floating_point():
    return t.pdf(Plain[dtype, w](x), Plain[dtype, w](SIMD[dtype, w](df[0]))).v


def _t_cdf_step[
    dtype: DType, w: Int
](x: SIMD[dtype, w], df: SIMD[dtype, 1]) -> SIMD[
    dtype, w
] where dtype.is_floating_point():
    return t.cdf(Plain[dtype, w](x), Plain[dtype, w](SIMD[dtype, w](df[0]))).v


def _t_ppf_step[
    dtype: DType, w: Int
](x: SIMD[dtype, w], df: SIMD[dtype, 1]) -> SIMD[
    dtype, w
] where dtype.is_floating_point():
    return t.ppf(Plain[dtype, w](x), Plain[dtype, w](SIMD[dtype, w](df[0]))).v


def _poisson_pmf_step[
    dtype: DType, w: Int
](x: SIMD[dtype, w], rate: SIMD[dtype, 1]) -> SIMD[
    dtype, w
] where dtype.is_floating_point():
    return poisson.pmf(
        Plain[dtype, w](x), Plain[dtype, w](SIMD[dtype, w](rate[0]))
    ).v


def _poisson_cdf_step[
    dtype: DType, w: Int
](x: SIMD[dtype, w], rate: SIMD[dtype, 1]) -> SIMD[
    dtype, w
] where dtype.is_floating_point():
    return poisson.cdf(
        Plain[dtype, w](x), Plain[dtype, w](SIMD[dtype, w](rate[0]))
    ).v


def _poisson_ppf_step[
    dtype: DType, w: Int
](x: SIMD[dtype, w], rate: SIMD[dtype, 1]) -> SIMD[
    dtype, w
] where dtype.is_floating_point():
    return poisson.ppf(
        Plain[dtype, w](x), Plain[dtype, w](SIMD[dtype, w](rate[0]))
    ).v


comptime _P64 = Plain[DType.float64, 1]


def _p(x: Float64) -> _P64:
    return _P64(x)


def _v(x: _P64) -> Float64:
    return Float64(x.v[0])


def _log64(x: Float64) -> Float64:
    return _v(_p(x).ln())


def _sqrt64(x: Float64) -> Float64:
    return _v(_p(x).sqrt())


def _psi(x: Float64) -> Float64:
    return _v(digamma(_p(x)))


def _inf64() -> Float64:
    return Float64.MAX * 2.0


def _nan64() -> Float64:
    return _inf64() - _inf64()


def _gamma_entropy(shape: Float64) -> Float64:
    """The unit-scale gamma's entropy, `k + lgamma(k) + (1 - k) psi(k)`."""
    return shape + _v(lgamma(_p(shape))) + (1.0 - shape) * _psi(shape)


def _poisson_entropy(rate: Float64) -> Float64:
    """`-sum pmf ln pmf` over `k` out to `rate + 20 sqrt(rate) + 50`, past
    which the terms are below `1e-80` of the total."""
    var total = 0.0
    var bound = Int(rate + 20.0 * _sqrt64(rate) + 50.0)
    for k in range(bound + 1):
        var logp = _v(poisson.logpmf(_p(Float64(k)), _p(rate)))
        if logp > -700.0:
            total -= _v(_p(logp).exp()) * logp
    return total


def _binom_entropy(n: Float64, p: Float64) -> Float64:
    """`-sum pmf ln pmf` over `k` in `0 .. n`."""
    var total = 0.0
    for k in range(Int(n) + 1):
        var logp = _v(binom.logpmf(_p(Float64(k)), _p(n), _p(p)))
        if logp > -700.0:
            total -= _v(_p(logp).exp()) * logp
    return total


# ------------------------------------------------ tier-1 helpers for 0.3
#
# `FloatLike` carries no `atan`, `pow`, `expm1` or `log1p`, and growing the
# trait for one module's closed forms is the cost `numeric.mojo` warns
# about, so they are written here from what it does carry: fixed work,
# branchless, both sides of every blend safe.

comptime _EULER = 0.5772156649015329
comptime _PI = 3.141592653589793


def _pow[T: FloatLike](x: T, y: T) -> T:
    """`x^y` for `x > 0`, as `exp(y ln x)`; `x` floored away from zero."""
    return (y * _safe_ln(x)).exp()


def _expm1[T: FloatLike](u: T) -> T:
    """`e^u - 1`, with the cubic Taylor polynomial below `|u| = 1e-5`, where
    `exp(u) - 1` would cancel, and the direct form above."""
    var small = T.one() - ge_indicator(u.abs(), T.constant(1e-5))
    var series = u.copy() * (
        T.one() + u.copy() * (T.constant(0.5) + u.copy() / T.constant(6.0))
    )
    return blend(small, series, u.exp() - T.one())


def _log1p[T: FloatLike](u: T) -> T:
    """`ln(1 + u)` for `u > -1`, with the cubic Taylor polynomial below
    `|u| = 1e-5` and the direct form, clamped, above."""
    var small = T.one() - ge_indicator(u.abs(), T.constant(1e-5))
    var series = u.copy() * (
        T.one() - u.copy() * (T.constant(0.5) - u.copy() / T.constant(3.0))
    )
    return blend(small, series, _safe_ln(T.one() + u))


def _atan[T: FloatLike](x: T) -> T:
    """`atan(x)`: `pi/2 - atan(1/|x|)` above `|x| = 1`, two half-angle steps
    `t -> t / (1 + sqrt(1 + t^2))` bringing the argument under `tan(pi/16)`,
    and the alternating series to `t^23`, whose tail there is below
    `1e-17`; the sign restored at the end."""
    var a = x.abs()
    var big = ge_indicator(a.copy(), T.one())
    var t = blend(big.copy(), T.one() / max_of(a.copy(), T.constant(_TINY)), a)
    for _ in range(2):
        t = t.copy() / (T.one() + (T.one() + t.copy() * t.copy()).sqrt())
    var t2 = t.copy() * t.copy()
    var term = t.copy()
    var total = t.copy()
    comptime for k in range(1, 12):
        term = -(term * t2.copy())
        total = total + term.copy() / T.constant(Float64(2 * k + 1))
    var r = T.constant(4.0) * total
    var magnitude = blend(big, T.constant(_PI / 2.0) - r.copy(), r)
    return magnitude.copysign(x)


def _exp64(x: Float64) -> Float64:
    return _v(_p(x).exp())


def _gamma64(x: Float64) -> Float64:
    return _exp64(_v(lgamma(_p(x))))


def _ln_scalar[
    dtype: DType
](x: Scalar[dtype]) -> Scalar[dtype] where dtype.is_floating_point():
    return _log64_scalar(x)


def _log64_scalar[
    dtype: DType
](x: Scalar[dtype]) -> Scalar[dtype] where dtype.is_floating_point():
    return Scalar[dtype](_log64(Float64(x)))


def _over3[
    T: TensorLike,
    step: def[w: Int](
        SIMD[T.dtype, w], SIMD[T.dtype, 1], SIMD[T.dtype, 1], SIMD[T.dtype, 1]
    ) thin -> SIMD[T.dtype, w],
    gpu: Bool,
    name: StaticString,
](
    x: T, p0: Scalar[T.dtype], p1: Scalar[T.dtype], p2: Scalar[T.dtype]
) raises -> Tensor[T.dtype, T.LayoutType] where (
    is_row_major[T] and T.dtype.is_floating_point()
):
    """`_over1` for a three-parameter distribution."""
    comptime dtype = T.dtype
    comptime LayoutType = T.LayoutType
    if not _check_device[gpu=gpu](x):
        _notice[gpu](name)
        var values = x.to_host()
        var walked = List[Scalar[dtype]](length=len(values), fill=0)
        for i in range(len(values)):
            walked[i] = step[1](values[i], p0, p1, p2)[0]
        return Tensor[dtype, LayoutType](x.tile().layout, walked^, x.context())

    var ctx = x.context()
    var out = Tensor[dtype, LayoutType]._uninitialized(ctx, x.tile().layout)
    var xs = _flat(x)
    var ys = _flat_out(out)

    @always_inline
    def body[
        width: Int, alignment: Int = 1
    ](coord: Coord) {var xs, var ys, var p0, var p1, var p2}:
        ys.store[width](coord, step[width](xs.load[width](coord), p0, p1, p2))

    _launch[gpu=gpu, lanes=_width[dtype, gpu]()](body, x.size(), ctx)
    return out^


def _lbinom[T: FloatLike](a: T, b: T) -> T:
    """`ln C(a, b)` through `lgamma`, arguments assumed in range."""
    return (
        lgamma(a.copy() + T.one())
        - lgamma(b.copy() + T.one())
        - lgamma(a - b + T.one())
    )


def _bernoulli_entropy(p: Float64) -> Float64:
    var h = 0.0
    if p > 0.0:
        h -= p * _log64(p)
    if p < 1.0:
        h -= (1.0 - p) * _log64(1.0 - p)
    return h


def _nbinom_entropy(n: Float64, p: Float64) -> Float64:
    """`-sum pmf ln pmf` out to `mean + 30 sd + 50`."""
    var mean = n * (1.0 - p) / p
    var sd = _sqrt64(n * (1.0 - p) / (p * p))
    var total = 0.0
    for k in range(Int(mean + 30.0 * sd + 50.0) + 1):
        var logp = _v(nbinom.logpmf(_p(Float64(k)), _p(n), _p(p)))
        if logp > -700.0:
            total -= _exp64(logp) * logp
    return total


def _hypergeom_entropy(M: Float64, n: Float64, N: Float64) -> Float64:
    """`-sum pmf ln pmf` over the support."""
    var total = 0.0
    for k in range(Int(min(n, N)) + 1):
        var logp = _v(hypergeom.logpmf(_p(Float64(k)), _p(M), _p(n), _p(N)))
        if logp > -700.0:
            total -= _exp64(logp) * logp
    return total


# ---------------------------------------------------------------- normal


struct norm:
    """The normal (Gaussian) distribution. `scipy.stats.norm`."""

    @staticmethod
    def pdf[T: FloatLike](x: T, mu: T, sigma: T) -> T:
        """The normal density with mean `mu` and standard deviation `sigma`."""
        var z = (x - mu) / sigma
        return (-(z * z) / T.constant(2.0)).exp() / (
            sigma * T.constant(_SQRT_2PI)
        )

    @staticmethod
    def cdf[T: FloatLike](x: T, mu: T, sigma: T) -> T:
        """The normal CDF, as `0.5*erfc(-z/sqrt(2))`.

        Written with `erfc` rather than the equivalent `0.5*(1 + erf(z/sqrt2))`
        on purpose: for `z` around `-6` the `erf` form is `0.5*(1 - 0.999...)`,
        which cancels away most of its significant digits, while `erfc` returns
        the small tail probability directly. `Plain.erfc` delegates to
        `std.math`, so that accuracy is real rather than nominal.
        """
        var z = (x - mu) / sigma
        return T.constant(0.5) * (-(z / T.constant(_SQRT_2))).erfc()

    @staticmethod
    def ppf[T: FloatLike](p: T, mu: T, sigma: T) -> T:
        """The inverse normal CDF, for `0 < p < 1`."""
        return mu + sigma * _standard_normal_quantile(p)

    @staticmethod
    def sf[T: FloatLike](x: T, mu: T, sigma: T) -> T:
        """`P(X > x)`, as `0.5*erfc(z/sqrt(2))` -- the upper tail read off
        `erfc` directly, so it keeps its digits where `1 - cdf` would
        cancel them. `scipy.stats.norm.sf`."""
        var z = (x - mu) / sigma
        return T.constant(0.5) * (z / T.constant(_SQRT_2)).erfc()

    @staticmethod
    def isf[T: FloatLike](p: T, mu: T, sigma: T) -> T:
        """The inverse survival function, `ppf(1 - p)`; by symmetry
        `mu - sigma * z(p)`, which loses nothing to `1 - p`."""
        return mu - sigma * _standard_normal_quantile(p)

    @staticmethod
    def logpdf[T: FloatLike](x: T, mu: T, sigma: T) -> T:
        var z = (x - mu) / sigma
        return (
            -(z * z) / T.constant(2.0)
            - _safe_ln(sigma)
            - T.constant(0.5 * _LN_2PI)
        )

    @staticmethod
    def logcdf[T: FloatLike](x: T, mu: T, sigma: T) -> T:
        return _safe_ln(norm.cdf(x, mu, sigma))

    @staticmethod
    def logsf[T: FloatLike](x: T, mu: T, sigma: T) -> T:
        return _safe_ln(norm.sf(x, mu, sigma))

    @staticmethod
    def pdf[
        T: TensorLike, gpu: Bool = False
    ](
        x: T,
        mu: Scalar[T.dtype],
        sigma: Scalar[T.dtype],
    ) raises -> Tensor[
        T.dtype, T.LayoutType
    ] where (is_row_major[T] and T.dtype.is_floating_point()):
        """The density over a `Tensor`, parameters as scalars; see the module
        docstring for the two paths `gpu` picks between."""
        comptime dtype = T.dtype
        return _over2[step=_norm_pdf_step[dtype, _], gpu=gpu, name="norm.pdf"](
            x, mu, sigma
        )

    @staticmethod
    def cdf[
        T: TensorLike, gpu: Bool = False
    ](
        x: T,
        mu: Scalar[T.dtype],
        sigma: Scalar[T.dtype],
    ) raises -> Tensor[
        T.dtype, T.LayoutType
    ] where (is_row_major[T] and T.dtype.is_floating_point()):
        """The CDF over a `Tensor`, parameters as scalars; see the module
        docstring for the two paths `gpu` picks between."""
        comptime dtype = T.dtype
        return _over2[step=_norm_cdf_step[dtype, _], gpu=gpu, name="norm.cdf"](
            x, mu, sigma
        )

    @staticmethod
    def ppf[
        T: TensorLike, gpu: Bool = False
    ](
        p: T,
        mu: Scalar[T.dtype],
        sigma: Scalar[T.dtype],
    ) raises -> Tensor[
        T.dtype, T.LayoutType
    ] where (is_row_major[T] and T.dtype.is_floating_point()):
        """The quantile over a `Tensor`, parameters as scalars; see the module
        docstring for the two paths `gpu` picks between."""
        comptime dtype = T.dtype
        return _over2[step=_norm_ppf_step[dtype, _], gpu=gpu, name="norm.ppf"](
            p, mu, sigma
        )

    @staticmethod
    def rvs[
        dtype: DType, *dims: Int, gpu: Bool = False
    ](
        mu: Scalar[dtype],
        sigma: Scalar[dtype],
        mut rng: Generator,
        ctx: Optional[DeviceContext] = None,
    ) raises -> Static[dtype, *dims] where dtype.is_floating_point():
        """Random draws at the compile-time shape `dims`, from `rng`'s stream.
        `scipy.stats.norm.rvs(..., size=dims, random_state=rng)`.

        By `Generator.normal`, Box-Muller; on `ctx`'s device at `gpu=True`.

        Parameters:
            dtype: The element type of the draws.
            dims: The shape of the result.
            gpu: Whether to draw one thread per element on `ctx`'s device.

        Args:
            mu: The distribution's `mu`.
            sigma: The distribution's `sigma`.
            rng: The generator; its seed advances.
            ctx: The device to allocate and draw on; `None` means the host.

        Returns:
            A new `Static` tensor of draws.

        Raises:
            When allocation or the fill fails.
        """
        return rng.normal[dtype, *dims, gpu=gpu](mu, sigma, ctx)

    @staticmethod
    def mean(mu: Float64, sigma: Float64) -> Float64:
        """The distribution's mean. `scipy.stats.norm.mean`.

        Args:
            mu: The distribution's `mu`.
            sigma: The distribution's `sigma`.

        Returns:
            The mean; `inf` or NaN where SciPy gives them.
        """
        return mu

    @staticmethod
    def var(mu: Float64, sigma: Float64) -> Float64:
        """The distribution's variance. `scipy.stats.norm.var`.

        Args:
            mu: The distribution's `mu`.
            sigma: The distribution's `sigma`.

        Returns:
            The variance; `inf` or NaN where SciPy gives them.
        """
        return sigma * sigma

    @staticmethod
    def std(mu: Float64, sigma: Float64) -> Float64:
        """The standard deviation, `sqrt(var)`.

        Args:
            mu: As `var` takes it.
            sigma: As `var` takes it.

        Returns:
            The standard deviation.
        """
        return _sqrt64(norm.var(mu, sigma))

    @staticmethod
    def interval(
        confidence: Float64, mu: Float64, sigma: Float64
    ) -> Tuple[Float64, Float64]:
        """The central interval holding `confidence` of the probability,
        `(ppf((1 - c) / 2), ppf((1 + c) / 2))`. `scipy.stats.norm.interval`.

        Args:
            confidence: The probability inside the interval, in `[0, 1]`.
            mu: As `ppf` takes it.
            sigma: As `ppf` takes it.

        Returns:
            The interval's lower and upper ends.
        """
        var lo = norm.ppf(_p((1.0 - confidence) / 2.0), _p(mu), _p(sigma))
        var hi = norm.ppf(_p((1.0 + confidence) / 2.0), _p(mu), _p(sigma))
        return (_v(lo), _v(hi))

    @staticmethod
    def entropy(mu: Float64, sigma: Float64) -> Float64:
        """The differential entropy in nats. `scipy.stats.norm.entropy`.

        Args:
            mu: The distribution's `mu`.
            sigma: The distribution's `sigma`.

        Returns:
            The entropy.
        """
        return 0.5 * (_LN_2PI + 1.0) + _log64(sigma)


def _standard_normal_quantile[T: FloatLike, num_iters: Int = 3](p: T) -> T:
    """The standard normal quantile: an Abramowitz & Stegun 26.2.23 seed
    (max error ~4.5e-4) refined by Newton against `norm.cdf`.

    The seed is defined for the lower half only, so the upper half is
    handled by the symmetry `z(p) = -z(1-p)`, applied branchlessly: `q` is
    the smaller of `p` and `1-p` either way, and the sign comes from an
    indicator on `p >= 0.5`.
    """
    var q = min_of(p, T.one() - p)
    var t = (-(T.constant(2.0) * _safe_ln(q))).sqrt()

    var numerator = (
        T.constant(2.515517)
        + T.constant(0.802853) * t
        + (T.constant(0.010328) * t * t)
    )
    var denominator = (
        T.one()
        + T.constant(1.432788) * t
        + T.constant(0.189269) * t * t
        + T.constant(0.001308) * t * t * t
    )
    # A&S 26.2.23 gives the *upper*-tail value, hence the negation for the
    # lower tail `q` names.
    var lower = -(t - (numerator / denominator))
    var z = blend(ge_indicator(p, T.constant(0.5)), -lower, lower.copy())

    var zero = T.constant(0.0)
    var one = T.one()
    for _ in range(num_iters):
        var residual = norm.cdf(z.copy(), zero.copy(), one.copy())
        z = z + (
            -(
                (residual - p)
                / max_of(
                    norm.pdf(z.copy(), zero.copy(), one.copy()),
                    T.constant(_TINY),
                )
            )
        )

    return z^


# ----------------------------------------------------------- exponential


struct expon:
    """The exponential distribution, parameterized by `rate` (`1/scale`). `scipy.stats.expon`.
    """

    @staticmethod
    def pdf[T: FloatLike](x: T, rate: T) -> T:
        """`rate*exp(-rate*x)` on `x >= 0`, and `0` below it."""
        var inside = rate * (-(rate * max_of(x, T.constant(0.0)))).exp()
        return inside * ge_indicator(x, T.constant(0.0))

    @staticmethod
    def cdf[T: FloatLike](x: T, rate: T) -> T:
        var inside = T.one() - (-(rate * max_of(x, T.constant(0.0)))).exp()
        return inside * ge_indicator(x, T.constant(0.0))

    @staticmethod
    def sf[T: FloatLike](x: T, rate: T) -> T:
        """`exp(-rate*x)` on `x >= 0`, and `1` below it."""
        var inside = (-(rate * max_of(x, T.constant(0.0)))).exp()
        return blend(ge_indicator(x, T.constant(0.0)), inside, T.one())

    @staticmethod
    def ppf[T: FloatLike](p: T, rate: T) -> T:
        """The inverse CDF, closed form: `-ln(1 - p) / rate`."""
        return -(_safe_ln(T.one() - p)) / rate

    @staticmethod
    def isf[T: FloatLike](p: T, rate: T) -> T:
        """`-ln(p) / rate`, which is `ppf(1 - p)` without the `1 - p`."""
        return -(_safe_ln(p)) / rate

    @staticmethod
    def logpdf[T: FloatLike](x: T, rate: T) -> T:
        var interior = _safe_ln(rate) - rate * max_of(x, T.constant(0.0))
        return blend(
            ge_indicator(x, T.constant(0.0)), interior, T.constant(_LOG_ZERO)
        )

    @staticmethod
    def logcdf[T: FloatLike](x: T, rate: T) -> T:
        return _safe_ln(expon.cdf(x, rate))

    @staticmethod
    def logsf[T: FloatLike](x: T, rate: T) -> T:
        return _safe_ln(expon.sf(x, rate))

    @staticmethod
    def pdf[
        T: TensorLike, gpu: Bool = False
    ](x: T, rate: Scalar[T.dtype]) raises -> Tensor[
        T.dtype, T.LayoutType
    ] where (is_row_major[T] and T.dtype.is_floating_point()):
        """The density over a `Tensor`, parameters as scalars; see the module
        docstring for the two paths `gpu` picks between."""
        comptime dtype = T.dtype
        return _over1[
            step=_expon_pdf_step[dtype, _], gpu=gpu, name="expon.pdf"
        ](x, rate)

    @staticmethod
    def cdf[
        T: TensorLike, gpu: Bool = False
    ](x: T, rate: Scalar[T.dtype]) raises -> Tensor[
        T.dtype, T.LayoutType
    ] where (is_row_major[T] and T.dtype.is_floating_point()):
        """The CDF over a `Tensor`, parameters as scalars; see the module
        docstring for the two paths `gpu` picks between."""
        comptime dtype = T.dtype
        return _over1[
            step=_expon_cdf_step[dtype, _], gpu=gpu, name="expon.cdf"
        ](x, rate)

    @staticmethod
    def ppf[
        T: TensorLike, gpu: Bool = False
    ](p: T, rate: Scalar[T.dtype]) raises -> Tensor[
        T.dtype, T.LayoutType
    ] where (is_row_major[T] and T.dtype.is_floating_point()):
        """The quantile over a `Tensor`, parameters as scalars; see the module
        docstring for the two paths `gpu` picks between."""
        comptime dtype = T.dtype
        return _over1[
            step=_expon_ppf_step[dtype, _], gpu=gpu, name="expon.ppf"
        ](p, rate)

    @staticmethod
    def rvs[
        dtype: DType, *dims: Int, gpu: Bool = False
    ](
        rate: Scalar[dtype],
        mut rng: Generator,
        ctx: Optional[DeviceContext] = None,
    ) raises -> Static[dtype, *dims] where dtype.is_floating_point():
        """Random draws at the compile-time shape `dims`, from `rng`'s stream.
        `scipy.stats.expon.rvs(..., size=dims, random_state=rng)`.

        By `Generator.exponential` at scale `1 / rate`; on `ctx`'s device at `gpu=True`.

        Parameters:
            dtype: The element type of the draws.
            dims: The shape of the result.
            gpu: Whether to draw one thread per element on `ctx`'s device.

        Args:
            rate: The distribution's `rate`.
            rng: The generator; its seed advances.
            ctx: The device to allocate and draw on; `None` means the host.

        Returns:
            A new `Static` tensor of draws.

        Raises:
            When allocation or the fill fails.
        """
        return rng.exponential[dtype, *dims, gpu=gpu](1 / rate, ctx)

    @staticmethod
    def mean(rate: Float64) -> Float64:
        """The distribution's mean. `scipy.stats.expon.mean`.

        Args:
            rate: The distribution's `rate`.

        Returns:
            The mean; `inf` or NaN where SciPy gives them.
        """
        return 1.0 / rate

    @staticmethod
    def var(rate: Float64) -> Float64:
        """The distribution's variance. `scipy.stats.expon.var`.

        Args:
            rate: The distribution's `rate`.

        Returns:
            The variance; `inf` or NaN where SciPy gives them.
        """
        return 1.0 / (rate * rate)

    @staticmethod
    def std(rate: Float64) -> Float64:
        """The standard deviation, `sqrt(var)`.

        Args:
            rate: As `var` takes it.

        Returns:
            The standard deviation.
        """
        return _sqrt64(expon.var(rate))

    @staticmethod
    def interval(confidence: Float64, rate: Float64) -> Tuple[Float64, Float64]:
        """The central interval holding `confidence` of the probability,
        `(ppf((1 - c) / 2), ppf((1 + c) / 2))`. `scipy.stats.expon.interval`.

        Args:
            confidence: The probability inside the interval, in `[0, 1]`.
            rate: As `ppf` takes it.

        Returns:
            The interval's lower and upper ends.
        """
        var lo = expon.ppf(_p((1.0 - confidence) / 2.0), _p(rate))
        var hi = expon.ppf(_p((1.0 + confidence) / 2.0), _p(rate))
        return (_v(lo), _v(hi))

    @staticmethod
    def entropy(rate: Float64) -> Float64:
        """The differential entropy in nats. `scipy.stats.expon.entropy`.

        Args:
            rate: The distribution's `rate`.

        Returns:
            The entropy.
        """
        return 1.0 - _log64(rate)


struct gamma:
    """The gamma distribution. `scipy.stats.gamma`. Distinct from `numax.special.gamma`, the function -- this is the distribution named after it.
    """

    @staticmethod
    def pdf[T: FloatLike](x: T, shape: T, scale: T) -> T:
        """The gamma density in the shape/scale parameterization.

        Computed in log space, which is what keeps it usable for large `shape`:
        the direct form needs `x^(shape-1)` and `Gamma(shape)` separately, and
        both overflow long before their ratio does.
        """
        return gamma.logpdf(x, shape, scale).exp()

    @staticmethod
    def logpdf[T: FloatLike](x: T, shape: T, scale: T) -> T:
        """The gamma log density; `_LOG_ZERO` below the support."""
        var z = max_of(x, T.constant(0.0)) / scale
        var interior = (
            (shape - T.one()) * _safe_ln(z)
            - z
            - lgamma(shape.copy())
            - _safe_ln(scale)
        )
        return blend(
            ge_indicator(x, T.constant(0.0)), interior, T.constant(_LOG_ZERO)
        )

    @staticmethod
    def cdf[T: FloatLike](x: T, shape: T, scale: T) -> T:
        """The gamma CDF -- `gammainc` under the substitution `x/scale`."""
        var z = max_of(x, T.constant(0.0)) / scale
        return gammainc(shape.copy(), z^) * ge_indicator(x, T.constant(0.0))

    @staticmethod
    def sf[T: FloatLike](x: T, shape: T, scale: T) -> T:
        """`P(X > x)` -- `gammaincc` under the same substitution, `1` below
        the support. See the module docstring on what `gammaincc` is."""
        var z = max_of(x, T.constant(0.0)) / scale
        return blend(
            ge_indicator(x, T.constant(0.0)),
            gammaincc(shape.copy(), z^),
            T.one(),
        )

    @staticmethod
    def logcdf[T: FloatLike](x: T, shape: T, scale: T) -> T:
        return _safe_ln(gamma.cdf(x, shape, scale))

    @staticmethod
    def logsf[T: FloatLike](x: T, shape: T, scale: T) -> T:
        return _safe_ln(gamma.sf(x, shape, scale))

    @staticmethod
    def ppf[T: FloatLike, num_iters: Int = 12](p: T, shape: T, scale: T) -> T:
        """The inverse gamma CDF, from a Wilson-Hilferty seed.

        Wilson-Hilferty approximates the gamma as a cube-rooted normal, which
        is accurate for `shape` above about 1 and increasingly rough below it.
        The Newton refinement covers the difference; the seed is floored away
        from zero because the cube can go negative for small `shape` and large
        lower-tail `p`, and `ln` of a negative seed would poison the lane.
        """
        var z = _standard_normal_quantile(p.copy())
        var a9 = T.constant(9.0) * shape
        var base = T.one() - (T.one() / a9) + z / a9.sqrt()
        var seed = max_of(shape * base * base * base, T.constant(1e-6))

        var x = seed * scale
        for _ in range(num_iters):
            var residual = gamma.cdf(x.copy(), shape.copy(), scale.copy())
            var density = max_of(
                gamma.pdf(x.copy(), shape.copy(), scale.copy()),
                T.constant(_TINY),
            )
            x = max_of(x - ((residual - p) / density), T.constant(_TINY))

        return x^

    @staticmethod
    def isf[T: FloatLike, num_iters: Int = 12](p: T, shape: T, scale: T) -> T:
        """`ppf(1 - p)`."""
        return gamma.ppf[T, num_iters](T.one() - p, shape, scale)

    @staticmethod
    def pdf[
        T: TensorLike, gpu: Bool = False
    ](
        x: T,
        shape: Scalar[T.dtype],
        scale: Scalar[T.dtype],
    ) raises -> Tensor[
        T.dtype, T.LayoutType
    ] where (is_row_major[T] and T.dtype.is_floating_point()):
        """The density over a `Tensor`, parameters as scalars; see the module
        docstring for the two paths `gpu` picks between."""
        comptime dtype = T.dtype
        return _over2[
            step=_gamma_pdf_step[dtype, _], gpu=gpu, name="gamma.pdf"
        ](x, shape, scale)

    @staticmethod
    def cdf[
        T: TensorLike, gpu: Bool = False
    ](
        x: T,
        shape: Scalar[T.dtype],
        scale: Scalar[T.dtype],
    ) raises -> Tensor[
        T.dtype, T.LayoutType
    ] where (is_row_major[T] and T.dtype.is_floating_point()):
        """The CDF over a `Tensor`, parameters as scalars; see the module
        docstring for the two paths `gpu` picks between."""
        comptime dtype = T.dtype
        return _over2[
            step=_gamma_cdf_step[dtype, _], gpu=gpu, name="gamma.cdf"
        ](x, shape, scale)

    @staticmethod
    def ppf[
        T: TensorLike, gpu: Bool = False
    ](
        p: T,
        shape: Scalar[T.dtype],
        scale: Scalar[T.dtype],
    ) raises -> Tensor[
        T.dtype, T.LayoutType
    ] where (is_row_major[T] and T.dtype.is_floating_point()):
        """The quantile over a `Tensor`, parameters as scalars; see the module
        docstring for the two paths `gpu` picks between."""
        comptime dtype = T.dtype
        return _over2[
            step=_gamma_ppf_step[dtype, _], gpu=gpu, name="gamma.ppf"
        ](p, shape, scale)

    @staticmethod
    def rvs[
        dtype: DType, *dims: Int, gpu: Bool = False
    ](
        shape: Scalar[dtype],
        scale: Scalar[dtype],
        mut rng: Generator,
        ctx: Optional[DeviceContext] = None,
    ) raises -> Static[dtype, *dims] where dtype.is_floating_point():
        """Random draws at the compile-time shape `dims`, from `rng`'s stream.
        `scipy.stats.gamma.rvs(..., size=dims, random_state=rng)`.

        By `Generator.gamma`, Marsaglia-Tsang; on `ctx`'s device at `gpu=True`.

        Parameters:
            dtype: The element type of the draws.
            dims: The shape of the result.
            gpu: Whether to draw one thread per element on `ctx`'s device.

        Args:
            shape: The distribution's `shape`.
            scale: The distribution's `scale`.
            rng: The generator; its seed advances.
            ctx: The device to allocate and draw on; `None` means the host.

        Returns:
            A new `Static` tensor of draws.

        Raises:
            When allocation or the fill fails.
        """
        return rng.gamma[dtype, *dims, gpu=gpu](shape, scale, ctx)

    @staticmethod
    def mean(shape: Float64, scale: Float64) -> Float64:
        """The distribution's mean. `scipy.stats.gamma.mean`.

        Args:
            shape: The distribution's `shape`.
            scale: The distribution's `scale`.

        Returns:
            The mean; `inf` or NaN where SciPy gives them.
        """
        return shape * scale

    @staticmethod
    def var(shape: Float64, scale: Float64) -> Float64:
        """The distribution's variance. `scipy.stats.gamma.var`.

        Args:
            shape: The distribution's `shape`.
            scale: The distribution's `scale`.

        Returns:
            The variance; `inf` or NaN where SciPy gives them.
        """
        return shape * scale * scale

    @staticmethod
    def std(shape: Float64, scale: Float64) -> Float64:
        """The standard deviation, `sqrt(var)`.

        Args:
            shape: As `var` takes it.
            scale: As `var` takes it.

        Returns:
            The standard deviation.
        """
        return _sqrt64(gamma.var(shape, scale))

    @staticmethod
    def interval(
        confidence: Float64, shape: Float64, scale: Float64
    ) -> Tuple[Float64, Float64]:
        """The central interval holding `confidence` of the probability,
        `(ppf((1 - c) / 2), ppf((1 + c) / 2))`. `scipy.stats.gamma.interval`.

        Args:
            confidence: The probability inside the interval, in `[0, 1]`.
            shape: As `ppf` takes it.
            scale: As `ppf` takes it.

        Returns:
            The interval's lower and upper ends.
        """
        var lo = gamma.ppf(_p((1.0 - confidence) / 2.0), _p(shape), _p(scale))
        var hi = gamma.ppf(_p((1.0 + confidence) / 2.0), _p(shape), _p(scale))
        return (_v(lo), _v(hi))

    @staticmethod
    def entropy(shape: Float64, scale: Float64) -> Float64:
        """The differential entropy in nats. `scipy.stats.gamma.entropy`.

        Args:
            shape: The distribution's `shape`.
            scale: The distribution's `scale`.

        Returns:
            The entropy.
        """
        return _gamma_entropy(shape) + _log64(scale)


struct chi2:
    """The chi-squared distribution. `scipy.stats.chi2`."""

    @staticmethod
    def pdf[T: FloatLike](x: T, df: T) -> T:
        """Chi-square with `df` degrees of freedom -- gamma with
        `shape = df/2`, `scale = 2`."""
        return gamma.pdf(x, df / T.constant(2.0), T.constant(2.0))

    @staticmethod
    def cdf[T: FloatLike](x: T, df: T) -> T:
        return gamma.cdf(x, df / T.constant(2.0), T.constant(2.0))

    @staticmethod
    def ppf[T: FloatLike](p: T, df: T) -> T:
        return gamma.ppf(p, df / T.constant(2.0), T.constant(2.0))

    @staticmethod
    def sf[T: FloatLike](x: T, df: T) -> T:
        return gamma.sf(x, df / T.constant(2.0), T.constant(2.0))

    @staticmethod
    def isf[T: FloatLike](p: T, df: T) -> T:
        return gamma.isf(p, df / T.constant(2.0), T.constant(2.0))

    @staticmethod
    def logpdf[T: FloatLike](x: T, df: T) -> T:
        return gamma.logpdf(x, df / T.constant(2.0), T.constant(2.0))

    @staticmethod
    def logcdf[T: FloatLike](x: T, df: T) -> T:
        return gamma.logcdf(x, df / T.constant(2.0), T.constant(2.0))

    @staticmethod
    def logsf[T: FloatLike](x: T, df: T) -> T:
        return gamma.logsf(x, df / T.constant(2.0), T.constant(2.0))

    @staticmethod
    def pdf[
        T: TensorLike, gpu: Bool = False
    ](x: T, df: Scalar[T.dtype]) raises -> Tensor[T.dtype, T.LayoutType] where (
        is_row_major[T] and T.dtype.is_floating_point()
    ):
        """The density over a `Tensor`, parameters as scalars; see the module
        docstring for the two paths `gpu` picks between."""
        comptime dtype = T.dtype
        return _over1[step=_chi2_pdf_step[dtype, _], gpu=gpu, name="chi2.pdf"](
            x, df
        )

    @staticmethod
    def cdf[
        T: TensorLike, gpu: Bool = False
    ](x: T, df: Scalar[T.dtype]) raises -> Tensor[T.dtype, T.LayoutType] where (
        is_row_major[T] and T.dtype.is_floating_point()
    ):
        """The CDF over a `Tensor`, parameters as scalars; see the module
        docstring for the two paths `gpu` picks between."""
        comptime dtype = T.dtype
        return _over1[step=_chi2_cdf_step[dtype, _], gpu=gpu, name="chi2.cdf"](
            x, df
        )

    @staticmethod
    def ppf[
        T: TensorLike, gpu: Bool = False
    ](p: T, df: Scalar[T.dtype]) raises -> Tensor[T.dtype, T.LayoutType] where (
        is_row_major[T] and T.dtype.is_floating_point()
    ):
        """The quantile over a `Tensor`, parameters as scalars; see the module
        docstring for the two paths `gpu` picks between."""
        comptime dtype = T.dtype
        return _over1[step=_chi2_ppf_step[dtype, _], gpu=gpu, name="chi2.ppf"](
            p, df
        )

    @staticmethod
    def rvs[
        dtype: DType, *dims: Int, gpu: Bool = False
    ](
        df: Scalar[dtype],
        mut rng: Generator,
        ctx: Optional[DeviceContext] = None,
    ) raises -> Static[dtype, *dims] where dtype.is_floating_point():
        """Random draws at the compile-time shape `dims`, from `rng`'s stream.
        `scipy.stats.chi2.rvs(..., size=dims, random_state=rng)`.

        By `Generator.gamma` at shape `df / 2` and scale `2`; on `ctx`'s device at `gpu=True`.

        Parameters:
            dtype: The element type of the draws.
            dims: The shape of the result.
            gpu: Whether to draw one thread per element on `ctx`'s device.

        Args:
            df: The distribution's `df`.
            rng: The generator; its seed advances.
            ctx: The device to allocate and draw on; `None` means the host.

        Returns:
            A new `Static` tensor of draws.

        Raises:
            When allocation or the fill fails.
        """
        return rng.gamma[dtype, *dims, gpu=gpu](df / 2, 2, ctx)

    @staticmethod
    def mean(df: Float64) -> Float64:
        """The distribution's mean. `scipy.stats.chi2.mean`.

        Args:
            df: The distribution's `df`.

        Returns:
            The mean; `inf` or NaN where SciPy gives them.
        """
        return df

    @staticmethod
    def var(df: Float64) -> Float64:
        """The distribution's variance. `scipy.stats.chi2.var`.

        Args:
            df: The distribution's `df`.

        Returns:
            The variance; `inf` or NaN where SciPy gives them.
        """
        return 2.0 * df

    @staticmethod
    def std(df: Float64) -> Float64:
        """The standard deviation, `sqrt(var)`.

        Args:
            df: As `var` takes it.

        Returns:
            The standard deviation.
        """
        return _sqrt64(chi2.var(df))

    @staticmethod
    def interval(confidence: Float64, df: Float64) -> Tuple[Float64, Float64]:
        """The central interval holding `confidence` of the probability,
        `(ppf((1 - c) / 2), ppf((1 + c) / 2))`. `scipy.stats.chi2.interval`.

        Args:
            confidence: The probability inside the interval, in `[0, 1]`.
            df: As `ppf` takes it.

        Returns:
            The interval's lower and upper ends.
        """
        var lo = chi2.ppf(_p((1.0 - confidence) / 2.0), _p(df))
        var hi = chi2.ppf(_p((1.0 + confidence) / 2.0), _p(df))
        return (_v(lo), _v(hi))

    @staticmethod
    def entropy(df: Float64) -> Float64:
        """The differential entropy in nats. `scipy.stats.chi2.entropy`.

        Args:
            df: The distribution's `df`.

        Returns:
            The entropy.
        """
        return _gamma_entropy(df / 2.0) + _log64(2.0)


struct beta:
    """The beta distribution. `scipy.stats.beta`. Distinct from `numax.special.beta`, the function.
    """

    @staticmethod
    def pdf[T: FloatLike](x: T, a: T, b: T) -> T:
        """The beta density on `0 < x < 1`, and `0` outside it."""
        return beta.logpdf(x, a, b).exp()

    @staticmethod
    def logpdf[T: FloatLike](x: T, a: T, b: T) -> T:
        """The beta log density; `_LOG_ZERO` outside `[0, 1]`."""
        var xc = min_of(max_of(x, T.constant(0.0)), T.one())
        var interior = (
            (a - T.one()) * _safe_ln(xc)
            + (b - T.one()) * _safe_ln(T.one() - xc)
            - _log_beta(a, b)
        )
        var support = ge_indicator(x, T.constant(0.0)) * ge_indicator(
            T.one(), x.copy()
        )
        return blend(support, interior, T.constant(_LOG_ZERO))

    @staticmethod
    def cdf[T: FloatLike](x: T, a: T, b: T) -> T:
        """The beta CDF -- `betainc` directly, clamped to its support."""
        var xc = min_of(max_of(x, T.constant(0.0)), T.one())
        return betainc(xc^, a, b)

    @staticmethod
    def sf[T: FloatLike](x: T, a: T, b: T) -> T:
        """`P(X > x)` -- `betaincc`, which is `I_{1-x}(b, a)` and so reads
        the upper tail directly rather than as `1 - cdf`."""
        var xc = min_of(max_of(x, T.constant(0.0)), T.one())
        return betaincc(xc^, a, b)

    @staticmethod
    def logcdf[T: FloatLike](x: T, a: T, b: T) -> T:
        return _safe_ln(beta.cdf(x, a, b))

    @staticmethod
    def logsf[T: FloatLike](x: T, a: T, b: T) -> T:
        return _safe_ln(beta.sf(x, a, b))

    @staticmethod
    def ppf[T: FloatLike, num_iters: Int = 20](p: T, a: T, b: T) -> T:
        """The inverse beta CDF, Newton from the distribution's mean.

        Each step is clamped back inside `(0, 1)`: Newton on a CDF that
        saturates near both endpoints readily proposes a point outside the
        support, and a clamped step is a bounded loss where an escaped one is
        unrecoverable.
        """
        var lo = T.constant(1e-8)
        var hi = T.one() - lo
        var x = min_of(max_of(a / (a + b), lo.copy()), hi.copy())

        for _ in range(num_iters):
            var residual = beta.cdf(x.copy(), a.copy(), b.copy())
            var density = max_of(
                beta.pdf(x.copy(), a.copy(), b.copy()), T.constant(_TINY)
            )
            var proposal = x - ((residual - p) / density)
            x = min_of(max_of(proposal^, lo.copy()), hi.copy())

        return x^

    @staticmethod
    def isf[T: FloatLike, num_iters: Int = 20](p: T, a: T, b: T) -> T:
        """`ppf(1 - p)`."""
        return beta.ppf[T, num_iters](T.one() - p, a, b)

    @staticmethod
    def pdf[
        T: TensorLike, gpu: Bool = False
    ](x: T, a: Scalar[T.dtype], b: Scalar[T.dtype]) raises -> Tensor[
        T.dtype, T.LayoutType
    ] where (is_row_major[T] and T.dtype.is_floating_point()):
        """The density over a `Tensor`, parameters as scalars; see the module
        docstring for the two paths `gpu` picks between."""
        comptime dtype = T.dtype
        return _over2[step=_beta_pdf_step[dtype, _], gpu=gpu, name="beta.pdf"](
            x, a, b
        )

    @staticmethod
    def cdf[
        T: TensorLike, gpu: Bool = False
    ](x: T, a: Scalar[T.dtype], b: Scalar[T.dtype]) raises -> Tensor[
        T.dtype, T.LayoutType
    ] where (is_row_major[T] and T.dtype.is_floating_point()):
        """The CDF over a `Tensor`, parameters as scalars; see the module
        docstring for the two paths `gpu` picks between."""
        comptime dtype = T.dtype
        return _over2[step=_beta_cdf_step[dtype, _], gpu=gpu, name="beta.cdf"](
            x, a, b
        )

    @staticmethod
    def ppf[
        T: TensorLike, gpu: Bool = False
    ](p: T, a: Scalar[T.dtype], b: Scalar[T.dtype]) raises -> Tensor[
        T.dtype, T.LayoutType
    ] where (is_row_major[T] and T.dtype.is_floating_point()):
        """The quantile over a `Tensor`, parameters as scalars; see the module
        docstring for the two paths `gpu` picks between."""
        comptime dtype = T.dtype
        return _over2[step=_beta_ppf_step[dtype, _], gpu=gpu, name="beta.ppf"](
            p, a, b
        )

    @staticmethod
    def rvs[
        dtype: DType, *dims: Int, gpu: Bool = False
    ](
        a: Scalar[dtype],
        b: Scalar[dtype],
        mut rng: Generator,
        ctx: Optional[DeviceContext] = None,
    ) raises -> Static[dtype, *dims] where dtype.is_floating_point():
        """Random draws at the compile-time shape `dims`, from `rng`'s stream.
        `scipy.stats.beta.rvs(..., size=dims, random_state=rng)`.

        By `Generator.beta`, the ratio of two gammas; on `ctx`'s device at `gpu=True`.

        Parameters:
            dtype: The element type of the draws.
            dims: The shape of the result.
            gpu: Whether to draw one thread per element on `ctx`'s device.

        Args:
            a: The distribution's `a`.
            b: The distribution's `b`.
            rng: The generator; its seed advances.
            ctx: The device to allocate and draw on; `None` means the host.

        Returns:
            A new `Static` tensor of draws.

        Raises:
            When allocation or the fill fails.
        """
        return rng.beta[dtype, *dims, gpu=gpu](a, b, ctx)

    @staticmethod
    def mean(a: Float64, b: Float64) -> Float64:
        """The distribution's mean. `scipy.stats.beta.mean`.

        Args:
            a: The distribution's `a`.
            b: The distribution's `b`.

        Returns:
            The mean; `inf` or NaN where SciPy gives them.
        """
        return a / (a + b)

    @staticmethod
    def var(a: Float64, b: Float64) -> Float64:
        """The distribution's variance. `scipy.stats.beta.var`.

        Args:
            a: The distribution's `a`.
            b: The distribution's `b`.

        Returns:
            The variance; `inf` or NaN where SciPy gives them.
        """
        return a * b / ((a + b) * (a + b) * (a + b + 1.0))

    @staticmethod
    def std(a: Float64, b: Float64) -> Float64:
        """The standard deviation, `sqrt(var)`.

        Args:
            a: As `var` takes it.
            b: As `var` takes it.

        Returns:
            The standard deviation.
        """
        return _sqrt64(beta.var(a, b))

    @staticmethod
    def interval(
        confidence: Float64, a: Float64, b: Float64
    ) -> Tuple[Float64, Float64]:
        """The central interval holding `confidence` of the probability,
        `(ppf((1 - c) / 2), ppf((1 + c) / 2))`. `scipy.stats.beta.interval`.

        Args:
            confidence: The probability inside the interval, in `[0, 1]`.
            a: As `ppf` takes it.
            b: As `ppf` takes it.

        Returns:
            The interval's lower and upper ends.
        """
        var lo = beta.ppf(_p((1.0 - confidence) / 2.0), _p(a), _p(b))
        var hi = beta.ppf(_p((1.0 + confidence) / 2.0), _p(a), _p(b))
        return (_v(lo), _v(hi))

    @staticmethod
    def entropy(a: Float64, b: Float64) -> Float64:
        """The differential entropy in nats. `scipy.stats.beta.entropy`.

        Args:
            a: The distribution's `a`.
            b: The distribution's `b`.

        Returns:
            The entropy.
        """
        return (
            _v(betaln(_p(a), _p(b)))
            - (a - 1.0) * _psi(a)
            - (b - 1.0) * _psi(b)
            + (a + b - 2.0) * _psi(a + b)
        )


struct t:
    """Student's t distribution. `scipy.stats.t`."""

    @staticmethod
    def pdf[T: FloatLike](x: T, df: T) -> T:
        return t.logpdf(x, df).exp()

    @staticmethod
    def logpdf[T: FloatLike](x: T, df: T) -> T:
        var half = (df + T.one()) / T.constant(2.0)
        return (
            lgamma(half.copy())
            - lgamma(df / T.constant(2.0))
            - (T.constant(0.5) * (_safe_ln(df) + T.constant(_LN_PI)))
            - (half * (T.one() + x * x / df).ln())
        )

    @staticmethod
    def sf[T: FloatLike](x: T, df: T) -> T:
        """`P(T > x)`, which by symmetry is `cdf(-x)` -- exact, and it
        keeps the upper tail's digits."""
        return t.cdf(-x, df)

    @staticmethod
    def logcdf[T: FloatLike](x: T, df: T) -> T:
        return _safe_ln(t.cdf(x, df))

    @staticmethod
    def logsf[T: FloatLike](x: T, df: T) -> T:
        return _safe_ln(t.sf(x, df))

    @staticmethod
    def cdf[T: FloatLike](x: T, df: T) -> T:
        """`P(T <= x)`, via the beta identity
        `P(T <= x) = 1 - 0.5*I_z(df/2, 1/2)` for `x >= 0`, with
        `z = df/(df + x^2)`.

        The two halves are combined branchlessly by sign rather than by an
        `if`, using `0.5 + 0.5*sign(x)*(1 - I_z)`: `z` depends on `x` only
        through `x^2`, so both halves share one `betainc` call and the sign is
        all that distinguishes them. At `x = 0` this gives `z = 1`,
        `I_1 = 1`, and exactly `0.5`.
        """
        var z = df / (df + x * x)
        var tail = betainc(z^, df / T.constant(2.0), T.constant(0.5))
        var sign = T.one().copysign(x)
        return T.constant(0.5) + T.constant(0.5) * sign * (T.one() - tail)

    @staticmethod
    def ppf[T: FloatLike, num_iters: Int = 12](p: T, df: T) -> T:
        """The inverse Student-t CDF, Newton from the normal quantile.

        The normal is the `df -> infinity` limit of the t, so it's a good seed
        for large `df` and a merely-adequate one for small `df`, where the t's
        heavier tails put the true quantile further out.
        """
        var x = _standard_normal_quantile(p.copy())

        for _ in range(num_iters):
            var residual = t.cdf(x.copy(), df.copy())
            var density = max_of(t.pdf(x.copy(), df.copy()), T.constant(_TINY))
            x = x - ((residual - p) / density)

        return x^

    @staticmethod
    def isf[T: FloatLike, num_iters: Int = 12](p: T, df: T) -> T:
        """`ppf(1 - p)`, which by symmetry is `-ppf(p)`."""
        return -t.ppf[T, num_iters](p, df)

    @staticmethod
    def pdf[
        T: TensorLike, gpu: Bool = False
    ](x: T, df: Scalar[T.dtype]) raises -> Tensor[T.dtype, T.LayoutType] where (
        is_row_major[T] and T.dtype.is_floating_point()
    ):
        """The density over a `Tensor`, parameters as scalars; see the module
        docstring for the two paths `gpu` picks between."""
        comptime dtype = T.dtype
        return _over1[step=_t_pdf_step[dtype, _], gpu=gpu, name="t.pdf"](x, df)

    @staticmethod
    def cdf[
        T: TensorLike, gpu: Bool = False
    ](x: T, df: Scalar[T.dtype]) raises -> Tensor[T.dtype, T.LayoutType] where (
        is_row_major[T] and T.dtype.is_floating_point()
    ):
        """The CDF over a `Tensor`, parameters as scalars; see the module
        docstring for the two paths `gpu` picks between."""
        comptime dtype = T.dtype
        return _over1[step=_t_cdf_step[dtype, _], gpu=gpu, name="t.cdf"](x, df)

    @staticmethod
    def ppf[
        T: TensorLike, gpu: Bool = False
    ](p: T, df: Scalar[T.dtype]) raises -> Tensor[T.dtype, T.LayoutType] where (
        is_row_major[T] and T.dtype.is_floating_point()
    ):
        """The quantile over a `Tensor`, parameters as scalars; see the module
        docstring for the two paths `gpu` picks between."""
        comptime dtype = T.dtype
        return _over1[step=_t_ppf_step[dtype, _], gpu=gpu, name="t.ppf"](p, df)

    @staticmethod
    def rvs[
        dtype: DType, *dims: Int, gpu: Bool = False
    ](
        df: Scalar[dtype],
        mut rng: Generator,
        ctx: Optional[DeviceContext] = None,
    ) raises -> Static[dtype, *dims] where dtype.is_floating_point():
        """Random draws at the compile-time shape `dims`, from `rng`'s stream.
        `scipy.stats.t.rvs(..., size=dims, random_state=rng)`.

        By inversion: `t.ppf` of uniforms in `(0, 1)`; on `ctx`'s device at `gpu=True`.

        Parameters:
            dtype: The element type of the draws.
            dims: The shape of the result.
            gpu: Whether to draw one thread per element on `ctx`'s device.

        Args:
            df: The distribution's `df`.
            rng: The generator; its seed advances.
            ctx: The device to allocate and draw on; `None` means the host.

        Returns:
            A new `Static` tensor of draws.

        Raises:
            When allocation or the fill fails.
        """
        comptime count = _product[*dims]()
        var u = rng.uniform[dtype, count, gpu=gpu](
            Scalar[dtype](2.9802322387695312e-08), 1, ctx
        )
        var flat = t.ppf[gpu=gpu](u, df)
        return Static[dtype, *dims](
            flat._buffer,
            rebind[_LayoutOf[*dims]](row_major[*dims]()),
            flat.host_addressable,
        )

    @staticmethod
    def mean(df: Float64) -> Float64:
        """The distribution's mean. `scipy.stats.t.mean`.

        Args:
            df: The distribution's `df`.

        Returns:
            The mean; `inf` or NaN where SciPy gives them.
        """
        return 0.0 if df > 1.0 else _nan64()

    @staticmethod
    def var(df: Float64) -> Float64:
        """The distribution's variance. `scipy.stats.t.var`.

        Args:
            df: The distribution's `df`.

        Returns:
            The variance; `inf` or NaN where SciPy gives them.
        """
        return df / (df - 2.0) if df > 2.0 else (
            _inf64() if df > 1.0 else _nan64()
        )

    @staticmethod
    def std(df: Float64) -> Float64:
        """The standard deviation, `sqrt(var)`.

        Args:
            df: As `var` takes it.

        Returns:
            The standard deviation.
        """
        return _sqrt64(t.var(df))

    @staticmethod
    def interval(confidence: Float64, df: Float64) -> Tuple[Float64, Float64]:
        """The central interval holding `confidence` of the probability,
        `(ppf((1 - c) / 2), ppf((1 + c) / 2))`. `scipy.stats.t.interval`.

        Args:
            confidence: The probability inside the interval, in `[0, 1]`.
            df: As `ppf` takes it.

        Returns:
            The interval's lower and upper ends.
        """
        var lo = t.ppf(_p((1.0 - confidence) / 2.0), _p(df))
        var hi = t.ppf(_p((1.0 + confidence) / 2.0), _p(df))
        return (_v(lo), _v(hi))

    @staticmethod
    def entropy(df: Float64) -> Float64:
        """The differential entropy in nats. `scipy.stats.t.entropy`.

        Args:
            df: The distribution's `df`.

        Returns:
            The entropy.
        """
        return (
            (df + 1.0) / 2.0 * (_psi((df + 1.0) / 2.0) - _psi(df / 2.0))
            + 0.5 * _log64(df)
            + _v(betaln(_p(df / 2.0), _p(0.5)))
        )


struct f:
    """The F distribution. `scipy.stats.f`."""

    @staticmethod
    def pdf[T: FloatLike](x: T, df1: T, df2: T) -> T:
        """The F density with `df1` and `df2` degrees of freedom."""
        return f.logpdf(x, df1, df2).exp()

    @staticmethod
    def logpdf[T: FloatLike](x: T, df1: T, df2: T) -> T:
        """The F log density; `_LOG_ZERO` below the support."""
        var xc = max_of(x, T.constant(0.0))
        var d1x = df1 * xc
        var interior = (
            T.constant(0.5)
            * (
                df1 * _safe_ln(d1x)
                + df2 * _safe_ln(df2)
                - ((df1 + df2) * _safe_ln(d1x + df2))
            )
            - _safe_ln(xc)
            - _log_beta(df1 / T.constant(2.0), df2 / T.constant(2.0))
        )
        return blend(
            ge_indicator(x, T.constant(0.0)), interior, T.constant(_LOG_ZERO)
        )

    @staticmethod
    def cdf[T: FloatLike](x: T, df1: T, df2: T) -> T:
        """The F CDF -- `betainc` at `z = df1*x/(df1*x + df2)`."""
        var d1x = df1 * max_of(x, T.constant(0.0))
        var z = d1x / (d1x + df2)
        return betainc(
            z^, df1 / T.constant(2.0), df2 / T.constant(2.0)
        ) * ge_indicator(x, T.constant(0.0))

    @staticmethod
    def sf[T: FloatLike](x: T, df1: T, df2: T) -> T:
        """`P(F > x)` -- `betaincc` at the same `z`, `1` below the
        support."""
        var d1x = df1 * max_of(x, T.constant(0.0))
        var z = d1x / (d1x + df2)
        return blend(
            ge_indicator(x, T.constant(0.0)),
            betaincc(z^, df1 / T.constant(2.0), df2 / T.constant(2.0)),
            T.one(),
        )

    @staticmethod
    def ppf[T: FloatLike, num_iters: Int = 20](p: T, df1: T, df2: T) -> T:
        """The inverse F CDF, through `beta.ppf`.

        `Z = df1 X / (df1 X + df2)` is `Beta(df1/2, df2/2)` when `X` is
        `F(df1, df2)` -- the same change of variables `cdf` uses, inverted:
        `x = df2 z / (df1 (1 - z))`. `beta.ppf` keeps `z` inside
        `(1e-8, 1 - 1e-8)`, so the division is always finite.
        """
        var z = beta.ppf[T, num_iters](
            p, df1 / T.constant(2.0), df2 / T.constant(2.0)
        )
        return df2 * z / (df1 * (T.one() - z))

    @staticmethod
    def isf[T: FloatLike, num_iters: Int = 20](p: T, df1: T, df2: T) -> T:
        """`ppf(1 - p)`."""
        return f.ppf[T, num_iters](T.one() - p, df1, df2)

    @staticmethod
    def logcdf[T: FloatLike](x: T, df1: T, df2: T) -> T:
        return _safe_ln(f.cdf(x, df1, df2))

    @staticmethod
    def logsf[T: FloatLike](x: T, df1: T, df2: T) -> T:
        return _safe_ln(f.sf(x, df1, df2))

    @staticmethod
    def pdf[
        T: TensorLike, gpu: Bool = False
    ](x: T, df1: Scalar[T.dtype], df2: Scalar[T.dtype]) raises -> Tensor[
        T.dtype, T.LayoutType
    ] where (is_row_major[T] and T.dtype.is_floating_point()):
        """The density over a `Tensor`, parameters as scalars; see the module
        docstring for the two paths `gpu` picks between."""
        comptime dtype = T.dtype
        return _over2[step=_f_pdf_step[dtype, _], gpu=gpu, name="f.pdf"](
            x, df1, df2
        )

    @staticmethod
    def cdf[
        T: TensorLike, gpu: Bool = False
    ](x: T, df1: Scalar[T.dtype], df2: Scalar[T.dtype]) raises -> Tensor[
        T.dtype, T.LayoutType
    ] where (is_row_major[T] and T.dtype.is_floating_point()):
        """The CDF over a `Tensor`, parameters as scalars; see the module
        docstring for the two paths `gpu` picks between."""
        comptime dtype = T.dtype
        return _over2[step=_f_cdf_step[dtype, _], gpu=gpu, name="f.cdf"](
            x, df1, df2
        )

    @staticmethod
    def ppf[
        T: TensorLike, gpu: Bool = False
    ](p: T, df1: Scalar[T.dtype], df2: Scalar[T.dtype]) raises -> Tensor[
        T.dtype, T.LayoutType
    ] where (is_row_major[T] and T.dtype.is_floating_point()):
        """The quantile over a `Tensor`, parameters as scalars; see the module
        docstring for the two paths `gpu` picks between."""
        comptime dtype = T.dtype
        return _over2[step=_f_ppf_step[dtype, _], gpu=gpu, name="f.ppf"](
            p, df1, df2
        )

    @staticmethod
    def rvs[
        dtype: DType, *dims: Int, gpu: Bool = False
    ](
        df1: Scalar[dtype],
        df2: Scalar[dtype],
        mut rng: Generator,
        ctx: Optional[DeviceContext] = None,
    ) raises -> Static[dtype, *dims] where dtype.is_floating_point():
        """Random draws at the compile-time shape `dims`, from `rng`'s stream.
        `scipy.stats.f.rvs(..., size=dims, random_state=rng)`.

        By inversion: `f.ppf` of uniforms in `(0, 1)`; on `ctx`'s device at `gpu=True`.

        Parameters:
            dtype: The element type of the draws.
            dims: The shape of the result.
            gpu: Whether to draw one thread per element on `ctx`'s device.

        Args:
            df1: The distribution's `df1`.
            df2: The distribution's `df2`.
            rng: The generator; its seed advances.
            ctx: The device to allocate and draw on; `None` means the host.

        Returns:
            A new `Static` tensor of draws.

        Raises:
            When allocation or the fill fails.
        """
        comptime count = _product[*dims]()
        var u = rng.uniform[dtype, count, gpu=gpu](
            Scalar[dtype](2.9802322387695312e-08), 1, ctx
        )
        var flat = f.ppf[gpu=gpu](u, df1, df2)
        return Static[dtype, *dims](
            flat._buffer,
            rebind[_LayoutOf[*dims]](row_major[*dims]()),
            flat.host_addressable,
        )

    @staticmethod
    def mean(df1: Float64, df2: Float64) -> Float64:
        """The distribution's mean. `scipy.stats.f.mean`.

        Args:
            df1: The distribution's `df1`.
            df2: The distribution's `df2`.

        Returns:
            The mean; `inf` or NaN where SciPy gives them.
        """
        return df2 / (df2 - 2.0) if df2 > 2.0 else _inf64()

    @staticmethod
    def var(df1: Float64, df2: Float64) -> Float64:
        """The distribution's variance. `scipy.stats.f.var`.

        Args:
            df1: The distribution's `df1`.
            df2: The distribution's `df2`.

        Returns:
            The variance; `inf` or NaN where SciPy gives them.
        """
        return 2.0 * df2 * df2 * (df1 + df2 - 2.0) / (
            df1 * (df2 - 2.0) * (df2 - 2.0) * (df2 - 4.0)
        ) if df2 > 4.0 else (_inf64() if df2 > 2.0 else _nan64())

    @staticmethod
    def std(df1: Float64, df2: Float64) -> Float64:
        """The standard deviation, `sqrt(var)`.

        Args:
            df1: As `var` takes it.
            df2: As `var` takes it.

        Returns:
            The standard deviation.
        """
        return _sqrt64(f.var(df1, df2))

    @staticmethod
    def interval(
        confidence: Float64, df1: Float64, df2: Float64
    ) -> Tuple[Float64, Float64]:
        """The central interval holding `confidence` of the probability,
        `(ppf((1 - c) / 2), ppf((1 + c) / 2))`. `scipy.stats.f.interval`.

        Args:
            confidence: The probability inside the interval, in `[0, 1]`.
            df1: As `ppf` takes it.
            df2: As `ppf` takes it.

        Returns:
            The interval's lower and upper ends.
        """
        var lo = f.ppf(_p((1.0 - confidence) / 2.0), _p(df1), _p(df2))
        var hi = f.ppf(_p((1.0 + confidence) / 2.0), _p(df1), _p(df2))
        return (_v(lo), _v(hi))

    @staticmethod
    def entropy(df1: Float64, df2: Float64) -> Float64:
        """The differential entropy in nats. `scipy.stats.f.entropy`.

        Args:
            df1: The distribution's `df1`.
            df2: The distribution's `df2`.

        Returns:
            The entropy.
        """
        return (
            _log64(df2)
            - _log64(df1)
            + _v(betaln(_p(df1 / 2.0), _p(df2 / 2.0)))
            + (1.0 - df1 / 2.0) * _psi(df1 / 2.0)
            - (1.0 + df2 / 2.0) * _psi(df2 / 2.0)
            + (df1 + df2) / 2.0 * _psi((df1 + df2) / 2.0)
        )


struct poisson:
    """The Poisson distribution. `scipy.stats.poisson`."""

    @staticmethod
    def pmf[T: FloatLike](k: T, rate: T) -> T:
        """`P(X = k)` for a Poisson with mean `rate`.

        `exp(k*ln(rate) - rate - lgamma(k+1))`: `lgamma(k+1)` is `ln(k!)`
        extended to non-integer `k`, so this is the usual PMF wherever `k` is a
        whole number and its standard continuous extension elsewhere.
        """
        return poisson.logpmf(k, rate).exp()

    @staticmethod
    def logpmf[T: FloatLike](k: T, rate: T) -> T:
        """The Poisson log PMF; `_LOG_ZERO` for `k < 0`.

        The interior is evaluated at `k` clamped to the support: `lgamma`
        has a pole at `0`, so an unclamped `k = -1` would put `+inf` on the
        discarded side, and `inf * 0` is the NaN no indicator can remove.
        `pmf` used to survive this because `exp(-inf)` is `0` before the
        multiply; a log density has no such rescue.
        """
        var kc = max_of(k, T.constant(0.0))
        var interior = kc * _safe_ln(rate) - rate - lgamma(kc + T.one())
        return blend(
            ge_indicator(k, T.constant(0.0)), interior, T.constant(_LOG_ZERO)
        )

    @staticmethod
    def cdf[T: FloatLike](k: T, rate: T) -> T:
        """`P(X <= k)`, which is exactly `gammaincc(k+1, rate)`.

        Not an approximation of the sum -- the regularized upper incomplete
        gamma equals that partial sum identically for integer `k`, which is
        what makes a discrete CDF fall out of a continuous special function.
        """
        return gammaincc(
            max_of(k, T.constant(0.0)) + T.one(), rate.copy()
        ) * ge_indicator(k, T.constant(0.0))

    @staticmethod
    def sf[T: FloatLike](k: T, rate: T) -> T:
        """`P(X > k)`, which is exactly `gammainc(k+1, rate)` -- the lower
        incomplete gamma, read directly rather than as `1 - cdf`. `1` for
        `k < 0`."""
        return blend(
            ge_indicator(k, T.constant(0.0)),
            gammainc(max_of(k, T.constant(0.0)) + T.one(), rate.copy()),
            T.one(),
        )

    @staticmethod
    def ppf[T: FloatLike, max_k: Int = 64](p: T, rate: T) -> T:
        """The smallest integer `k` with `cdf(k) >= p`, as SciPy defines
        it -- by the fixed-count branchless scan the module docstring
        describes, capped at `max_k`."""
        var k = T.constant(0.0)
        for _ in range(max_k):
            var caught_up = ge_indicator(poisson.cdf(k.copy(), rate.copy()), p)
            k = k + (T.one() - caught_up)
        return k^

    @staticmethod
    def isf[T: FloatLike, max_k: Int = 64](p: T, rate: T) -> T:
        """`ppf(1 - p)`."""
        return poisson.ppf[T, max_k](T.one() - p, rate)

    @staticmethod
    def logcdf[T: FloatLike](k: T, rate: T) -> T:
        return _safe_ln(poisson.cdf(k, rate))

    @staticmethod
    def logsf[T: FloatLike](k: T, rate: T) -> T:
        return _safe_ln(poisson.sf(k, rate))

    @staticmethod
    def pmf[
        T: TensorLike, gpu: Bool = False
    ](k: T, rate: Scalar[T.dtype]) raises -> Tensor[
        T.dtype, T.LayoutType
    ] where (is_row_major[T] and T.dtype.is_floating_point()):
        """The PMF over a `Tensor`, parameters as scalars; see the module
        docstring for the two paths `gpu` picks between."""
        comptime dtype = T.dtype
        return _over1[
            step=_poisson_pmf_step[dtype, _], gpu=gpu, name="poisson.pmf"
        ](k, rate)

    @staticmethod
    def cdf[
        T: TensorLike, gpu: Bool = False
    ](k: T, rate: Scalar[T.dtype]) raises -> Tensor[
        T.dtype, T.LayoutType
    ] where (is_row_major[T] and T.dtype.is_floating_point()):
        """The CDF over a `Tensor`, parameters as scalars; see the module
        docstring for the two paths `gpu` picks between."""
        comptime dtype = T.dtype
        return _over1[
            step=_poisson_cdf_step[dtype, _], gpu=gpu, name="poisson.cdf"
        ](k, rate)

    @staticmethod
    def ppf[
        T: TensorLike, gpu: Bool = False
    ](p: T, rate: Scalar[T.dtype]) raises -> Tensor[
        T.dtype, T.LayoutType
    ] where (is_row_major[T] and T.dtype.is_floating_point()):
        """The quantile over a `Tensor`, parameters as scalars; see the module
        docstring for the two paths `gpu` picks between."""
        comptime dtype = T.dtype
        return _over1[
            step=_poisson_ppf_step[dtype, _], gpu=gpu, name="poisson.ppf"
        ](p, rate)

    @staticmethod
    def rvs[
        dtype: DType, *dims: Int, gpu: Bool = False
    ](
        rate: Float64,
        mut rng: Generator,
        ctx: Optional[DeviceContext] = None,
    ) raises -> Static[dtype, *dims]:
        """Random draws at the compile-time shape `dims`, from `rng`'s stream.
        `scipy.stats.poisson.rvs(..., size=dims, random_state=rng)`.

        By `Generator.poisson`, NumPy's inversion and PTRS; on `ctx`'s device at `gpu=True`.

        Parameters:
            dtype: The element type of the draws.
            dims: The shape of the result.
            gpu: Whether to draw one thread per element on `ctx`'s device.

        Args:
            rate: The distribution's `rate`.
            rng: The generator; its seed advances.
            ctx: The device to allocate and draw on; `None` means the host.

        Returns:
            A new `Static` tensor of draws.

        Raises:
            When allocation or the fill fails.
        """
        return rng.poisson[dtype, *dims, gpu=gpu](rate, ctx)

    @staticmethod
    def mean(rate: Float64) -> Float64:
        """The distribution's mean. `scipy.stats.poisson.mean`.

        Args:
            rate: The distribution's `rate`.

        Returns:
            The mean; `inf` or NaN where SciPy gives them.
        """
        return rate

    @staticmethod
    def var(rate: Float64) -> Float64:
        """The distribution's variance. `scipy.stats.poisson.var`.

        Args:
            rate: The distribution's `rate`.

        Returns:
            The variance; `inf` or NaN where SciPy gives them.
        """
        return rate

    @staticmethod
    def std(rate: Float64) -> Float64:
        """The standard deviation, `sqrt(var)`.

        Args:
            rate: As `var` takes it.

        Returns:
            The standard deviation.
        """
        return _sqrt64(poisson.var(rate))

    @staticmethod
    def interval(confidence: Float64, rate: Float64) -> Tuple[Float64, Float64]:
        """The central interval holding `confidence` of the probability,
        `(ppf((1 - c) / 2), ppf((1 + c) / 2))`. `scipy.stats.poisson.interval`.

        Args:
            confidence: The probability inside the interval, in `[0, 1]`.
            rate: As `ppf` takes it.

        Returns:
            The interval's lower and upper ends.
        """
        var lo = poisson.ppf(_p((1.0 - confidence) / 2.0), _p(rate))
        var hi = poisson.ppf(_p((1.0 + confidence) / 2.0), _p(rate))
        return (_v(lo), _v(hi))

    @staticmethod
    def entropy(rate: Float64) -> Float64:
        """The Shannon entropy in nats. `scipy.stats.poisson.entropy`.

        Args:
            rate: The distribution's `rate`.

        Returns:
            The entropy.
        """
        return _poisson_entropy(rate)


struct binom:
    """The binomial distribution. `scipy.stats.binom`."""

    @staticmethod
    def pmf[T: FloatLike](k: T, n: T, p: T) -> T:
        """`P(X = k)` for `n` trials with success probability `p`."""
        return binom.logpmf(k, n, p).exp()

    @staticmethod
    def logpmf[T: FloatLike](k: T, n: T, p: T) -> T:
        """The binomial log PMF; `_LOG_ZERO` outside `0 <= k <= n`.

        The interior is evaluated at `k` clamped into `[0, n]`, for the
        reason `poisson.logpmf` gives: `lgamma(n - k + 1)` has its pole
        exactly one step past `n`.
        """
        var kc = min_of(max_of(k, T.constant(0.0)), n.copy())
        var interior = (
            lgamma(n + T.one())
            - lgamma(kc + T.one())
            - lgamma(n - kc + T.one())
            + kc * _safe_ln(p)
            + (n - kc) * _safe_ln(T.one() - p)
        )
        var support = ge_indicator(k, T.constant(0.0)) * ge_indicator(
            n, k.copy()
        )
        return blend(support, interior, T.constant(_LOG_ZERO))

    @staticmethod
    def cdf[T: FloatLike](k: T, n: T, p: T) -> T:
        """`P(X <= k)`, which is `I_{1-p}(n-k, k+1)` -- the same
        special-function-in-disguise relationship `poisson.cdf` has with the
        incomplete gamma."""
        var kc = min_of(max_of(k, T.constant(0.0)), n.copy())
        # At `k = n` the first beta parameter is exactly `0`, where `betainc`
        # is undefined and returns NaN -- and a NaN survives being multiplied
        # by a `0` indicator, so the blend below can't clean it up afterwards.
        # Flooring the parameter keeps that lane finite; the blend then
        # discards it in favour of the exact answer, `P(X <= n) = 1`.
        var trials_left = max_of(n - kc, T.constant(1e-8))
        var upper = betainc(T.one() - p, trials_left^, kc + T.one())
        return blend(ge_indicator(k, n.copy()), T.one(), upper^)

    @staticmethod
    def sf[T: FloatLike](k: T, n: T, p: T) -> T:
        """`P(X > k)`, which is `I_p(k+1, n-k)` -- the upper tail read
        straight off `betainc`. `0` at and past `k = n`, `1` below `k = 0`.
        """
        var kc = min_of(max_of(k, T.constant(0.0)), n.copy())
        # As in `cdf`: the second parameter is exactly `0` at `k = n`, and
        # `betainc` returns NaN there, which no indicator can discard.
        var trials_left = max_of(n - kc, T.constant(1e-8))
        var upper = betainc(p.copy(), kc + T.one(), trials_left^)
        var above = blend(ge_indicator(k, n.copy()), T.constant(0.0), upper^)
        return blend(ge_indicator(k, T.constant(0.0)), above^, T.one())

    @staticmethod
    def ppf[T: FloatLike, max_k: Int = 64](p: T, n: T, prob: T) -> T:
        """The smallest integer `k` with `cdf(k) >= p`, as SciPy defines
        it -- the module docstring's fixed-count scan, capped at `max_k`.
        `cdf(n) == 1` exactly, so the scan stops at `n` on its own."""
        var k = T.constant(0.0)
        for _ in range(max_k):
            var caught_up = ge_indicator(
                binom.cdf(k.copy(), n.copy(), prob.copy()), p
            )
            k = k + (T.one() - caught_up)
        return k^

    @staticmethod
    def isf[T: FloatLike, max_k: Int = 64](p: T, n: T, prob: T) -> T:
        """`ppf(1 - p)`."""
        return binom.ppf[T, max_k](T.one() - p, n, prob)

    @staticmethod
    def logcdf[T: FloatLike](k: T, n: T, p: T) -> T:
        return _safe_ln(binom.cdf(k, n, p))

    @staticmethod
    def logsf[T: FloatLike](k: T, n: T, p: T) -> T:
        return _safe_ln(binom.sf(k, n, p))

    @staticmethod
    def pmf[
        T: TensorLike, gpu: Bool = False
    ](k: T, n: Scalar[T.dtype], prob: Scalar[T.dtype]) raises -> Tensor[
        T.dtype, T.LayoutType
    ] where (is_row_major[T] and T.dtype.is_floating_point()):
        """The PMF over a `Tensor`, parameters as scalars; see the module
        docstring for the two paths `gpu` picks between."""
        comptime dtype = T.dtype
        return _over2[
            step=_binom_pmf_step[dtype, _], gpu=gpu, name="binom.pmf"
        ](k, n, prob)

    @staticmethod
    def cdf[
        T: TensorLike, gpu: Bool = False
    ](k: T, n: Scalar[T.dtype], prob: Scalar[T.dtype]) raises -> Tensor[
        T.dtype, T.LayoutType
    ] where (is_row_major[T] and T.dtype.is_floating_point()):
        """The CDF over a `Tensor`, parameters as scalars; see the module
        docstring for the two paths `gpu` picks between."""
        comptime dtype = T.dtype
        return _over2[
            step=_binom_cdf_step[dtype, _], gpu=gpu, name="binom.cdf"
        ](k, n, prob)

    @staticmethod
    def ppf[
        T: TensorLike, gpu: Bool = False
    ](p: T, n: Scalar[T.dtype], prob: Scalar[T.dtype]) raises -> Tensor[
        T.dtype, T.LayoutType
    ] where (is_row_major[T] and T.dtype.is_floating_point()):
        """The quantile over a `Tensor`, parameters as scalars; see the module
        docstring for the two paths `gpu` picks between."""
        comptime dtype = T.dtype
        return _over2[
            step=_binom_ppf_step[dtype, _], gpu=gpu, name="binom.ppf"
        ](p, n, prob)

    @staticmethod
    def rvs[
        dtype: DType, *dims: Int, gpu: Bool = False
    ](
        n: Float64,
        p: Float64,
        mut rng: Generator,
        ctx: Optional[DeviceContext] = None,
    ) raises -> Static[dtype, *dims]:
        """Random draws at the compile-time shape `dims`, from `rng`'s stream.
        `scipy.stats.binom.rvs(..., size=dims, random_state=rng)`.

        By `Generator.binomial`, inversion and BTRS; on `ctx`'s device at `gpu=True`.

        Parameters:
            dtype: The element type of the draws.
            dims: The shape of the result.
            gpu: Whether to draw one thread per element on `ctx`'s device.

        Args:
            n: The distribution's `n`.
            p: The distribution's `p`.
            rng: The generator; its seed advances.
            ctx: The device to allocate and draw on; `None` means the host.

        Returns:
            A new `Static` tensor of draws.

        Raises:
            When allocation or the fill fails.
        """
        return rng.binomial[dtype, *dims, gpu=gpu](Int(n), p, ctx)

    @staticmethod
    def mean(n: Float64, p: Float64) -> Float64:
        """The distribution's mean. `scipy.stats.binom.mean`.

        Args:
            n: The distribution's `n`.
            p: The distribution's `p`.

        Returns:
            The mean; `inf` or NaN where SciPy gives them.
        """
        return n * p

    @staticmethod
    def var(n: Float64, p: Float64) -> Float64:
        """The distribution's variance. `scipy.stats.binom.var`.

        Args:
            n: The distribution's `n`.
            p: The distribution's `p`.

        Returns:
            The variance; `inf` or NaN where SciPy gives them.
        """
        return n * p * (1.0 - p)

    @staticmethod
    def std(n: Float64, p: Float64) -> Float64:
        """The standard deviation, `sqrt(var)`.

        Args:
            n: As `var` takes it.
            p: As `var` takes it.

        Returns:
            The standard deviation.
        """
        return _sqrt64(binom.var(n, p))

    @staticmethod
    def interval(
        confidence: Float64, n: Float64, p: Float64
    ) -> Tuple[Float64, Float64]:
        """The central interval holding `confidence` of the probability,
        `(ppf((1 - c) / 2), ppf((1 + c) / 2))`. `scipy.stats.binom.interval`.

        Args:
            confidence: The probability inside the interval, in `[0, 1]`.
            n: As `ppf` takes it.
            p: As `ppf` takes it.

        Returns:
            The interval's lower and upper ends.
        """
        var lo = binom.ppf(_p((1.0 - confidence) / 2.0), _p(n), _p(p))
        var hi = binom.ppf(_p((1.0 + confidence) / 2.0), _p(n), _p(p))
        return (_v(lo), _v(hi))

    @staticmethod
    def entropy(n: Float64, p: Float64) -> Float64:
        """The Shannon entropy in nats. `scipy.stats.binom.entropy`.

        Args:
            n: The distribution's `n`.
            p: The distribution's `p`.

        Returns:
            The entropy.
        """
        return _binom_entropy(n, p)


struct uniform_dist:
    """The continuous uniform distribution on `[loc, loc + scale]`. `scipy.stats.uniform`,
    named `uniform_dist` because `numax.stats.uniform` is NumPy's sampler."""

    @staticmethod
    def pdf[T: FloatLike](x: T, loc: T, scale: T) -> T:
        """The density. `scipy.stats.uniform.pdf`."""
        var z = (x - loc) / scale
        return (
            ge_indicator(z.copy(), T.constant(0.0))
            * ge_indicator(T.one() - z, T.constant(0.0))
            / scale
        )

    @staticmethod
    def logpdf[T: FloatLike](x: T, loc: T, scale: T) -> T:
        """The log density, finite (`_LOG_ZERO`) outside the support. `scipy.stats.uniform.logpdf`.
        """
        var z = (x - loc) / scale
        var inside = ge_indicator(z.copy(), T.constant(0.0)) * ge_indicator(
            T.one() - z, T.constant(0.0)
        )
        return blend(inside, -_safe_ln(scale), T.constant(_LOG_ZERO))

    @staticmethod
    def cdf[T: FloatLike](x: T, loc: T, scale: T) -> T:
        """The CDF. `scipy.stats.uniform.cdf`."""
        return min_of(max_of((x - loc) / scale, T.constant(0.0)), T.one())

    @staticmethod
    def logcdf[T: FloatLike](x: T, loc: T, scale: T) -> T:
        """`ln cdf`. `scipy.stats.uniform.logcdf`."""
        return _safe_ln(uniform_dist.cdf(x, loc, scale))

    @staticmethod
    def sf[T: FloatLike](x: T, loc: T, scale: T) -> T:
        """The survival function `P(X > x)`. `scipy.stats.uniform.sf`."""
        return min_of(
            max_of(T.one() - (x - loc) / scale, T.constant(0.0)), T.one()
        )

    @staticmethod
    def logsf[T: FloatLike](x: T, loc: T, scale: T) -> T:
        """`ln sf`. `scipy.stats.uniform.logsf`."""
        return _safe_ln(uniform_dist.sf(x, loc, scale))

    @staticmethod
    def ppf[T: FloatLike](p: T, loc: T, scale: T) -> T:
        """The quantile, closed form. `scipy.stats.uniform.ppf`."""
        return loc + p * scale

    @staticmethod
    def isf[T: FloatLike](p: T, loc: T, scale: T) -> T:
        """The inverse survival function, `ppf(1 - p)` without the `1 - p`. `scipy.stats.uniform.isf`.
        """
        return loc + (T.one() - p) * scale

    @staticmethod
    def pdf[
        T: TensorLike, gpu: Bool = False
    ](x: T, loc: Scalar[T.dtype], scale: Scalar[T.dtype]) raises -> Tensor[
        T.dtype, T.LayoutType
    ] where (is_row_major[T] and T.dtype.is_floating_point()):
        """The density over a `Tensor`, parameters as scalars; see the module
        docstring for the two paths `gpu` picks between."""
        comptime dtype = T.dtype
        return _over2[
            step=_uniform_dist_pdf_step[dtype, _],
            gpu=gpu,
            name="uniform_dist.pdf",
        ](x, loc, scale)

    @staticmethod
    def cdf[
        T: TensorLike, gpu: Bool = False
    ](x: T, loc: Scalar[T.dtype], scale: Scalar[T.dtype]) raises -> Tensor[
        T.dtype, T.LayoutType
    ] where (is_row_major[T] and T.dtype.is_floating_point()):
        """The CDF over a `Tensor`, parameters as scalars; see the module
        docstring for the two paths `gpu` picks between."""
        comptime dtype = T.dtype
        return _over2[
            step=_uniform_dist_cdf_step[dtype, _],
            gpu=gpu,
            name="uniform_dist.cdf",
        ](x, loc, scale)

    @staticmethod
    def ppf[
        T: TensorLike, gpu: Bool = False
    ](p: T, loc: Scalar[T.dtype], scale: Scalar[T.dtype]) raises -> Tensor[
        T.dtype, T.LayoutType
    ] where (is_row_major[T] and T.dtype.is_floating_point()):
        """The quantile over a `Tensor`, parameters as scalars; see the module
        docstring for the two paths `gpu` picks between."""
        comptime dtype = T.dtype
        return _over2[
            step=_uniform_dist_ppf_step[dtype, _],
            gpu=gpu,
            name="uniform_dist.ppf",
        ](p, loc, scale)

    @staticmethod
    def rvs[
        dtype: DType, *dims: Int, gpu: Bool = False
    ](
        loc: Scalar[dtype],
        scale: Scalar[dtype],
        mut rng: Generator,
        ctx: Optional[DeviceContext] = None,
    ) raises -> Static[dtype, *dims] where dtype.is_floating_point():
        """Random draws at the compile-time shape `dims`, from `rng`'s stream,
        by `Generator`'s own sampler. `scipy.stats.uniform.rvs`.

        Parameters:
            dtype: The element type of the draws.
            dims: The shape of the result.
            gpu: Whether to draw one thread per element on `ctx`'s device.

        Args:
            loc: The distribution's `loc`.
            scale: The distribution's `scale`.
            rng: The generator; its seed advances.
            ctx: The device to allocate and draw on; `None` means the host.

        Returns:
            A new `Static` tensor of draws.

        Raises:
            When allocation or the fill fails.
        """
        return rng.uniform[dtype, *dims, gpu=gpu](loc, loc + scale, ctx)

    @staticmethod
    def mean(loc: Float64, scale: Float64) -> Float64:
        """The mean. `scipy.stats.uniform.mean`; `inf` or NaN where SciPy gives them.

        Args:
            loc: The distribution's `loc`.
            scale: The distribution's `scale`.

        Returns:
            The mean.
        """
        return loc + scale / 2.0

    @staticmethod
    def var(loc: Float64, scale: Float64) -> Float64:
        """The variance. `scipy.stats.uniform.var`; `inf` or NaN where SciPy gives them.

        Args:
            loc: The distribution's `loc`.
            scale: The distribution's `scale`.

        Returns:
            The variance.
        """
        return scale * scale / 12.0

    @staticmethod
    def std(loc: Float64, scale: Float64) -> Float64:
        """The standard deviation, `sqrt(var)`.

        Args:
            loc: The distribution's `loc`.
            scale: The distribution's `scale`.

        Returns:
            The standard deviation.
        """
        return _sqrt64(uniform_dist.var(loc, scale))

    @staticmethod
    def interval(
        confidence: Float64, loc: Float64, scale: Float64
    ) -> Tuple[Float64, Float64]:
        """The central interval holding `confidence` of the probability.
        `scipy.stats.uniform.interval`.

        Args:
            confidence: The probability inside the interval, in `[0, 1]`.
            loc: The distribution's `loc`.
            scale: The distribution's `scale`.

        Returns:
            The interval's lower and upper ends.
        """
        var lo = uniform_dist.ppf(
            _p((1.0 - confidence) / 2.0), _p(loc), _p(scale)
        )
        var hi = uniform_dist.ppf(
            _p((1.0 + confidence) / 2.0), _p(loc), _p(scale)
        )
        return (_v(lo), _v(hi))

    @staticmethod
    def entropy(loc: Float64, scale: Float64) -> Float64:
        """The differential entropy in nats. `scipy.stats.uniform.entropy`.

        Args:
            loc: The distribution's `loc`.
            scale: The distribution's `scale`.

        Returns:
            The entropy.
        """
        return _log64(scale)


def _uniform_dist_pdf_step[
    dtype: DType, w: Int
](x: SIMD[dtype, w], loc: SIMD[dtype, 1], scale: SIMD[dtype, 1]) -> SIMD[
    dtype, w
] where dtype.is_floating_point():
    return uniform_dist.pdf(
        Plain[dtype, w](x),
        Plain[dtype, w](SIMD[dtype, w](loc[0])),
        Plain[dtype, w](SIMD[dtype, w](scale[0])),
    ).v


def _uniform_dist_cdf_step[
    dtype: DType, w: Int
](x: SIMD[dtype, w], loc: SIMD[dtype, 1], scale: SIMD[dtype, 1]) -> SIMD[
    dtype, w
] where dtype.is_floating_point():
    return uniform_dist.cdf(
        Plain[dtype, w](x),
        Plain[dtype, w](SIMD[dtype, w](loc[0])),
        Plain[dtype, w](SIMD[dtype, w](scale[0])),
    ).v


def _uniform_dist_ppf_step[
    dtype: DType, w: Int
](x: SIMD[dtype, w], loc: SIMD[dtype, 1], scale: SIMD[dtype, 1]) -> SIMD[
    dtype, w
] where dtype.is_floating_point():
    return uniform_dist.ppf(
        Plain[dtype, w](x),
        Plain[dtype, w](SIMD[dtype, w](loc[0])),
        Plain[dtype, w](SIMD[dtype, w](scale[0])),
    ).v


struct lognorm:
    """The log-normal distribution: `ln(X)` normal with standard deviation `s` and mean `ln(scale)`. `scipy.stats.lognorm`.
    """

    @staticmethod
    def pdf[T: FloatLike](x: T, s: T, scale: T) -> T:
        """The density. `scipy.stats.lognorm.pdf`."""
        var xs = max_of(x.copy(), T.constant(_TINY))
        var y = (xs / scale).ln() / s
        var inside = (-(y * y) / T.constant(2.0)).exp() / (
            xs * s * T.constant(_SQRT_2PI)
        )
        return inside * ge_indicator(x, T.constant(_TINY))

    @staticmethod
    def logpdf[T: FloatLike](x: T, s: T, scale: T) -> T:
        """The log density, finite (`_LOG_ZERO`) outside the support. `scipy.stats.lognorm.logpdf`.
        """
        var xs = max_of(x.copy(), T.constant(_TINY))
        var y = (xs / scale).ln() / s
        var inside = (
            -(y * y) / T.constant(2.0)
            - xs.ln()
            - _safe_ln(s)
            - T.constant(0.5 * _LN_2PI)
        )
        return blend(
            ge_indicator(x, T.constant(_TINY)), inside, T.constant(_LOG_ZERO)
        )

    @staticmethod
    def cdf[T: FloatLike](x: T, s: T, scale: T) -> T:
        """The CDF. `scipy.stats.lognorm.cdf`."""
        var y = (max_of(x.copy(), T.constant(_TINY)) / scale).ln() / s
        return (
            T.constant(0.5)
            * (-(y / T.constant(_SQRT_2))).erfc()
            * ge_indicator(x, T.constant(_TINY))
        )

    @staticmethod
    def logcdf[T: FloatLike](x: T, s: T, scale: T) -> T:
        """`ln cdf`. `scipy.stats.lognorm.logcdf`."""
        return _safe_ln(lognorm.cdf(x, s, scale))

    @staticmethod
    def sf[T: FloatLike](x: T, s: T, scale: T) -> T:
        """The survival function `P(X > x)`. `scipy.stats.lognorm.sf`."""
        var y = (max_of(x.copy(), T.constant(_TINY)) / scale).ln() / s
        return blend(
            ge_indicator(x, T.constant(_TINY)),
            T.constant(0.5) * (y / T.constant(_SQRT_2)).erfc(),
            T.one(),
        )

    @staticmethod
    def logsf[T: FloatLike](x: T, s: T, scale: T) -> T:
        """`ln sf`. `scipy.stats.lognorm.logsf`."""
        return _safe_ln(lognorm.sf(x, s, scale))

    @staticmethod
    def ppf[T: FloatLike](p: T, s: T, scale: T) -> T:
        """The quantile, closed form. `scipy.stats.lognorm.ppf`."""
        return scale * (s * _standard_normal_quantile(p)).exp()

    @staticmethod
    def isf[T: FloatLike](p: T, s: T, scale: T) -> T:
        """The inverse survival function, `ppf(1 - p)` without the `1 - p`. `scipy.stats.lognorm.isf`.
        """
        return scale * (-(s * _standard_normal_quantile(p))).exp()

    @staticmethod
    def pdf[
        T: TensorLike, gpu: Bool = False
    ](x: T, s: Scalar[T.dtype], scale: Scalar[T.dtype]) raises -> Tensor[
        T.dtype, T.LayoutType
    ] where (is_row_major[T] and T.dtype.is_floating_point()):
        """The density over a `Tensor`, parameters as scalars; see the module
        docstring for the two paths `gpu` picks between."""
        comptime dtype = T.dtype
        return _over2[
            step=_lognorm_pdf_step[dtype, _], gpu=gpu, name="lognorm.pdf"
        ](x, s, scale)

    @staticmethod
    def cdf[
        T: TensorLike, gpu: Bool = False
    ](x: T, s: Scalar[T.dtype], scale: Scalar[T.dtype]) raises -> Tensor[
        T.dtype, T.LayoutType
    ] where (is_row_major[T] and T.dtype.is_floating_point()):
        """The CDF over a `Tensor`, parameters as scalars; see the module
        docstring for the two paths `gpu` picks between."""
        comptime dtype = T.dtype
        return _over2[
            step=_lognorm_cdf_step[dtype, _], gpu=gpu, name="lognorm.cdf"
        ](x, s, scale)

    @staticmethod
    def ppf[
        T: TensorLike, gpu: Bool = False
    ](p: T, s: Scalar[T.dtype], scale: Scalar[T.dtype]) raises -> Tensor[
        T.dtype, T.LayoutType
    ] where (is_row_major[T] and T.dtype.is_floating_point()):
        """The quantile over a `Tensor`, parameters as scalars; see the module
        docstring for the two paths `gpu` picks between."""
        comptime dtype = T.dtype
        return _over2[
            step=_lognorm_ppf_step[dtype, _], gpu=gpu, name="lognorm.ppf"
        ](p, s, scale)

    @staticmethod
    def rvs[
        dtype: DType, *dims: Int, gpu: Bool = False
    ](
        s: Scalar[dtype],
        scale: Scalar[dtype],
        mut rng: Generator,
        ctx: Optional[DeviceContext] = None,
    ) raises -> Static[dtype, *dims] where dtype.is_floating_point():
        """Random draws at the compile-time shape `dims`, from `rng`'s stream,
        by `Generator`'s own sampler. `scipy.stats.lognorm.rvs`.

        Parameters:
            dtype: The element type of the draws.
            dims: The shape of the result.
            gpu: Whether to draw one thread per element on `ctx`'s device.

        Args:
            s: The distribution's `s`.
            scale: The distribution's `scale`.
            rng: The generator; its seed advances.
            ctx: The device to allocate and draw on; `None` means the host.

        Returns:
            A new `Static` tensor of draws.

        Raises:
            When allocation or the fill fails.
        """
        return rng.lognormal[dtype, *dims, gpu=gpu](_ln_scalar(scale), s, ctx)

    @staticmethod
    def mean(s: Float64, scale: Float64) -> Float64:
        """The mean. `scipy.stats.lognorm.mean`; `inf` or NaN where SciPy gives them.

        Args:
            s: The distribution's `s`.
            scale: The distribution's `scale`.

        Returns:
            The mean.
        """
        return scale * _exp64(s * s / 2.0)

    @staticmethod
    def var(s: Float64, scale: Float64) -> Float64:
        """The variance. `scipy.stats.lognorm.var`; `inf` or NaN where SciPy gives them.

        Args:
            s: The distribution's `s`.
            scale: The distribution's `scale`.

        Returns:
            The variance.
        """
        return scale * scale * _exp64(s * s) * (_exp64(s * s) - 1.0)

    @staticmethod
    def std(s: Float64, scale: Float64) -> Float64:
        """The standard deviation, `sqrt(var)`.

        Args:
            s: The distribution's `s`.
            scale: The distribution's `scale`.

        Returns:
            The standard deviation.
        """
        return _sqrt64(lognorm.var(s, scale))

    @staticmethod
    def interval(
        confidence: Float64, s: Float64, scale: Float64
    ) -> Tuple[Float64, Float64]:
        """The central interval holding `confidence` of the probability.
        `scipy.stats.lognorm.interval`.

        Args:
            confidence: The probability inside the interval, in `[0, 1]`.
            s: The distribution's `s`.
            scale: The distribution's `scale`.

        Returns:
            The interval's lower and upper ends.
        """
        var lo = lognorm.ppf(_p((1.0 - confidence) / 2.0), _p(s), _p(scale))
        var hi = lognorm.ppf(_p((1.0 + confidence) / 2.0), _p(s), _p(scale))
        return (_v(lo), _v(hi))

    @staticmethod
    def entropy(s: Float64, scale: Float64) -> Float64:
        """The differential entropy in nats. `scipy.stats.lognorm.entropy`.

        Args:
            s: The distribution's `s`.
            scale: The distribution's `scale`.

        Returns:
            The entropy.
        """
        return 0.5 + 0.5 * _LN_2PI + _log64(s) + _log64(scale)


def _lognorm_pdf_step[
    dtype: DType, w: Int
](x: SIMD[dtype, w], s: SIMD[dtype, 1], scale: SIMD[dtype, 1]) -> SIMD[
    dtype, w
] where dtype.is_floating_point():
    return lognorm.pdf(
        Plain[dtype, w](x),
        Plain[dtype, w](SIMD[dtype, w](s[0])),
        Plain[dtype, w](SIMD[dtype, w](scale[0])),
    ).v


def _lognorm_cdf_step[
    dtype: DType, w: Int
](x: SIMD[dtype, w], s: SIMD[dtype, 1], scale: SIMD[dtype, 1]) -> SIMD[
    dtype, w
] where dtype.is_floating_point():
    return lognorm.cdf(
        Plain[dtype, w](x),
        Plain[dtype, w](SIMD[dtype, w](s[0])),
        Plain[dtype, w](SIMD[dtype, w](scale[0])),
    ).v


def _lognorm_ppf_step[
    dtype: DType, w: Int
](x: SIMD[dtype, w], s: SIMD[dtype, 1], scale: SIMD[dtype, 1]) -> SIMD[
    dtype, w
] where dtype.is_floating_point():
    return lognorm.ppf(
        Plain[dtype, w](x),
        Plain[dtype, w](SIMD[dtype, w](s[0])),
        Plain[dtype, w](SIMD[dtype, w](scale[0])),
    ).v


struct weibull_min:
    """The Weibull minimum-extreme-value distribution with shape `c`. `scipy.stats.weibull_min`.
    """

    @staticmethod
    def pdf[T: FloatLike](x: T, c: T, scale: T) -> T:
        """The density. `scipy.stats.weibull_min.pdf`."""
        var z = max_of(x.copy(), T.constant(0.0)) / scale
        var u = _pow(z.copy(), c.copy())
        var inside = c / scale * _pow(z, c - T.one()) * (-u).exp()
        return inside * ge_indicator(x, T.constant(0.0))

    @staticmethod
    def logpdf[T: FloatLike](x: T, c: T, scale: T) -> T:
        """The log density, finite (`_LOG_ZERO`) outside the support. `scipy.stats.weibull_min.logpdf`.
        """
        var z = max_of(x.copy(), T.constant(0.0)) / scale
        var inside = (
            _safe_ln(c / scale)
            + (c - T.one()) * _safe_ln(z.copy())
            - _pow(z, c.copy())
        )
        return blend(
            ge_indicator(x, T.constant(0.0)), inside, T.constant(_LOG_ZERO)
        )

    @staticmethod
    def cdf[T: FloatLike](x: T, c: T, scale: T) -> T:
        """The CDF. `scipy.stats.weibull_min.cdf`."""
        var z = max_of(x, T.constant(0.0)) / scale
        return -_expm1(-_pow(z, c.copy()))

    @staticmethod
    def logcdf[T: FloatLike](x: T, c: T, scale: T) -> T:
        """`ln cdf`. `scipy.stats.weibull_min.logcdf`."""
        return _safe_ln(weibull_min.cdf(x, c, scale))

    @staticmethod
    def sf[T: FloatLike](x: T, c: T, scale: T) -> T:
        """The survival function `P(X > x)`. `scipy.stats.weibull_min.sf`."""
        var z = max_of(x, T.constant(0.0)) / scale
        return (-_pow(z, c.copy())).exp()

    @staticmethod
    def logsf[T: FloatLike](x: T, c: T, scale: T) -> T:
        """`ln sf`. `scipy.stats.weibull_min.logsf`."""
        return _safe_ln(weibull_min.sf(x, c, scale))

    @staticmethod
    def ppf[T: FloatLike](p: T, c: T, scale: T) -> T:
        """The quantile, closed form. `scipy.stats.weibull_min.ppf`."""
        return scale * _pow(-_log1p(-p), T.one() / c)

    @staticmethod
    def isf[T: FloatLike](p: T, c: T, scale: T) -> T:
        """The inverse survival function, `ppf(1 - p)` without the `1 - p`. `scipy.stats.weibull_min.isf`.
        """
        return scale * _pow(-_safe_ln(p), T.one() / c)

    @staticmethod
    def pdf[
        T: TensorLike, gpu: Bool = False
    ](x: T, c: Scalar[T.dtype], scale: Scalar[T.dtype]) raises -> Tensor[
        T.dtype, T.LayoutType
    ] where (is_row_major[T] and T.dtype.is_floating_point()):
        """The density over a `Tensor`, parameters as scalars; see the module
        docstring for the two paths `gpu` picks between."""
        comptime dtype = T.dtype
        return _over2[
            step=_weibull_min_pdf_step[dtype, _],
            gpu=gpu,
            name="weibull_min.pdf",
        ](x, c, scale)

    @staticmethod
    def cdf[
        T: TensorLike, gpu: Bool = False
    ](x: T, c: Scalar[T.dtype], scale: Scalar[T.dtype]) raises -> Tensor[
        T.dtype, T.LayoutType
    ] where (is_row_major[T] and T.dtype.is_floating_point()):
        """The CDF over a `Tensor`, parameters as scalars; see the module
        docstring for the two paths `gpu` picks between."""
        comptime dtype = T.dtype
        return _over2[
            step=_weibull_min_cdf_step[dtype, _],
            gpu=gpu,
            name="weibull_min.cdf",
        ](x, c, scale)

    @staticmethod
    def ppf[
        T: TensorLike, gpu: Bool = False
    ](p: T, c: Scalar[T.dtype], scale: Scalar[T.dtype]) raises -> Tensor[
        T.dtype, T.LayoutType
    ] where (is_row_major[T] and T.dtype.is_floating_point()):
        """The quantile over a `Tensor`, parameters as scalars; see the module
        docstring for the two paths `gpu` picks between."""
        comptime dtype = T.dtype
        return _over2[
            step=_weibull_min_ppf_step[dtype, _],
            gpu=gpu,
            name="weibull_min.ppf",
        ](p, c, scale)

    @staticmethod
    def rvs[
        dtype: DType, *dims: Int, gpu: Bool = False
    ](
        c: Scalar[dtype],
        scale: Scalar[dtype],
        mut rng: Generator,
        ctx: Optional[DeviceContext] = None,
    ) raises -> Static[dtype, *dims] where dtype.is_floating_point():
        """Random draws at the compile-time shape `dims`, from `rng`'s stream,
        by inverting the closed-form `ppf` over uniforms in `(0, 1)`. `scipy.stats.weibull_min.rvs`.

        Parameters:
            dtype: The element type of the draws.
            dims: The shape of the result.
            gpu: Whether to draw one thread per element on `ctx`'s device.

        Args:
            c: The distribution's `c`.
            scale: The distribution's `scale`.
            rng: The generator; its seed advances.
            ctx: The device to allocate and draw on; `None` means the host.

        Returns:
            A new `Static` tensor of draws.

        Raises:
            When allocation or the fill fails.
        """
        comptime count = _product[*dims]()
        var u = rng.uniform[dtype, count, gpu=gpu](
            Scalar[dtype](2.9802322387695312e-08), 1, ctx
        )
        var flat = weibull_min.ppf[gpu=gpu](u, c, scale)
        return Static[dtype, *dims](
            flat._buffer,
            rebind[_LayoutOf[*dims]](row_major[*dims]()),
            flat.host_addressable,
        )

    @staticmethod
    def mean(c: Float64, scale: Float64) -> Float64:
        """The mean. `scipy.stats.weibull_min.mean`; `inf` or NaN where SciPy gives them.

        Args:
            c: The distribution's `c`.
            scale: The distribution's `scale`.

        Returns:
            The mean.
        """
        return scale * _gamma64(1.0 + 1.0 / c)

    @staticmethod
    def var(c: Float64, scale: Float64) -> Float64:
        """The variance. `scipy.stats.weibull_min.var`; `inf` or NaN where SciPy gives them.

        Args:
            c: The distribution's `c`.
            scale: The distribution's `scale`.

        Returns:
            The variance.
        """
        return (
            scale
            * scale
            * (_gamma64(1.0 + 2.0 / c) - _gamma64(1.0 + 1.0 / c) ** 2)
        )

    @staticmethod
    def std(c: Float64, scale: Float64) -> Float64:
        """The standard deviation, `sqrt(var)`.

        Args:
            c: The distribution's `c`.
            scale: The distribution's `scale`.

        Returns:
            The standard deviation.
        """
        return _sqrt64(weibull_min.var(c, scale))

    @staticmethod
    def interval(
        confidence: Float64, c: Float64, scale: Float64
    ) -> Tuple[Float64, Float64]:
        """The central interval holding `confidence` of the probability.
        `scipy.stats.weibull_min.interval`.

        Args:
            confidence: The probability inside the interval, in `[0, 1]`.
            c: The distribution's `c`.
            scale: The distribution's `scale`.

        Returns:
            The interval's lower and upper ends.
        """
        var lo = weibull_min.ppf(_p((1.0 - confidence) / 2.0), _p(c), _p(scale))
        var hi = weibull_min.ppf(_p((1.0 + confidence) / 2.0), _p(c), _p(scale))
        return (_v(lo), _v(hi))

    @staticmethod
    def entropy(c: Float64, scale: Float64) -> Float64:
        """The differential entropy in nats. `scipy.stats.weibull_min.entropy`.

        Args:
            c: The distribution's `c`.
            scale: The distribution's `scale`.

        Returns:
            The entropy.
        """
        return _EULER * (1.0 - 1.0 / c) - _log64(c) + 1.0 + _log64(scale)


def _weibull_min_pdf_step[
    dtype: DType, w: Int
](x: SIMD[dtype, w], c: SIMD[dtype, 1], scale: SIMD[dtype, 1]) -> SIMD[
    dtype, w
] where dtype.is_floating_point():
    return weibull_min.pdf(
        Plain[dtype, w](x),
        Plain[dtype, w](SIMD[dtype, w](c[0])),
        Plain[dtype, w](SIMD[dtype, w](scale[0])),
    ).v


def _weibull_min_cdf_step[
    dtype: DType, w: Int
](x: SIMD[dtype, w], c: SIMD[dtype, 1], scale: SIMD[dtype, 1]) -> SIMD[
    dtype, w
] where dtype.is_floating_point():
    return weibull_min.cdf(
        Plain[dtype, w](x),
        Plain[dtype, w](SIMD[dtype, w](c[0])),
        Plain[dtype, w](SIMD[dtype, w](scale[0])),
    ).v


def _weibull_min_ppf_step[
    dtype: DType, w: Int
](x: SIMD[dtype, w], c: SIMD[dtype, 1], scale: SIMD[dtype, 1]) -> SIMD[
    dtype, w
] where dtype.is_floating_point():
    return weibull_min.ppf(
        Plain[dtype, w](x),
        Plain[dtype, w](SIMD[dtype, w](c[0])),
        Plain[dtype, w](SIMD[dtype, w](scale[0])),
    ).v


struct cauchy:
    """The Cauchy (Lorentz) distribution. `scipy.stats.cauchy`."""

    @staticmethod
    def pdf[T: FloatLike](x: T, loc: T, scale: T) -> T:
        """The density. `scipy.stats.cauchy.pdf`."""
        var z = (x - loc) / scale
        return T.one() / (T.constant(_PI) * scale * (T.one() + z * z))

    @staticmethod
    def logpdf[T: FloatLike](x: T, loc: T, scale: T) -> T:
        """The log density, finite (`_LOG_ZERO`) outside the support. `scipy.stats.cauchy.logpdf`.
        """
        var z = (x - loc) / scale
        return -_safe_ln(T.constant(_PI) * scale) - (T.one() + z * z).ln()

    @staticmethod
    def cdf[T: FloatLike](x: T, loc: T, scale: T) -> T:
        """The CDF. `scipy.stats.cauchy.cdf`."""
        return T.constant(0.5) + _atan((x - loc) / scale) / T.constant(_PI)

    @staticmethod
    def logcdf[T: FloatLike](x: T, loc: T, scale: T) -> T:
        """`ln cdf`. `scipy.stats.cauchy.logcdf`."""
        return _safe_ln(cauchy.cdf(x, loc, scale))

    @staticmethod
    def sf[T: FloatLike](x: T, loc: T, scale: T) -> T:
        """The survival function `P(X > x)`. `scipy.stats.cauchy.sf`."""
        return T.constant(0.5) - _atan((x - loc) / scale) / T.constant(_PI)

    @staticmethod
    def logsf[T: FloatLike](x: T, loc: T, scale: T) -> T:
        """`ln sf`. `scipy.stats.cauchy.logsf`."""
        return _safe_ln(cauchy.sf(x, loc, scale))

    @staticmethod
    def ppf[T: FloatLike](p: T, loc: T, scale: T) -> T:
        """The quantile, closed form. `scipy.stats.cauchy.ppf`."""
        var angle = T.constant(_PI) * (p - T.constant(0.5))
        return loc + scale * angle.sin() / angle.cos()

    @staticmethod
    def isf[T: FloatLike](p: T, loc: T, scale: T) -> T:
        """The inverse survival function, `ppf(1 - p)` without the `1 - p`. `scipy.stats.cauchy.isf`.
        """
        var angle = T.constant(_PI) * (p - T.constant(0.5))
        return loc - scale * angle.sin() / angle.cos()

    @staticmethod
    def pdf[
        T: TensorLike, gpu: Bool = False
    ](x: T, loc: Scalar[T.dtype], scale: Scalar[T.dtype]) raises -> Tensor[
        T.dtype, T.LayoutType
    ] where (is_row_major[T] and T.dtype.is_floating_point()):
        """The density over a `Tensor`, parameters as scalars; see the module
        docstring for the two paths `gpu` picks between."""
        comptime dtype = T.dtype
        return _over2[
            step=_cauchy_pdf_step[dtype, _], gpu=gpu, name="cauchy.pdf"
        ](x, loc, scale)

    @staticmethod
    def cdf[
        T: TensorLike, gpu: Bool = False
    ](x: T, loc: Scalar[T.dtype], scale: Scalar[T.dtype]) raises -> Tensor[
        T.dtype, T.LayoutType
    ] where (is_row_major[T] and T.dtype.is_floating_point()):
        """The CDF over a `Tensor`, parameters as scalars; see the module
        docstring for the two paths `gpu` picks between."""
        comptime dtype = T.dtype
        return _over2[
            step=_cauchy_cdf_step[dtype, _], gpu=gpu, name="cauchy.cdf"
        ](x, loc, scale)

    @staticmethod
    def ppf[
        T: TensorLike, gpu: Bool = False
    ](p: T, loc: Scalar[T.dtype], scale: Scalar[T.dtype]) raises -> Tensor[
        T.dtype, T.LayoutType
    ] where (is_row_major[T] and T.dtype.is_floating_point()):
        """The quantile over a `Tensor`, parameters as scalars; see the module
        docstring for the two paths `gpu` picks between."""
        comptime dtype = T.dtype
        return _over2[
            step=_cauchy_ppf_step[dtype, _], gpu=gpu, name="cauchy.ppf"
        ](p, loc, scale)

    @staticmethod
    def rvs[
        dtype: DType, *dims: Int, gpu: Bool = False
    ](
        loc: Scalar[dtype],
        scale: Scalar[dtype],
        mut rng: Generator,
        ctx: Optional[DeviceContext] = None,
    ) raises -> Static[dtype, *dims] where dtype.is_floating_point():
        """Random draws at the compile-time shape `dims`, from `rng`'s stream,
        by inverting the closed-form `ppf` over uniforms in `(0, 1)`. `scipy.stats.cauchy.rvs`.

        Parameters:
            dtype: The element type of the draws.
            dims: The shape of the result.
            gpu: Whether to draw one thread per element on `ctx`'s device.

        Args:
            loc: The distribution's `loc`.
            scale: The distribution's `scale`.
            rng: The generator; its seed advances.
            ctx: The device to allocate and draw on; `None` means the host.

        Returns:
            A new `Static` tensor of draws.

        Raises:
            When allocation or the fill fails.
        """
        comptime count = _product[*dims]()
        var u = rng.uniform[dtype, count, gpu=gpu](
            Scalar[dtype](2.9802322387695312e-08), 1, ctx
        )
        var flat = cauchy.ppf[gpu=gpu](u, loc, scale)
        return Static[dtype, *dims](
            flat._buffer,
            rebind[_LayoutOf[*dims]](row_major[*dims]()),
            flat.host_addressable,
        )

    @staticmethod
    def mean(loc: Float64, scale: Float64) -> Float64:
        """The mean. `scipy.stats.cauchy.mean`; `inf` or NaN where SciPy gives them.

        Args:
            loc: The distribution's `loc`.
            scale: The distribution's `scale`.

        Returns:
            The mean.
        """
        return _nan64()

    @staticmethod
    def var(loc: Float64, scale: Float64) -> Float64:
        """The variance. `scipy.stats.cauchy.var`; `inf` or NaN where SciPy gives them.

        Args:
            loc: The distribution's `loc`.
            scale: The distribution's `scale`.

        Returns:
            The variance.
        """
        return _nan64()

    @staticmethod
    def std(loc: Float64, scale: Float64) -> Float64:
        """The standard deviation, `sqrt(var)`.

        Args:
            loc: The distribution's `loc`.
            scale: The distribution's `scale`.

        Returns:
            The standard deviation.
        """
        return _sqrt64(cauchy.var(loc, scale))

    @staticmethod
    def interval(
        confidence: Float64, loc: Float64, scale: Float64
    ) -> Tuple[Float64, Float64]:
        """The central interval holding `confidence` of the probability.
        `scipy.stats.cauchy.interval`.

        Args:
            confidence: The probability inside the interval, in `[0, 1]`.
            loc: The distribution's `loc`.
            scale: The distribution's `scale`.

        Returns:
            The interval's lower and upper ends.
        """
        var lo = cauchy.ppf(_p((1.0 - confidence) / 2.0), _p(loc), _p(scale))
        var hi = cauchy.ppf(_p((1.0 + confidence) / 2.0), _p(loc), _p(scale))
        return (_v(lo), _v(hi))

    @staticmethod
    def entropy(loc: Float64, scale: Float64) -> Float64:
        """The differential entropy in nats. `scipy.stats.cauchy.entropy`.

        Args:
            loc: The distribution's `loc`.
            scale: The distribution's `scale`.

        Returns:
            The entropy.
        """
        return _log64(4.0 * _PI * scale)


def _cauchy_pdf_step[
    dtype: DType, w: Int
](x: SIMD[dtype, w], loc: SIMD[dtype, 1], scale: SIMD[dtype, 1]) -> SIMD[
    dtype, w
] where dtype.is_floating_point():
    return cauchy.pdf(
        Plain[dtype, w](x),
        Plain[dtype, w](SIMD[dtype, w](loc[0])),
        Plain[dtype, w](SIMD[dtype, w](scale[0])),
    ).v


def _cauchy_cdf_step[
    dtype: DType, w: Int
](x: SIMD[dtype, w], loc: SIMD[dtype, 1], scale: SIMD[dtype, 1]) -> SIMD[
    dtype, w
] where dtype.is_floating_point():
    return cauchy.cdf(
        Plain[dtype, w](x),
        Plain[dtype, w](SIMD[dtype, w](loc[0])),
        Plain[dtype, w](SIMD[dtype, w](scale[0])),
    ).v


def _cauchy_ppf_step[
    dtype: DType, w: Int
](x: SIMD[dtype, w], loc: SIMD[dtype, 1], scale: SIMD[dtype, 1]) -> SIMD[
    dtype, w
] where dtype.is_floating_point():
    return cauchy.ppf(
        Plain[dtype, w](x),
        Plain[dtype, w](SIMD[dtype, w](loc[0])),
        Plain[dtype, w](SIMD[dtype, w](scale[0])),
    ).v


struct laplace:
    """The Laplace (double exponential) distribution. `scipy.stats.laplace`."""

    @staticmethod
    def pdf[T: FloatLike](x: T, loc: T, scale: T) -> T:
        """The density. `scipy.stats.laplace.pdf`."""
        var z = (x - loc) / scale
        return (-(z.abs())).exp() / (T.constant(2.0) * scale)

    @staticmethod
    def logpdf[T: FloatLike](x: T, loc: T, scale: T) -> T:
        """The log density, finite (`_LOG_ZERO`) outside the support. `scipy.stats.laplace.logpdf`.
        """
        var z = (x - loc) / scale
        return -_safe_ln(T.constant(2.0) * scale) - z.abs()

    @staticmethod
    def cdf[T: FloatLike](x: T, loc: T, scale: T) -> T:
        """The CDF. `scipy.stats.laplace.cdf`."""
        var z = (x - loc) / scale
        var half = T.constant(0.5) * (-(z.abs())).exp()
        return blend(
            ge_indicator(z, T.constant(0.0)), T.one() - half, half.copy()
        )

    @staticmethod
    def logcdf[T: FloatLike](x: T, loc: T, scale: T) -> T:
        """`ln cdf`. `scipy.stats.laplace.logcdf`."""
        return _safe_ln(laplace.cdf(x, loc, scale))

    @staticmethod
    def sf[T: FloatLike](x: T, loc: T, scale: T) -> T:
        """The survival function `P(X > x)`. `scipy.stats.laplace.sf`."""
        var z = (x - loc) / scale
        var half = T.constant(0.5) * (-(z.abs())).exp()
        return blend(
            ge_indicator(z, T.constant(0.0)), half.copy(), T.one() - half
        )

    @staticmethod
    def logsf[T: FloatLike](x: T, loc: T, scale: T) -> T:
        """`ln sf`. `scipy.stats.laplace.logsf`."""
        return _safe_ln(laplace.sf(x, loc, scale))

    @staticmethod
    def ppf[T: FloatLike](p: T, loc: T, scale: T) -> T:
        """The quantile, closed form. `scipy.stats.laplace.ppf`."""
        var low = loc + scale * _safe_ln(T.constant(2.0) * p)
        var high = loc - scale * _safe_ln(T.constant(2.0) * (T.one() - p))
        return blend(ge_indicator(p, T.constant(0.5)), high, low)

    @staticmethod
    def isf[T: FloatLike](p: T, loc: T, scale: T) -> T:
        """The inverse survival function, `ppf(1 - p)` without the `1 - p`. `scipy.stats.laplace.isf`.
        """
        return laplace.ppf(T.one() - p, loc, scale)

    @staticmethod
    def pdf[
        T: TensorLike, gpu: Bool = False
    ](x: T, loc: Scalar[T.dtype], scale: Scalar[T.dtype]) raises -> Tensor[
        T.dtype, T.LayoutType
    ] where (is_row_major[T] and T.dtype.is_floating_point()):
        """The density over a `Tensor`, parameters as scalars; see the module
        docstring for the two paths `gpu` picks between."""
        comptime dtype = T.dtype
        return _over2[
            step=_laplace_pdf_step[dtype, _], gpu=gpu, name="laplace.pdf"
        ](x, loc, scale)

    @staticmethod
    def cdf[
        T: TensorLike, gpu: Bool = False
    ](x: T, loc: Scalar[T.dtype], scale: Scalar[T.dtype]) raises -> Tensor[
        T.dtype, T.LayoutType
    ] where (is_row_major[T] and T.dtype.is_floating_point()):
        """The CDF over a `Tensor`, parameters as scalars; see the module
        docstring for the two paths `gpu` picks between."""
        comptime dtype = T.dtype
        return _over2[
            step=_laplace_cdf_step[dtype, _], gpu=gpu, name="laplace.cdf"
        ](x, loc, scale)

    @staticmethod
    def ppf[
        T: TensorLike, gpu: Bool = False
    ](p: T, loc: Scalar[T.dtype], scale: Scalar[T.dtype]) raises -> Tensor[
        T.dtype, T.LayoutType
    ] where (is_row_major[T] and T.dtype.is_floating_point()):
        """The quantile over a `Tensor`, parameters as scalars; see the module
        docstring for the two paths `gpu` picks between."""
        comptime dtype = T.dtype
        return _over2[
            step=_laplace_ppf_step[dtype, _], gpu=gpu, name="laplace.ppf"
        ](p, loc, scale)

    @staticmethod
    def rvs[
        dtype: DType, *dims: Int, gpu: Bool = False
    ](
        loc: Scalar[dtype],
        scale: Scalar[dtype],
        mut rng: Generator,
        ctx: Optional[DeviceContext] = None,
    ) raises -> Static[dtype, *dims] where dtype.is_floating_point():
        """Random draws at the compile-time shape `dims`, from `rng`'s stream,
        by inverting the closed-form `ppf` over uniforms in `(0, 1)`. `scipy.stats.laplace.rvs`.

        Parameters:
            dtype: The element type of the draws.
            dims: The shape of the result.
            gpu: Whether to draw one thread per element on `ctx`'s device.

        Args:
            loc: The distribution's `loc`.
            scale: The distribution's `scale`.
            rng: The generator; its seed advances.
            ctx: The device to allocate and draw on; `None` means the host.

        Returns:
            A new `Static` tensor of draws.

        Raises:
            When allocation or the fill fails.
        """
        comptime count = _product[*dims]()
        var u = rng.uniform[dtype, count, gpu=gpu](
            Scalar[dtype](2.9802322387695312e-08), 1, ctx
        )
        var flat = laplace.ppf[gpu=gpu](u, loc, scale)
        return Static[dtype, *dims](
            flat._buffer,
            rebind[_LayoutOf[*dims]](row_major[*dims]()),
            flat.host_addressable,
        )

    @staticmethod
    def mean(loc: Float64, scale: Float64) -> Float64:
        """The mean. `scipy.stats.laplace.mean`; `inf` or NaN where SciPy gives them.

        Args:
            loc: The distribution's `loc`.
            scale: The distribution's `scale`.

        Returns:
            The mean.
        """
        return loc

    @staticmethod
    def var(loc: Float64, scale: Float64) -> Float64:
        """The variance. `scipy.stats.laplace.var`; `inf` or NaN where SciPy gives them.

        Args:
            loc: The distribution's `loc`.
            scale: The distribution's `scale`.

        Returns:
            The variance.
        """
        return 2.0 * scale * scale

    @staticmethod
    def std(loc: Float64, scale: Float64) -> Float64:
        """The standard deviation, `sqrt(var)`.

        Args:
            loc: The distribution's `loc`.
            scale: The distribution's `scale`.

        Returns:
            The standard deviation.
        """
        return _sqrt64(laplace.var(loc, scale))

    @staticmethod
    def interval(
        confidence: Float64, loc: Float64, scale: Float64
    ) -> Tuple[Float64, Float64]:
        """The central interval holding `confidence` of the probability.
        `scipy.stats.laplace.interval`.

        Args:
            confidence: The probability inside the interval, in `[0, 1]`.
            loc: The distribution's `loc`.
            scale: The distribution's `scale`.

        Returns:
            The interval's lower and upper ends.
        """
        var lo = laplace.ppf(_p((1.0 - confidence) / 2.0), _p(loc), _p(scale))
        var hi = laplace.ppf(_p((1.0 + confidence) / 2.0), _p(loc), _p(scale))
        return (_v(lo), _v(hi))

    @staticmethod
    def entropy(loc: Float64, scale: Float64) -> Float64:
        """The differential entropy in nats. `scipy.stats.laplace.entropy`.

        Args:
            loc: The distribution's `loc`.
            scale: The distribution's `scale`.

        Returns:
            The entropy.
        """
        return 1.0 + _log64(2.0 * scale)


def _laplace_pdf_step[
    dtype: DType, w: Int
](x: SIMD[dtype, w], loc: SIMD[dtype, 1], scale: SIMD[dtype, 1]) -> SIMD[
    dtype, w
] where dtype.is_floating_point():
    return laplace.pdf(
        Plain[dtype, w](x),
        Plain[dtype, w](SIMD[dtype, w](loc[0])),
        Plain[dtype, w](SIMD[dtype, w](scale[0])),
    ).v


def _laplace_cdf_step[
    dtype: DType, w: Int
](x: SIMD[dtype, w], loc: SIMD[dtype, 1], scale: SIMD[dtype, 1]) -> SIMD[
    dtype, w
] where dtype.is_floating_point():
    return laplace.cdf(
        Plain[dtype, w](x),
        Plain[dtype, w](SIMD[dtype, w](loc[0])),
        Plain[dtype, w](SIMD[dtype, w](scale[0])),
    ).v


def _laplace_ppf_step[
    dtype: DType, w: Int
](x: SIMD[dtype, w], loc: SIMD[dtype, 1], scale: SIMD[dtype, 1]) -> SIMD[
    dtype, w
] where dtype.is_floating_point():
    return laplace.ppf(
        Plain[dtype, w](x),
        Plain[dtype, w](SIMD[dtype, w](loc[0])),
        Plain[dtype, w](SIMD[dtype, w](scale[0])),
    ).v


struct rayleigh:
    """The Rayleigh distribution: the length of a 2-D standard normal vector, times `scale`. `scipy.stats.rayleigh`.
    """

    @staticmethod
    def pdf[T: FloatLike](x: T, scale: T) -> T:
        """The density. `scipy.stats.rayleigh.pdf`."""
        var z = max_of(x.copy(), T.constant(0.0)) / scale
        return (
            z.copy()
            / scale
            * (-(z * z) / T.constant(2.0)).exp()
            * ge_indicator(x, T.constant(0.0))
        )

    @staticmethod
    def logpdf[T: FloatLike](x: T, scale: T) -> T:
        """The log density, finite (`_LOG_ZERO`) outside the support. `scipy.stats.rayleigh.logpdf`.
        """
        var z = max_of(x.copy(), T.constant(0.0)) / scale
        var inside = (
            _safe_ln(z.copy()) - _safe_ln(scale) - z * z / T.constant(2.0)
        )
        return blend(
            ge_indicator(x, T.constant(_TINY)), inside, T.constant(_LOG_ZERO)
        )

    @staticmethod
    def cdf[T: FloatLike](x: T, scale: T) -> T:
        """The CDF. `scipy.stats.rayleigh.cdf`."""
        var z = max_of(x, T.constant(0.0)) / scale
        return -_expm1(-(z * z) / T.constant(2.0))

    @staticmethod
    def logcdf[T: FloatLike](x: T, scale: T) -> T:
        """`ln cdf`. `scipy.stats.rayleigh.logcdf`."""
        return _safe_ln(rayleigh.cdf(x, scale))

    @staticmethod
    def sf[T: FloatLike](x: T, scale: T) -> T:
        """The survival function `P(X > x)`. `scipy.stats.rayleigh.sf`."""
        var z = max_of(x, T.constant(0.0)) / scale
        return (-(z * z) / T.constant(2.0)).exp()

    @staticmethod
    def logsf[T: FloatLike](x: T, scale: T) -> T:
        """`ln sf`. `scipy.stats.rayleigh.logsf`."""
        return _safe_ln(rayleigh.sf(x, scale))

    @staticmethod
    def ppf[T: FloatLike](p: T, scale: T) -> T:
        """The quantile, closed form. `scipy.stats.rayleigh.ppf`."""
        return scale * (T.constant(-2.0) * _log1p(-p)).sqrt()

    @staticmethod
    def isf[T: FloatLike](p: T, scale: T) -> T:
        """The inverse survival function, `ppf(1 - p)` without the `1 - p`. `scipy.stats.rayleigh.isf`.
        """
        return scale * (T.constant(-2.0) * _safe_ln(p)).sqrt()

    @staticmethod
    def pdf[
        T: TensorLike, gpu: Bool = False
    ](x: T, scale: Scalar[T.dtype]) raises -> Tensor[
        T.dtype, T.LayoutType
    ] where (is_row_major[T] and T.dtype.is_floating_point()):
        """The density over a `Tensor`, parameters as scalars; see the module
        docstring for the two paths `gpu` picks between."""
        comptime dtype = T.dtype
        return _over1[
            step=_rayleigh_pdf_step[dtype, _], gpu=gpu, name="rayleigh.pdf"
        ](x, scale)

    @staticmethod
    def cdf[
        T: TensorLike, gpu: Bool = False
    ](x: T, scale: Scalar[T.dtype]) raises -> Tensor[
        T.dtype, T.LayoutType
    ] where (is_row_major[T] and T.dtype.is_floating_point()):
        """The CDF over a `Tensor`, parameters as scalars; see the module
        docstring for the two paths `gpu` picks between."""
        comptime dtype = T.dtype
        return _over1[
            step=_rayleigh_cdf_step[dtype, _], gpu=gpu, name="rayleigh.cdf"
        ](x, scale)

    @staticmethod
    def ppf[
        T: TensorLike, gpu: Bool = False
    ](p: T, scale: Scalar[T.dtype]) raises -> Tensor[
        T.dtype, T.LayoutType
    ] where (is_row_major[T] and T.dtype.is_floating_point()):
        """The quantile over a `Tensor`, parameters as scalars; see the module
        docstring for the two paths `gpu` picks between."""
        comptime dtype = T.dtype
        return _over1[
            step=_rayleigh_ppf_step[dtype, _], gpu=gpu, name="rayleigh.ppf"
        ](p, scale)

    @staticmethod
    def rvs[
        dtype: DType, *dims: Int, gpu: Bool = False
    ](
        scale: Scalar[dtype],
        mut rng: Generator,
        ctx: Optional[DeviceContext] = None,
    ) raises -> Static[dtype, *dims] where dtype.is_floating_point():
        """Random draws at the compile-time shape `dims`, from `rng`'s stream,
        by inverting the closed-form `ppf` over uniforms in `(0, 1)`. `scipy.stats.rayleigh.rvs`.

        Parameters:
            dtype: The element type of the draws.
            dims: The shape of the result.
            gpu: Whether to draw one thread per element on `ctx`'s device.

        Args:
            scale: The distribution's `scale`.
            rng: The generator; its seed advances.
            ctx: The device to allocate and draw on; `None` means the host.

        Returns:
            A new `Static` tensor of draws.

        Raises:
            When allocation or the fill fails.
        """
        comptime count = _product[*dims]()
        var u = rng.uniform[dtype, count, gpu=gpu](
            Scalar[dtype](2.9802322387695312e-08), 1, ctx
        )
        var flat = rayleigh.ppf[gpu=gpu](u, scale)
        return Static[dtype, *dims](
            flat._buffer,
            rebind[_LayoutOf[*dims]](row_major[*dims]()),
            flat.host_addressable,
        )

    @staticmethod
    def mean(scale: Float64) -> Float64:
        """The mean. `scipy.stats.rayleigh.mean`; `inf` or NaN where SciPy gives them.

        Args:
            scale: The distribution's `scale`.

        Returns:
            The mean.
        """
        return scale * _sqrt64(_PI / 2.0)

    @staticmethod
    def var(scale: Float64) -> Float64:
        """The variance. `scipy.stats.rayleigh.var`; `inf` or NaN where SciPy gives them.

        Args:
            scale: The distribution's `scale`.

        Returns:
            The variance.
        """
        return (4.0 - _PI) / 2.0 * scale * scale

    @staticmethod
    def std(scale: Float64) -> Float64:
        """The standard deviation, `sqrt(var)`.

        Args:
            scale: The distribution's `scale`.

        Returns:
            The standard deviation.
        """
        return _sqrt64(rayleigh.var(scale))

    @staticmethod
    def interval(
        confidence: Float64, scale: Float64
    ) -> Tuple[Float64, Float64]:
        """The central interval holding `confidence` of the probability.
        `scipy.stats.rayleigh.interval`.

        Args:
            confidence: The probability inside the interval, in `[0, 1]`.
            scale: The distribution's `scale`.

        Returns:
            The interval's lower and upper ends.
        """
        var lo = rayleigh.ppf(_p((1.0 - confidence) / 2.0), _p(scale))
        var hi = rayleigh.ppf(_p((1.0 + confidence) / 2.0), _p(scale))
        return (_v(lo), _v(hi))

    @staticmethod
    def entropy(scale: Float64) -> Float64:
        """The differential entropy in nats. `scipy.stats.rayleigh.entropy`.

        Args:
            scale: The distribution's `scale`.

        Returns:
            The entropy.
        """
        return 1.0 + _log64(scale / _SQRT_2) + _EULER / 2.0


def _rayleigh_pdf_step[
    dtype: DType, w: Int
](x: SIMD[dtype, w], scale: SIMD[dtype, 1]) -> SIMD[
    dtype, w
] where dtype.is_floating_point():
    return rayleigh.pdf(
        Plain[dtype, w](x), Plain[dtype, w](SIMD[dtype, w](scale[0]))
    ).v


def _rayleigh_cdf_step[
    dtype: DType, w: Int
](x: SIMD[dtype, w], scale: SIMD[dtype, 1]) -> SIMD[
    dtype, w
] where dtype.is_floating_point():
    return rayleigh.cdf(
        Plain[dtype, w](x), Plain[dtype, w](SIMD[dtype, w](scale[0]))
    ).v


def _rayleigh_ppf_step[
    dtype: DType, w: Int
](x: SIMD[dtype, w], scale: SIMD[dtype, 1]) -> SIMD[
    dtype, w
] where dtype.is_floating_point():
    return rayleigh.ppf(
        Plain[dtype, w](x), Plain[dtype, w](SIMD[dtype, w](scale[0]))
    ).v


struct logistic:
    """The logistic distribution. `scipy.stats.logistic`."""

    @staticmethod
    def pdf[T: FloatLike](x: T, loc: T, scale: T) -> T:
        """The density. `scipy.stats.logistic.pdf`."""
        var z = (x - loc) / scale
        var e = (-(z.abs())).exp()
        return e.copy() / (scale * (T.one() + e.copy()) * (T.one() + e))

    @staticmethod
    def logpdf[T: FloatLike](x: T, loc: T, scale: T) -> T:
        """The log density, finite (`_LOG_ZERO`) outside the support. `scipy.stats.logistic.logpdf`.
        """
        var z = (x - loc) / scale
        return (
            -(z.abs())
            - T.constant(2.0) * (T.one() + (-(z.abs())).exp()).ln()
            - _safe_ln(scale)
        )

    @staticmethod
    def cdf[T: FloatLike](x: T, loc: T, scale: T) -> T:
        """The CDF. `scipy.stats.logistic.cdf`."""
        var z = (x - loc) / scale
        var e = (-(z.abs())).exp()
        return blend(
            ge_indicator(z, T.constant(0.0)),
            T.one() / (T.one() + e.copy()),
            e.copy() / (T.one() + e),
        )

    @staticmethod
    def logcdf[T: FloatLike](x: T, loc: T, scale: T) -> T:
        """`ln cdf`. `scipy.stats.logistic.logcdf`."""
        return _safe_ln(logistic.cdf(x, loc, scale))

    @staticmethod
    def sf[T: FloatLike](x: T, loc: T, scale: T) -> T:
        """The survival function `P(X > x)`. `scipy.stats.logistic.sf`."""
        return logistic.cdf(-(x - loc) + loc, loc, scale)

    @staticmethod
    def logsf[T: FloatLike](x: T, loc: T, scale: T) -> T:
        """`ln sf`. `scipy.stats.logistic.logsf`."""
        return _safe_ln(logistic.sf(x, loc, scale))

    @staticmethod
    def ppf[T: FloatLike](p: T, loc: T, scale: T) -> T:
        """The quantile, closed form. `scipy.stats.logistic.ppf`."""
        return loc + scale * (_safe_ln(p) - _log1p(-p))

    @staticmethod
    def isf[T: FloatLike](p: T, loc: T, scale: T) -> T:
        """The inverse survival function, `ppf(1 - p)` without the `1 - p`. `scipy.stats.logistic.isf`.
        """
        return loc + scale * (_log1p(-p) - _safe_ln(p))

    @staticmethod
    def pdf[
        T: TensorLike, gpu: Bool = False
    ](x: T, loc: Scalar[T.dtype], scale: Scalar[T.dtype]) raises -> Tensor[
        T.dtype, T.LayoutType
    ] where (is_row_major[T] and T.dtype.is_floating_point()):
        """The density over a `Tensor`, parameters as scalars; see the module
        docstring for the two paths `gpu` picks between."""
        comptime dtype = T.dtype
        return _over2[
            step=_logistic_pdf_step[dtype, _], gpu=gpu, name="logistic.pdf"
        ](x, loc, scale)

    @staticmethod
    def cdf[
        T: TensorLike, gpu: Bool = False
    ](x: T, loc: Scalar[T.dtype], scale: Scalar[T.dtype]) raises -> Tensor[
        T.dtype, T.LayoutType
    ] where (is_row_major[T] and T.dtype.is_floating_point()):
        """The CDF over a `Tensor`, parameters as scalars; see the module
        docstring for the two paths `gpu` picks between."""
        comptime dtype = T.dtype
        return _over2[
            step=_logistic_cdf_step[dtype, _], gpu=gpu, name="logistic.cdf"
        ](x, loc, scale)

    @staticmethod
    def ppf[
        T: TensorLike, gpu: Bool = False
    ](p: T, loc: Scalar[T.dtype], scale: Scalar[T.dtype]) raises -> Tensor[
        T.dtype, T.LayoutType
    ] where (is_row_major[T] and T.dtype.is_floating_point()):
        """The quantile over a `Tensor`, parameters as scalars; see the module
        docstring for the two paths `gpu` picks between."""
        comptime dtype = T.dtype
        return _over2[
            step=_logistic_ppf_step[dtype, _], gpu=gpu, name="logistic.ppf"
        ](p, loc, scale)

    @staticmethod
    def rvs[
        dtype: DType, *dims: Int, gpu: Bool = False
    ](
        loc: Scalar[dtype],
        scale: Scalar[dtype],
        mut rng: Generator,
        ctx: Optional[DeviceContext] = None,
    ) raises -> Static[dtype, *dims] where dtype.is_floating_point():
        """Random draws at the compile-time shape `dims`, from `rng`'s stream,
        by inverting the closed-form `ppf` over uniforms in `(0, 1)`. `scipy.stats.logistic.rvs`.

        Parameters:
            dtype: The element type of the draws.
            dims: The shape of the result.
            gpu: Whether to draw one thread per element on `ctx`'s device.

        Args:
            loc: The distribution's `loc`.
            scale: The distribution's `scale`.
            rng: The generator; its seed advances.
            ctx: The device to allocate and draw on; `None` means the host.

        Returns:
            A new `Static` tensor of draws.

        Raises:
            When allocation or the fill fails.
        """
        comptime count = _product[*dims]()
        var u = rng.uniform[dtype, count, gpu=gpu](
            Scalar[dtype](2.9802322387695312e-08), 1, ctx
        )
        var flat = logistic.ppf[gpu=gpu](u, loc, scale)
        return Static[dtype, *dims](
            flat._buffer,
            rebind[_LayoutOf[*dims]](row_major[*dims]()),
            flat.host_addressable,
        )

    @staticmethod
    def mean(loc: Float64, scale: Float64) -> Float64:
        """The mean. `scipy.stats.logistic.mean`; `inf` or NaN where SciPy gives them.

        Args:
            loc: The distribution's `loc`.
            scale: The distribution's `scale`.

        Returns:
            The mean.
        """
        return loc

    @staticmethod
    def var(loc: Float64, scale: Float64) -> Float64:
        """The variance. `scipy.stats.logistic.var`; `inf` or NaN where SciPy gives them.

        Args:
            loc: The distribution's `loc`.
            scale: The distribution's `scale`.

        Returns:
            The variance.
        """
        return _PI * _PI * scale * scale / 3.0

    @staticmethod
    def std(loc: Float64, scale: Float64) -> Float64:
        """The standard deviation, `sqrt(var)`.

        Args:
            loc: The distribution's `loc`.
            scale: The distribution's `scale`.

        Returns:
            The standard deviation.
        """
        return _sqrt64(logistic.var(loc, scale))

    @staticmethod
    def interval(
        confidence: Float64, loc: Float64, scale: Float64
    ) -> Tuple[Float64, Float64]:
        """The central interval holding `confidence` of the probability.
        `scipy.stats.logistic.interval`.

        Args:
            confidence: The probability inside the interval, in `[0, 1]`.
            loc: The distribution's `loc`.
            scale: The distribution's `scale`.

        Returns:
            The interval's lower and upper ends.
        """
        var lo = logistic.ppf(_p((1.0 - confidence) / 2.0), _p(loc), _p(scale))
        var hi = logistic.ppf(_p((1.0 + confidence) / 2.0), _p(loc), _p(scale))
        return (_v(lo), _v(hi))

    @staticmethod
    def entropy(loc: Float64, scale: Float64) -> Float64:
        """The differential entropy in nats. `scipy.stats.logistic.entropy`.

        Args:
            loc: The distribution's `loc`.
            scale: The distribution's `scale`.

        Returns:
            The entropy.
        """
        return 2.0 + _log64(scale)


def _logistic_pdf_step[
    dtype: DType, w: Int
](x: SIMD[dtype, w], loc: SIMD[dtype, 1], scale: SIMD[dtype, 1]) -> SIMD[
    dtype, w
] where dtype.is_floating_point():
    return logistic.pdf(
        Plain[dtype, w](x),
        Plain[dtype, w](SIMD[dtype, w](loc[0])),
        Plain[dtype, w](SIMD[dtype, w](scale[0])),
    ).v


def _logistic_cdf_step[
    dtype: DType, w: Int
](x: SIMD[dtype, w], loc: SIMD[dtype, 1], scale: SIMD[dtype, 1]) -> SIMD[
    dtype, w
] where dtype.is_floating_point():
    return logistic.cdf(
        Plain[dtype, w](x),
        Plain[dtype, w](SIMD[dtype, w](loc[0])),
        Plain[dtype, w](SIMD[dtype, w](scale[0])),
    ).v


def _logistic_ppf_step[
    dtype: DType, w: Int
](x: SIMD[dtype, w], loc: SIMD[dtype, 1], scale: SIMD[dtype, 1]) -> SIMD[
    dtype, w
] where dtype.is_floating_point():
    return logistic.ppf(
        Plain[dtype, w](x),
        Plain[dtype, w](SIMD[dtype, w](loc[0])),
        Plain[dtype, w](SIMD[dtype, w](scale[0])),
    ).v


struct pareto:
    """The Pareto distribution with shape `b`, supported on `x >= scale`. `scipy.stats.pareto`.
    """

    @staticmethod
    def pdf[T: FloatLike](x: T, b: T, scale: T) -> T:
        """The density. `scipy.stats.pareto.pdf`."""
        var y = max_of(x.copy() / scale, T.one())
        return (
            b
            / scale
            * _pow(y, -(b + T.one()))
            * ge_indicator(x / scale, T.one())
        )

    @staticmethod
    def logpdf[T: FloatLike](x: T, b: T, scale: T) -> T:
        """The log density, finite (`_LOG_ZERO`) outside the support. `scipy.stats.pareto.logpdf`.
        """
        var y = max_of(x.copy() / scale, T.one())
        var inside = _safe_ln(b / scale) - (b + T.one()) * y.ln()
        return blend(
            ge_indicator(x / scale, T.one()), inside, T.constant(_LOG_ZERO)
        )

    @staticmethod
    def cdf[T: FloatLike](x: T, b: T, scale: T) -> T:
        """The CDF. `scipy.stats.pareto.cdf`."""
        var y = max_of(x / scale, T.one())
        return T.one() - _pow(y, -b)

    @staticmethod
    def logcdf[T: FloatLike](x: T, b: T, scale: T) -> T:
        """`ln cdf`. `scipy.stats.pareto.logcdf`."""
        return _safe_ln(pareto.cdf(x, b, scale))

    @staticmethod
    def sf[T: FloatLike](x: T, b: T, scale: T) -> T:
        """The survival function `P(X > x)`. `scipy.stats.pareto.sf`."""
        var y = max_of(x / scale, T.one())
        return _pow(y, -b)

    @staticmethod
    def logsf[T: FloatLike](x: T, b: T, scale: T) -> T:
        """`ln sf`. `scipy.stats.pareto.logsf`."""
        return _safe_ln(pareto.sf(x, b, scale))

    @staticmethod
    def ppf[T: FloatLike](p: T, b: T, scale: T) -> T:
        """The quantile, closed form. `scipy.stats.pareto.ppf`."""
        return scale * _pow(T.one() - p, -(T.one() / b))

    @staticmethod
    def isf[T: FloatLike](p: T, b: T, scale: T) -> T:
        """The inverse survival function, `ppf(1 - p)` without the `1 - p`. `scipy.stats.pareto.isf`.
        """
        return scale * _pow(p, -(T.one() / b))

    @staticmethod
    def pdf[
        T: TensorLike, gpu: Bool = False
    ](x: T, b: Scalar[T.dtype], scale: Scalar[T.dtype]) raises -> Tensor[
        T.dtype, T.LayoutType
    ] where (is_row_major[T] and T.dtype.is_floating_point()):
        """The density over a `Tensor`, parameters as scalars; see the module
        docstring for the two paths `gpu` picks between."""
        comptime dtype = T.dtype
        return _over2[
            step=_pareto_pdf_step[dtype, _], gpu=gpu, name="pareto.pdf"
        ](x, b, scale)

    @staticmethod
    def cdf[
        T: TensorLike, gpu: Bool = False
    ](x: T, b: Scalar[T.dtype], scale: Scalar[T.dtype]) raises -> Tensor[
        T.dtype, T.LayoutType
    ] where (is_row_major[T] and T.dtype.is_floating_point()):
        """The CDF over a `Tensor`, parameters as scalars; see the module
        docstring for the two paths `gpu` picks between."""
        comptime dtype = T.dtype
        return _over2[
            step=_pareto_cdf_step[dtype, _], gpu=gpu, name="pareto.cdf"
        ](x, b, scale)

    @staticmethod
    def ppf[
        T: TensorLike, gpu: Bool = False
    ](p: T, b: Scalar[T.dtype], scale: Scalar[T.dtype]) raises -> Tensor[
        T.dtype, T.LayoutType
    ] where (is_row_major[T] and T.dtype.is_floating_point()):
        """The quantile over a `Tensor`, parameters as scalars; see the module
        docstring for the two paths `gpu` picks between."""
        comptime dtype = T.dtype
        return _over2[
            step=_pareto_ppf_step[dtype, _], gpu=gpu, name="pareto.ppf"
        ](p, b, scale)

    @staticmethod
    def rvs[
        dtype: DType, *dims: Int, gpu: Bool = False
    ](
        b: Scalar[dtype],
        scale: Scalar[dtype],
        mut rng: Generator,
        ctx: Optional[DeviceContext] = None,
    ) raises -> Static[dtype, *dims] where dtype.is_floating_point():
        """Random draws at the compile-time shape `dims`, from `rng`'s stream,
        by inverting the closed-form `ppf` over uniforms in `(0, 1)`. `scipy.stats.pareto.rvs`.

        Parameters:
            dtype: The element type of the draws.
            dims: The shape of the result.
            gpu: Whether to draw one thread per element on `ctx`'s device.

        Args:
            b: The distribution's `b`.
            scale: The distribution's `scale`.
            rng: The generator; its seed advances.
            ctx: The device to allocate and draw on; `None` means the host.

        Returns:
            A new `Static` tensor of draws.

        Raises:
            When allocation or the fill fails.
        """
        comptime count = _product[*dims]()
        var u = rng.uniform[dtype, count, gpu=gpu](
            Scalar[dtype](2.9802322387695312e-08), 1, ctx
        )
        var flat = pareto.ppf[gpu=gpu](u, b, scale)
        return Static[dtype, *dims](
            flat._buffer,
            rebind[_LayoutOf[*dims]](row_major[*dims]()),
            flat.host_addressable,
        )

    @staticmethod
    def mean(b: Float64, scale: Float64) -> Float64:
        """The mean. `scipy.stats.pareto.mean`; `inf` or NaN where SciPy gives them.

        Args:
            b: The distribution's `b`.
            scale: The distribution's `scale`.

        Returns:
            The mean.
        """
        return b * scale / (b - 1.0) if b > 1.0 else _inf64()

    @staticmethod
    def var(b: Float64, scale: Float64) -> Float64:
        """The variance. `scipy.stats.pareto.var`; `inf` or NaN where SciPy gives them.

        Args:
            b: The distribution's `b`.
            scale: The distribution's `scale`.

        Returns:
            The variance.
        """
        return (
            scale * scale * b / ((b - 1.0) * (b - 1.0) * (b - 2.0)) if b
            > 2.0 else _inf64()
        )

    @staticmethod
    def std(b: Float64, scale: Float64) -> Float64:
        """The standard deviation, `sqrt(var)`.

        Args:
            b: The distribution's `b`.
            scale: The distribution's `scale`.

        Returns:
            The standard deviation.
        """
        return _sqrt64(pareto.var(b, scale))

    @staticmethod
    def interval(
        confidence: Float64, b: Float64, scale: Float64
    ) -> Tuple[Float64, Float64]:
        """The central interval holding `confidence` of the probability.
        `scipy.stats.pareto.interval`.

        Args:
            confidence: The probability inside the interval, in `[0, 1]`.
            b: The distribution's `b`.
            scale: The distribution's `scale`.

        Returns:
            The interval's lower and upper ends.
        """
        var lo = pareto.ppf(_p((1.0 - confidence) / 2.0), _p(b), _p(scale))
        var hi = pareto.ppf(_p((1.0 + confidence) / 2.0), _p(b), _p(scale))
        return (_v(lo), _v(hi))

    @staticmethod
    def entropy(b: Float64, scale: Float64) -> Float64:
        """The differential entropy in nats. `scipy.stats.pareto.entropy`.

        Args:
            b: The distribution's `b`.
            scale: The distribution's `scale`.

        Returns:
            The entropy.
        """
        return _log64(scale / b) + 1.0 / b + 1.0


def _pareto_pdf_step[
    dtype: DType, w: Int
](x: SIMD[dtype, w], b: SIMD[dtype, 1], scale: SIMD[dtype, 1]) -> SIMD[
    dtype, w
] where dtype.is_floating_point():
    return pareto.pdf(
        Plain[dtype, w](x),
        Plain[dtype, w](SIMD[dtype, w](b[0])),
        Plain[dtype, w](SIMD[dtype, w](scale[0])),
    ).v


def _pareto_cdf_step[
    dtype: DType, w: Int
](x: SIMD[dtype, w], b: SIMD[dtype, 1], scale: SIMD[dtype, 1]) -> SIMD[
    dtype, w
] where dtype.is_floating_point():
    return pareto.cdf(
        Plain[dtype, w](x),
        Plain[dtype, w](SIMD[dtype, w](b[0])),
        Plain[dtype, w](SIMD[dtype, w](scale[0])),
    ).v


def _pareto_ppf_step[
    dtype: DType, w: Int
](x: SIMD[dtype, w], b: SIMD[dtype, 1], scale: SIMD[dtype, 1]) -> SIMD[
    dtype, w
] where dtype.is_floating_point():
    return pareto.ppf(
        Plain[dtype, w](x),
        Plain[dtype, w](SIMD[dtype, w](b[0])),
        Plain[dtype, w](SIMD[dtype, w](scale[0])),
    ).v


struct bernoulli:
    """The Bernoulli distribution: `1` with probability `p`, else `0`.
    `scipy.stats.bernoulli`."""

    @staticmethod
    def pmf[T: FloatLike](k: T, p: T) -> T:
        """`p` at `k = 1`, `1 - p` at `k = 0`, and `0` elsewhere."""
        return bernoulli.logpmf(k, p).exp()

    @staticmethod
    def logpmf[T: FloatLike](k: T, p: T) -> T:
        """`k ln p + (1 - k) ln(1 - p)` on `{0, 1}`; `_LOG_ZERO` elsewhere."""
        # `0 - k`, not `-k`: at `k = 0` the negation is `-0.0`, which the
        # sign-reading indicator counts as below zero.
        var at_zero = ge_indicator(k.copy(), T.constant(0.0)) * ge_indicator(
            T.constant(0.0) - k.copy(), T.constant(0.0)
        )
        var at_one = ge_indicator(k.copy(), T.one()) * ge_indicator(
            T.one() - k, T.constant(0.0)
        )
        var inside = blend(
            at_one.copy(), _safe_ln(p.copy()), _safe_ln(T.one() - p)
        )
        return blend(at_zero + at_one, inside, T.constant(_LOG_ZERO))

    @staticmethod
    def cdf[T: FloatLike](k: T, p: T) -> T:
        """`0` below 0, `1 - p` on `[0, 1)`, `1` from 1."""
        return ge_indicator(k.copy(), T.constant(0.0)) * (
            T.one() - p * (T.one() - ge_indicator(k, T.one()))
        )

    @staticmethod
    def sf[T: FloatLike](k: T, p: T) -> T:
        """`1` below 0, `p` on `[0, 1)`, `0` from 1."""
        return blend(
            ge_indicator(k.copy(), T.constant(0.0)),
            p * (T.one() - ge_indicator(k, T.one())),
            T.one(),
        )

    @staticmethod
    def logcdf[T: FloatLike](k: T, p: T) -> T:
        return _safe_ln(bernoulli.cdf(k, p))

    @staticmethod
    def logsf[T: FloatLike](k: T, p: T) -> T:
        return _safe_ln(bernoulli.sf(k, p))

    @staticmethod
    def ppf[T: FloatLike](q: T, p: T) -> T:
        """The smallest `k` with `cdf(k) >= q`: `0` while `q <= 1 - p`, else
        `1`."""
        return T.one() - ge_indicator(T.one() - p, q)

    @staticmethod
    def isf[T: FloatLike](q: T, p: T) -> T:
        """`ppf(1 - q)`."""
        return bernoulli.ppf(T.one() - q, p)

    @staticmethod
    def pmf[
        T: TensorLike, gpu: Bool = False
    ](k: T, p: Scalar[T.dtype]) raises -> Tensor[T.dtype, T.LayoutType] where (
        is_row_major[T] and T.dtype.is_floating_point()
    ):
        """The PMF over a `Tensor`, parameters as scalars; see the module
        docstring for the two paths `gpu` picks between."""
        comptime dtype = T.dtype
        return _over1[
            step=_bernoulli_pmf_step[dtype, _], gpu=gpu, name="bernoulli.pmf"
        ](k, p)

    @staticmethod
    def cdf[
        T: TensorLike, gpu: Bool = False
    ](k: T, p: Scalar[T.dtype]) raises -> Tensor[T.dtype, T.LayoutType] where (
        is_row_major[T] and T.dtype.is_floating_point()
    ):
        """The CDF over a `Tensor`, parameters as scalars; see the module
        docstring for the two paths `gpu` picks between."""
        comptime dtype = T.dtype
        return _over1[
            step=_bernoulli_cdf_step[dtype, _], gpu=gpu, name="bernoulli.cdf"
        ](k, p)

    @staticmethod
    def ppf[
        T: TensorLike, gpu: Bool = False
    ](q: T, p: Scalar[T.dtype]) raises -> Tensor[T.dtype, T.LayoutType] where (
        is_row_major[T] and T.dtype.is_floating_point()
    ):
        """The quantile over a `Tensor`, parameters as scalars; see the module
        docstring for the two paths `gpu` picks between."""
        comptime dtype = T.dtype
        return _over1[
            step=_bernoulli_ppf_step[dtype, _], gpu=gpu, name="bernoulli.ppf"
        ](q, p)

    @staticmethod
    def rvs[
        dtype: DType, *dims: Int, gpu: Bool = False
    ](
        p: Float64,
        mut rng: Generator,
        ctx: Optional[DeviceContext] = None,
    ) raises -> Static[dtype, *dims]:
        """Random draws at the compile-time shape `dims`, from `rng`'s stream,
        by `Generator.binomial` with one trial. `scipy.stats.bernoulli.rvs`.

        Parameters:
            dtype: The element type of the draws.
            dims: The shape of the result.
            gpu: Whether to draw one thread per element on `ctx`'s device.

        Args:
            p: The success probability.
            rng: The generator; its seed advances.
            ctx: The device to allocate and draw on; `None` means the host.

        Returns:
            A new `Static` tensor of draws.

        Raises:
            When allocation or the fill fails.
        """
        return rng.binomial[dtype, *dims, gpu=gpu](1, p, ctx)

    @staticmethod
    def mean(p: Float64) -> Float64:
        """The mean. `scipy.stats.bernoulli.mean`.

        Args:
            p: The distribution's `p`.

        Returns:
            The mean.
        """
        return p

    @staticmethod
    def var(p: Float64) -> Float64:
        """The variance. `scipy.stats.bernoulli.var`.

        Args:
            p: The distribution's `p`.

        Returns:
            The variance.
        """
        return p * (1.0 - p)

    @staticmethod
    def std(p: Float64) -> Float64:
        """The standard deviation, `sqrt(var)`.

        Args:
            p: The distribution's `p`.

        Returns:
            The standard deviation.
        """
        return _sqrt64(bernoulli.var(p))

    @staticmethod
    def interval(confidence: Float64, p: Float64) -> Tuple[Float64, Float64]:
        """The central interval holding at least `confidence` of the
        probability, `(ppf((1 - c) / 2), ppf((1 + c) / 2))`.
        `scipy.stats.bernoulli.interval`.

        Args:
            confidence: The probability inside the interval, in `[0, 1]`.
            p: The distribution's `p`.

        Returns:
            The interval's lower and upper ends.
        """
        var lo = bernoulli.ppf(_p((1.0 - confidence) / 2.0), _p(p))
        var hi = bernoulli.ppf(_p((1.0 + confidence) / 2.0), _p(p))
        return (_v(lo), _v(hi))

    @staticmethod
    def entropy(p: Float64) -> Float64:
        """The Shannon entropy in nats. `scipy.stats.bernoulli.entropy`.

        Args:
            p: The distribution's `p`.

        Returns:
            The entropy.
        """
        return _bernoulli_entropy(p)


def _bernoulli_pmf_step[
    dtype: DType, w: Int
](x: SIMD[dtype, w], p: SIMD[dtype, 1]) -> SIMD[
    dtype, w
] where dtype.is_floating_point():
    return bernoulli.pmf(
        Plain[dtype, w](x), Plain[dtype, w](SIMD[dtype, w](p[0]))
    ).v


def _bernoulli_cdf_step[
    dtype: DType, w: Int
](x: SIMD[dtype, w], p: SIMD[dtype, 1]) -> SIMD[
    dtype, w
] where dtype.is_floating_point():
    return bernoulli.cdf(
        Plain[dtype, w](x), Plain[dtype, w](SIMD[dtype, w](p[0]))
    ).v


def _bernoulli_ppf_step[
    dtype: DType, w: Int
](x: SIMD[dtype, w], p: SIMD[dtype, 1]) -> SIMD[
    dtype, w
] where dtype.is_floating_point():
    return bernoulli.ppf(
        Plain[dtype, w](x), Plain[dtype, w](SIMD[dtype, w](p[0]))
    ).v


struct geom:
    """The geometric distribution: trials up to and including the first
    success, on `k = 1, 2, ...`. `scipy.stats.geom`."""

    @staticmethod
    def pmf[T: FloatLike](k: T, p: T) -> T:
        """`(1 - p)^(k-1) p` for `k >= 1`."""
        return geom.logpmf(k, p).exp()

    @staticmethod
    def logpmf[T: FloatLike](k: T, p: T) -> T:
        var kc = max_of(k.copy(), T.one())
        var inside = (kc - T.one()) * _log1p(-p.copy()) + _safe_ln(p)
        return blend(ge_indicator(k, T.one()), inside, T.constant(_LOG_ZERO))

    @staticmethod
    def cdf[T: FloatLike](k: T, p: T) -> T:
        """`1 - (1 - p)^floor(k)` for `k >= 1`, read through `expm1` so a
        small `p` keeps its digits."""
        var kc = max_of(k.copy(), T.one()).floor()
        return -_expm1(kc * _log1p(-p)) * ge_indicator(k, T.one())

    @staticmethod
    def sf[T: FloatLike](k: T, p: T) -> T:
        """`(1 - p)^floor(k)` for `k >= 1`, and `1` below."""
        var kc = max_of(k.copy(), T.one()).floor()
        return blend(ge_indicator(k, T.one()), (kc * _log1p(-p)).exp(), T.one())

    @staticmethod
    def logcdf[T: FloatLike](k: T, p: T) -> T:
        return _safe_ln(geom.cdf(k, p))

    @staticmethod
    def logsf[T: FloatLike](k: T, p: T) -> T:
        return _safe_ln(geom.sf(k, p))

    @staticmethod
    def ppf[T: FloatLike](q: T, p: T) -> T:
        """SciPy's: `ceil(log1p(-q) / log1p(-p))`, stepped back by one where
        the CDF one below already reaches `q`."""
        var vals = (_log1p(-q.copy()) / _log1p(-p.copy())).ceil()
        var below = geom.cdf(vals.copy() - T.one(), p.copy())
        var step_back = ge_indicator(below, q) * ge_indicator(
            vals.copy(), T.constant(1.0)
        )
        return max_of(vals - step_back, T.one())

    @staticmethod
    def isf[T: FloatLike](q: T, p: T) -> T:
        """`ppf(1 - q)`."""
        return geom.ppf(T.one() - q, p)

    @staticmethod
    def pmf[
        T: TensorLike, gpu: Bool = False
    ](k: T, p: Scalar[T.dtype]) raises -> Tensor[T.dtype, T.LayoutType] where (
        is_row_major[T] and T.dtype.is_floating_point()
    ):
        """The PMF over a `Tensor`, parameters as scalars; see the module
        docstring for the two paths `gpu` picks between."""
        comptime dtype = T.dtype
        return _over1[step=_geom_pmf_step[dtype, _], gpu=gpu, name="geom.pmf"](
            k, p
        )

    @staticmethod
    def cdf[
        T: TensorLike, gpu: Bool = False
    ](k: T, p: Scalar[T.dtype]) raises -> Tensor[T.dtype, T.LayoutType] where (
        is_row_major[T] and T.dtype.is_floating_point()
    ):
        """The CDF over a `Tensor`, parameters as scalars; see the module
        docstring for the two paths `gpu` picks between."""
        comptime dtype = T.dtype
        return _over1[step=_geom_cdf_step[dtype, _], gpu=gpu, name="geom.cdf"](
            k, p
        )

    @staticmethod
    def ppf[
        T: TensorLike, gpu: Bool = False
    ](q: T, p: Scalar[T.dtype]) raises -> Tensor[T.dtype, T.LayoutType] where (
        is_row_major[T] and T.dtype.is_floating_point()
    ):
        """The quantile over a `Tensor`, parameters as scalars; see the module
        docstring for the two paths `gpu` picks between."""
        comptime dtype = T.dtype
        return _over1[step=_geom_ppf_step[dtype, _], gpu=gpu, name="geom.ppf"](
            q, p
        )

    @staticmethod
    def rvs[
        dtype: DType, *dims: Int, gpu: Bool = False
    ](
        p: Scalar[dtype],
        mut rng: Generator,
        ctx: Optional[DeviceContext] = None,
    ) raises -> Static[dtype, *dims] where dtype.is_floating_point():
        """Random draws at the compile-time shape `dims`, from `rng`'s stream,
        by inverting the closed-form `ppf` over uniforms in `(0, 1)`. `scipy.stats.geom.rvs`.

        Parameters:
            dtype: The element type of the draws.
            dims: The shape of the result.
            gpu: Whether to draw one thread per element on `ctx`'s device.

        Args:
            p: The success probability.
            rng: The generator; its seed advances.
            ctx: The device to allocate and draw on; `None` means the host.

        Returns:
            A new `Static` tensor of draws.

        Raises:
            When allocation or the fill fails.
        """
        comptime count = _product[*dims]()
        var u = rng.uniform[dtype, count, gpu=gpu](
            Scalar[dtype](2.9802322387695312e-08), 1, ctx
        )
        var flat = geom.ppf[gpu=gpu](u, p)
        return Static[dtype, *dims](
            flat._buffer,
            rebind[_LayoutOf[*dims]](row_major[*dims]()),
            flat.host_addressable,
        )

    @staticmethod
    def mean(p: Float64) -> Float64:
        """The mean. `scipy.stats.geom.mean`.

        Args:
            p: The distribution's `p`.

        Returns:
            The mean.
        """
        return 1.0 / p

    @staticmethod
    def var(p: Float64) -> Float64:
        """The variance. `scipy.stats.geom.var`.

        Args:
            p: The distribution's `p`.

        Returns:
            The variance.
        """
        return (1.0 - p) / (p * p)

    @staticmethod
    def std(p: Float64) -> Float64:
        """The standard deviation, `sqrt(var)`.

        Args:
            p: The distribution's `p`.

        Returns:
            The standard deviation.
        """
        return _sqrt64(geom.var(p))

    @staticmethod
    def interval(confidence: Float64, p: Float64) -> Tuple[Float64, Float64]:
        """The central interval holding at least `confidence` of the
        probability, `(ppf((1 - c) / 2), ppf((1 + c) / 2))`.
        `scipy.stats.geom.interval`.

        Args:
            confidence: The probability inside the interval, in `[0, 1]`.
            p: The distribution's `p`.

        Returns:
            The interval's lower and upper ends.
        """
        var lo = geom.ppf(_p((1.0 - confidence) / 2.0), _p(p))
        var hi = geom.ppf(_p((1.0 + confidence) / 2.0), _p(p))
        return (_v(lo), _v(hi))

    @staticmethod
    def entropy(p: Float64) -> Float64:
        """The Shannon entropy in nats. `scipy.stats.geom.entropy`.

        Args:
            p: The distribution's `p`.

        Returns:
            The entropy.
        """
        return (-(1.0 - p) * _log64(1.0 - p) - p * _log64(p)) / p


def _geom_pmf_step[
    dtype: DType, w: Int
](x: SIMD[dtype, w], p: SIMD[dtype, 1]) -> SIMD[
    dtype, w
] where dtype.is_floating_point():
    return geom.pmf(Plain[dtype, w](x), Plain[dtype, w](SIMD[dtype, w](p[0]))).v


def _geom_cdf_step[
    dtype: DType, w: Int
](x: SIMD[dtype, w], p: SIMD[dtype, 1]) -> SIMD[
    dtype, w
] where dtype.is_floating_point():
    return geom.cdf(Plain[dtype, w](x), Plain[dtype, w](SIMD[dtype, w](p[0]))).v


def _geom_ppf_step[
    dtype: DType, w: Int
](x: SIMD[dtype, w], p: SIMD[dtype, 1]) -> SIMD[
    dtype, w
] where dtype.is_floating_point():
    return geom.ppf(Plain[dtype, w](x), Plain[dtype, w](SIMD[dtype, w](p[0]))).v


struct nbinom:
    """The negative binomial distribution: failures before the `n`-th
    success, on `k = 0, 1, ...`; `n` need not be an integer.
    `scipy.stats.nbinom`."""

    @staticmethod
    def pmf[T: FloatLike](k: T, n: T, p: T) -> T:
        """`C(k + n - 1, k) p^n (1 - p)^k`, through `lgamma`."""
        return nbinom.logpmf(k, n, p).exp()

    @staticmethod
    def logpmf[T: FloatLike](k: T, n: T, p: T) -> T:
        var kc = max_of(k.copy(), T.constant(0.0))
        var inside = (
            lgamma(kc.copy() + n.copy())
            - lgamma(kc.copy() + T.one())
            - lgamma(n.copy())
            + n * _safe_ln(p.copy())
            + kc * _log1p(-p)
        )
        return blend(
            ge_indicator(k, T.constant(0.0)), inside, T.constant(_LOG_ZERO)
        )

    @staticmethod
    def cdf[T: FloatLike](k: T, n: T, p: T) -> T:
        """`I_p(n, floor(k) + 1)`, the regularized incomplete beta, which is
        the partial sum exactly."""
        var kc = max_of(k.copy(), T.constant(0.0)).floor()
        return betainc(p.copy(), n.copy(), kc + T.one()) * ge_indicator(
            k, T.constant(0.0)
        )

    @staticmethod
    def sf[T: FloatLike](k: T, n: T, p: T) -> T:
        """`1 - I_p(n, floor(k) + 1)`, read off `betaincc`."""
        var kc = max_of(k.copy(), T.constant(0.0)).floor()
        return blend(
            ge_indicator(k, T.constant(0.0)),
            betaincc(p.copy(), n.copy(), kc + T.one()),
            T.one(),
        )

    @staticmethod
    def logcdf[T: FloatLike](k: T, n: T, p: T) -> T:
        return _safe_ln(nbinom.cdf(k, n, p))

    @staticmethod
    def logsf[T: FloatLike](k: T, n: T, p: T) -> T:
        return _safe_ln(nbinom.sf(k, n, p))

    @staticmethod
    def ppf[T: FloatLike, max_k: Int = 64](q: T, n: T, p: T) -> T:
        """The smallest integer `k` with `cdf(k) >= q`, by the fixed-count
        branchless scan `poisson.ppf` uses, capped at `max_k`."""
        var k = T.constant(0.0)
        for _ in range(max_k):
            var caught_up = ge_indicator(
                nbinom.cdf(k.copy(), n.copy(), p.copy()), q.copy()
            )
            k = k + (T.one() - caught_up)
        return k^

    @staticmethod
    def isf[T: FloatLike, max_k: Int = 64](q: T, n: T, p: T) -> T:
        """`ppf(1 - q)`."""
        return nbinom.ppf[T, max_k](T.one() - q, n, p)

    @staticmethod
    def pmf[
        T: TensorLike, gpu: Bool = False
    ](k: T, n: Scalar[T.dtype], p: Scalar[T.dtype]) raises -> Tensor[
        T.dtype, T.LayoutType
    ] where (is_row_major[T] and T.dtype.is_floating_point()):
        """The PMF over a `Tensor`, parameters as scalars; see the module
        docstring for the two paths `gpu` picks between."""
        comptime dtype = T.dtype
        return _over2[
            step=_nbinom_pmf_step[dtype, _], gpu=gpu, name="nbinom.pmf"
        ](k, n, p)

    @staticmethod
    def cdf[
        T: TensorLike, gpu: Bool = False
    ](k: T, n: Scalar[T.dtype], p: Scalar[T.dtype]) raises -> Tensor[
        T.dtype, T.LayoutType
    ] where (is_row_major[T] and T.dtype.is_floating_point()):
        """The CDF over a `Tensor`, parameters as scalars; see the module
        docstring for the two paths `gpu` picks between."""
        comptime dtype = T.dtype
        return _over2[
            step=_nbinom_cdf_step[dtype, _], gpu=gpu, name="nbinom.cdf"
        ](k, n, p)

    @staticmethod
    def ppf[
        T: TensorLike, gpu: Bool = False
    ](q: T, n: Scalar[T.dtype], p: Scalar[T.dtype]) raises -> Tensor[
        T.dtype, T.LayoutType
    ] where (is_row_major[T] and T.dtype.is_floating_point()):
        """The quantile over a `Tensor`, parameters as scalars; see the module
        docstring for the two paths `gpu` picks between."""
        comptime dtype = T.dtype
        return _over2[
            step=_nbinom_ppf_step[dtype, _], gpu=gpu, name="nbinom.ppf"
        ](q, n, p)

    @staticmethod
    def rvs[
        dtype: DType, *dims: Int, gpu: Bool = False
    ](
        n: Float64,
        p: Float64,
        mut rng: Generator,
        ctx: Optional[DeviceContext] = None,
    ) raises -> Static[dtype, *dims]:
        """Random draws at the compile-time shape `dims`, from `rng`'s stream,
        by `Generator.negative_binomial`, the gamma-Poisson mixture. `scipy.stats.nbinom.rvs`.

        Parameters:
            dtype: The element type of the draws.
            dims: The shape of the result.
            gpu: Whether to draw one thread per element on `ctx`'s device.

        Args:
            n: The number of successes.
            p: The success probability.
            rng: The generator; its seed advances.
            ctx: The device to allocate and draw on; `None` means the host.

        Returns:
            A new `Static` tensor of draws.

        Raises:
            When allocation or the fill fails.
        """
        return rng.negative_binomial[dtype, *dims, gpu=gpu](n, p, ctx)

    @staticmethod
    def mean(n: Float64, p: Float64) -> Float64:
        """The mean. `scipy.stats.nbinom.mean`.

        Args:
            n: The distribution's `n`.
            p: The distribution's `p`.

        Returns:
            The mean.
        """
        return n * (1.0 - p) / p

    @staticmethod
    def var(n: Float64, p: Float64) -> Float64:
        """The variance. `scipy.stats.nbinom.var`.

        Args:
            n: The distribution's `n`.
            p: The distribution's `p`.

        Returns:
            The variance.
        """
        return n * (1.0 - p) / (p * p)

    @staticmethod
    def std(n: Float64, p: Float64) -> Float64:
        """The standard deviation, `sqrt(var)`.

        Args:
            n: The distribution's `n`.
            p: The distribution's `p`.

        Returns:
            The standard deviation.
        """
        return _sqrt64(nbinom.var(n, p))

    @staticmethod
    def interval(
        confidence: Float64, n: Float64, p: Float64
    ) -> Tuple[Float64, Float64]:
        """The central interval holding at least `confidence` of the
        probability, `(ppf((1 - c) / 2), ppf((1 + c) / 2))`.
        `scipy.stats.nbinom.interval`.

        Args:
            confidence: The probability inside the interval, in `[0, 1]`.
            n: The distribution's `n`.
            p: The distribution's `p`.

        Returns:
            The interval's lower and upper ends.
        """
        var lo = nbinom.ppf(_p((1.0 - confidence) / 2.0), _p(n), _p(p))
        var hi = nbinom.ppf(_p((1.0 + confidence) / 2.0), _p(n), _p(p))
        return (_v(lo), _v(hi))

    @staticmethod
    def entropy(n: Float64, p: Float64) -> Float64:
        """The Shannon entropy in nats. `scipy.stats.nbinom.entropy`.

        Args:
            n: The distribution's `n`.
            p: The distribution's `p`.

        Returns:
            The entropy.
        """
        return _nbinom_entropy(n, p)


def _nbinom_pmf_step[
    dtype: DType, w: Int
](x: SIMD[dtype, w], n: SIMD[dtype, 1], p: SIMD[dtype, 1]) -> SIMD[
    dtype, w
] where dtype.is_floating_point():
    return nbinom.pmf(
        Plain[dtype, w](x),
        Plain[dtype, w](SIMD[dtype, w](n[0])),
        Plain[dtype, w](SIMD[dtype, w](p[0])),
    ).v


def _nbinom_cdf_step[
    dtype: DType, w: Int
](x: SIMD[dtype, w], n: SIMD[dtype, 1], p: SIMD[dtype, 1]) -> SIMD[
    dtype, w
] where dtype.is_floating_point():
    return nbinom.cdf(
        Plain[dtype, w](x),
        Plain[dtype, w](SIMD[dtype, w](n[0])),
        Plain[dtype, w](SIMD[dtype, w](p[0])),
    ).v


def _nbinom_ppf_step[
    dtype: DType, w: Int
](x: SIMD[dtype, w], n: SIMD[dtype, 1], p: SIMD[dtype, 1]) -> SIMD[
    dtype, w
] where dtype.is_floating_point():
    return nbinom.ppf(
        Plain[dtype, w](x),
        Plain[dtype, w](SIMD[dtype, w](n[0])),
        Plain[dtype, w](SIMD[dtype, w](p[0])),
    ).v


struct hypergeom:
    """The hypergeometric distribution: good balls among `N` drawn without
    replacement from `M` balls of which `n` are good. `scipy.stats.hypergeom`,
    in SciPy's `(M, n, N)` order."""

    @staticmethod
    def pmf[T: FloatLike](k: T, M: T, n: T, N: T) -> T:
        """`C(n, k) C(M - n, N - k) / C(M, N)`, through `lgamma`."""
        return hypergeom.logpmf(k, M, n, N).exp()

    @staticmethod
    def logpmf[T: FloatLike](k: T, M: T, n: T, N: T) -> T:
        """`_LOG_ZERO` outside `max(0, N - (M - n)) <= k <= min(n, N)`; the
        interior's `lgamma` arguments are clamped so the discarded side stays
        finite."""
        var zero = T.constant(0.0)
        var inside = (
            ge_indicator(k.copy(), zero.copy())
            * ge_indicator(n.copy() - k.copy(), zero.copy())
            * ge_indicator(N.copy() - k.copy(), zero.copy())
            * ge_indicator(
                M.copy() - n.copy() - N.copy() + k.copy(), zero.copy()
            )
        )
        var kc = max_of(
            min_of(k.copy(), min_of(n.copy(), N.copy())), zero.copy()
        )
        var good = _lbinom(n.copy(), kc.copy())
        var bad = _lbinom(
            M.copy() - n.copy(), max_of(N.copy() - kc, zero.copy())
        )
        var total = _lbinom(M, N)
        return blend(inside, good + bad - total, T.constant(_LOG_ZERO))

    @staticmethod
    def cdf[T: FloatLike, max_k: Int = 64](k: T, M: T, n: T, N: T) -> T:
        """`sum_{j <= k} pmf(j)` over `j = 0 .. max_k - 1`: a fixed-count sum,
        so the support must lie below `max_k` (raise it for larger draws)."""
        var total = T.constant(0.0)
        var jf = T.constant(0.0)
        for _ in range(max_k):
            var term = hypergeom.pmf(jf.copy(), M.copy(), n.copy(), N.copy())
            total = total + term * ge_indicator(k.copy(), jf.copy())
            jf = jf + T.one()
        return min_of(total, T.one())

    @staticmethod
    def sf[T: FloatLike, max_k: Int = 64](k: T, M: T, n: T, N: T) -> T:
        """`sum_{j > k} pmf(j)`, the upper tail summed directly."""
        var total = T.constant(0.0)
        var jf = T.constant(0.0)
        for _ in range(max_k):
            var term = hypergeom.pmf(jf.copy(), M.copy(), n.copy(), N.copy())
            total = total + term * (T.one() - ge_indicator(k.copy(), jf.copy()))
            jf = jf + T.one()
        return min_of(total, T.one())

    @staticmethod
    def logcdf[T: FloatLike](k: T, M: T, n: T, N: T) -> T:
        return _safe_ln(hypergeom.cdf(k, M, n, N))

    @staticmethod
    def logsf[T: FloatLike](k: T, M: T, n: T, N: T) -> T:
        return _safe_ln(hypergeom.sf(k, M, n, N))

    @staticmethod
    def ppf[T: FloatLike, max_k: Int = 64](q: T, M: T, n: T, N: T) -> T:
        """The smallest integer `k` with `cdf(k) >= q`, by the running sum of
        the PMF, a fixed `max_k` terms."""
        var total = T.constant(0.0)
        var k = T.constant(0.0)
        var jf = T.constant(0.0)
        for _ in range(max_k):
            total = total + hypergeom.pmf(
                jf.copy(), M.copy(), n.copy(), N.copy()
            )
            k = k + (
                T.one()
                - ge_indicator(total.copy(), q.copy() - T.constant(1e-12))
            )
            jf = jf + T.one()
        return k^

    @staticmethod
    def isf[T: FloatLike, max_k: Int = 64](q: T, M: T, n: T, N: T) -> T:
        """`ppf(1 - q)`."""
        return hypergeom.ppf[T, max_k](T.one() - q, M, n, N)

    @staticmethod
    def pmf[
        T: TensorLike, gpu: Bool = False
    ](
        k: T, M: Scalar[T.dtype], n: Scalar[T.dtype], N: Scalar[T.dtype]
    ) raises -> Tensor[T.dtype, T.LayoutType] where (
        is_row_major[T] and T.dtype.is_floating_point()
    ):
        """The PMF over a `Tensor`, parameters as scalars; see the module
        docstring for the two paths `gpu` picks between."""
        comptime dtype = T.dtype
        return _over3[
            step=_hypergeom_pmf_step[dtype, _], gpu=gpu, name="hypergeom.pmf"
        ](k, M, n, N)

    @staticmethod
    def cdf[
        T: TensorLike, gpu: Bool = False
    ](
        k: T, M: Scalar[T.dtype], n: Scalar[T.dtype], N: Scalar[T.dtype]
    ) raises -> Tensor[T.dtype, T.LayoutType] where (
        is_row_major[T] and T.dtype.is_floating_point()
    ):
        """The CDF over a `Tensor`, parameters as scalars; see the module
        docstring for the two paths `gpu` picks between."""
        comptime dtype = T.dtype
        return _over3[
            step=_hypergeom_cdf_step[dtype, _], gpu=gpu, name="hypergeom.cdf"
        ](k, M, n, N)

    @staticmethod
    def ppf[
        T: TensorLike, gpu: Bool = False
    ](
        q: T, M: Scalar[T.dtype], n: Scalar[T.dtype], N: Scalar[T.dtype]
    ) raises -> Tensor[T.dtype, T.LayoutType] where (
        is_row_major[T] and T.dtype.is_floating_point()
    ):
        """The quantile over a `Tensor`, parameters as scalars; see the module
        docstring for the two paths `gpu` picks between."""
        comptime dtype = T.dtype
        return _over3[
            step=_hypergeom_ppf_step[dtype, _], gpu=gpu, name="hypergeom.ppf"
        ](q, M, n, N)

    @staticmethod
    def rvs[
        dtype: DType, *dims: Int, gpu: Bool = False
    ](
        M: Int,
        n: Int,
        N: Int,
        mut rng: Generator,
        ctx: Optional[DeviceContext] = None,
    ) raises -> Static[dtype, *dims]:
        """Random draws at the compile-time shape `dims`, from `rng`'s stream,
        by `Generator.hypergeometric`, the urn simulated. `scipy.stats.hypergeom.rvs`.

        Parameters:
            dtype: The element type of the draws.
            dims: The shape of the result.
            gpu: Whether to draw one thread per element on `ctx`'s device.

        Args:
            M: The population size.
            n: The good balls in it.
            N: The balls drawn.
            rng: The generator; its seed advances.
            ctx: The device to allocate and draw on; `None` means the host.

        Returns:
            A new `Static` tensor of draws.

        Raises:
            When allocation or the fill fails.
        """
        return rng.hypergeometric[dtype, *dims, gpu=gpu](n, M - n, N, ctx)

    @staticmethod
    def mean(M: Float64, n: Float64, N: Float64) -> Float64:
        """The mean. `scipy.stats.hypergeom.mean`.

        Args:
            M: The distribution's `M`.
            n: The distribution's `n`.
            N: The distribution's `N`.

        Returns:
            The mean.
        """
        return N * n / M

    @staticmethod
    def var(M: Float64, n: Float64, N: Float64) -> Float64:
        """The variance. `scipy.stats.hypergeom.var`.

        Args:
            M: The distribution's `M`.
            n: The distribution's `n`.
            N: The distribution's `N`.

        Returns:
            The variance.
        """
        return N * n * (M - n) * (M - N) / (M * M * (M - 1.0))

    @staticmethod
    def std(M: Float64, n: Float64, N: Float64) -> Float64:
        """The standard deviation, `sqrt(var)`.

        Args:
            M: The distribution's `M`.
            n: The distribution's `n`.
            N: The distribution's `N`.

        Returns:
            The standard deviation.
        """
        return _sqrt64(hypergeom.var(M, n, N))

    @staticmethod
    def interval(
        confidence: Float64, M: Float64, n: Float64, N: Float64
    ) -> Tuple[Float64, Float64]:
        """The central interval holding at least `confidence` of the
        probability, `(ppf((1 - c) / 2), ppf((1 + c) / 2))`.
        `scipy.stats.hypergeom.interval`.

        Args:
            confidence: The probability inside the interval, in `[0, 1]`.
            M: The distribution's `M`.
            n: The distribution's `n`.
            N: The distribution's `N`.

        Returns:
            The interval's lower and upper ends.
        """
        var lo = hypergeom.ppf(
            _p((1.0 - confidence) / 2.0), _p(M), _p(n), _p(N)
        )
        var hi = hypergeom.ppf(
            _p((1.0 + confidence) / 2.0), _p(M), _p(n), _p(N)
        )
        return (_v(lo), _v(hi))

    @staticmethod
    def entropy(M: Float64, n: Float64, N: Float64) -> Float64:
        """The Shannon entropy in nats. `scipy.stats.hypergeom.entropy`.

        Args:
            M: The distribution's `M`.
            n: The distribution's `n`.
            N: The distribution's `N`.

        Returns:
            The entropy.
        """
        return _hypergeom_entropy(M, n, N)


def _hypergeom_pmf_step[
    dtype: DType, w: Int
](
    x: SIMD[dtype, w], M: SIMD[dtype, 1], n: SIMD[dtype, 1], N: SIMD[dtype, 1]
) -> SIMD[dtype, w] where dtype.is_floating_point():
    return hypergeom.pmf(
        Plain[dtype, w](x),
        Plain[dtype, w](SIMD[dtype, w](M[0])),
        Plain[dtype, w](SIMD[dtype, w](n[0])),
        Plain[dtype, w](SIMD[dtype, w](N[0])),
    ).v


def _hypergeom_cdf_step[
    dtype: DType, w: Int
](
    x: SIMD[dtype, w], M: SIMD[dtype, 1], n: SIMD[dtype, 1], N: SIMD[dtype, 1]
) -> SIMD[dtype, w] where dtype.is_floating_point():
    return hypergeom.cdf(
        Plain[dtype, w](x),
        Plain[dtype, w](SIMD[dtype, w](M[0])),
        Plain[dtype, w](SIMD[dtype, w](n[0])),
        Plain[dtype, w](SIMD[dtype, w](N[0])),
    ).v


def _hypergeom_ppf_step[
    dtype: DType, w: Int
](
    x: SIMD[dtype, w], M: SIMD[dtype, 1], n: SIMD[dtype, 1], N: SIMD[dtype, 1]
) -> SIMD[dtype, w] where dtype.is_floating_point():
    return hypergeom.ppf(
        Plain[dtype, w](x),
        Plain[dtype, w](SIMD[dtype, w](M[0])),
        Plain[dtype, w](SIMD[dtype, w](n[0])),
        Plain[dtype, w](SIMD[dtype, w](N[0])),
    ).v
