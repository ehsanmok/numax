"""`scipy.stats` and the NumPy statistics, over a `Tensor`.

The call this file exists for is the first line of `main`:

```mojo
var p = norm.cdf(xs, 0.0, 1.0)      # xs is a Tensor, p is a Tensor
```

`scipy.stats.norm.cdf(x)` on an array is the most ordinary thing a SciPy
user does, and until 0.2 it could not be written here at all: the
distributions were `FloatLike` kernels over scalars, and a `Tensor` had no
overload. Both now exist in the same namespace, so the scalar call and the
whole-tensor call are the same name -- and the tensor one is one
`elementwise` launch through the tier-1 `erf`, which `gpu=True` runs on a
device with no host round trip.

The rest is the NumPy and SciPy surface that sits on top of a sample:
quantiles, a histogram, the correlation family, a regression and a t-test.
Most of it is host-side by declaration rather than by accident -- a
histogram needs a scatter-add MAX does not ship, and a quantile needs a
sort -- and `docs/parity.md` records which is which, with
`docs/performance.md` measuring what that costs.

Run: `pixi run example-stats-surface`
"""

from std.math import sin

from max.gpu.host import DeviceContext

from numax import Plain
from numax.core.array import Static
from numax.stats import (
    corrcoef,
    cov,
    describe,
    histogram,
    linregress,
    norm,
    pearsonr,
    percentile,
    quantile,
    ttest_ind,
    zscore,
)

comptime dtype = DType.float64
comptime P = Plain[dtype]
comptime n = 256


def sample(ctx: DeviceContext, shift: Float64) raises -> Static[dtype, n]:
    """A deterministic sample on roughly `[-3, 3)`, offset by `shift`. A
    hash rather than an RNG so the printed numbers are stable across runs
    and machines -- `numax.stats.random` is what a real program reaches
    for, and `examples/intermediate/random_ensemble.mojo` shows it."""
    var values = List[Scalar[dtype]](capacity=n)
    for i in range(n):
        var h = (i * 2654435761 + 12345) % 16777216
        values.append(
            Scalar[dtype](6.0 * (Float64(h) / 16777216.0 - 0.5) + shift)
        )
    return Static[dtype, n](ctx, values^)


def main() raises:
    var ctx = DeviceContext(api="cpu")

    print("A distribution over a whole tensor")
    print("----------------------------------")

    var grid = Static[dtype, 5](ctx, [-2.0, -1.0, 0.0, 1.0, 2.0])
    var probabilities = norm.cdf(grid, Scalar[dtype](0), Scalar[dtype](1))
    var p = probabilities.to_host()
    print("  norm.cdf([-2, -1, 0, 1, 2], 0, 1)")
    print("   ", p[0], p[1], p[2], p[3], p[4])
    print("    one elementwise launch; `gpu=True` runs it on a device")

    # The same namespace answers a scalar, where the argument is a
    # conformer rather than a `Float64` -- so the identical call at `Dual`
    # differentiates. Eight methods each: pdf, logpdf, cdf, logcdf, sf,
    # logsf, ppf, isf.
    print(
        "  norm.ppf(0.975)  =",
        norm.ppf(P.constant(0.975), P.constant(0.0), P.one()).v,
    )
    print(
        "  norm.sf(1.96)    =",
        norm.sf(P.constant(1.96), P.constant(0.0), P.one()).v,
    )

    print()
    print("Quantiles and binning")
    print("---------------------")

    var xs = sample(ctx, 0.0)
    print("  median  =", quantile(xs, 0.5))
    print("  p05, p95 =", percentile(xs, 5.0), percentile(xs, 95.0))
    print("    thirteen NumPy `method=` interpolations; 'linear' by default")

    var binned = histogram[bins=8](xs, -3.0, 3.0)
    var counts = binned.counts.to_host()
    var edges = binned.edges.to_host()
    print("  histogram(xs, bins=8, range=(-3, 3))")
    for b in range(8):
        print("    [", edges[b], ",", edges[b + 1], ") ", counts[b])

    print()
    print("Correlation, regression, and a test")
    print("-----------------------------------")

    # A second sample that is the first plus a trend, so the correlation
    # and the regression have something to find.
    var base = sample(ctx, 0.0)
    var host = base.to_host()
    var response_values = List[Scalar[dtype]](capacity=n)
    for i in range(n):
        response_values.append(
            Scalar[dtype](
                2.5 * Float64(host[i]) + 1.0 + 0.3 * sin(Float64(i) * 0.7)
            )
        )
    var response = Static[dtype, n](ctx, response_values^)

    var predictor = sample(ctx, 0.0)
    var paired = corrcoef(predictor, response).to_host()
    print("  corrcoef(x, y)[0, 1] =", paired[1])

    var predictor2 = sample(ctx, 0.0)
    var response2 = Static[dtype, n](ctx, response.to_host())
    var covariance = cov(predictor2, response2).to_host()
    print("  cov(x, y)[0, 1]      =", covariance[1])

    var predictor3 = sample(ctx, 0.0)
    var response3 = Static[dtype, n](ctx, response.to_host())
    var correlation = pearsonr(predictor3, response3)
    print(
        "  pearsonr(x, y)       r =",
        correlation.statistic,
        " p =",
        correlation.pvalue,
    )

    var predictor4 = sample(ctx, 0.0)
    var response4 = Static[dtype, n](ctx, response.to_host())
    var fit = linregress(predictor4, response4)
    print(
        "  linregress(x, y)     slope =",
        fit.slope,
        " intercept =",
        fit.intercept,
    )
    print("    the trend put in was 2.5 x + 1")

    # Two samples that differ only by a shift of 0.8, which at this size is
    # well inside what the test resolves.
    var left = sample(ctx, 0.0)
    var right = sample(ctx, 0.8)
    var test = ttest_ind(left, right)
    print(
        "  ttest_ind(a, b)      t =",
        test.statistic,
        " p =",
        test.pvalue,
    )
    print("    p below 0.05, so the shift is detected")

    var summary = describe(sample(ctx, 0.0))
    print(
        "  describe(xs)         n =",
        summary.nobs,
        " mean =",
        summary.mean,
        " var =",
        summary.variance,
    )

    var standardized = zscore(sample(ctx, 0.0))
    var z = standardized.to_host()
    print("  zscore(xs)[0:3]      =", z[0], z[1], z[2])
