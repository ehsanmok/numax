"""Tests for the `Tensor` overloads of `numax.stats`'s distributions.

The claim is one definition, two spellings: every `Tensor` method is the
`FloatLike` kernel of the same name evaluated per lane. So each check
compares the tensor result element by element against the scalar call on
that element -- a divergence between the two is the failure worth catching,
not a wrong number in isolation, which `test_distributions.mojo` already
covers for the scalar kernels.
"""

from std.testing import TestSuite, assert_almost_equal, assert_equal

from max.gpu.host import DeviceContext

from numax import Plain
from numax.core.array import Static, linspace, zeros
from numax.core.ops import add
from numax.stats import beta, binom, chi2, expon, f, gamma, norm, poisson, t

comptime dtype = DType.float64
comptime P = Plain[dtype, 1]


def _ramp[n: Int](lo: Float64, hi: Float64) raises -> Static[dtype, n]:
    return linspace[n](lo, hi, ctx=DeviceContext(api="cpu"))


def test_norm_cdf_over_a_tensor_is_the_scalar_kernel_per_element() raises:
    comptime n = 9
    var xs = _ramp[n](-3.0, 3.0)
    var got = norm.cdf(xs, Scalar[dtype](0.5), Scalar[dtype](1.3)).to_host()
    var source = xs.to_host()
    for i in range(n):
        var want = norm.cdf(P(source[i]), P(0.5), P(1.3)).v[0]
        assert_almost_equal(got[i], want, atol=1e-15)


def test_norm_pdf_and_ppf_over_a_tensor_match_the_scalar_kernels() raises:
    comptime n = 7
    var xs = _ramp[n](-2.0, 2.0)
    var ps = _ramp[n](0.05, 0.95)

    var density = norm.pdf(xs, Scalar[dtype](0.0), Scalar[dtype](2.0)).to_host()
    var quantile = norm.ppf(
        ps, Scalar[dtype](0.0), Scalar[dtype](2.0)
    ).to_host()
    var x_host = xs.to_host()
    var p_host = ps.to_host()
    for i in range(n):
        assert_almost_equal(
            density[i], norm.pdf(P(x_host[i]), P(0.0), P(2.0)).v[0], atol=1e-15
        )
        assert_almost_equal(
            quantile[i], norm.ppf(P(p_host[i]), P(0.0), P(2.0)).v[0], atol=1e-13
        )


def test_every_one_parameter_family_matches_its_scalar_kernel() raises:
    # expon, chi2, t, poisson: one parameter each, so the one-scalar driver.
    comptime n = 6
    var xs = _ramp[n](0.5, 5.5)
    var x_host = xs.to_host()

    var e = expon.cdf(xs, Scalar[dtype](0.7)).to_host()
    var c = chi2.pdf(xs, Scalar[dtype](3.0)).to_host()
    var s_ = t.cdf(xs, Scalar[dtype](10.0)).to_host()
    var po = poisson.pmf(xs, Scalar[dtype](2.5)).to_host()
    for i in range(n):
        assert_almost_equal(
            e[i], expon.cdf(P(x_host[i]), P(0.7)).v[0], atol=1e-15
        )
        assert_almost_equal(
            c[i], chi2.pdf(P(x_host[i]), P(3.0)).v[0], atol=1e-15
        )
        assert_almost_equal(
            s_[i], t.cdf(P(x_host[i]), P(10.0)).v[0], atol=1e-15
        )
        assert_almost_equal(
            po[i], poisson.pmf(P(x_host[i]), P(2.5)).v[0], atol=1e-15
        )


def test_every_two_parameter_family_matches_its_scalar_kernel() raises:
    # gamma, beta, f, binom: two parameters each, so the two-scalar driver.
    comptime n = 6
    var xs = _ramp[n](0.1, 0.9)
    var ks = _ramp[n](0.0, 10.0)
    var x_host = xs.to_host()
    var k_host = ks.to_host()

    var g = gamma.cdf(xs, Scalar[dtype](3.0), Scalar[dtype](1.5)).to_host()
    var b = beta.pdf(xs, Scalar[dtype](2.0), Scalar[dtype](3.0)).to_host()
    var ff = f.cdf(xs, Scalar[dtype](5.0), Scalar[dtype](10.0)).to_host()
    var bi = binom.cdf(ks, Scalar[dtype](10.0), Scalar[dtype](0.3)).to_host()
    for i in range(n):
        assert_almost_equal(
            g[i], gamma.cdf(P(x_host[i]), P(3.0), P(1.5)).v[0], atol=1e-15
        )
        assert_almost_equal(
            b[i], beta.pdf(P(x_host[i]), P(2.0), P(3.0)).v[0], atol=1e-15
        )
        assert_almost_equal(
            ff[i], f.cdf(P(x_host[i]), P(5.0), P(10.0)).v[0], atol=1e-15
        )
        assert_almost_equal(
            bi[i], binom.cdf(P(k_host[i]), P(10.0), P(0.3)).v[0], atol=1e-15
        )


def test_quantiles_over_a_tensor_invert_cdfs_over_a_tensor() raises:
    # Round trip through both tensor spellings, which is the property a
    # caller relies on when p-values and critical values come from a batch.
    comptime n = 5
    var ps = _ramp[n](0.1, 0.9)
    var shape = Scalar[dtype](3.0)
    var scale = Scalar[dtype](2.0)

    var xs = gamma.ppf(ps, shape, scale)
    var back = gamma.cdf(xs, shape, scale).to_host()
    var p_host = ps.to_host()
    for i in range(n):
        assert_almost_equal(back[i], p_host[i], atol=1e-9)


def test_a_run_time_shaped_tensor_takes_the_host_walk() raises:
    # A broadcast result is a `Dynamic`, whose extent is not in the type; the
    # distribution still has to accept it, through the host path.
    var ctx = DeviceContext(api="cpu")
    var a = zeros[dtype, 2, 3](ctx)
    var row = zeros[dtype, 3](ctx)
    for i in range(6):
        a[i] = Scalar[dtype](i) * 0.5 - 1.0
    for i in range(3):
        row[i] = Scalar[dtype](i) * 0.1
    var dyn = add(a, row)

    var got = norm.cdf(dyn, Scalar[dtype](0.0), Scalar[dtype](1.0))
    assert_equal(got.dim_at(0), 2)
    assert_equal(got.dim_at(1), 3)
    var out = got.to_host()
    var source = dyn.to_host()
    for i in range(6):
        assert_almost_equal(
            out[i], norm.cdf(P(source[i]), P(0.0), P(1.0)).v[0], atol=1e-15
        )


def test_the_tensor_form_covers_a_length_that_does_not_divide_the_width() raises:
    # 13 elements: the SIMD bulk plus a scalar tail, both carrying the
    # parameters -- dropping them in the tail is the regression to catch.
    comptime n = 13
    var xs = _ramp[n](-2.0, 2.0)
    var got = norm.cdf(xs, Scalar[dtype](0.3), Scalar[dtype](0.8)).to_host()
    var source = xs.to_host()
    for i in range(n):
        assert_almost_equal(
            got[i], norm.cdf(P(source[i]), P(0.3), P(0.8)).v[0], atol=1e-15
        )


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
