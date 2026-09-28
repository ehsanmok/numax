"""Tests for the continuous distribution namespaces added in 0.3:
`uniform_dist`, `lognorm`, `weibull_min`, `cauchy`, `laplace`, `rayleigh`,
`logistic` and `pareto`.

Every method -- `pdf`, `logpdf`, `cdf`, `sf`, `ppf`, `isf`, `mean`,
`var`, `entropy`, `interval` -- against SciPy's frozen distribution at
points across each support, both tails included; the `Tensor` overloads
against the `FloatLike` ones; and `rvs` by the moments of its draws.
`cauchy`'s moments are NaN, as SciPy's are.
"""

from std.math import sqrt
from std.testing import (
    TestSuite,
    assert_almost_equal,
    assert_equal,
    assert_true,
)

from max.gpu.host import DeviceContext

from numax.core.plain import Plain
from numax.core.tensor import Static
from numax.stats import (
    Generator,
    cauchy,
    laplace,
    logistic,
    lognorm,
    pareto,
    rayleigh,
    uniform_dist,
    weibull_min,
)

comptime _P = Plain[DType.float64, 1]
comptime f64 = DType.float64


def _v(x: _P) -> Float64:
    return Float64(x.v[0])


def _near(got: Float64, want: Float64) raises:
    assert_almost_equal(got, want, atol=1e-12, rtol=1e-11)


def test_continuous_families_match_scipy() raises:
    _near(_v(uniform_dist.pdf(_P(1.5), _P(1.0), _P(3.0))), 0.3333333333333333)
    _near(
        _v(uniform_dist.logpdf(_P(1.5), _P(1.0), _P(3.0))), -1.0986122886681098
    )
    _near(_v(uniform_dist.cdf(_P(1.5), _P(1.0), _P(3.0))), 0.16666666666666666)
    _near(_v(uniform_dist.sf(_P(1.5), _P(1.0), _P(3.0))), 0.8333333333333334)
    _near(_v(uniform_dist.pdf(_P(2.0), _P(1.0), _P(3.0))), 0.3333333333333333)
    _near(
        _v(uniform_dist.logpdf(_P(2.0), _P(1.0), _P(3.0))), -1.0986122886681098
    )
    _near(_v(uniform_dist.cdf(_P(2.0), _P(1.0), _P(3.0))), 0.3333333333333333)
    _near(_v(uniform_dist.sf(_P(2.0), _P(1.0), _P(3.0))), 0.6666666666666667)
    _near(_v(uniform_dist.pdf(_P(3.7), _P(1.0), _P(3.0))), 0.3333333333333333)
    _near(
        _v(uniform_dist.logpdf(_P(3.7), _P(1.0), _P(3.0))), -1.0986122886681098
    )
    _near(_v(uniform_dist.cdf(_P(3.7), _P(1.0), _P(3.0))), 0.9)
    _near(_v(uniform_dist.sf(_P(3.7), _P(1.0), _P(3.0))), 0.09999999999999998)
    _near(_v(uniform_dist.ppf(_P(0.05), _P(1.0), _P(3.0))), 1.15)
    _near(_v(uniform_dist.isf(_P(0.05), _P(1.0), _P(3.0))), 3.8499999999999996)
    _near(_v(uniform_dist.ppf(_P(0.5), _P(1.0), _P(3.0))), 2.5)
    _near(_v(uniform_dist.isf(_P(0.5), _P(1.0), _P(3.0))), 2.5)
    _near(_v(uniform_dist.ppf(_P(0.93), _P(1.0), _P(3.0))), 3.79)
    _near(_v(uniform_dist.isf(_P(0.93), _P(1.0), _P(3.0))), 1.21)
    _near(uniform_dist.mean(1.0, 3.0), 2.5)
    _near(uniform_dist.var(1.0, 3.0), 0.75)
    _near(uniform_dist.entropy(1.0, 3.0), 1.0986122886681098)
    var i_uniform_dist = uniform_dist.interval(0.8, 1.0, 3.0)
    _near(i_uniform_dist[0], 1.2999999999999998)
    _near(i_uniform_dist[1], 3.7)

    _near(_v(lognorm.pdf(_P(0.5), _P(0.6), _P(2.0))), 0.09216736779521725)
    _near(_v(lognorm.logpdf(_P(0.5), _P(0.6), _P(2.0))), -2.384149139535411)
    _near(_v(lognorm.cdf(_P(0.5), _P(0.6), _P(2.0))), 0.010430504126476363)
    _near(_v(lognorm.sf(_P(0.5), _P(0.6), _P(2.0))), 0.9895694958735236)
    _near(_v(lognorm.pdf(_P(2.0), _P(0.6), _P(2.0))), 0.33245190033452726)
    _near(_v(lognorm.logpdf(_P(2.0), _P(0.6), _P(2.0))), -1.1012600899986271)
    _near(_v(lognorm.cdf(_P(2.0), _P(0.6), _P(2.0))), 0.5)
    _near(_v(lognorm.sf(_P(2.0), _P(0.6), _P(2.0))), 0.5)
    _near(_v(lognorm.pdf(_P(4.5), _P(0.6), _P(2.0))), 0.05927696514821828)
    _near(_v(lognorm.logpdf(_P(4.5), _P(0.6), _P(2.0))), -2.8255344945103196)
    _near(_v(lognorm.cdf(_P(4.5), _P(0.6), _P(2.0))), 0.9117404003852039)
    _near(_v(lognorm.sf(_P(4.5), _P(0.6), _P(2.0))), 0.08825959961479618)
    _near(_v(lognorm.ppf(_P(0.05), _P(0.6), _P(2.0))), 0.7454516552711703)
    _near(_v(lognorm.isf(_P(0.05), _P(0.6), _P(2.0))), 5.365874462435708)
    _near(_v(lognorm.ppf(_P(0.5), _P(0.6), _P(2.0))), 2.0)
    _near(_v(lognorm.isf(_P(0.5), _P(0.6), _P(2.0))), 2.0)
    _near(_v(lognorm.ppf(_P(0.93), _P(0.6), _P(2.0))), 4.848269307579481)
    _near(_v(lognorm.isf(_P(0.93), _P(0.6), _P(2.0))), 0.8250366772626782)
    _near(lognorm.mean(0.6, 2.0), 2.3944347262436203)
    _near(lognorm.var(0.6, 2.0), 2.484415184334189)
    _near(lognorm.entropy(0.6, 2.0), 1.6012600899986271)
    var i_lognorm = lognorm.interval(0.8, 0.6, 2.0)
    _near(i_lognorm[0], 0.927016644109475)
    _near(i_lognorm[1], 4.314917132736642)

    _near(_v(weibull_min.pdf(_P(0.4), _P(1.8), _P(2.5))), 0.16017275050130067)
    _near(_v(weibull_min.logpdf(_P(0.4), _P(1.8), _P(2.5))), -1.831502355561909)
    _near(_v(weibull_min.cdf(_P(0.4), _P(1.8), _P(2.5))), 0.03625940951430173)
    _near(_v(weibull_min.sf(_P(0.4), _P(1.8), _P(2.5))), 0.9637405904856983)
    _near(_v(weibull_min.pdf(_P(2.0), _P(1.8), _P(2.5))), 0.30843991435719154)
    _near(
        _v(weibull_min.logpdf(_P(2.0), _P(1.8), _P(2.5))), -1.1762282216818187
    )
    _near(_v(weibull_min.cdf(_P(2.0), _P(1.8), _P(2.5))), 0.4878866612317274)
    _near(_v(weibull_min.sf(_P(2.0), _P(1.8), _P(2.5))), 0.5121133387682726)
    _near(_v(weibull_min.pdf(_P(5.0), _P(1.8), _P(2.5))), 0.03853499167812118)
    _near(
        _v(weibull_min.logpdf(_P(5.0), _P(1.8), _P(2.5))), -3.2561885757085762
    )
    _near(_v(weibull_min.cdf(_P(5.0), _P(1.8), _P(2.5))), 0.9692603600343873)
    _near(_v(weibull_min.sf(_P(5.0), _P(1.8), _P(2.5))), 0.030739639965612665)
    _near(_v(weibull_min.ppf(_P(0.05), _P(1.8), _P(2.5))), 0.480072694430958)
    _near(_v(weibull_min.isf(_P(0.05), _P(1.8), _P(2.5))), 4.599005156847313)
    _near(_v(weibull_min.ppf(_P(0.5), _P(1.8), _P(2.5))), 2.0394342534670535)
    _near(_v(weibull_min.isf(_P(0.5), _P(1.8), _P(2.5))), 2.0394342534670535)
    _near(_v(weibull_min.ppf(_P(0.93), _P(1.8), _P(2.5))), 4.304455623355753)
    _near(_v(weibull_min.isf(_P(0.93), _P(1.8), _P(2.5))), 0.5821428969268037)
    _near(weibull_min.mean(1.8, 2.5), 2.223216831130532)
    _near(weibull_min.var(1.8, 2.5), 1.6334551773485007)
    _near(weibull_min.entropy(1.8, 2.5), 1.5850443624838286)
    var i_weibull_min = weibull_min.interval(0.8, 1.8, 2.5)
    _near(i_weibull_min[0], 0.7161158389567861)
    _near(i_weibull_min[1], 3.9734792263862824)

    _near(_v(cauchy.pdf(_P(-3.0), _P(0.5), _P(2.0))), 0.039176601376466544)
    _near(_v(cauchy.logpdf(_P(-3.0), _P(0.5), _P(2.0))), -3.2396756140652014)
    _near(_v(cauchy.cdf(_P(-3.0), _P(0.5), _P(2.0))), 0.1652493405385679)
    _near(_v(cauchy.sf(_P(-3.0), _P(0.5), _P(2.0))), 0.834750659461432)
    _near(_v(cauchy.pdf(_P(0.5), _P(0.5), _P(2.0))), 0.15915494309189535)
    _near(_v(cauchy.logpdf(_P(0.5), _P(0.5), _P(2.0))), -1.8378770664093453)
    _near(_v(cauchy.cdf(_P(0.5), _P(0.5), _P(2.0))), 0.5)
    _near(_v(cauchy.sf(_P(0.5), _P(0.5), _P(2.0))), 0.5)
    _near(_v(cauchy.pdf(_P(7.0), _P(0.5), _P(2.0))), 0.013764751834974732)
    _near(_v(cauchy.logpdf(_P(7.0), _P(0.5), _P(2.0))), -4.285644169247889)
    _near(_v(cauchy.cdf(_P(7.0), _P(0.5), _P(2.0))), 0.9049848390608202)
    _near(_v(cauchy.sf(_P(7.0), _P(0.5), _P(2.0))), 0.09501516093917986)
    _near(_v(cauchy.ppf(_P(0.05), _P(0.5), _P(2.0))), -12.127503029350088)
    _near(_v(cauchy.isf(_P(0.05), _P(0.5), _P(2.0))), 13.127503029350088)
    _near(_v(cauchy.ppf(_P(0.5), _P(0.5), _P(2.0))), 0.5)
    _near(_v(cauchy.isf(_P(0.5), _P(0.5), _P(2.0))), 0.5)
    _near(_v(cauchy.ppf(_P(0.93), _P(0.5), _P(2.0))), 9.447485658423115)
    _near(_v(cauchy.isf(_P(0.93), _P(0.5), _P(2.0))), -8.447485658423115)
    assert_true(cauchy.mean(0.5, 2.0) != cauchy.mean(0.5, 2.0))
    assert_true(cauchy.var(0.5, 2.0) != cauchy.var(0.5, 2.0))
    _near(cauchy.entropy(0.5, 2.0), 3.224171427529236)
    var i_cauchy = cauchy.interval(0.8, 0.5, 2.0)
    _near(i_cauchy[0], -5.655367074350509)
    _near(i_cauchy[1], 6.655367074350509)

    _near(_v(laplace.pdf(_P(-2.0), _P(1.0), _P(1.5))), 0.0451117610788709)
    _near(_v(laplace.logpdf(_P(-2.0), _P(1.0), _P(1.5))), -3.09861228866811)
    _near(_v(laplace.cdf(_P(-2.0), _P(1.0), _P(1.5))), 0.06766764161830635)
    _near(_v(laplace.sf(_P(-2.0), _P(1.0), _P(1.5))), 0.9323323583816936)
    _near(_v(laplace.pdf(_P(1.0), _P(1.0), _P(1.5))), 0.3333333333333333)
    _near(_v(laplace.logpdf(_P(1.0), _P(1.0), _P(1.5))), -1.0986122886681096)
    _near(_v(laplace.cdf(_P(1.0), _P(1.0), _P(1.5))), 0.5)
    _near(_v(laplace.sf(_P(1.0), _P(1.0), _P(1.5))), 0.5)
    _near(_v(laplace.pdf(_P(3.5), _P(1.0), _P(1.5))), 0.06295853427918728)
    _near(_v(laplace.logpdf(_P(3.5), _P(1.0), _P(1.5))), -2.765278955334776)
    _near(_v(laplace.cdf(_P(3.5), _P(1.0), _P(1.5))), 0.9055621985812191)
    _near(_v(laplace.sf(_P(3.5), _P(1.0), _P(1.5))), 0.09443780141878091)
    _near(_v(laplace.ppf(_P(0.05), _P(1.0), _P(1.5))), -2.4538776394910684)
    _near(_v(laplace.isf(_P(0.05), _P(1.0), _P(1.5))), 4.453877639491068)
    _near(_v(laplace.ppf(_P(0.5), _P(1.0), _P(1.5))), 1.0)
    _near(_v(laplace.isf(_P(0.5), _P(1.0), _P(1.5))), 1.0)
    _near(_v(laplace.ppf(_P(0.93), _P(1.0), _P(1.5))), 3.9491692845592503)
    _near(_v(laplace.isf(_P(0.93), _P(1.0), _P(1.5))), -1.9491692845592503)
    _near(laplace.mean(1.0, 1.5), 1.0)
    _near(laplace.var(1.0, 1.5), 4.5)
    _near(laplace.entropy(1.0, 1.5), 2.09861228866811)
    var i_laplace = laplace.interval(0.8, 1.0, 1.5)
    _near(i_laplace[0], -1.4141568686511508)
    _near(i_laplace[1], 3.4141568686511508)

    _near(_v(rayleigh.pdf(_P(0.3), _P(1.7))), 0.1022023874716199)
    _near(_v(rayleigh.logpdf(_P(0.3), _P(1.7))), -2.280800240706332)
    _near(_v(rayleigh.cdf(_P(0.3), _P(1.7))), 0.015450334023395056)
    _near(_v(rayleigh.sf(_P(0.3), _P(1.7))), 0.984549665976605)
    _near(_v(rayleigh.pdf(_P(1.7), _P(1.7))), 0.35678274100743146)
    _near(_v(rayleigh.logpdf(_P(1.7), _P(1.7))), -1.0306282510621703)
    _near(_v(rayleigh.cdf(_P(1.7), _P(1.7))), 0.3934693402873666)
    _near(_v(rayleigh.sf(_P(1.7), _P(1.7))), 0.6065306597126334)
    _near(_v(rayleigh.pdf(_P(4.0), _P(1.7))), 0.08688861821332163)
    _near(_v(rayleigh.logpdf(_P(4.0), _P(1.7))), -2.4431282309698483)
    _near(_v(rayleigh.cdf(_P(4.0), _P(1.7))), 0.9372229733408751)
    _near(_v(rayleigh.sf(_P(4.0), _P(1.7))), 0.06277702665912487)
    _near(_v(rayleigh.ppf(_P(0.05), _P(1.7))), 0.544495400862158)
    _near(_v(rayleigh.isf(_P(0.05), _P(1.7))), 4.161169612157388)
    _near(_v(rayleigh.ppf(_P(0.5), _P(1.7))), 2.001597038276307)
    _near(_v(rayleigh.isf(_P(0.5), _P(1.7))), 2.001597038276307)
    _near(_v(rayleigh.ppf(_P(0.93), _P(1.7))), 3.920525859304012)
    _near(_v(rayleigh.isf(_P(0.93), _P(1.7))), 0.6476562395170361)
    _near(rayleigh.mean(1.7), 2.1306340334363503)
    _near(rayleigh.var(1.7), 1.2403986155627489)
    _near(rayleigh.entropy(1.7), 1.472662493232964)
    var i_rayleigh = rayleigh.interval(0.8, 1.7)
    _near(i_rayleigh[0], 0.7803741285449152)
    _near(i_rayleigh[1], 3.6481422446918903)

    _near(_v(logistic.pdf(_P(-3.0), _P(-1.0), _P(0.8))), 0.08762964568138518)
    _near(_v(logistic.logpdf(_P(-3.0), _P(-1.0), _P(0.8))), -2.4346359172708896)
    _near(_v(logistic.cdf(_P(-3.0), _P(-1.0), _P(0.8))), 0.07585818002124355)
    _near(_v(logistic.sf(_P(-3.0), _P(-1.0), _P(0.8))), 0.9241418199787566)
    _near(_v(logistic.pdf(_P(-1.0), _P(-1.0), _P(0.8))), 0.3125)
    _near(_v(logistic.logpdf(_P(-1.0), _P(-1.0), _P(0.8))), -1.1631508098056809)
    _near(_v(logistic.cdf(_P(-1.0), _P(-1.0), _P(0.8))), 0.5)
    _near(_v(logistic.sf(_P(-1.0), _P(-1.0), _P(0.8))), 0.5)
    _near(_v(logistic.pdf(_P(2.0), _P(-1.0), _P(0.8))), 0.028061762977554323)
    _near(_v(logistic.logpdf(_P(2.0), _P(-1.0), _P(0.8))), -3.5733473774306406)
    _near(_v(logistic.cdf(_P(2.0), _P(-1.0), _P(0.8))), 0.9770226300899744)
    _near(_v(logistic.sf(_P(2.0), _P(-1.0), _P(0.8))), 0.022977369910025615)
    _near(_v(logistic.ppf(_P(0.05), _P(-1.0), _P(0.8))), -3.3555511833331524)
    _near(_v(logistic.isf(_P(0.05), _P(-1.0), _P(0.8))), 1.3555511833331524)
    _near(_v(logistic.ppf(_P(0.5), _P(-1.0), _P(0.8))), -1.0)
    _near(_v(logistic.isf(_P(0.5), _P(-1.0), _P(0.8))), -1.0)
    _near(_v(logistic.ppf(_P(0.93), _P(-1.0), _P(0.8))), 1.0693514752783546)
    _near(_v(logistic.isf(_P(0.93), _P(-1.0), _P(0.8))), -3.0693514752783546)
    _near(logistic.mean(-1.0, 0.8), -1.0)
    _near(logistic.var(-1.0, 0.8), 2.10551560556573)
    _near(logistic.entropy(-1.0, 0.8), 1.7768564486857903)
    var i_logistic = logistic.interval(0.8, -1.0, 0.8)
    _near(i_logistic[0], -2.757779661868976)
    _near(i_logistic[1], 0.7577796618689758)

    _near(_v(pareto.pdf(_P(1.6), _P(2.5), _P(1.5))), 1.3296833082529738)
    _near(_v(pareto.logpdf(_P(1.6), _P(2.5), _P(1.5))), 0.2849407997844916)
    _near(_v(pareto.cdf(_P(1.6), _P(2.5), _P(1.5))), 0.1490026827180968)
    _near(_v(pareto.sf(_P(1.6), _P(2.5), _P(1.5))), 0.8509973172819032)
    _near(_v(pareto.pdf(_P(2.5), _P(2.5), _P(1.5))), 0.278854800926934)
    _near(_v(pareto.logpdf(_P(2.5), _P(2.5), _P(1.5))), -1.277064059414977)
    _near(_v(pareto.cdf(_P(2.5), _P(2.5), _P(1.5))), 0.721145199073066)
    _near(_v(pareto.sf(_P(2.5), _P(2.5), _P(1.5))), 0.278854800926934)
    _near(_v(pareto.pdf(_P(6.0), _P(2.5), _P(1.5))), 0.013020833333333334)
    _near(_v(pareto.logpdf(_P(6.0), _P(2.5), _P(1.5))), -4.341204640153626)
    _near(_v(pareto.cdf(_P(6.0), _P(2.5), _P(1.5))), 0.96875)
    _near(_v(pareto.sf(_P(6.0), _P(2.5), _P(1.5))), 0.03125)
    _near(_v(pareto.ppf(_P(0.05), _P(2.5), _P(1.5))), 1.5310938672437064)
    _near(_v(pareto.isf(_P(0.05), _P(2.5), _P(1.5))), 4.97168102600998)
    _near(_v(pareto.ppf(_P(0.5), _P(2.5), _P(1.5))), 1.9792618661593413)
    _near(_v(pareto.isf(_P(0.5), _P(2.5), _P(1.5))), 1.9792618661593413)
    _near(_v(pareto.ppf(_P(0.93), _P(2.5), _P(1.5))), 4.345622961946377)
    _near(_v(pareto.isf(_P(0.93), _P(2.5), _P(1.5))), 1.544180556095745)
    _near(pareto.mean(2.5, 1.5), 2.5)
    _near(pareto.var(2.5, 1.5), 5.0)
    _near(pareto.entropy(2.5, 1.5), 0.8891743762340092)
    var i_pareto = pareto.interval(0.8, 2.5, 1.5)
    _near(i_pareto[0], 1.5645673222659489)
    _near(i_pareto[1], 3.767829647264371)


def test_tensor_overloads_agree_with_the_scalar_ones() raises:
    var x = Static[f64, 4]([0.5, 1.0, 2.0, 4.0], DeviceContext(api="cpu"))
    var got = weibull_min.cdf(x, 1.8, 2.5).to_host()
    var want = [0.5, 1.0, 2.0, 4.0]
    for i in range(4):
        _near(
            Float64(got[i]), _v(weibull_min.cdf(_P(want[i]), _P(1.8), _P(2.5)))
        )
    var q = Static[f64, 3]([0.1, 0.5, 0.9], DeviceContext(api="cpu"))
    var ppf = cauchy.ppf(q, 0.5, 2.0).to_host()
    var qs = [0.1, 0.5, 0.9]
    for i in range(3):
        _near(Float64(ppf[i]), _v(cauchy.ppf(_P(qs[i]), _P(0.5), _P(2.0))))


def _check[
    dtype: DType
](values: List[Scalar[dtype]], mean: Float64, var_: Float64) raises:
    var total = 0.0
    for i in range(len(values)):
        total += Float64(values[i])
    var m = total / Float64(len(values))
    var sq = 0.0
    for i in range(len(values)):
        sq += (Float64(values[i]) - m) ** 2
    assert_true(abs(m - mean) < 5 * sqrt(var_ / Float64(len(values))))
    assert_true(abs(sq / Float64(len(values) - 1) - var_) < 0.08 * var_)


def test_rvs_have_the_distribution_moments() raises:
    var rng = Generator(seed=91)
    comptime n = 100000
    _check(uniform_dist.rvs[f64, n](1.0, 3.0, rng).to_host(), 2.5, 0.75)
    _check(
        lognorm.rvs[f64, n](0.4, 2.0, rng).to_host(),
        lognorm.mean(0.4, 2.0),
        lognorm.var(0.4, 2.0),
    )
    _check(
        weibull_min.rvs[f64, n](1.8, 2.5, rng).to_host(),
        weibull_min.mean(1.8, 2.5),
        weibull_min.var(1.8, 2.5),
    )
    _check(laplace.rvs[f64, n](1.0, 1.5, rng).to_host(), 1.0, 4.5)
    _check(
        rayleigh.rvs[f64, n](1.7, rng).to_host(),
        rayleigh.mean(1.7),
        rayleigh.var(1.7),
    )
    _check(
        logistic.rvs[f64, n](-1.0, 0.8, rng).to_host(),
        -1.0,
        logistic.var(-1.0, 0.8),
    )
    _check(
        pareto.rvs[f64, n](5.0, 1.5, rng).to_host(),
        pareto.mean(5.0, 1.5),
        pareto.var(5.0, 1.5),
    )


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
