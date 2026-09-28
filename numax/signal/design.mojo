"""IIR filter design and frequency response over
`numax.core.tensor.Tensor`: `butter`, `cheby1`, `cheby2`, `ellip`, the
`iirfilter` front door they share, and `freqz`; and the conversions
between the three forms a filter takes -- `(b, a)`, zeros-poles-gain and
second-order sections -- `zpk2tf`, `tf2zpk`, `zpk2sos`, `tf2sos` and
`sos2tf`.

Every design takes `output=`, SciPy's argument as an Int selector:
`OUTPUT_BA` (the default), `OUTPUT_ZPK` or `OUTPUT_SOS`. The sections are
paired straight from the design's own roots, never from expanded
polynomials, so they are SciPy's to rounding; a `StaticString` cannot
steer a return type, which is why the selector is an Int.

**Tier 2.** A design is a few dozen complex numbers on the host --
prototype poles, a frequency warp, the bilinear transform, a polynomial
expansion -- and a table at the end, so every routine here runs on the
host in `Float64` and uploads `(b, a)` once, the way `firwin` does.
`freqz` is device work: one lane per frequency evaluating two polynomials
on the unit circle.

## The reversal this module records

`docs/parity.md` said filter *design* past a windowed sinc was out of
scope. It is not, and the reason it was is worth stating: the design math
needs complex poles and zeros, and a `dtype`-monomorphic `Tensor` holds no
`Complex`. But the design is scalar host arithmetic on a handful of values
and its *result* -- `b` and `a` -- is real, so nothing about the `Tensor`
tier stands in its way; the poles live in host `Float64` pairs for the
microseconds the design takes. All four families follow
`scipy.signal.iirfilter`'s route exactly: an analog prototype, the edge (or
pair of edges) pre-warped by `4 tan(pi wn / 2)` so the bilinear transform
lands it where it was asked for, `lp2lp`/`lp2hp`/`lp2bp`/`lp2bs` to move
the prototype there, the bilinear transform at `fs = 2`, and the
zero-pole-gain form expanded to polynomials.

## Why the order is a parameter, and doubles for a band

`TransferFunction` carries its order in its type, and `lp2bp`/`lp2bs`
double it. A `StaticString` cannot be constrained in a `where` clause, so
`btype` cannot steer a return type. The dispatch is the *argument* instead:
each family has a `wn: Float64` overload that designs lowpass or highpass
at `order`, and a `wn: Tuple[Float64, Float64]` overload that designs
bandpass or bandstop at `2 * order`. A `btype` the overload cannot serve
raises rather than falling through, which is the rule the rest of numax's
string-keyed entry points follow.

## Complete `K(m)` here rather than `numax.special.elliptic_k`

`_ellipap` needs `K` twice inside the degree equation, once under an
exponential, and the tier-1 `elliptic_k` is Abramowitz and Stegun
17.3.34's polynomial-plus-log at `~2e-8`. Measured: routing `_ellipdeg`
through it puts `ellip(3, 1, 40, 0.3)`'s coefficients `1e-8` from SciPy's,
where the arithmetic-geometric mean below puts them `4e-13` away. The AGM
is four lines and converges quadratically, and this is host tier-2 code
with no fixed-iteration obligation, so it is the right `K` for this call
site and `elliptic_k` stays the right one for a GPU-launchable kernel.

## The MAX gate

Nothing: MAX has no filter design of any kind -- not in `linalg`, `nn`,
`algorithm` or `layout`. `nn` ships convolution and `numax.fft` the
transforms, but a prototype-to-bilinear route has no counterpart to
delegate to. **Extend.**
"""

from std.math import (
    acosh as _acosh,
    asin as _asin,
    asinh as _asinh,
    atan as _atan,
    ceil as _ceil,
    cos as _cos,
    cosh as _cosh,
    exp as _exp,
    expm1 as _expm1,
    hypot as _hypot,
    log10 as _log10,
    sin as _sin,
    sinh as _sinh,
    sqrt as _sqrt,
    tan as _tan,
)

from layout import Coord, coord_to_index_list
from max.algorithm.functional import elementwise
from max.gpu.host import DeviceContext

from ..core.tensorlike import TensorLike, TensorView, dim, is_row_major
from ..core.tensor import Static

comptime _PI = 3.141592653589793
comptime _LN10 = 2.302585092994046

# `numpy.finfo(float).eps`: SciPy's `_filter_design` uses it to drop the
# zero `sn` at the origin and the real pole of an odd-order elliptic
# prototype, and the counts only come out right if the same threshold is
# used here.
comptime _EPS = 2.220446049250313e-16


struct TransferFunction[dtype: DType, order: Int](Movable):
    """What the design routines return: the numerator `b` and denominator
    `a` of a digital filter of the given `order`, each `order + 1` long,
    SciPy's `(b, a)` as a struct. Feed them to `lfilter`, `filtfilt` or
    `freqz`."""

    var b: Static[Self.dtype, Self.order + 1]
    var a: Static[Self.dtype, Self.order + 1]

    def __init__(
        out self,
        var b: Static[Self.dtype, Self.order + 1],
        var a: Static[Self.dtype, Self.order + 1],
    ):
        self.b = b^
        self.a = a^


struct FrequencyResponse[dtype: DType, n: Int](Movable):
    """What `freqz` returns: the frequencies `w` in radians per sample and
    the complex response `H(e^{jw})` as a real/imaginary pair, SciPy's
    `(w, h)`."""

    var w: Static[Self.dtype, Self.n]
    var real: Static[Self.dtype, Self.n]
    var imag: Static[Self.dtype, Self.n]

    def __init__(
        out self,
        var w: Static[Self.dtype, Self.n],
        var real: Static[Self.dtype, Self.n],
        var imag: Static[Self.dtype, Self.n],
    ):
        self.w = w^
        self.real = real^
        self.imag = imag^


# --------------------------------------------------------------------------
# Complex scalars, as pairs of `Float64`
# --------------------------------------------------------------------------


def _complex_multiply(
    ar: Float64, ai: Float64, br: Float64, bi: Float64
) -> Tuple[Float64, Float64]:
    return (ar * br - ai * bi, ar * bi + ai * br)


def _complex_divide(
    ar: Float64, ai: Float64, br: Float64, bi: Float64
) -> Tuple[Float64, Float64]:
    var d = br * br + bi * bi
    return ((ar * br + ai * bi) / d, (ai * br - ar * bi) / d)


def _complex_sqrt(ar: Float64, ai: Float64) -> Tuple[Float64, Float64]:
    """The principal square root. `lp2bp`/`lp2bs` split each prototype root
    into two through it, which is where the order doubling comes from."""
    if ar == 0.0 and ai == 0.0:
        return (0.0, 0.0)
    var r = _hypot(ar, ai)
    var re = _sqrt(0.5 * (r + ar))
    var im = _sqrt(0.5 * (r - ar))
    return (re, -im) if ai < 0.0 else (re, im)


def _product_of_negatives(
    re: List[Float64], im: List[Float64]
) -> Tuple[Float64, Float64]:
    """`prod(-x)` over a root list -- the gain factor every transform that
    moves a root off the origin needs."""
    var r = 1.0
    var i = 0.0
    for j in range(len(re)):
        var q = _complex_multiply(r, i, -re[j], -im[j])
        r = q[0]
        i = q[1]
    return (r, i)


def _product_of_offsets(
    c: Float64, re: List[Float64], im: List[Float64]
) -> Tuple[Float64, Float64]:
    """`prod(c - x)`, the bilinear transform's gain factor."""
    var r = 1.0
    var i = 0.0
    for j in range(len(re)):
        var q = _complex_multiply(r, i, c - re[j], -im[j])
        r = q[0]
        i = q[1]
    return (r, i)


def _poly_from_roots(
    re: List[Float64], im: List[Float64]
) -> Tuple[List[Float64], List[Float64]]:
    """The monic polynomial with the given complex roots, highest degree
    first, as real and imaginary coefficient lists."""
    var pr = List[Float64]()
    var pi = List[Float64]()
    pr.append(1.0)
    pi.append(0.0)
    for k in range(len(re)):
        var nr = List[Float64](length=len(pr) + 1, fill=0.0)
        var ni = List[Float64](length=len(pr) + 1, fill=0.0)
        for j in range(len(pr)):
            # (p * z) - root * p, shifted
            nr[j] += pr[j]
            ni[j] += pi[j]
            nr[j + 1] -= pr[j] * re[k] - pi[j] * im[k]
            ni[j + 1] -= pr[j] * im[k] + pi[j] * re[k]
        pr = nr^
        pi = ni^
    return (pr^, pi^)


struct _Zpk(Movable):
    """A zero-pole-gain filter, `scipy.signal`'s `(z, p, k)` with the two
    root lists split into real and imaginary halves, since numax has no
    host complex type and the design needs no more than this."""

    var zr: List[Float64]
    var zi: List[Float64]
    var pr: List[Float64]
    var pi: List[Float64]
    var gain: Float64

    def __init__(
        out self,
        var zr: List[Float64],
        var zi: List[Float64],
        var pr: List[Float64],
        var pi: List[Float64],
        gain: Float64,
    ):
        self.zr = zr^
        self.zi = zi^
        self.pr = pr^
        self.pi = pi^
        self.gain = gain

    def relative_degree(self) -> Int:
        """Poles minus zeros: how many zeros a transform has to put back
        at the origin, at infinity or at `-1` to keep the filter proper."""
        return len(self.pr) - len(self.zr)


def _pow10m1(x: Float64) -> Float64:
    """`10^x - 1`, accurate for small `x` -- SciPy's `_pow10m1`, which is
    what makes a ripple of `0.01 dB` mean something."""
    return _expm1(_LN10 * x)


# --------------------------------------------------------------------------
# Analog prototypes, normalized to a cutoff of 1 rad/s
# --------------------------------------------------------------------------


def _buttap(n: Int) -> _Zpk:
    """`scipy.signal.buttap`: `n` poles evenly spaced on the left half of
    the unit circle, no zeros, unit gain."""
    var pr = List[Float64](capacity=n)
    var pi = List[Float64](capacity=n)
    for j in range(n):
        var angle = _PI * Float64(-n + 1 + 2 * j) / Float64(2 * n)
        pr.append(-_cos(angle))
        pi.append(-_sin(angle))
    return _Zpk(List[Float64](), List[Float64](), pr^, pi^, 1.0)


def _cheb1ap(n: Int, rp: Float64) -> _Zpk:
    """`scipy.signal.cheb1ap`: Butterworth's circle squashed onto an
    ellipse by `mu = asinh(1 / eps) / n`, giving `rp` dB of equiripple in
    the passband and no zeros."""
    var eps_sq = _pow10m1(0.1 * rp)
    var mu = _asinh(1.0 / _sqrt(eps_sq)) / Float64(n)
    var pr = List[Float64](capacity=n)
    var pi = List[Float64](capacity=n)
    for j in range(n):
        var theta = _PI * Float64(-n + 1 + 2 * j) / Float64(2 * n)
        pr.append(-_sinh(mu) * _cos(theta))
        pi.append(-_cosh(mu) * _sin(theta))
    var prod = _product_of_negatives(pr, pi)
    var gain = prod[0]
    if n % 2 == 0:
        # An even-order type I filter starts the ripple at -rp dB.
        gain /= _sqrt(1.0 + eps_sq)
    return _Zpk(List[Float64](), List[Float64](), pr^, pi^, gain)


def _cheb2ap(n: Int, rs: Float64) -> _Zpk:
    """`scipy.signal.cheb2ap`: the inverse Chebyshev, flat in the passband
    with `rs` dB of equiripple stopband held down by finite imaginary
    zeros. An odd order has one fewer zero than pole, which is the count
    the `m` list below gets right by skipping the one at the origin."""
    var mu = _asinh(_sqrt(_pow10m1(0.1 * rs))) / Float64(n)

    var ms = List[Int]()
    if n % 2 == 1:
        var v = -n + 1
        while v < 0:
            ms.append(v)
            v += 2
        v = 2
        while v < n:
            ms.append(v)
            v += 2
    else:
        var v = -n + 1
        while v < n:
            ms.append(v)
            v += 2

    var zr = List[Float64](capacity=len(ms))
    var zi = List[Float64](capacity=len(ms))
    for j in range(len(ms)):
        zr.append(0.0)
        zi.append(1.0 / _sin(Float64(ms[j]) * _PI / Float64(2 * n)))

    var pr = List[Float64](capacity=n)
    var pi = List[Float64](capacity=n)
    for j in range(n):
        var angle = _PI * Float64(-n + 1 + 2 * j) / Float64(2 * n)
        var q = _complex_divide(
            1.0, 0.0, -_sinh(mu) * _cos(angle), -_cosh(mu) * _sin(angle)
        )
        pr.append(q[0])
        pi.append(q[1])

    var pp = _product_of_negatives(pr, pi)
    var zz = _product_of_negatives(zr, zi)
    var g = _complex_divide(pp[0], pp[1], zz[0], zz[1])
    return _Zpk(zr^, zi^, pr^, pi^, g[0])


def _ellipk(m: Float64) -> Float64:
    """The complete elliptic integral of the first kind by the
    arithmetic-geometric mean: `K(m) = pi / (2 AGM(1, sqrt(1 - m)))`.

    The module docstring says why this is here rather than
    `numax.special.elliptic_k`. Twelve iterations is well past the six a
    quadratically convergent mean needs for float64.
    """
    var a = 1.0
    var b = _sqrt(1.0 - m)
    for _ in range(12):
        var next_a = 0.5 * (a + b)
        b = _sqrt(a * b)
        a = next_a
    return _PI / (2.0 * a)


def _ellipdeg(n: Int, m1: Float64) -> Float64:
    """Solve `n K(m) / K(1 - m) = K(m1) / K(1 - m1)` for `m` through the
    nome series, Orfanidis eq. (49) as SciPy's `_ellipdeg` writes it."""
    var q = _exp(-_PI * _ellipk(1.0 - m1) / _ellipk(m1)) ** (1.0 / Float64(n))
    var num = 0.0
    for j in range(8):
        num += q ** Float64(j * (j + 1))
    var den = 1.0
    for j in range(1, 9):
        den += 2.0 * q ** Float64(j * j)
    var ratio = num / den
    return 16.0 * q * ratio * ratio * ratio * ratio


def _ellipj(u: Float64, m: Float64) -> Tuple[Float64, Float64, Float64]:
    """Jacobi `sn`, `cn` and `dn` by Abramowitz and Stegun 16.4's
    descending Landen ladder, the pieces `_ellipap` places its zeros and
    poles from. `scipy.special.ellipj`."""
    var a = List[Float64]()
    var c = List[Float64]()
    a.append(1.0)
    c.append(_sqrt(m))
    var b = _sqrt(1.0 - m)
    var steps = 0
    while steps < 16:
        var next_a = 0.5 * (a[steps] + b)
        var next_c = 0.5 * (a[steps] - b)
        b = _sqrt(a[steps] * b)
        a.append(next_a)
        c.append(next_c)
        steps += 1
        if abs(next_c) < 1e-17:
            break

    var phi = u
    for _ in range(steps):
        phi *= 2.0
    phi *= a[steps]

    var previous = phi
    for i in range(steps, 0, -1):
        var arg = c[i] / a[i] * _sin(phi)
        if arg > 1.0:
            arg = 1.0
        if arg < -1.0:
            arg = -1.0
        previous = phi
        phi = 0.5 * (phi + _asin(arg))
    return (_sin(phi), _cos(phi), _cos(phi) / _cos(previous - phi))


def _arc_jac_sc1(w: Float64, m: Float64) -> Float64:
    """Solve `w = sc(z, 1 - m)` for a real `z`, SciPy's `_arc_jac_sc1`.

    SciPy reaches it through `sn^{-1}(i w, m)` on complex arithmetic and
    then takes the imaginary part. It never needs the complex form: the
    Landen ladder sends a purely imaginary argument to a purely imaginary
    one, since `sqrt(1 - (k i y)^2) = sqrt(1 + (k y)^2)` is real, and the
    final `arcsin(i y)` is `i arcsinh(y)`. So this runs on the imaginary
    parts alone.
    """
    var ks = List[Float64]()
    ks.append(_sqrt(m))
    var steps = 0
    while steps < 12:
        var k = ks[steps]
        if k == 0.0:
            break
        var kp = _sqrt((1.0 - k) * (1.0 + k))
        ks.append((1.0 - kp) / (1.0 + kp))
        steps += 1

    var big_k = 1.0
    for j in range(1, len(ks)):
        big_k *= 1.0 + ks[j]
    big_k *= _PI / 2.0

    var y = w
    for j in range(len(ks) - 1):
        var kn = ks[j]
        y = (
            2.0
            * y
            / ((1.0 + ks[j + 1]) * (1.0 + _sqrt(1.0 + (kn * y) * (kn * y))))
        )
    return big_k * (2.0 / _PI) * _asinh(y)


def _ellipap(n: Int, rp: Float64, rs: Float64) raises -> _Zpk:
    """`scipy.signal.ellipap`: `rp` dB of passband ripple and `rs` dB of
    stopband attenuation in the fewest poles of any of the four, at the
    price of ripple in both bands. Orfanidis' route, which SciPy follows:
    the degree equation for the modulus, Jacobi `sn` at `n` evenly spaced
    quarter-period points for the zeros, and the inverse `sc` of `1/eps`
    for the pole offset."""
    if n == 1:
        var single = -_sqrt(1.0 / _pow10m1(0.1 * rp))
        var pr = List[Float64]()
        var pi = List[Float64]()
        pr.append(single)
        pi.append(0.0)
        return _Zpk(List[Float64](), List[Float64](), pr^, pi^, -single)

    var eps_sq = _pow10m1(0.1 * rp)
    var ck1_sq = eps_sq / _pow10m1(0.1 * rs)
    if ck1_sq == 0.0:
        raise Error(
            "ellip: no filter meets both rp and rs; lower rs or raise rp"
        )

    var k1 = _ellipk(ck1_sq)
    var m = _ellipdeg(n, ck1_sq)
    var capk = _ellipk(m)

    var sn = List[Float64]()
    var cn = List[Float64]()
    var dn = List[Float64]()
    var j = 1 - n % 2
    while j < n:
        var e = _ellipj(Float64(j) * capk / Float64(n), m)
        sn.append(e[0])
        cn.append(e[1])
        dn.append(e[2])
        j += 2

    # Zeros on the imaginary axis at `i / (sqrt(m) sn)`, in conjugate
    # pairs. An odd order's `sn` at the origin is zero and drops out,
    # which is what leaves it one zero short of its pole count.
    var zr = List[Float64]()
    var zi = List[Float64]()
    for i in range(len(sn)):
        if abs(sn[i]) > _EPS:
            zr.append(0.0)
            zi.append(1.0 / (_sqrt(m) * sn[i]))
    var distinct_zeros = len(zr)
    for i in range(distinct_zeros):
        zr.append(zr[i])
        zi.append(-zi[i])

    var offset = capk * _arc_jac_sc1(1.0 / _sqrt(eps_sq), ck1_sq)
    var e = _ellipj(offset / (Float64(n) * k1), 1.0 - m)
    var sv = e[0]
    var cv = e[1]
    var dv = e[2]

    var pr = List[Float64]()
    var pi = List[Float64]()
    for i in range(len(sn)):
        var den = 1.0 - (dn[i] * sv) * (dn[i] * sv)
        pr.append(-(cn[i] * dn[i] * sv * cv) / den)
        pi.append(-(sn[i] * dv) / den)

    var distinct_poles = len(pr)
    if n % 2 == 1:
        # The odd order's one real pole is its own conjugate.
        var norm = 0.0
        for i in range(distinct_poles):
            norm += pr[i] * pr[i] + pi[i] * pi[i]
        norm = _sqrt(norm)
        for i in range(distinct_poles):
            if abs(pi[i]) > _EPS * norm:
                pr.append(pr[i])
                pi.append(-pi[i])
    else:
        for i in range(distinct_poles):
            pr.append(pr[i])
            pi.append(-pi[i])

    var pp = _product_of_negatives(pr, pi)
    var zz = _product_of_negatives(zr, zi)
    var g = _complex_divide(pp[0], pp[1], zz[0], zz[1])
    var gain = g[0]
    if n % 2 == 0:
        gain /= _sqrt(1.0 + eps_sq)
    return _Zpk(zr^, zi^, pr^, pi^, gain)


# --------------------------------------------------------------------------
# Band transforms and the bilinear map
# --------------------------------------------------------------------------


def _lp2lp_zpk(var f: _Zpk, wo: Float64) -> _Zpk:
    """Move a unit-cutoff prototype to `wo`: every root scales, the gain by
    `wo` once per missing zero."""
    var degree = f.relative_degree()
    for j in range(len(f.zr)):
        f.zr[j] *= wo
        f.zi[j] *= wo
    for j in range(len(f.pr)):
        f.pr[j] *= wo
        f.pi[j] *= wo
    for _ in range(degree):
        f.gain *= wo
    return f^


def _lp2hp_zpk(var f: _Zpk, wo: Float64) -> _Zpk:
    """`s -> wo / s`: every root inverts, the missing zeros land at the
    origin, and the gain picks up `prod(-z) / prod(-p)`."""
    var degree = f.relative_degree()
    var zz = _product_of_negatives(f.zr, f.zi)
    var pp = _product_of_negatives(f.pr, f.pi)
    var ratio = _complex_divide(zz[0], zz[1], pp[0], pp[1])
    f.gain *= ratio[0]
    for j in range(len(f.zr)):
        var q = _complex_divide(wo, 0.0, f.zr[j], f.zi[j])
        f.zr[j] = q[0]
        f.zi[j] = q[1]
    for j in range(len(f.pr)):
        var q = _complex_divide(wo, 0.0, f.pr[j], f.pi[j])
        f.pr[j] = q[0]
        f.pi[j] = q[1]
    for _ in range(degree):
        f.zr.append(0.0)
        f.zi.append(0.0)
    return f^


def _split_around(
    re: List[Float64], im: List[Float64], wo: Float64
) -> Tuple[List[Float64], List[Float64]]:
    """`x +- sqrt(x^2 - wo^2)` for every root, the plus branch first --
    the step that doubles a band form's order."""
    var outr = List[Float64](capacity=2 * len(re))
    var outi = List[Float64](capacity=2 * len(re))
    var lowr = List[Float64](capacity=len(re))
    var lowi = List[Float64](capacity=len(re))
    for j in range(len(re)):
        var square = _complex_multiply(re[j], im[j], re[j], im[j])
        var root = _complex_sqrt(square[0] - wo * wo, square[1])
        outr.append(re[j] + root[0])
        outi.append(im[j] + root[1])
        lowr.append(re[j] - root[0])
        lowi.append(im[j] - root[1])
    for j in range(len(lowr)):
        outr.append(lowr[j])
        outi.append(lowi[j])
    return (outr^, outi^)


def _lp2bp_zpk(var f: _Zpk, wo: Float64, bw: Float64) -> _Zpk:
    """`s -> (s^2 + wo^2) / (s bw)`: each root scales by `bw / 2` and then
    splits in two, the missing zeros land at the origin."""
    var degree = f.relative_degree()
    var half = bw / 2.0
    for j in range(len(f.zr)):
        f.zr[j] *= half
        f.zi[j] *= half
    for j in range(len(f.pr)):
        f.pr[j] *= half
        f.pi[j] *= half
    var zs = _split_around(f.zr, f.zi, wo)
    var ps = _split_around(f.pr, f.pi, wo)
    f.zr = zs[0].copy()
    f.zi = zs[1].copy()
    f.pr = ps[0].copy()
    f.pi = ps[1].copy()
    for _ in range(degree):
        f.zr.append(0.0)
        f.zi.append(0.0)
    for _ in range(degree):
        f.gain *= bw
    return f^


def _lp2bs_zpk(var f: _Zpk, wo: Float64, bw: Float64) -> _Zpk:
    """`s -> (s bw) / (s^2 + wo^2)`: the highpass inversion, then the same
    split, with the missing zeros landing on the band's own center
    frequency at `+-i wo` rather than at the origin."""
    var degree = f.relative_degree()
    var half = bw / 2.0
    var zz = _product_of_negatives(f.zr, f.zi)
    var pp = _product_of_negatives(f.pr, f.pi)
    var ratio = _complex_divide(zz[0], zz[1], pp[0], pp[1])
    f.gain *= ratio[0]
    for j in range(len(f.zr)):
        var q = _complex_divide(half, 0.0, f.zr[j], f.zi[j])
        f.zr[j] = q[0]
        f.zi[j] = q[1]
    for j in range(len(f.pr)):
        var q = _complex_divide(half, 0.0, f.pr[j], f.pi[j])
        f.pr[j] = q[0]
        f.pi[j] = q[1]
    var zs = _split_around(f.zr, f.zi, wo)
    var ps = _split_around(f.pr, f.pi, wo)
    f.zr = zs[0].copy()
    f.zi = zs[1].copy()
    f.pr = ps[0].copy()
    f.pi = ps[1].copy()
    for _ in range(degree):
        f.zr.append(0.0)
        f.zi.append(wo)
    for _ in range(degree):
        f.zr.append(0.0)
        f.zi.append(-wo)
    return f^


def _bilinear_zpk(var f: _Zpk, fs: Float64) -> _Zpk:
    """`s -> 2 fs (z - 1) / (z + 1)`: analog to digital, the missing zeros
    landing at `-1` (Nyquist) and the gain by
    `prod(2 fs - z) / prod(2 fs - p)`."""
    var degree = f.relative_degree()
    var fs2 = 2.0 * fs
    var zz = _product_of_offsets(fs2, f.zr, f.zi)
    var pp = _product_of_offsets(fs2, f.pr, f.pi)
    var ratio = _complex_divide(zz[0], zz[1], pp[0], pp[1])
    f.gain *= ratio[0]
    for j in range(len(f.zr)):
        var q = _complex_divide(fs2 + f.zr[j], f.zi[j], fs2 - f.zr[j], -f.zi[j])
        f.zr[j] = q[0]
        f.zi[j] = q[1]
    for j in range(len(f.pr)):
        var q = _complex_divide(fs2 + f.pr[j], f.pi[j], fs2 - f.pr[j], -f.pi[j])
        f.pr[j] = q[0]
        f.pi[j] = q[1]
    for _ in range(degree):
        f.zr.append(-1.0)
        f.zi.append(0.0)
    return f^


def _zpk2tf(var f: _Zpk) -> Tuple[List[Float64], List[Float64]]:
    """`scipy.signal.zpk2tf`: expand both root sets into polynomials and
    scale the numerator by the gain. Both come out real, since every
    complex root arrives with its conjugate."""
    var num = _poly_from_roots(f.zr, f.zi)
    var den = _poly_from_roots(f.pr, f.pi)
    var b = List[Float64](capacity=len(num[0]))
    for j in range(len(num[0])):
        b.append(f.gain * num[0][j])
    return (b^, den[0].copy())


# --------------------------------------------------------------------------
# The shared route, and the checks the string arguments need
# --------------------------------------------------------------------------


def _prototype(
    ftype: StaticString, order: Int, rp: Float64, rs: Float64
) raises -> _Zpk:
    if ftype == "butter":
        return _buttap(order)
    elif ftype == "cheby1":
        return _cheb1ap(order, rp)
    elif ftype == "cheby2":
        return _cheb2ap(order, rs)
    elif ftype == "ellip":
        return _ellipap(order, rp, rs)
    else:
        raise Error(
            "iirfilter: unknown ftype '",
            ftype,
            "'; expected 'butter', 'cheby1', 'cheby2' or 'ellip'",
        )


def _check_edge(who: StaticString, btype: StaticString, wn: Float64) raises:
    if not (btype == "lowpass" or btype == "highpass"):
        raise Error(
            who,
            ": btype '",
            btype,
            (
                "' needs a (low, high) pair; this overload takes one edge and"
                " designs 'lowpass' or 'highpass'"
            ),
        )
    if wn <= 0 or wn >= 1:
        raise Error(who, ": wn must lie strictly inside (0, 1)")


def _check_band(
    who: StaticString, btype: StaticString, wn: Tuple[Float64, Float64]
) raises:
    if not (btype == "bandpass" or btype == "bandstop"):
        raise Error(
            who,
            ": btype '",
            btype,
            (
                "' takes one edge, not a pair; this overload designs"
                " 'bandpass' or 'bandstop'"
            ),
        )
    if wn[0] <= 0 or wn[1] >= 1 or wn[0] >= wn[1]:
        raise Error(
            who, ": wn must be an increasing pair strictly inside (0, 1)"
        )


def _design(
    var proto: _Zpk, wn_lo: Float64, wn_hi: Float64, btype: StaticString
) raises -> _Zpk:
    """The route every family shares: warp the edges, move the prototype,
    bilinear at `fs = 2`. The digital zeros, poles and gain, which
    `_deliver` expands, keeps or pairs into sections."""
    var w_lo = 4.0 * _tan(_PI * wn_lo / 2.0)
    if btype == "lowpass":
        proto = _lp2lp_zpk(proto^, w_lo)
    elif btype == "highpass":
        proto = _lp2hp_zpk(proto^, w_lo)
    else:
        var w_hi = 4.0 * _tan(_PI * wn_hi / 2.0)
        var bw = w_hi - w_lo
        var wo = _sqrt(w_lo * w_hi)
        if btype == "bandpass":
            proto = _lp2bp_zpk(proto^, wo, bw)
        else:
            proto = _lp2bs_zpk(proto^, wo, bw)
    return _bilinear_zpk(proto^, 2.0)


def _to_transfer_function[
    dtype: DType, order: Int
](
    var ba: Tuple[List[Float64], List[Float64]],
    ctx: Optional[DeviceContext],
) raises -> TransferFunction[dtype, order] where dtype.is_floating_point():
    var b = List[Scalar[dtype]](capacity=order + 1)
    var a = List[Scalar[dtype]](capacity=order + 1)
    for j in range(order + 1):
        b.append(Scalar[dtype](ba[0][j]))
        a.append(Scalar[dtype](ba[1][j]))
    var device = ctx.value() if ctx else DeviceContext(api="cpu")
    return TransferFunction[dtype, order](
        Static[dtype, order + 1](b^, device),
        Static[dtype, order + 1](a^, device),
    )


# --------------------------------------------------------------------------
# The four families, each as an edge overload and a band overload
# --------------------------------------------------------------------------


def butter[
    dtype: DType, order: Int, output: Int = OUTPUT_BA
](
    wn: Float64,
    btype: StaticString = "lowpass",
    ctx: Optional[DeviceContext] = None,
) raises -> _Designed[dtype, order, output] where (
    dtype.is_floating_point() and order >= 1
):
    """A digital Butterworth filter of the given `order`, lowpass or
    highpass, with its edge at `wn` as a fraction of Nyquist.
    `scipy.signal.butter(order, wn, btype)`, returning `(b, a)`.

    Maximally flat in the passband, which is what the name promises and the
    only thing it does; `cheby1`, `cheby2` and `ellip` trade ripple for a
    steeper transition at the same order.

    `"lowpass"` or `"highpass"`; pass a `(low, high)` tuple for `"bandpass"`
    or `"bandstop"`, which doubles the order and so is a separate overload.
    For an order above about eight, ask for second-order sections --
    `output=OUTPUT_SOS`, SciPy's `output="sos"` -- since the `(b, a)` of a
    long polynomial lose digits a cascade keeps.
    `wn` must lie in `(0, 1)`.

    Parameters:
        dtype: The floating-point dtype of the returned coefficients.
        order: The order of the filter, at least 1; `b` and `a` are each
            `order + 1` long.
        output: `OUTPUT_BA` (the default), `OUTPUT_ZPK` or `OUTPUT_SOS`,
            SciPy's `output=`.

    Args:
        wn: The critical frequency as a fraction of Nyquist, strictly
            inside `(0, 1)`.
        btype: `"lowpass"` (the default) or `"highpass"`.
        ctx: The device `b` and `a` are uploaded to; `None` puts them in
            host memory.

    Returns:
        The design in the form `output` selects: a `TransferFunction` of
        `b` and `a` (`OUTPUT_BA`), a `ZerosPolesGain` (`OUTPUT_ZPK`), or the
        second-order sections as a `(sections, 6)` tensor (`OUTPUT_SOS`),
        paired as `zpk2sos` pairs them.

    Raises:
        If `btype` is not `"lowpass"` or `"highpass"`, if `wn` is outside
        `(0, 1)`, or if the upload to `ctx` fails.
    """
    _check_edge("butter", btype, wn)
    return _deliver[dtype, order, output](
        _design(_buttap(order), wn, 0.0, btype), ctx
    )


def butter[
    dtype: DType, order: Int, output: Int = OUTPUT_BA
](
    wn: Tuple[Float64, Float64],
    btype: StaticString = "bandpass",
    ctx: Optional[DeviceContext] = None,
) raises -> _Designed[dtype, 2 * order, output] where (
    dtype.is_floating_point() and order >= 1
):
    """A digital Butterworth bandpass or bandstop filter across
    `wn = (low, high)`, both as fractions of Nyquist.
    `scipy.signal.butter(order, (low, high), btype)`.

    `lp2bp`/`lp2bs` split every prototype root in two, so the result has
    order `2 * order` -- named in the return type, which is why this is an
    overload rather than a `btype` on the one above.

    Parameters:
        dtype: The floating-point dtype of the returned coefficients.
        order: The order of the lowpass prototype, at least 1; the result
            has order `2 * order`, so `b` and `a` are `2 * order + 1` long.
        output: `OUTPUT_BA` (the default), `OUTPUT_ZPK` or `OUTPUT_SOS`,
            SciPy's `output=`.

    Args:
        wn: The `(low, high)` band edges as fractions of Nyquist, an
            increasing pair strictly inside `(0, 1)`.
        btype: `"bandpass"` (the default) or `"bandstop"`.
        ctx: The device `b` and `a` are uploaded to; `None` puts them in
            host memory.

    Returns:
        The design in the form `output` selects: a `TransferFunction` of
        `b` and `a` (`OUTPUT_BA`), a `ZerosPolesGain` (`OUTPUT_ZPK`), or the
        second-order sections as a `(sections, 6)` tensor (`OUTPUT_SOS`),
        paired as `zpk2sos` pairs them.

    Raises:
        If `btype` is not `"bandpass"` or `"bandstop"`, if `wn` is not an
        increasing pair inside `(0, 1)`, or if the upload to
        `ctx` fails.
    """
    _check_band("butter", btype, wn)
    return _deliver[dtype, 2 * order, output](
        _design(_buttap(order), wn[0], wn[1], btype), ctx
    )


def cheby1[
    dtype: DType, order: Int, output: Int = OUTPUT_BA
](
    rp: Float64,
    wn: Float64,
    btype: StaticString = "lowpass",
    ctx: Optional[DeviceContext] = None,
) raises -> _Designed[dtype, order, output] where (
    dtype.is_floating_point() and order >= 1
):
    """A digital Chebyshev type I filter: `rp` dB of equiripple in the
    passband, monotone stopband. `scipy.signal.cheby1(order, rp, wn,
    btype)`.

    `wn` is the edge where the response last leaves the ripple band, not
    the half-power point, which is SciPy's convention too. `rp` must be
    positive; `"lowpass"` or `"highpass"` here, a pair for the band forms.

    Parameters:
        dtype: The floating-point dtype of the returned coefficients.
        order: The order of the filter, at least 1; `b` and `a` are each
            `order + 1` long.
        output: `OUTPUT_BA` (the default), `OUTPUT_ZPK` or `OUTPUT_SOS`,
            SciPy's `output=`.

    Args:
        rp: The maximum passband ripple in dB, positive.
        wn: The critical frequency as a fraction of Nyquist, strictly
            inside `(0, 1)`.
        btype: `"lowpass"` (the default) or `"highpass"`.
        ctx: The device `b` and `a` are uploaded to; `None` puts them in
            host memory.

    Returns:
        The design in the form `output` selects: a `TransferFunction` of
        `b` and `a` (`OUTPUT_BA`), a `ZerosPolesGain` (`OUTPUT_ZPK`), or the
        second-order sections as a `(sections, 6)` tensor (`OUTPUT_SOS`),
        paired as `zpk2sos` pairs them.

    Raises:
        If `btype` is not `"lowpass"` or `"highpass"`, if `wn` is outside
        `(0, 1)`, or if the upload to `ctx` fails.
    """
    _check_edge("cheby1", btype, wn)
    return _deliver[dtype, order, output](
        _design(_cheb1ap(order, rp), wn, 0.0, btype), ctx
    )


def cheby1[
    dtype: DType, order: Int, output: Int = OUTPUT_BA
](
    rp: Float64,
    wn: Tuple[Float64, Float64],
    btype: StaticString = "bandpass",
    ctx: Optional[DeviceContext] = None,
) raises -> _Designed[dtype, 2 * order, output] where (
    dtype.is_floating_point() and order >= 1
):
    """A digital Chebyshev type I bandpass or bandstop filter at order
    `2 * order`. `scipy.signal.cheby1(order, rp, (low, high), btype)`.

    Parameters:
        dtype: The floating-point dtype of the returned coefficients.
        order: The order of the lowpass prototype, at least 1; the result
            has order `2 * order`, so `b` and `a` are `2 * order + 1` long.
        output: `OUTPUT_BA` (the default), `OUTPUT_ZPK` or `OUTPUT_SOS`,
            SciPy's `output=`.

    Args:
        rp: The maximum passband ripple in dB, positive.
        wn: The `(low, high)` band edges as fractions of Nyquist, an
            increasing pair strictly inside `(0, 1)`.
        btype: `"bandpass"` (the default) or `"bandstop"`.
        ctx: The device `b` and `a` are uploaded to; `None` puts them in
            host memory.

    Returns:
        The design in the form `output` selects: a `TransferFunction` of
        `b` and `a` (`OUTPUT_BA`), a `ZerosPolesGain` (`OUTPUT_ZPK`), or the
        second-order sections as a `(sections, 6)` tensor (`OUTPUT_SOS`),
        paired as `zpk2sos` pairs them.

    Raises:
        If `btype` is not `"bandpass"` or `"bandstop"`, if `wn` is not an
        increasing pair inside `(0, 1)`, or if the upload to
        `ctx` fails.
    """
    _check_band("cheby1", btype, wn)
    return _deliver[dtype, 2 * order, output](
        _design(_cheb1ap(order, rp), wn[0], wn[1], btype), ctx
    )


def cheby2[
    dtype: DType, order: Int, output: Int = OUTPUT_BA
](
    rs: Float64,
    wn: Float64,
    btype: StaticString = "lowpass",
    ctx: Optional[DeviceContext] = None,
) raises -> _Designed[dtype, order, output] where (
    dtype.is_floating_point() and order >= 1
):
    """A digital Chebyshev type II filter: flat passband, `rs` dB of
    equiripple attenuation in the stopband. `scipy.signal.cheby2(order, rs,
    wn, btype)`.

    `wn` is the edge where the stopband attenuation first reaches `rs`.
    Unlike type I this one has finite zeros, so `b` is not a scaled
    `(1 + z)^order`; an odd order has one fewer zero than pole.

    Parameters:
        dtype: The floating-point dtype of the returned coefficients.
        order: The order of the filter, at least 1; `b` and `a` are each
            `order + 1` long.
        output: `OUTPUT_BA` (the default), `OUTPUT_ZPK` or `OUTPUT_SOS`,
            SciPy's `output=`.

    Args:
        rs: The minimum stopband attenuation in dB, positive.
        wn: The critical frequency as a fraction of Nyquist, strictly
            inside `(0, 1)`.
        btype: `"lowpass"` (the default) or `"highpass"`.
        ctx: The device `b` and `a` are uploaded to; `None` puts them in
            host memory.

    Returns:
        The design in the form `output` selects: a `TransferFunction` of
        `b` and `a` (`OUTPUT_BA`), a `ZerosPolesGain` (`OUTPUT_ZPK`), or the
        second-order sections as a `(sections, 6)` tensor (`OUTPUT_SOS`),
        paired as `zpk2sos` pairs them.

    Raises:
        If `btype` is not `"lowpass"` or `"highpass"`, if `wn` is outside
        `(0, 1)`, or if the upload to `ctx` fails.
    """
    _check_edge("cheby2", btype, wn)
    return _deliver[dtype, order, output](
        _design(_cheb2ap(order, rs), wn, 0.0, btype), ctx
    )


def cheby2[
    dtype: DType, order: Int, output: Int = OUTPUT_BA
](
    rs: Float64,
    wn: Tuple[Float64, Float64],
    btype: StaticString = "bandpass",
    ctx: Optional[DeviceContext] = None,
) raises -> _Designed[dtype, 2 * order, output] where (
    dtype.is_floating_point() and order >= 1
):
    """A digital Chebyshev type II bandpass or bandstop filter at order
    `2 * order`. `scipy.signal.cheby2(order, rs, (low, high), btype)`.

    Parameters:
        dtype: The floating-point dtype of the returned coefficients.
        order: The order of the lowpass prototype, at least 1; the result
            has order `2 * order`, so `b` and `a` are `2 * order + 1` long.
        output: `OUTPUT_BA` (the default), `OUTPUT_ZPK` or `OUTPUT_SOS`,
            SciPy's `output=`.

    Args:
        rs: The minimum stopband attenuation in dB, positive.
        wn: The `(low, high)` band edges as fractions of Nyquist, an
            increasing pair strictly inside `(0, 1)`.
        btype: `"bandpass"` (the default) or `"bandstop"`.
        ctx: The device `b` and `a` are uploaded to; `None` puts them in
            host memory.

    Returns:
        The design in the form `output` selects: a `TransferFunction` of
        `b` and `a` (`OUTPUT_BA`), a `ZerosPolesGain` (`OUTPUT_ZPK`), or the
        second-order sections as a `(sections, 6)` tensor (`OUTPUT_SOS`),
        paired as `zpk2sos` pairs them.

    Raises:
        If `btype` is not `"bandpass"` or `"bandstop"`, if `wn` is not an
        increasing pair inside `(0, 1)`, or if the upload to
        `ctx` fails.
    """
    _check_band("cheby2", btype, wn)
    return _deliver[dtype, 2 * order, output](
        _design(_cheb2ap(order, rs), wn[0], wn[1], btype), ctx
    )


def ellip[
    dtype: DType, order: Int, output: Int = OUTPUT_BA
](
    rp: Float64,
    rs: Float64,
    wn: Float64,
    btype: StaticString = "lowpass",
    ctx: Optional[DeviceContext] = None,
) raises -> _Designed[dtype, order, output] where (
    dtype.is_floating_point() and order >= 1
):
    """A digital elliptic (Cauer) filter: `rp` dB of passband ripple and
    `rs` dB of stopband attenuation in the fewest poles of the four.
    `scipy.signal.ellip(order, rp, rs, wn, btype)`.

    The steepest transition an IIR filter of this order can have, paid for
    with ripple in both bands and the worst phase response of the four.
    Raises when `rp` and `rs` cannot both be met at any order.

    Parameters:
        dtype: The floating-point dtype of the returned coefficients.
        order: The order of the filter, at least 1; `b` and `a` are each
            `order + 1` long.
        output: `OUTPUT_BA` (the default), `OUTPUT_ZPK` or `OUTPUT_SOS`,
            SciPy's `output=`.

    Args:
        rp: The maximum passband ripple in dB, positive.
        rs: The minimum stopband attenuation in dB, positive.
        wn: The critical frequency as a fraction of Nyquist, strictly
            inside `(0, 1)`.
        btype: `"lowpass"` (the default) or `"highpass"`.
        ctx: The device `b` and `a` are uploaded to; `None` puts them in
            host memory.

    Returns:
        The design in the form `output` selects: a `TransferFunction` of
        `b` and `a` (`OUTPUT_BA`), a `ZerosPolesGain` (`OUTPUT_ZPK`), or the
        second-order sections as a `(sections, 6)` tensor (`OUTPUT_SOS`),
        paired as `zpk2sos` pairs them.

    Raises:
        If `btype` is not `"lowpass"` or `"highpass"`, if `wn` is outside
        `(0, 1)`, if no filter of this order meets both `rp` and `rs`, or
        if the upload to `ctx` fails.
    """
    _check_edge("ellip", btype, wn)
    return _deliver[dtype, order, output](
        _design(_ellipap(order, rp, rs), wn, 0.0, btype), ctx
    )


def ellip[
    dtype: DType, order: Int, output: Int = OUTPUT_BA
](
    rp: Float64,
    rs: Float64,
    wn: Tuple[Float64, Float64],
    btype: StaticString = "bandpass",
    ctx: Optional[DeviceContext] = None,
) raises -> _Designed[dtype, 2 * order, output] where (
    dtype.is_floating_point() and order >= 1
):
    """A digital elliptic bandpass or bandstop filter at order
    `2 * order`. `scipy.signal.ellip(order, rp, rs, (low, high), btype)`.

    Parameters:
        dtype: The floating-point dtype of the returned coefficients.
        order: The order of the lowpass prototype, at least 1; the result
            has order `2 * order`, so `b` and `a` are `2 * order + 1` long.
        output: `OUTPUT_BA` (the default), `OUTPUT_ZPK` or `OUTPUT_SOS`,
            SciPy's `output=`.

    Args:
        rp: The maximum passband ripple in dB, positive.
        rs: The minimum stopband attenuation in dB, positive.
        wn: The `(low, high)` band edges as fractions of Nyquist, an
            increasing pair strictly inside `(0, 1)`.
        btype: `"bandpass"` (the default) or `"bandstop"`.
        ctx: The device `b` and `a` are uploaded to; `None` puts them in
            host memory.

    Returns:
        The design in the form `output` selects: a `TransferFunction` of
        `b` and `a` (`OUTPUT_BA`), a `ZerosPolesGain` (`OUTPUT_ZPK`), or the
        second-order sections as a `(sections, 6)` tensor (`OUTPUT_SOS`),
        paired as `zpk2sos` pairs them.

    Raises:
        If `btype` is not `"bandpass"` or `"bandstop"`, if `wn` is not an
        increasing pair inside `(0, 1)`, if no filter meets both `rp` and
        `rs`, or if the upload to `ctx` fails.
    """
    _check_band("ellip", btype, wn)
    return _deliver[dtype, 2 * order, output](
        _design(_ellipap(order, rp, rs), wn[0], wn[1], btype), ctx
    )


def iirfilter[
    dtype: DType, order: Int, output: Int = OUTPUT_BA
](
    wn: Float64,
    rp: Float64 = 1.0,
    rs: Float64 = 40.0,
    btype: StaticString = "lowpass",
    ftype: StaticString = "butter",
    ctx: Optional[DeviceContext] = None,
) raises -> _Designed[dtype, order, output] where (
    dtype.is_floating_point() and order >= 1
):
    """The family named rather than called: `scipy.signal.iirfilter(order,
    wn, rp, rs, btype, ftype=...)` over `"butter"`, `"cheby1"`,
    `"cheby2"` and `"ellip"`.

    Same answer as calling the family directly -- this only chooses the
    prototype -- so it exists for a caller whose filter type is a
    configuration value rather than a decision in the source. `rp` is read
    only by `"cheby1"` and `"ellip"`, `rs` only by `"cheby2"` and
    `"ellip"`, and both keep SciPy-shaped defaults so a `"butter"` call
    names neither. An unknown `ftype` raises rather than defaulting,
    since a typo that silently designed a different filter would be worse
    than a failure.

    Parameters:
        dtype: The floating-point dtype of the returned coefficients.
        order: The order of the filter, at least 1; `b` and `a` are each
            `order + 1` long.
        output: `OUTPUT_BA` (the default), `OUTPUT_ZPK` or `OUTPUT_SOS`,
            SciPy's `output=`.

    Args:
        wn: The critical frequency as a fraction of Nyquist, strictly
            inside `(0, 1)`.
        rp: The passband ripple in dB, read only by `"cheby1"` and
            `"ellip"`.
        rs: The stopband attenuation in dB, read only by `"cheby2"` and
            `"ellip"`.
        btype: `"lowpass"` (the default) or `"highpass"`.
        ftype: The family: `"butter"` (the default), `"cheby1"`,
            `"cheby2"` or `"ellip"`.
        ctx: The device `b` and `a` are uploaded to; `None` puts them in
            host memory.

    Returns:
        The design in the form `output` selects: a `TransferFunction` of
        `b` and `a` (`OUTPUT_BA`), a `ZerosPolesGain` (`OUTPUT_ZPK`), or the
        second-order sections as a `(sections, 6)` tensor (`OUTPUT_SOS`),
        paired as `zpk2sos` pairs them.

    Raises:
        If `btype` is not `"lowpass"` or `"highpass"`, if `wn` is outside
        `(0, 1)`, if `ftype` is unknown, if an `"ellip"` design cannot meet
        both `rp` and `rs`, or if the upload to `ctx` fails.
    """
    _check_edge("iirfilter", btype, wn)
    return _deliver[dtype, order, output](
        _design(_prototype(ftype, order, rp, rs), wn, 0.0, btype), ctx
    )


def iirfilter[
    dtype: DType, order: Int, output: Int = OUTPUT_BA
](
    wn: Tuple[Float64, Float64],
    rp: Float64 = 1.0,
    rs: Float64 = 40.0,
    btype: StaticString = "bandpass",
    ftype: StaticString = "butter",
    ctx: Optional[DeviceContext] = None,
) raises -> _Designed[dtype, 2 * order, output] where (
    dtype.is_floating_point() and order >= 1
):
    """The named-`ftype` front door for the band forms, at order
    `2 * order`. `scipy.signal.iirfilter(order, (low, high), rp, rs,
    btype, ftype=...)`.

    Parameters:
        dtype: The floating-point dtype of the returned coefficients.
        order: The order of the lowpass prototype, at least 1; the result
            has order `2 * order`, so `b` and `a` are `2 * order + 1` long.
        output: `OUTPUT_BA` (the default), `OUTPUT_ZPK` or `OUTPUT_SOS`,
            SciPy's `output=`.

    Args:
        wn: The `(low, high)` band edges as fractions of Nyquist, an
            increasing pair strictly inside `(0, 1)`.
        rp: The passband ripple in dB, read only by `"cheby1"` and
            `"ellip"`.
        rs: The stopband attenuation in dB, read only by `"cheby2"` and
            `"ellip"`.
        btype: `"bandpass"` (the default) or `"bandstop"`.
        ftype: The family: `"butter"` (the default), `"cheby1"`,
            `"cheby2"` or `"ellip"`.
        ctx: The device `b` and `a` are uploaded to; `None` puts them in
            host memory.

    Returns:
        The design in the form `output` selects: a `TransferFunction` of
        `b` and `a` (`OUTPUT_BA`), a `ZerosPolesGain` (`OUTPUT_ZPK`), or the
        second-order sections as a `(sections, 6)` tensor (`OUTPUT_SOS`),
        paired as `zpk2sos` pairs them.

    Raises:
        If `btype` is not `"bandpass"` or `"bandstop"`, if `wn` is not an
        increasing pair inside `(0, 1)`, if `ftype` is unknown, if an
        `"ellip"` design cannot meet both `rp` and `rs`, or if the upload
        to `ctx` fails.
    """
    _check_band("iirfilter", btype, wn)
    return _deliver[dtype, 2 * order, output](
        _design(_prototype(ftype, order, rp, rs), wn[0], wn[1], btype), ctx
    )


# --------------------------------------------------------------------------
# Zeros, poles and gain; second-order sections
# --------------------------------------------------------------------------

comptime OUTPUT_BA = 0
"""`output=` selector for the design routines: `(b, a)` as a
`TransferFunction`, SciPy's `output="ba"` and the default."""
comptime OUTPUT_ZPK = 1
"""`output=` selector: the zeros, poles and gain as a `ZerosPolesGain`,
SciPy's `output="zpk"`."""
comptime OUTPUT_SOS = 2
"""`output=` selector: second-order sections, a `(sections, 6)` tensor,
SciPy's `output="sos"` -- the form to run a filter above order eight in."""


struct ZerosPolesGain(Movable):
    """A filter as its zeros, poles and gain, SciPy's `(z, p, k)`.

    The roots are host lists with real and imaginary parts apart, the split
    `zpk2tf` already takes: a `Tensor` is `dtype`-monomorphic and holds no
    complex value, and a zero-pole-gain form is a few dozen numbers the
    design arithmetic produces on the host anyway.
    """

    var zeros_re: List[Float64]
    """Real parts of the zeros."""
    var zeros_im: List[Float64]
    """Imaginary parts of the zeros; a conjugate pair is two entries."""
    var poles_re: List[Float64]
    """Real parts of the poles."""
    var poles_im: List[Float64]
    """Imaginary parts of the poles; a conjugate pair is two entries."""
    var gain: Float64
    """The system gain `k`."""

    def __init__(
        out self,
        var zeros_re: List[Float64],
        var zeros_im: List[Float64],
        var poles_re: List[Float64],
        var poles_im: List[Float64],
        gain: Float64,
    ):
        """Build from the four root lists and the gain.

        Args:
            zeros_re: Real parts of the zeros.
            zeros_im: Imaginary parts of the zeros, as many as `zeros_re`.
            poles_re: Real parts of the poles.
            poles_im: Imaginary parts of the poles, as many as `poles_re`.
            gain: The system gain.
        """
        self.zeros_re = zeros_re^
        self.zeros_im = zeros_im^
        self.poles_re = poles_re^
        self.poles_im = poles_im^
        self.gain = gain


comptime _Designed[dtype: DType, order: Int, output: Int] = (
    TransferFunction[dtype, order] if output
    == OUTPUT_BA else (
        ZerosPolesGain if output
        == OUTPUT_ZPK else Static[dtype, (order + 1) // 2, 6]
    )
)
"""What a design of `order` returns under each `output` selector."""


def _deliver[
    dtype: DType, order: Int, output: Int
](var f: _Zpk, ctx: Optional[DeviceContext]) raises -> _Designed[
    dtype, order, output
] where dtype.is_floating_point():
    """A digital design's zeros, poles and gain in the form `output` asks
    for. The sections come straight from the roots, never through the
    expanded polynomials, which is the point of asking for them."""
    comptime assert (
        output == OUTPUT_BA or output == OUTPUT_ZPK or output == OUTPUT_SOS
    ), "output is OUTPUT_BA, OUTPUT_ZPK or OUTPUT_SOS"
    comptime if output == OUTPUT_BA:
        return rebind_var[_Designed[dtype, order, output]](
            _to_transfer_function[dtype=dtype, order=order](_zpk2tf(f^), ctx)
        )
    elif output == OUTPUT_ZPK:
        return rebind_var[_Designed[dtype, order, output]](
            ZerosPolesGain(
                f.zr.copy(), f.zi.copy(), f.pr.copy(), f.pi.copy(), f.gain
            )
        )
    else:
        comptime sections = (order + 1) // 2
        return rebind_var[_Designed[dtype, order, output]](
            _sos_tensor[dtype, sections](_zpk2sos_rows(f), "design", ctx)
        )


def _sos_tensor[
    dtype: DType, sections: Int
](
    rows: List[Float64], who: StaticString, ctx: Optional[DeviceContext]
) raises -> Static[dtype, sections, 6]:
    """`sections` rows of six coefficients uploaded as a `(sections, 6)`
    tensor, raising when the pairing produced a different count."""
    if len(rows) != sections * 6:
        raise Error(
            who,
            ": the zeros and poles pair into ",
            len(rows) // 6,
            " sections, not ",
            sections,
        )
    var values = List[Scalar[dtype]](capacity=sections * 6)
    for i in range(sections * 6):
        values.append(Scalar[dtype](rows[i]))
    var device = ctx.value() if ctx else DeviceContext(api="cpu")
    return Static[dtype, sections, 6](values^, device)


@fieldwise_init
struct _Cx(Copyable, ImplicitlyCopyable, Movable):
    """A host complex number for the pairing arithmetic."""

    var re: Float64
    var im: Float64

    def is_real(self) -> Bool:
        return self.im == 0.0

    def conj(self) -> _Cx:
        return _Cx(self.re, -self.im)

    def abs(self) -> Float64:
        return _hypot(self.re, self.im)


def _distance(a: _Cx, b: _Cx) -> Float64:
    return _hypot(a.re - b.re, a.im - b.im)


def _cplxreal(z: List[_Cx]) raises -> List[_Cx]:
    """`scipy.signal._filter_design._cplxreal`, concatenated: one member of
    each conjugate pair (the one with positive imaginary part, averaged
    with its partner's conjugate), then the real roots, each group sorted
    by real part and then by the size of the imaginary part."""
    var tol = 100.0 * _EPS
    var sorted = z.copy()
    # Lexicographic by (real, |imag|); an insertion sort, the lists being
    # a filter's worth of roots.
    for i in range(1, len(sorted)):
        var j = i
        while j > 0 and (
            sorted[j].re < sorted[j - 1].re
            or (
                sorted[j].re == sorted[j - 1].re
                and abs(sorted[j].im) < abs(sorted[j - 1].im)
            )
        ):
            var t = sorted[j]
            sorted[j] = sorted[j - 1]
            sorted[j - 1] = t
            j -= 1
    var reals = List[_Cx]()
    var positive = List[_Cx]()
    var negative = List[_Cx]()
    for i in range(len(sorted)):
        var x = sorted[i]
        if abs(x.im) <= tol * x.abs():
            reals.append(_Cx(x.re, 0.0))
        elif x.im > 0:
            positive.append(x)
        else:
            negative.append(x)
    if len(positive) != len(negative):
        raise Error("zpk2sos: a complex root has no matching conjugate")
    var out = List[_Cx](capacity=len(positive) + len(reals))
    for i in range(len(positive)):
        var p = positive[i]
        var n = negative[i]
        if _distance(p, n.conj()) > tol * n.abs():
            raise Error("zpk2sos: a complex root has no matching conjugate")
        out.append(_Cx((p.re + n.re) / 2.0, (p.im - n.im) / 2.0))
    for i in range(len(reals)):
        out.append(reals[i])
    return out^


def _worst_pole(p: List[_Cx]) -> Int:
    """The pole nearest the unit circle: `argmin |1 - |p||`, first on a
    tie, as `numpy.argmin` breaks it."""
    var best = 0
    for i in range(1, len(p)):
        if abs(1.0 - p[i].abs()) < abs(1.0 - p[best].abs()):
            best = i
    return best


def _nearest(fro: List[_Cx], to: _Cx, which: StaticString) raises -> Int:
    """`_nearest_real_complex_idx`: the closest element of `fro` to `to`
    that is real, complex, or either."""
    var best = -1
    for i in range(len(fro)):
        var ok = which == "any" or (
            fro[i].is_real() if which == "real" else not fro[i].is_real()
        )
        if not ok:
            continue
        if best < 0 or _distance(fro[i], to) < _distance(fro[best], to):
            best = i
    if best < 0:
        raise Error("zpk2sos: no ", which, " root left to pair")
    return best


def _section(z: List[_Cx], p: List[_Cx]) -> List[Float64]:
    """`_single_zpksos`: up to two zeros and two poles as one row
    `[b0, b1, b2, a0, a1, a2]`, each polynomial right-aligned so a missing
    root leaves a leading zero."""
    var zr = List[Float64]()
    var zi = List[Float64]()
    for i in range(len(z)):
        zr.append(z[i].re)
        zi.append(z[i].im)
    var pr = List[Float64]()
    var pi = List[Float64]()
    for i in range(len(p)):
        pr.append(p[i].re)
        pi.append(p[i].im)
    var b = _poly_from_roots(zr, zi)[0].copy()
    var a = _poly_from_roots(pr, pi)[0].copy()
    var row = List[Float64](length=6, fill=0.0)
    for j in range(len(b)):
        row[3 - len(b) + j] = b[j]
    for j in range(len(a)):
        row[6 - len(a) + j] = a[j]
    return row^


def _pop(mut xs: List[_Cx], i: Int) -> _Cx:
    var x = xs[i]
    _ = xs.pop(i)
    return x


def _count_real(xs: List[_Cx]) -> Int:
    var count = 0
    for i in range(len(xs)):
        if xs[i].is_real():
            count += 1
    return count


def _zpk2sos_rows(f: _Zpk) raises -> List[Float64]:
    """`scipy.signal.zpk2sos(z, p, k)` with its default digital
    `pairing="nearest"`, as flat rows of six: the poles nearest the unit
    circle go last, each paired with the zeros nearest them, and the gain
    rides on the first section. Transcribed branch for branch, so the
    sections and their order are SciPy's."""
    var z = List[_Cx]()
    for i in range(len(f.zr)):
        z.append(_Cx(f.zr[i], f.zi[i]))
    var p = List[_Cx]()
    for i in range(len(f.pr)):
        p.append(_Cx(f.pr[i], f.pi[i]))
    if len(z) == 0 and len(p) == 0:
        return [f.gain, 0.0, 0.0, 1.0, 0.0, 0.0]
    while len(p) < len(z):
        p.append(_Cx(0.0, 0.0))
    while len(z) < len(p):
        z.append(_Cx(0.0, 0.0))
    var sections = (len(p) + 1) // 2
    if len(p) % 2 == 1:
        p.append(_Cx(0.0, 0.0))
        z.append(_Cx(0.0, 0.0))
    z = _cplxreal(z)
    p = _cplxreal(p)
    var rows = List[List[Float64]]()
    for _ in range(sections):
        rows.append(List[Float64]())
    for si in range(sections - 1, -1, -1):
        var p1 = _pop(p, _worst_pole(p))
        if p1.is_real() and _count_real(p) == 0:
            var z1 = _pop(z, _nearest(z, p1, "real"))
            rows[si] = _section([z1, _Cx(0.0, 0.0)], [p1, _Cx(0.0, 0.0)])
        elif (
            len(p) + 1 == len(z)
            and not p1.is_real()
            and _count_real(p) == 1
            and _count_real(z) == 1
        ):
            var z1 = _pop(z, _nearest(z, p1, "complex"))
            rows[si] = _section([z1, z1.conj()], [p1, p1.conj()])
        else:
            var p2: _Cx
            if p1.is_real():
                var best = -1
                for i in range(len(p)):
                    if p[i].is_real() and (
                        best < 0
                        or abs(1.0 - p[i].abs()) < abs(1.0 - p[best].abs())
                    ):
                        best = i
                p2 = _pop(p, best)
            else:
                p2 = p1.conj()
            if len(z) > 0:
                var z1 = _pop(z, _nearest(z, p1, "any"))
                if not z1.is_real():
                    rows[si] = _section([z1, z1.conj()], [p1, p2])
                elif len(z) > 0:
                    var z2 = _pop(z, _nearest(z, p1, "real"))
                    rows[si] = _section([z1, z2], [p1, p2])
                else:
                    rows[si] = _section([z1], [p1, p2])
            else:
                rows[si] = _section(List[_Cx](), [p1, p2])
    var flat = List[Float64](capacity=sections * 6)
    for si in range(sections):
        for j in range(6):
            flat.append(rows[si][j] * (f.gain if si == 0 and j < 3 else 1.0))
    return flat^


def _poly_roots(c: List[Float64]) raises -> List[_Cx]:
    """The roots of the descending polynomial `c`, `numpy.roots`' contract
    and algorithm: leading zeros dropped, trailing zeros returned as roots
    at the origin, and the rest the eigenvalues of the companion matrix --
    balanced, then Francis double-shift QR on the host. That route is
    backward stable, so the roots are those of a polynomial a rounding
    away from `c` even where a root is multiple and each copy of it is
    only good to `eps^(1/m)`, and every complex root comes with its exact
    conjugate. **Host, tier 2.**
    """
    var start = 0
    while start < len(c) and c[start] == 0.0:
        start += 1
    var stop = len(c)
    var at_zero = 0
    while stop > start + 1 and c[stop - 1] == 0.0:
        stop -= 1
        at_zero += 1
    var roots = List[_Cx]()
    var degree = stop - start - 1
    if degree >= 1:
        var h = List[Float64](length=degree * degree, fill=0.0)
        for j in range(degree):
            h[j] = -c[start + 1 + j] / c[start]
        for i in range(1, degree):
            h[i * degree + i - 1] = 1.0
        _balance(h, degree)
        roots = _hqr(h, degree)
    for _ in range(at_zero):
        roots.append(_Cx(0.0, 0.0))
    return roots^


def _balance(mut h: List[Float64], n: Int):
    """EISPACK `balanc` without permutations: scale rows and columns by
    powers of two until each row and its column have comparable norms,
    which is what makes the companion matrix's eigenvalues accurate."""
    var radix = 2.0
    var sqrdx = radix * radix
    var done = False
    while not done:
        done = True
        for i in range(n):
            var r = 0.0
            var c = 0.0
            for j in range(n):
                if j != i:
                    c += abs(h[j * n + i])
                    r += abs(h[i * n + j])
            if c != 0.0 and r != 0.0:
                var g = r / radix
                var f = 1.0
                var s = c + r
                while c < g:
                    f *= radix
                    c *= sqrdx
                while c > g * sqrdx:
                    f /= radix
                    c /= sqrdx
                if (c + r) / f < 0.95 * s:
                    done = False
                    g = 1.0 / f
                    for j in range(n):
                        h[i * n + j] *= g
                    for j in range(n):
                        h[j * n + i] *= f


def _hqr(mut a: List[Float64], n: Int) raises -> List[_Cx]:
    """The eigenvalues of the upper Hessenberg `a` (row-major `n x n`,
    destroyed): Francis double-shift QR with exceptional shifts, the
    EISPACK `hqr` iteration. Host, tier 2."""
    var wr = List[Float64](length=n, fill=0.0)
    var wi = List[Float64](length=n, fill=0.0)
    var anorm = 0.0
    for i in range(n):
        for j in range(max(i - 1, 0), n):
            anorm += abs(a[i * n + j])
    var nn = n - 1
    var t = 0.0
    var p = 0.0
    var q = 0.0
    var r = 0.0
    var s: Float64
    var w: Float64
    var x: Float64
    var y: Float64
    var z: Float64
    while nn >= 0:
        var its = 0
        var l: Int
        while True:
            l = nn
            while l >= 1:
                s = abs(a[(l - 1) * n + l - 1]) + abs(a[l * n + l])
                if s == 0.0:
                    s = anorm
                if abs(a[l * n + l - 1]) + s == s:
                    a[l * n + l - 1] = 0.0
                    break
                l -= 1
            x = a[nn * n + nn]
            if l == nn:
                wr[nn] = x + t
                wi[nn] = 0.0
                nn -= 1
                break
            y = a[(nn - 1) * n + nn - 1]
            w = a[nn * n + nn - 1] * a[(nn - 1) * n + nn]
            if l == nn - 1:
                p = 0.5 * (y - x)
                q = p * p + w
                z = _sqrt(abs(q))
                x += t
                if q >= 0.0:
                    z = p + (z if p >= 0 else -z)
                    wr[nn - 1] = x + z
                    wr[nn] = wr[nn - 1]
                    if z != 0.0:
                        wr[nn] = x - w / z
                    wi[nn - 1] = 0.0
                    wi[nn] = 0.0
                else:
                    wr[nn - 1] = x + p
                    wr[nn] = x + p
                    wi[nn - 1] = -z
                    wi[nn] = z
                nn -= 2
                break
            if its == 60:
                raise Error("roots: the QR iteration did not converge")
            if its == 10 or its == 20:
                t += x
                for i in range(nn + 1):
                    a[i * n + i] -= x
                s = abs(a[nn * n + nn - 1]) + abs(a[(nn - 1) * n + nn - 2])
                x = 0.75 * s
                y = x
                w = -0.4375 * s * s
            its += 1
            var m = nn - 2
            while m >= l:
                z = a[m * n + m]
                r = x - z
                s = y - z
                p = (r * s - w) / a[(m + 1) * n + m] + a[m * n + m + 1]
                q = a[(m + 1) * n + m + 1] - z - r - s
                r = a[(m + 2) * n + m + 1]
                s = abs(p) + abs(q) + abs(r)
                p /= s
                q /= s
                r /= s
                if m == l:
                    break
                var u = abs(a[m * n + m - 1]) * (abs(q) + abs(r))
                var v = abs(p) * (
                    abs(a[(m - 1) * n + m - 1])
                    + abs(z)
                    + abs(a[(m + 1) * n + m + 1])
                )
                if u + v == v:
                    break
                m -= 1
            for i in range(m + 2, nn + 1):
                a[i * n + i - 2] = 0.0
                if i != m + 2:
                    a[i * n + i - 3] = 0.0
            var k = m
            while k <= nn - 1:
                if k != m:
                    p = a[k * n + k - 1]
                    q = a[(k + 1) * n + k - 1]
                    r = 0.0
                    if k != nn - 1:
                        r = a[(k + 2) * n + k - 1]
                    x = abs(p) + abs(q) + abs(r)
                    if x != 0.0:
                        p /= x
                        q /= x
                        r /= x
                var mag = _sqrt(p * p + q * q + r * r)
                s = mag if p >= 0 else -mag
                if s != 0.0:
                    if k == m:
                        if l != m:
                            a[k * n + k - 1] = -a[k * n + k - 1]
                    else:
                        a[k * n + k - 1] = -s * x
                    p += s
                    x = p / s
                    y = q / s
                    z = r / s
                    q /= p
                    r /= p
                    for j in range(k, nn + 1):
                        p = a[k * n + j] + q * a[(k + 1) * n + j]
                        if k != nn - 1:
                            p += r * a[(k + 2) * n + j]
                            a[(k + 2) * n + j] -= p * z
                        a[(k + 1) * n + j] -= p * y
                        a[k * n + j] -= p * x
                    var mmin = nn if nn < k + 3 else k + 3
                    for i in range(l, mmin + 1):
                        p = x * a[i * n + k] + y * a[i * n + k + 1]
                        if k != nn - 1:
                            p += z * a[i * n + k + 2]
                            a[i * n + k + 2] -= p * r
                        a[i * n + k + 1] -= p * q
                        a[i * n + k] -= p
                k += 1
            if l >= nn - 1:
                break
    var out = List[_Cx](capacity=n)
    for i in range(n):
        out.append(_Cx(wr[i], wi[i]))
    return out^


def tf2zpk[
    A: TensorLike, B: TensorLike
](b: A, a: B) raises -> ZerosPolesGain where (
    A.dtype.is_floating_point()
    and B.dtype == A.dtype
    and A.LayoutType.rank == 1
    and B.LayoutType.rank == 1
):
    """The zeros, poles and gain of the transfer function `(b, a)`.
    `scipy.signal.tf2zpk(b, a)`.

    `scipy.signal.normalize` first -- leading zeros of `a` dropped, both
    divided by `a[0]`, and leading numerator coefficients at or below
    `1e-14` dropped -- then the gain is the leading numerator coefficient
    and the roots are `numpy.roots` of each polynomial: the balanced
    companion matrix's eigenvalues by Francis QR, on the host. The
    inverse is `zpk2tf`. A multiple root, such as the `n` zeros a
    Butterworth design puts at `-1`, is recovered only to about
    `eps^(1/m)`, as it is from NumPy; the design routines' `OUTPUT_ZPK` and
    `OUTPUT_SOS` never go through this, and are exact.

    Parameters:
        A: The tensor type of `b`, rank 1, floating-point.
        B: The tensor type of `a`, rank 1, with `b`'s dtype.

    Args:
        b: The numerator, descending in powers of the delay.
        a: The denominator, descending, not all zero.

    Returns:
        A `ZerosPolesGain` with the zeros, poles and gain.

    Raises:
        If `a` is all zeros, or the root iteration fails to pair its roots.
    """
    var bh = b.to_host()
    var ah = a.to_host()
    var start = 0
    while start < len(ah) and ah[start] == 0:
        start += 1
    if start == len(ah):
        raise Error("tf2zpk: the denominator is all zeros")
    var lead = Float64(ah[start])
    var den = List[Float64](capacity=len(ah) - start)
    for i in range(start, len(ah)):
        den.append(Float64(ah[i]) / lead)
    var num = List[Float64](capacity=len(bh))
    var nstart = 0
    while nstart < len(bh) - 1 and abs(Float64(bh[nstart]) / lead) <= 1e-14:
        nstart += 1
    for i in range(nstart, len(bh)):
        num.append(Float64(bh[i]) / lead)
    var gain = num[0]
    var zeros = _poly_roots(num) if gain != 0.0 else List[_Cx]()
    var poles = _poly_roots(den)
    var zr = List[Float64]()
    var zi = List[Float64]()
    for i in range(len(zeros)):
        zr.append(zeros[i].re)
        zi.append(zeros[i].im)
    var pr = List[Float64]()
    var pi = List[Float64]()
    for i in range(len(poles)):
        pr.append(poles[i].re)
        pi.append(poles[i].im)
    return ZerosPolesGain(zr^, zi^, pr^, pi^, gain)


def zpk2sos[
    dtype: DType, sections: Int
](zpk: ZerosPolesGain, ctx: Optional[DeviceContext] = None) raises -> Static[
    dtype, sections, 6
] where (dtype.is_floating_point() and sections >= 1):
    """Second-order sections from zeros, poles and a gain.
    `scipy.signal.zpk2sos(z, p, k)` with its default digital pairing,
    `"nearest"`.

    SciPy's algorithm transcribed branch for branch: the pole nearest the
    unit circle goes in the last section with the zeros nearest it, and so
    on inward, and the gain rides on the first section's numerator. So the
    sections, their order and their coefficients are SciPy's. `sections`
    is `ceil(max(len(z), len(p)) / 2)`, SciPy's count; a different one
    raises, since it is a compile-time shape the roots cannot be checked
    against until they are read.

    Parameters:
        dtype: The floating-point dtype of the sections.
        sections: The number of sections, `(max(len(z), len(p)) + 1) // 2`.

    Args:
        zpk: The zeros, poles and gain, conjugate pairs complete.
        ctx: The device the sections are uploaded to; `None` puts them in
            host memory.

    Returns:
        A `(sections, 6)` tensor, one `[b0, b1, b2, a0, a1, a2]` row per
        section, the form `sosfilt` takes.

    Raises:
        If a complex root has no conjugate, if the roots pair into a
        different number of sections, or if the upload fails.
    """
    var f = _Zpk(
        zpk.zeros_re.copy(),
        zpk.zeros_im.copy(),
        zpk.poles_re.copy(),
        zpk.poles_im.copy(),
        zpk.gain,
    )
    return _sos_tensor[dtype, sections](_zpk2sos_rows(f), "zpk2sos", ctx)


def tf2sos[
    dtype: DType, order: Int
](
    tf: TransferFunction[dtype, order], ctx: Optional[DeviceContext] = None
) raises -> Static[dtype, (order + 1) // 2, 6] where (
    dtype.is_floating_point() and order >= 1
):
    """Second-order sections from a transfer function.
    `scipy.signal.tf2sos(b, a)`: `tf2zpk` then `zpk2sos`.

    The roots come from `tf2zpk`'s iteration, so a design's own
    `OUTPUT_SOS` is the better route when there is one: it pairs the exact
    roots and never forms the long polynomials whose coefficients a high
    order loses digits in.

    Parameters:
        dtype: The floating-point dtype of the coefficients.
        order: The filter order; there are `(order + 1) // 2` sections.

    Args:
        tf: The transfer function `(b, a)`.
        ctx: The device the sections are uploaded to; `None` puts them in
            host memory.

    Returns:
        A `((order + 1) // 2, 6)` tensor of sections.

    Raises:
        As `tf2zpk` and `zpk2sos` do.
    """
    var zpk = tf2zpk(tf.b, tf.a)
    var f = _Zpk(
        zpk.zeros_re.copy(),
        zpk.zeros_im.copy(),
        zpk.poles_re.copy(),
        zpk.poles_im.copy(),
        zpk.gain,
    )
    return _sos_tensor[dtype, (order + 1) // 2](_zpk2sos_rows(f), "tf2sos", ctx)


def sos2tf[
    T: TensorLike
](sos: T, ctx: Optional[DeviceContext] = None) raises -> TransferFunction[
    T.dtype, 2 * dim[T, 0]
] where (
    T.dtype.is_floating_point()
    and T.LayoutType.rank == 2
    and T.LayoutType.all_dims_known
    and dim[T, 1] == 6
):
    """The transfer function of a cascade of second-order sections.
    `scipy.signal.sos2tf(sos)`: the sections' numerators multiplied
    together, and their denominators.

    Parameters:
        T: The tensor type of `sos`, `(sections, 6)`.

    Args:
        sos: The sections, one `[b0, b1, b2, a0, a1, a2]` row each.
        ctx: The device `b` and `a` are uploaded to; `None` puts them in
            host memory.

    Returns:
        A `TransferFunction` of order `2 * sections`.

    Raises:
        If reading `sos` or the upload fails.
    """
    comptime sections = dim[T, 0]
    var table = sos.to_host()
    var b: List[Float64] = [1.0]
    var a: List[Float64] = [1.0]
    for s in range(sections):
        var bs = List[Float64]()
        var as_ = List[Float64]()
        for k in range(3):
            bs.append(Float64(table[s * 6 + k]))
            as_.append(Float64(table[s * 6 + 3 + k]))
        b = _poly_mul(b, bs)
        a = _poly_mul(a, as_)
    return _to_transfer_function[dtype=T.dtype, order=2 * sections](
        (b^, a^), ctx
    )


def _poly_mul(p: List[Float64], q: List[Float64]) -> List[Float64]:
    var out = List[Float64](length=len(p) + len(q) - 1, fill=0.0)
    for i in range(len(p)):
        for j in range(len(q)):
            out[i + j] += p[i] * q[j]
    return out^


def freqz[
    A: TensorLike,
    B: TensorLike,
    worN: Int = 512,
    gpu: Bool = False,
](b: A, a: B) raises -> FrequencyResponse[A.dtype, worN] where (
    (
        A.dtype.is_floating_point()
        and dim[A, 0] > 0
        and dim[B, 0] > 0
        and worN > 0
    )
    and A.LayoutType.rank == 1
    and A.LayoutType.all_dims_known
    and B.dtype == A.dtype
    and B.LayoutType.rank == 1
    and B.LayoutType.all_dims_known
):
    """The frequency response `H(e^{jw}) = B(e^{jw}) / A(e^{jw})` of the
    filter `(b, a)` at `worN` frequencies evenly spaced over `[0, pi)`.
    `scipy.signal.freqz(b, a, worN)`, with its default `whole=False` grid.

    One lane per frequency: both polynomials by complex Horner in
    `e^{-jw}`, then one complex division. `worN` is a parameter because it
    shapes the result; SciPy's default of 512 is kept.

    Parameters:
        A: The `TensorLike` type of `b`, rank 1 with a static length.
        B: The `TensorLike` type of `a`, rank 1 with a static length and
            the same dtype as `A`.
        worN: The number of frequencies on the grid, default 512.
        gpu: Whether the per-frequency launch targets the GPU rather than
            the CPU; `b` and `a` must live on the matching device.

    Args:
        b: The numerator coefficients, highest power of `z^-1` last.
        a: The denominator coefficients, highest power of `z^-1` last.

    Returns:
        A `FrequencyResponse` on `b`'s device: the grid `w = pi k / worN`
        for `k` in `[0, worN)`, and the real and imaginary parts of `H` at
        each.

    Raises:
        If allocating the result or launching the kernel on `b`'s device
        fails.
    """
    comptime nb = dim[A, 0]
    comptime na = dim[B, 0]
    var ctx = b.context()
    var w = List[Scalar[A.dtype]](capacity=worN)
    for k in range(worN):
        w.append(Scalar[A.dtype](_PI * Float64(k) / Float64(worN)))
    var grid = Static[A.dtype, worN](w^, ctx)
    var real = Static[A.dtype, worN]._uninitialized(ctx)
    var imag = Static[A.dtype, worN]._uninitialized(ctx)
    var ws = grid.tile()
    var bs = b.tile()
    var az = a.tile_as[A.dtype]()
    var rs = real.tile()
    var ims = imag.tile()

    @always_inline
    def lane[
        w: Int, alignment: Int = 1
    ](coord: Coord) {var ws, var bs, var az, var rs, var ims}:
        var k = coord_to_index_list(coord)[0]
        var angle = ws[Coord(k)]
        # e^{-jw}: cos and sin at the working precision, once per lane.
        var er = _cos(angle)
        var ei = -_sin(angle)
        var nr = bs[Coord(nb - 1)]
        var ni = Scalar[A.dtype](0)
        for step in range(1, nb):
            var tr = nr * er - ni * ei
            var ti = nr * ei + ni * er
            nr = tr + bs[Coord(nb - 1 - step)]
            ni = ti
        var dr = az[Coord(na - 1)]
        var di = Scalar[A.dtype](0)
        for step in range(1, na):
            var tr = dr * er - di * ei
            var ti = dr * ei + di * er
            dr = tr + az[Coord(na - 1 - step)]
            di = ti
        var mag = dr * dr + di * di
        rs.store[1](Coord(k), (nr * dr + ni * di) / mag)
        ims.store[1](Coord(k), (ni * dr - nr * di) / mag)

    elementwise[simd_width=1, target="gpu" if gpu else "cpu"](
        lane, Coord(worN), ctx
    )
    ctx.synchronize()
    return FrequencyResponse[A.dtype, worN](grid^, real^, imag^)


def sosfreqz[
    T: TensorLike,
    worN: Int = 512,
    gpu: Bool = False,
](sos: T) raises -> FrequencyResponse[T.dtype, worN] where (
    T.dtype.is_floating_point()
    and T.LayoutType.rank == 2
    and T.LayoutType.all_dims_known
    and dim[T, 1] == 6
    and worN > 0
):
    """The frequency response of a cascade of second-order sections at
    `worN` frequencies evenly spaced over `[0, pi)`.
    `scipy.signal.sosfreqz(sos, worN)`.

    The product of the sections' own responses, never the expanded
    polynomial's, which is what keeps a high-order response accurate: one
    lane per frequency evaluates every biquad at `e^{-jw}` and multiplies
    the quotients.

    Parameters:
        T: The tensor type of `sos`, `(sections, 6)`, floating-point.
        worN: The number of frequencies on the grid, default 512.
        gpu: Whether the per-frequency launch targets the GPU rather than
            the CPU; `sos` must live on the matching device.

    Args:
        sos: The sections, one `[b0, b1, b2, a0, a1, a2]` row each.

    Returns:
        A `FrequencyResponse` on `sos`'s device: the grid `w = pi k / worN`
        and the real and imaginary parts of `H` at each.

    Raises:
        If allocating the result or launching the kernel fails.
    """
    comptime sections = dim[T, 0]
    comptime dtype = T.dtype
    var ctx = sos.context()
    var w = List[Scalar[dtype]](capacity=worN)
    for k in range(worN):
        w.append(Scalar[dtype](_PI * Float64(k) / Float64(worN)))
    var grid = Static[dtype, worN](w^, ctx)
    var real = Static[dtype, worN]._uninitialized(ctx)
    var imag = Static[dtype, worN]._uninitialized(ctx)
    var ws = grid.tile()
    var ss = sos.tile().ptr.unsafe_origin_cast[MutAnyOrigin]()
    var rs = real.tile()
    var ims = imag.tile()

    @always_inline
    def lane[
        w: Int, alignment: Int = 1
    ](coord: Coord) {var ws, var ss, var rs, var ims}:
        var k = coord_to_index_list(coord)[0]
        var angle = rebind[Scalar[dtype]](ws[Coord(k)])
        var er = _cos(angle)
        var ei = -_sin(angle)
        var e2r = er * er - ei * ei
        var e2i = 2 * er * ei
        var hr = Scalar[dtype](1)
        var hi = Scalar[dtype](0)
        for s in range(sections):
            var o = s * 6
            var b0 = rebind[Scalar[dtype]](ss[unsafe_offset=o])
            var b1 = rebind[Scalar[dtype]](ss[unsafe_offset=o + 1])
            var b2 = rebind[Scalar[dtype]](ss[unsafe_offset=o + 2])
            var a0 = rebind[Scalar[dtype]](ss[unsafe_offset=o + 3])
            var a1 = rebind[Scalar[dtype]](ss[unsafe_offset=o + 4])
            var a2 = rebind[Scalar[dtype]](ss[unsafe_offset=o + 5])
            var nr = b0 + b1 * er + b2 * e2r
            var ni = b1 * ei + b2 * e2i
            var dr = a0 + a1 * er + a2 * e2r
            var di = a1 * ei + a2 * e2i
            var mag = dr * dr + di * di
            var qr = (nr * dr + ni * di) / mag
            var qi = (ni * dr - nr * di) / mag
            var tr = hr * qr - hi * qi
            hi = hr * qi + hi * qr
            hr = tr
        rs.store[1](Coord(k), hr)
        ims.store[1](Coord(k), hi)

    elementwise[simd_width=1, target="gpu" if gpu else "cpu"](
        lane, Coord(worN), ctx
    )
    ctx.synchronize()
    return FrequencyResponse[dtype, worN](grid^, real^, imag^)


def _notch_peak[
    dtype: DType
](
    w0: Float64,
    q: Float64,
    fs: Float64,
    peak: Bool,
    who: StaticString,
    ctx: Optional[DeviceContext],
) raises -> TransferFunction[dtype, 2] where dtype.is_floating_point():
    """`scipy.signal._design_notch_peak_filter`: the second-order notch or
    peak at `w0` with quality `q`."""
    var w = 2.0 * w0 / fs
    if w >= 1.0 or w <= 0.0:
        raise Error(who, ": w0 must lie strictly between 0 and fs / 2")
    if q <= 0:
        raise Error(who, ": Q must be positive")
    var bw = w / q * _PI
    var wr = w * _PI
    var beta = _tan(bw / 2.0)
    var gain = 1.0 / (1.0 + beta)
    var b: List[Float64]
    if peak:
        b = [1.0 - gain, 0.0, -(1.0 - gain)]
    else:
        b = [gain, -2.0 * gain * _cos(wr), gain]
    var a: List[Float64] = [1.0, -2.0 * gain * _cos(wr), 2.0 * gain - 1.0]
    return _to_transfer_function[dtype=dtype, order=2]((b^, a^), ctx)


def iirnotch[
    dtype: DType
](
    w0: Float64,
    q: Float64,
    fs: Float64 = 2.0,
    ctx: Optional[DeviceContext] = None,
) raises -> TransferFunction[dtype, 2] where dtype.is_floating_point():
    """A second-order IIR notch: unit gain everywhere but a null at `w0`,
    `w0 / Q` wide. `scipy.signal.iirnotch(w0, Q, fs)`.

    Parameters:
        dtype: The floating-point dtype of the returned coefficients.

    Args:
        w0: The frequency to remove, in the units of `fs`, strictly inside
            `(0, fs / 2)`.
        q: The quality factor: the notch's center over its -3 dB width.
        fs: The sampling frequency; `2` (the default) makes `w0` a fraction
            of Nyquist, SciPy's convention.
        ctx: The device `b` and `a` are uploaded to; `None` puts them in
            host memory.

    Returns:
        A `TransferFunction` of order 2.

    Raises:
        If `w0` is outside `(0, fs / 2)`, `Q` is not positive, or the upload
        fails.
    """
    return _notch_peak[dtype](w0, q, fs, False, "iirnotch", ctx)


def iirpeak[
    dtype: DType
](
    w0: Float64,
    q: Float64,
    fs: Float64 = 2.0,
    ctx: Optional[DeviceContext] = None,
) raises -> TransferFunction[dtype, 2] where dtype.is_floating_point():
    """A second-order IIR peak (resonator): unit gain at `w0`, falling off
    over a band `w0 / Q` wide. `scipy.signal.iirpeak(w0, Q, fs)`.

    Parameters:
        dtype: The floating-point dtype of the returned coefficients.

    Args:
        w0: The frequency to keep, in the units of `fs`, strictly inside
            `(0, fs / 2)`.
        q: The quality factor: the peak's center over its -3 dB width.
        fs: The sampling frequency; `2` (the default) makes `w0` a fraction
            of Nyquist.
        ctx: The device `b` and `a` are uploaded to; `None` puts them in
            host memory.

    Returns:
        A `TransferFunction` of order 2.

    Raises:
        If `w0` is outside `(0, fs / 2)`, `Q` is not positive, or the upload
        fails.
    """
    return _notch_peak[dtype](w0, q, fs, True, "iirpeak", ctx)


def _bilinear_expand(
    c: List[Float64], order: Int, zp1: List[Float64], zm1: List[Float64]
) -> List[Float64]:
    """`sum_q c_q zp1^(N-q) zm1^q` in ascending powers of `z`, `c_q` the
    analog coefficient of `s^q`: the bilinear transform's expansion."""
    var total = List[Float64](length=order + 1, fill=0.0)
    for q in range(len(c)):
        var term: List[Float64] = [c[q]]
        for _ in range(order - q):
            term = _poly_mul(term, zp1)
        for _ in range(q):
            term = _poly_mul(term, zm1)
        for i in range(len(term)):
            total[i] += term[i]
    return total^


def bilinear[
    A: TensorLike, B: TensorLike
](
    b: A, a: B, fs: Float64 = 1.0, ctx: Optional[DeviceContext] = None
) raises -> TransferFunction[
    A.dtype, (dim[A, 0] if dim[A, 0] > dim[B, 0] else dim[B, 0]) - 1
] where (
    A.dtype.is_floating_point()
    and B.dtype == A.dtype
    and A.LayoutType.rank == 1
    and A.LayoutType.all_dims_known
    and B.LayoutType.rank == 1
    and B.LayoutType.all_dims_known
):
    """The digital filter the bilinear transform `s = 2 fs (z - 1)/(z + 1)`
    makes of the analog filter `(b, a)`. `scipy.signal.bilinear(b, a, fs)`.

    SciPy's expansion: with `N` the order, the numerator is
    `sum_q b_q ((z + 1)/sqrt(2 fs))^(N-q) ((z - 1) sqrt(2 fs))^q` over the
    analog coefficients `b_q` of `s^q`, the denominator likewise, and both
    are divided by the denominator's leading coefficient. Host-side, on
    the coefficients.

    Parameters:
        A: The tensor type of the analog numerator `b`, rank 1.
        B: The tensor type of the analog denominator `a`, rank 1.

    Args:
        b: The analog numerator, descending in powers of `s`.
        a: The analog denominator, descending, `a[0]` nonzero.
        fs: The sampling frequency.
        ctx: The device `b` and `a` are uploaded to; `None` puts them in
            host memory.

    Returns:
        A `TransferFunction` of order `max(len(b), len(a)) - 1`.

    Raises:
        If `a[0]` is zero, if `b` has leading zeros that lower the order
        below the one its length names, or if the upload fails.
    """
    comptime nb = dim[A, 0]
    comptime na = dim[B, 0]
    comptime order = (nb if nb > na else na) - 1
    var bh = b.to_host()
    var ah = a.to_host()
    if ah[0] == 0:
        raise Error("bilinear: a[0] must be nonzero")
    var bs = 0
    while bs < nb - 1 and bh[bs] == 0:
        bs += 1
    if max(na, nb - bs) - 1 != order:
        raise Error("bilinear: b's leading zeros lower the order below ", order)
    var fac = _sqrt(2.0 * fs)
    # Ascending-coefficient polynomials in `z`.
    var zp1: List[Float64] = [1.0 / fac, 1.0 / fac]
    var zm1: List[Float64] = [-fac, fac]

    var b_rev = List[Float64]()
    for i in range(nb - 1, bs - 1, -1):
        b_rev.append(Float64(bh[i]))
    var a_rev = List[Float64]()
    for i in range(na - 1, -1, -1):
        a_rev.append(Float64(ah[i]))
    var num = _bilinear_expand(b_rev, order, zp1, zm1)
    var den = _bilinear_expand(a_rev, order, zp1, zm1)
    var lead = den[order]
    var bd = List[Float64](capacity=order + 1)
    var ad = List[Float64](capacity=order + 1)
    for i in range(order, -1, -1):
        bd.append(num[i] / lead)
        ad.append(den[i] / lead)
    return _to_transfer_function[dtype=A.dtype, order=order]((bd^, ad^), ctx)


struct GroupDelay[dtype: DType, n: Int](Movable):
    """What `group_delay` returns: the frequencies and the group delay in
    samples at each, SciPy's `(w, gd)`."""

    var w: Static[Self.dtype, Self.n]
    """The frequencies, in the units of `fs`."""
    var gd: Static[Self.dtype, Self.n]
    """The group delay at each frequency, in samples."""

    def __init__(
        out self,
        var w: Static[Self.dtype, Self.n],
        var gd: Static[Self.dtype, Self.n],
    ):
        """Build from the grid and the delays.

        Args:
            w: The frequencies.
            gd: The group delay at each.
        """
        self.w = w^
        self.gd = gd^


def group_delay[
    A: TensorLike,
    B: TensorLike,
    worN: Int = 512,
    whole: Bool = False,
    gpu: Bool = False,
](b: A, a: B, fs: Float64 = 2.0 * _PI) raises -> GroupDelay[
    A.dtype, worN
] where (
    A.dtype.is_floating_point()
    and B.dtype == A.dtype
    and A.LayoutType.rank == 1
    and A.LayoutType.all_dims_known
    and B.LayoutType.rank == 1
    and B.LayoutType.all_dims_known
    and worN > 0
):
    """The group delay `-d(phase)/dw` of the filter `(b, a)` at `worN`
    frequencies over `[0, pi)` (or `[0, 2 pi)` when `whole`).
    `scipy.signal.group_delay((b, a), worN, whole, fs)`.

    SciPy's method: `c = b * reversed(a)` (a convolution, on the host) and
    `gd = Re(C'(z) / C(z)) - (len(a) - 1)` at `z = e^{-jw}`, `C'` the
    polynomial with coefficients `k c_k`; one lane per frequency. A
    frequency where the delay is singular reads `0`, as SciPy sets it.

    Parameters:
        A: The tensor type of `b`, rank 1, floating-point.
        B: The tensor type of `a`, rank 1, same dtype.
        worN: The number of frequencies, default 512.
        whole: Span the whole circle rather than the upper half.
        gpu: Whether the per-frequency launch targets the GPU; `b` must live
            on the matching device.

    Args:
        b: The numerator coefficients.
        a: The denominator coefficients.
        fs: The sampling frequency the returned `w` is in; `2 pi` (the
            default) gives radians per sample.

    Returns:
        A `GroupDelay` on `b`'s device: the frequencies and the delays.

    Raises:
        If allocating the result or launching the kernel fails.
    """
    comptime dtype = A.dtype
    comptime nb = dim[A, 0]
    comptime na = dim[B, 0]
    comptime nc = nb + na - 1
    var ctx = b.context()
    var bh = b.to_host()
    var ah = a.to_host[dtype]()
    var c = List[Scalar[dtype]](length=nc, fill=0)
    for i in range(nb):
        for j in range(na):
            c[i + j] += bh[i] * ah[na - 1 - j]
    var coeffs = Static[dtype, nc](c^, ctx)
    var span = 2.0 * _PI if whole else _PI
    var w = List[Scalar[dtype]](capacity=worN)
    var scaled = List[Scalar[dtype]](capacity=worN)
    for k in range(worN):
        var wk = span * Float64(k) / Float64(worN)
        w.append(Scalar[dtype](wk))
        scaled.append(Scalar[dtype](wk * fs / (2.0 * _PI)))
    var grid = Static[dtype, worN](w^, ctx)
    var gd = Static[dtype, worN]._uninitialized(ctx)
    var ws = grid.tile()
    var cs = coeffs.tile()
    var gs = gd.tile()
    var offset = Scalar[dtype](na - 1)

    @always_inline
    def lane[
        width: Int, alignment: Int = 1
    ](coord: Coord) {var ws, var cs, var gs, var offset}:
        var k = coord_to_index_list(coord)[0]
        var angle = ws[Coord(k)]
        var zr = _cos(angle)
        var zi = -_sin(angle)
        # Horner in `z` from the top coefficient: `C(z)` and `C'(z) z`.
        var dr = Scalar[dtype](0)
        var di = Scalar[dtype](0)
        var nr = Scalar[dtype](0)
        var ni = Scalar[dtype](0)
        for step in range(nc):
            var j = nc - 1 - step
            var cj = cs[Coord(j)]
            var tr = dr * zr - di * zi + cj
            di = dr * zi + di * zr
            dr = tr
            var ur = nr * zr - ni * zi + Scalar[dtype](j) * cj
            ni = nr * zi + ni * zr
            nr = ur
        var mag = dr * dr + di * di
        var value = (nr * dr + ni * di) / mag - offset
        if not (value == value) or mag == 0:
            value = Scalar[dtype](0)
        gs.store[1](Coord(k), value)

    elementwise[simd_width=1, target="gpu" if gpu else "cpu"](
        lane, Coord(worN), ctx
    )
    ctx.synchronize()
    _ = coeffs^
    return GroupDelay[dtype, worN](Static[dtype, worN](scaled^, ctx), gd^)


# --------------------------------------------------------------------------
# Order selection
# --------------------------------------------------------------------------


@fieldwise_init
struct FilterOrder(Copyable, Movable):
    """What the order estimators return: the lowest order that meets the
    specification and the critical frequency (or band) to design it at,
    SciPy's `(N, Wn)`. Pass `order` as the design's compile-time `order`
    and `wn` (or `(wn, wn_high)` for a band) as its edge."""

    var order: Int
    """The minimum filter order."""
    var wn: Float64
    """The critical frequency, or the band's lower edge."""
    var wn_high: Float64
    """The band's upper edge; `0` for a lowpass or highpass."""


def _ellipkm1(p: Float64) -> Float64:
    """`K(1 - p)`, accurate for small `p`: `pi / (2 AGM(1, sqrt(p)))` with
    the complement's square root taken directly. Iterates to convergence,
    since a tiny `sqrt(p)` takes a few more rounds than `_ellipk`'s
    twelve."""
    var a = 1.0
    var b = _sqrt(p)
    for _ in range(64):
        if abs(a - b) <= 1e-16 * a:
            break
        var next_a = 0.5 * (a + b)
        b = _sqrt(a * b)
        a = next_a
    return _PI / (2.0 * a)


def _ellipk_full(m: Float64) -> Float64:
    """`K(m)` iterated to convergence, for `m` near 1."""
    return _ellipkm1(1.0 - m)


def _order_count(
    kind: StaticString, nat: Float64, gpass: Float64, gstop: Float64
) -> Float64:
    """The fractional order a band edge ratio `nat` needs, per family --
    `band_stop_obj`'s objective."""
    if kind == "butter":
        var gs = 10.0 ** (0.1 * abs(gstop))
        var gp = 10.0 ** (0.1 * abs(gpass))
        return _log10((gs - 1.0) / (gp - 1.0)) / (2.0 * _log10(nat))
    elif kind == "cheby":
        var gs = 10.0 ** (0.1 * abs(gstop))
        var gp = 10.0 ** (0.1 * abs(gpass))
        return _acosh(_sqrt((gs - 1.0) / (gp - 1.0))) / _acosh(nat)
    var gs = 10.0 ** (0.1 * gstop)
    var gp = 10.0 ** (0.1 * gpass)
    var arg1 = _sqrt((gp - 1.0) / (gs - 1.0))
    var arg0 = 1.0 / nat
    var d00 = _ellipk_full(arg0 * arg0)
    var d01 = _ellipk_full(1.0 - arg0 * arg0)
    var d10 = _ellipk_full(arg1 * arg1)
    var d11 = _ellipk_full(1.0 - arg1 * arg1)
    return d00 * d11 / (d01 * d10)


def _band_stop_obj(
    wp: Float64,
    ind: Int,
    passb: List[Float64],
    stopb: List[Float64],
    gpass: Float64,
    gstop: Float64,
    kind: StaticString,
) -> Float64:
    """`scipy.signal.band_stop_obj`: the order a bandstop needs when edge
    `ind` of the passband moves to `wp`."""
    var p0 = wp if ind == 0 else passb[0]
    var p1 = wp if ind == 1 else passb[1]
    var nat = 1e300
    for i in range(2):
        nat = min(
            nat, abs(stopb[i] * (p0 - p1) / (stopb[i] * stopb[i] - p0 * p1))
        )
    return _order_count(kind, nat, gpass, gstop)


def _fminbound(
    lo: Float64,
    hi: Float64,
    ind: Int,
    passb: List[Float64],
    stopb: List[Float64],
    gpass: Float64,
    gstop: Float64,
    kind: StaticString,
) -> Float64:
    """`scipy.optimize.fminbound` of `_band_stop_obj` on `[lo, hi]` at its
    default `xtol = 1e-5`: Brent's bounded minimizer transcribed, so the
    band edges come out where SciPy's do."""
    var sqrt_eps = _sqrt(2.2e-16)
    var golden_mean = 0.5 * (3.0 - _sqrt(5.0))
    var xatol = 1e-5
    var a = lo
    var b = hi
    var fulc = a + golden_mean * (b - a)
    var nfc = fulc
    var xf = fulc
    var rat = 0.0
    var e = 0.0
    var x = xf
    var fx = _band_stop_obj(x, ind, passb, stopb, gpass, gstop, kind)
    var num = 1
    var ffulc = fx
    var fnfc = fx
    var xm = 0.5 * (a + b)
    var tol1 = sqrt_eps * abs(xf) + xatol / 3.0
    var tol2 = 2.0 * tol1
    while abs(xf - xm) > (tol2 - 0.5 * (b - a)):
        var golden = True
        if abs(e) > tol1:
            golden = False
            var r = (xf - nfc) * (fx - ffulc)
            var q = (xf - fulc) * (fx - fnfc)
            var p = (xf - fulc) * q - (xf - nfc) * r
            q = 2.0 * (q - r)
            if q > 0.0:
                p = -p
            q = abs(q)
            r = e
            e = rat
            if (
                (abs(p) < abs(0.5 * q * r))
                and (p > q * (a - xf))
                and (p < q * (b - xf))
            ):
                rat = p / q
                x = xf + rat
                if ((x - a) < tol2) or ((b - x) < tol2):
                    var si = 1.0 if xm - xf >= 0 else -1.0
                    rat = tol1 * si
            else:
                golden = True
        if golden:
            if xf >= xm:
                e = a - xf
            else:
                e = b - xf
            rat = golden_mean * e
        var si = 1.0 if rat >= 0 else -1.0
        x = xf + si * max(abs(rat), tol1)
        var fu = _band_stop_obj(x, ind, passb, stopb, gpass, gstop, kind)
        num += 1
        if fu <= fx:
            if x >= xf:
                a = xf
            else:
                b = xf
            fulc = nfc
            ffulc = fnfc
            nfc = xf
            fnfc = fx
            xf = x
            fx = fu
        else:
            if x < xf:
                a = x
            else:
                b = x
            if (fu <= fnfc) or (nfc == xf):
                fulc = nfc
                ffulc = fnfc
                nfc = x
                fnfc = fu
            elif (fu <= ffulc) or (fulc == xf) or (fulc == nfc):
                fulc = x
                ffulc = fu
        xm = 0.5 * (a + b)
        tol1 = sqrt_eps * abs(xf) + xatol / 3.0
        tol2 = 2.0 * tol1
        if num >= 500:
            break
    return xf


def _estimate_order(
    kind: StaticString,
    family: StaticString,
    wp_in: List[Float64],
    ws_in: List[Float64],
    gpass: Float64,
    gstop: Float64,
    fs: Optional[Float64],
) raises -> FilterOrder:
    """The shared body of `buttord`, `cheb1ord`, `cheb2ord` and `ellipord`:
    SciPy's `_validate_wp_ws`, `_pre_warp` and `_find_nat_freq`, then the
    family's own order formula and edge placement."""
    if gpass <= 0.0:
        raise Error(family, ": gpass must be positive")
    if gstop <= 0.0:
        raise Error(family, ": gstop must be positive")
    if gpass > gstop:
        raise Error(family, ": gpass must not exceed gstop")
    var wp = wp_in.copy()
    var ws = ws_in.copy()
    if fs:
        for i in range(len(wp)):
            wp[i] = 2.0 * wp[i] / fs.value()
            ws[i] = 2.0 * ws[i] / fs.value()
    var filter_type = 2 * (len(wp) - 1) + 1
    if wp[0] >= ws[0]:
        filter_type += 1
    var passb = List[Float64]()
    var stopb = List[Float64]()
    for i in range(len(wp)):
        passb.append(_tan(_PI * wp[i] / 2.0))
        stopb.append(_tan(_PI * ws[i] / 2.0))
    var nat: Float64
    if filter_type == 1:
        nat = abs(stopb[0] / passb[0])
    elif filter_type == 2:
        nat = abs(passb[0] / stopb[0])
    elif filter_type == 3:
        var wp0 = _fminbound(
            passb[0], stopb[0] - 1e-12, 0, passb, stopb, gpass, gstop, kind
        )
        var wp1 = _fminbound(
            stopb[1] + 1e-12, passb[1], 1, passb, stopb, gpass, gstop, kind
        )
        passb = [wp0, wp1]
        nat = 1e300
        for i in range(2):
            nat = min(
                nat,
                abs(
                    stopb[i]
                    * (passb[0] - passb[1])
                    / (stopb[i] * stopb[i] - passb[0] * passb[1])
                ),
            )
    else:
        nat = 1e300
        for i in range(2):
            nat = min(
                nat,
                abs(
                    (stopb[i] * stopb[i] - passb[0] * passb[1])
                    / (stopb[i] * (passb[0] - passb[1]))
                ),
            )
    var order: Int
    var wn = List[Float64]()
    if family == "buttord":
        var gs = 10.0 ** (0.1 * abs(gstop))
        var gp = 10.0 ** (0.1 * abs(gpass))
        order = Int(
            _ceil(_log10((gs - 1.0) / (gp - 1.0)) / (2.0 * _log10(nat)))
        )
        var w0 = (gp - 1.0) ** (
            -1.0 / (2.0 * Float64(order))
        ) if order != 0 else 1.0
        if filter_type == 1:
            wn = [w0 * passb[0]]
        elif filter_type == 2:
            wn = [passb[0] / w0]
        elif filter_type == 3:
            var discr = _sqrt(
                (passb[1] - passb[0]) ** 2 + 4.0 * w0 * w0 * passb[0] * passb[1]
            )
            var a = abs(((passb[1] - passb[0]) + discr) / (2.0 * w0))
            var b = abs(((passb[1] - passb[0]) - discr) / (2.0 * w0))
            wn = [min(a, b), max(a, b)]
        else:
            var half = (passb[1] - passb[0]) / 2.0
            var root_part = _sqrt(
                w0 * w0 / 4.0 * (passb[1] - passb[0]) ** 2 + passb[0] * passb[1]
            )
            var a = abs(w0 * half + root_part)
            var b = abs(-w0 * half + root_part)
            wn = [min(a, b), max(a, b)]
    elif family == "ellipord":
        var arg1_sq = _pow10m1(0.1 * gpass) / _pow10m1(0.1 * gstop)
        var arg0 = 1.0 / nat
        var d00 = _ellipk_full(arg0 * arg0)
        var d01 = _ellipkm1(arg0 * arg0)
        var d10 = _ellipk_full(arg1_sq)
        var d11 = _ellipkm1(arg1_sq)
        order = Int(_ceil(d00 * d11 / (d01 * d10)))
        wn = passb.copy()
    else:
        var gs = 10.0 ** (0.1 * abs(gstop))
        var gp = 10.0 ** (0.1 * abs(gpass))
        var v_pass_stop = _acosh(_sqrt((gs - 1.0) / (gp - 1.0)))
        order = Int(_ceil(v_pass_stop / _acosh(nat)))
        if family == "cheb1ord":
            wn = passb.copy()
        else:
            var new_freq = 1.0 / _cosh(v_pass_stop / Float64(order))
            if filter_type == 1:
                wn = [passb[0] / new_freq]
            elif filter_type == 2:
                wn = [passb[0] * new_freq]
            elif filter_type == 3:
                var nat0 = new_freq / 2.0 * (passb[0] - passb[1]) + _sqrt(
                    new_freq * new_freq * (passb[1] - passb[0]) ** 2 / 4.0
                    + passb[1] * passb[0]
                )
                wn = [nat0, passb[1] * passb[0] / nat0]
            else:
                var nat0 = 1.0 / (2.0 * new_freq) * (
                    passb[0] - passb[1]
                ) + _sqrt(
                    (passb[1] - passb[0]) ** 2 / (4.0 * new_freq * new_freq)
                    + passb[1] * passb[0]
                )
                wn = [nat0, passb[0] * passb[1] / nat0]
    var scale = fs.value() / 2.0 if fs else 1.0
    for i in range(len(wn)):
        wn[i] = _atan(wn[i]) * 2.0 / _PI * scale
    if len(wn) == 1:
        return FilterOrder(order, wn[0], 0.0)
    return FilterOrder(order, wn[0], wn[1])


def buttord(
    wp: Float64,
    ws: Float64,
    gpass: Float64,
    gstop: Float64,
    fs: Optional[Float64] = None,
) raises -> FilterOrder:
    """The lowest Butterworth order that loses no more than `gpass` dB in
    the passband and at least `gstop` dB in the stopband, and the edge to
    design it at. `scipy.signal.buttord(wp, ws, gpass, gstop, fs=fs)`.

    SciPy's computation step for step.
    The order is a run-time value and a design's order is a compile-time
    parameter, so this answers "what order", to be written into the
    design call.

    Args:
        wp: The passband edge, as a fraction of Nyquist unless `fs`
            is given.
        ws: The stopband edge, in the same units.
        gpass: The most passband loss allowed, in dB.
        gstop: The least stopband attenuation required, in dB.
        fs: The sampling frequency the edges are in; `None` means
            fractions of Nyquist.

    Returns:
        A `FilterOrder` with the order and the design edge, in the
        units of the inputs.

    Raises:
        If `gpass` or `gstop` is not positive or `gpass > gstop`.
    """
    return _estimate_order("butter", "buttord", [wp], [ws], gpass, gstop, fs)


def buttord(
    wp: Tuple[Float64, Float64],
    ws: Tuple[Float64, Float64],
    gpass: Float64,
    gstop: Float64,
    fs: Optional[Float64] = None,
) raises -> FilterOrder:
    """`buttord` for a band: the `(low, high)` passband and stopband edges of a bandpass (the passband inside the stopband) or bandstop (the stopband inside). `scipy.signal.buttord(wp, ws, gpass, gstop, fs=fs)`.

    SciPy's computation step for step, including the bounded Brent search it runs to place a bandstop's passband edges.
    The order is a run-time value and a design's order is a compile-time
    parameter, so this answers "what order", to be written into the
    design call.

    Args:
        wp: The passband edges, as a fraction of Nyquist unless `fs`
            is given.
        ws: The stopband edges, in the same units.
        gpass: The most passband loss allowed, in dB.
        gstop: The least stopband attenuation required, in dB.
        fs: The sampling frequency the edges are in; `None` means
            fractions of Nyquist.

    Returns:
        A `FilterOrder` with the order and the design edges, in the
        units of the inputs.

    Raises:
        If `gpass` or `gstop` is not positive or `gpass > gstop`.
    """
    return _estimate_order(
        "butter", "buttord", [wp[0], wp[1]], [ws[0], ws[1]], gpass, gstop, fs
    )


def cheb1ord(
    wp: Float64,
    ws: Float64,
    gpass: Float64,
    gstop: Float64,
    fs: Optional[Float64] = None,
) raises -> FilterOrder:
    """The lowest Chebyshev type I order that loses no more than `gpass` dB in
    the passband and at least `gstop` dB in the stopband, and the edge to
    design it at. `scipy.signal.cheb1ord(wp, ws, gpass, gstop, fs=fs)`.

    SciPy's computation step for step.
    The order is a run-time value and a design's order is a compile-time
    parameter, so this answers "what order", to be written into the
    design call.

    Args:
        wp: The passband edge, as a fraction of Nyquist unless `fs`
            is given.
        ws: The stopband edge, in the same units.
        gpass: The most passband loss allowed, in dB.
        gstop: The least stopband attenuation required, in dB.
        fs: The sampling frequency the edges are in; `None` means
            fractions of Nyquist.

    Returns:
        A `FilterOrder` with the order and the design edge, in the
        units of the inputs.

    Raises:
        If `gpass` or `gstop` is not positive or `gpass > gstop`.
    """
    return _estimate_order("cheby", "cheb1ord", [wp], [ws], gpass, gstop, fs)


def cheb1ord(
    wp: Tuple[Float64, Float64],
    ws: Tuple[Float64, Float64],
    gpass: Float64,
    gstop: Float64,
    fs: Optional[Float64] = None,
) raises -> FilterOrder:
    """`cheb1ord` for a band: the `(low, high)` passband and stopband edges of a bandpass (the passband inside the stopband) or bandstop (the stopband inside). `scipy.signal.cheb1ord(wp, ws, gpass, gstop, fs=fs)`.

    SciPy's computation step for step.
    The order is a run-time value and a design's order is a compile-time
    parameter, so this answers "what order", to be written into the
    design call.

    Args:
        wp: The passband edges, as a fraction of Nyquist unless `fs`
            is given.
        ws: The stopband edges, in the same units.
        gpass: The most passband loss allowed, in dB.
        gstop: The least stopband attenuation required, in dB.
        fs: The sampling frequency the edges are in; `None` means
            fractions of Nyquist.

    Returns:
        A `FilterOrder` with the order and the design edges, in the
        units of the inputs.

    Raises:
        If `gpass` or `gstop` is not positive or `gpass > gstop`.
    """
    return _estimate_order(
        "cheby", "cheb1ord", [wp[0], wp[1]], [ws[0], ws[1]], gpass, gstop, fs
    )


def cheb2ord(
    wp: Float64,
    ws: Float64,
    gpass: Float64,
    gstop: Float64,
    fs: Optional[Float64] = None,
) raises -> FilterOrder:
    """The lowest Chebyshev type II order that loses no more than `gpass` dB in
    the passband and at least `gstop` dB in the stopband, and the edge to
    design it at. `scipy.signal.cheb2ord(wp, ws, gpass, gstop, fs=fs)`.

    SciPy's computation step for step.
    The order is a run-time value and a design's order is a compile-time
    parameter, so this answers "what order", to be written into the
    design call.

    Args:
        wp: The passband edge, as a fraction of Nyquist unless `fs`
            is given.
        ws: The stopband edge, in the same units.
        gpass: The most passband loss allowed, in dB.
        gstop: The least stopband attenuation required, in dB.
        fs: The sampling frequency the edges are in; `None` means
            fractions of Nyquist.

    Returns:
        A `FilterOrder` with the order and the design edge, in the
        units of the inputs.

    Raises:
        If `gpass` or `gstop` is not positive or `gpass > gstop`.
    """
    return _estimate_order("cheby", "cheb2ord", [wp], [ws], gpass, gstop, fs)


def cheb2ord(
    wp: Tuple[Float64, Float64],
    ws: Tuple[Float64, Float64],
    gpass: Float64,
    gstop: Float64,
    fs: Optional[Float64] = None,
) raises -> FilterOrder:
    """`cheb2ord` for a band: the `(low, high)` passband and stopband edges of a bandpass (the passband inside the stopband) or bandstop (the stopband inside). `scipy.signal.cheb2ord(wp, ws, gpass, gstop, fs=fs)`.

    SciPy's computation step for step.
    The order is a run-time value and a design's order is a compile-time
    parameter, so this answers "what order", to be written into the
    design call.

    Args:
        wp: The passband edges, as a fraction of Nyquist unless `fs`
            is given.
        ws: The stopband edges, in the same units.
        gpass: The most passband loss allowed, in dB.
        gstop: The least stopband attenuation required, in dB.
        fs: The sampling frequency the edges are in; `None` means
            fractions of Nyquist.

    Returns:
        A `FilterOrder` with the order and the design edges, in the
        units of the inputs.

    Raises:
        If `gpass` or `gstop` is not positive or `gpass > gstop`.
    """
    return _estimate_order(
        "cheby", "cheb2ord", [wp[0], wp[1]], [ws[0], ws[1]], gpass, gstop, fs
    )


def ellipord(
    wp: Float64,
    ws: Float64,
    gpass: Float64,
    gstop: Float64,
    fs: Optional[Float64] = None,
) raises -> FilterOrder:
    """The lowest elliptic order that loses no more than `gpass` dB in
    the passband and at least `gstop` dB in the stopband, and the edge to
    design it at. `scipy.signal.ellipord(wp, ws, gpass, gstop, fs=fs)`.

    SciPy's computation step for step.
    The order is a run-time value and a design's order is a compile-time
    parameter, so this answers "what order", to be written into the
    design call.

    Args:
        wp: The passband edge, as a fraction of Nyquist unless `fs`
            is given.
        ws: The stopband edge, in the same units.
        gpass: The most passband loss allowed, in dB.
        gstop: The least stopband attenuation required, in dB.
        fs: The sampling frequency the edges are in; `None` means
            fractions of Nyquist.

    Returns:
        A `FilterOrder` with the order and the design edge, in the
        units of the inputs.

    Raises:
        If `gpass` or `gstop` is not positive or `gpass > gstop`.
    """
    return _estimate_order("ellip", "ellipord", [wp], [ws], gpass, gstop, fs)


def ellipord(
    wp: Tuple[Float64, Float64],
    ws: Tuple[Float64, Float64],
    gpass: Float64,
    gstop: Float64,
    fs: Optional[Float64] = None,
) raises -> FilterOrder:
    """`ellipord` for a band: the `(low, high)` passband and stopband edges of a bandpass (the passband inside the stopband) or bandstop (the stopband inside). `scipy.signal.ellipord(wp, ws, gpass, gstop, fs=fs)`.

    SciPy's computation step for step.
    The order is a run-time value and a design's order is a compile-time
    parameter, so this answers "what order", to be written into the
    design call.

    Args:
        wp: The passband edges, as a fraction of Nyquist unless `fs`
            is given.
        ws: The stopband edges, in the same units.
        gpass: The most passband loss allowed, in dB.
        gstop: The least stopband attenuation required, in dB.
        fs: The sampling frequency the edges are in; `None` means
            fractions of Nyquist.

    Returns:
        A `FilterOrder` with the order and the design edges, in the
        units of the inputs.

    Raises:
        If `gpass` or `gstop` is not positive or `gpass > gstop`.
    """
    return _estimate_order(
        "ellip", "ellipord", [wp[0], wp[1]], [ws[0], ws[1]], gpass, gstop, fs
    )


def zpk2tf[
    dtype: DType, nz: Int, np: Int
](
    zeros_re: List[Float64],
    zeros_im: List[Float64],
    poles_re: List[Float64],
    poles_im: List[Float64],
    gain: Float64 = 1.0,
    ctx: Optional[DeviceContext] = None,
) raises -> TransferFunction[dtype, np] where (
    dtype.is_floating_point() and nz >= 0 and np >= 1 and np >= nz
):
    """Transfer-function coefficients from zeros, poles and a gain.
    `scipy.signal.zpk2tf(z, p, k)`.

    `b` is `gain` times the expanded product of `(1 - z_i / s)` and `a`
    the expanded product of `(1 - p_i / s)`, both descending in powers of
    the delay, which is the order `lfilter`, `filtfilt` and `freqz` take.

    Zeros and poles arrive as separate real and imaginary lists because a
    `Tensor` is `dtype`-monomorphic and holds no `Complex` -- the same
    split `Eigenvalues` and `STFT` make. A conjugate pair is two entries
    with opposite imaginary parts, and the expansion is real whenever the
    roots come in such pairs, which is the only case a real filter has.
    The imaginary part of the product is computed and **discarded**: for a
    conjugate-closed root set it is zero to rounding, and for one that is
    not closed the answer was never a real filter to begin with.

    `np >= nz` because a transfer function with more zeros than poles is
    not causal, and `TransferFunction` carries one order for both.

    Host-side, `O((nz + np)^2)` complex multiplies -- a coefficient
    expansion, not a signal pass.
    """
    var b = _expand_roots(zeros_re, zeros_im, np, gain)
    var a = _expand_roots(poles_re, poles_im, np, 1.0)
    return _to_transfer_function[dtype=dtype, order=np]((b^, a^), ctx)


def _expand_roots(
    re: List[Float64], im: List[Float64], order: Int, gain: Float64
) raises -> List[Float64]:
    """`gain * prod_i (1 - r_i x)` expanded into `order + 1` descending
    coefficients, the polynomial's real part.

    Synthetic multiplication one root at a time: multiplying by
    `(1 - r x)` shifts the accumulated coefficients and subtracts `r`
    times them, which is `O(degree)` per root and needs no root finding
    in reverse.
    """
    if len(re) != len(im):
        raise Error(
            "zpk2tf: ",
            len(re),
            " real parts against ",
            len(im),
            " imaginary parts",
        )
    if len(re) > order:
        raise Error("zpk2tf: ", len(re), " roots exceed the order ", order)
    var cr = List[Float64](length=order + 1, fill=0.0)
    var ci = List[Float64](length=order + 1, fill=0.0)
    cr[0] = 1.0
    var degree = 0
    for i in range(len(re)):
        degree += 1
        # Walk down so a slot is read before it is overwritten.
        for j in range(degree, 0, -1):
            cr[j] -= re[i] * cr[j - 1] - im[i] * ci[j - 1]
            ci[j] -= re[i] * ci[j - 1] + im[i] * cr[j - 1]
    var out = List[Float64](capacity=order + 1)
    for j in range(order + 1):
        out.append(gain * cr[j])
    return out^
