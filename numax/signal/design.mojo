"""IIR filter design and frequency response over
`numax.core.array.Tensor`: `butter`, `cheby1`, `cheby2`, `ellip`, the
`iirfilter` front door they share, and `freqz`.

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
    asin as _asin,
    asinh as _asinh,
    cos as _cos,
    cosh as _cosh,
    exp as _exp,
    expm1 as _expm1,
    hypot as _hypot,
    sin as _sin,
    sinh as _sinh,
    sqrt as _sqrt,
    tan as _tan,
)

from layout import Coord, coord_to_index_list
from max.algorithm.functional import elementwise
from max.gpu.host import DeviceContext

from ..core.array import Static

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
    split, with the missing zeros landing on the band's own centre
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
) raises -> Tuple[List[Float64], List[Float64]]:
    """The route every family shares: warp the edges, move the prototype,
    bilinear at `fs = 2`, expand."""
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
    return _zpk2tf(_bilinear_zpk(proto^, 2.0))


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
        Static[dtype, order + 1](device, b^),
        Static[dtype, order + 1](device, a^),
    )


# --------------------------------------------------------------------------
# The four families, each as an edge overload and a band overload
# --------------------------------------------------------------------------


def butter[
    dtype: DType, order: Int
](
    wn: Float64,
    btype: StaticString = "lowpass",
    ctx: Optional[DeviceContext] = None,
) raises -> TransferFunction[dtype, order] where (
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
    For an order above about eight, prefer running the design through
    second-order sections -- SciPy's `output="sos"` -- which this does not
    produce; the `(b, a)` of a long polynomial lose digits a cascade keeps.
    `wn` must lie in `(0, 1)`.
    """
    _check_edge("butter", btype, wn)
    return _to_transfer_function[dtype, order](
        _design(_buttap(order), wn, 0.0, btype), ctx
    )


def butter[
    dtype: DType, order: Int
](
    wn: Tuple[Float64, Float64],
    btype: StaticString = "bandpass",
    ctx: Optional[DeviceContext] = None,
) raises -> TransferFunction[dtype, 2 * order] where (
    dtype.is_floating_point() and order >= 1
):
    """A digital Butterworth bandpass or bandstop filter across
    `wn = (low, high)`, both as fractions of Nyquist.
    `scipy.signal.butter(order, (low, high), btype)`.

    `lp2bp`/`lp2bs` split every prototype root in two, so the result has
    order `2 * order` -- named in the return type, which is why this is an
    overload rather than a `btype` on the one above.
    """
    _check_band("butter", btype, wn)
    return _to_transfer_function[dtype, 2 * order](
        _design(_buttap(order), wn[0], wn[1], btype), ctx
    )


def cheby1[
    dtype: DType, order: Int
](
    rp: Float64,
    wn: Float64,
    btype: StaticString = "lowpass",
    ctx: Optional[DeviceContext] = None,
) raises -> TransferFunction[dtype, order] where (
    dtype.is_floating_point() and order >= 1
):
    """A digital Chebyshev type I filter: `rp` dB of equiripple in the
    passband, monotone stopband. `scipy.signal.cheby1(order, rp, wn,
    btype)`.

    `wn` is the edge where the response last leaves the ripple band, not
    the half-power point, which is SciPy's convention too. `rp` must be
    positive; `"lowpass"` or `"highpass"` here, a pair for the band forms.
    """
    _check_edge("cheby1", btype, wn)
    return _to_transfer_function[dtype, order](
        _design(_cheb1ap(order, rp), wn, 0.0, btype), ctx
    )


def cheby1[
    dtype: DType, order: Int
](
    rp: Float64,
    wn: Tuple[Float64, Float64],
    btype: StaticString = "bandpass",
    ctx: Optional[DeviceContext] = None,
) raises -> TransferFunction[dtype, 2 * order] where (
    dtype.is_floating_point() and order >= 1
):
    """A digital Chebyshev type I bandpass or bandstop filter at order
    `2 * order`. `scipy.signal.cheby1(order, rp, (low, high), btype)`."""
    _check_band("cheby1", btype, wn)
    return _to_transfer_function[dtype, 2 * order](
        _design(_cheb1ap(order, rp), wn[0], wn[1], btype), ctx
    )


def cheby2[
    dtype: DType, order: Int
](
    rs: Float64,
    wn: Float64,
    btype: StaticString = "lowpass",
    ctx: Optional[DeviceContext] = None,
) raises -> TransferFunction[dtype, order] where (
    dtype.is_floating_point() and order >= 1
):
    """A digital Chebyshev type II filter: flat passband, `rs` dB of
    equiripple attenuation in the stopband. `scipy.signal.cheby2(order, rs,
    wn, btype)`.

    `wn` is the edge where the stopband attenuation first reaches `rs`.
    Unlike type I this one has finite zeros, so `b` is not a scaled
    `(1 + z)^order`; an odd order has one fewer zero than pole.
    """
    _check_edge("cheby2", btype, wn)
    return _to_transfer_function[dtype, order](
        _design(_cheb2ap(order, rs), wn, 0.0, btype), ctx
    )


def cheby2[
    dtype: DType, order: Int
](
    rs: Float64,
    wn: Tuple[Float64, Float64],
    btype: StaticString = "bandpass",
    ctx: Optional[DeviceContext] = None,
) raises -> TransferFunction[dtype, 2 * order] where (
    dtype.is_floating_point() and order >= 1
):
    """A digital Chebyshev type II bandpass or bandstop filter at order
    `2 * order`. `scipy.signal.cheby2(order, rs, (low, high), btype)`."""
    _check_band("cheby2", btype, wn)
    return _to_transfer_function[dtype, 2 * order](
        _design(_cheb2ap(order, rs), wn[0], wn[1], btype), ctx
    )


def ellip[
    dtype: DType, order: Int
](
    rp: Float64,
    rs: Float64,
    wn: Float64,
    btype: StaticString = "lowpass",
    ctx: Optional[DeviceContext] = None,
) raises -> TransferFunction[dtype, order] where (
    dtype.is_floating_point() and order >= 1
):
    """A digital elliptic (Cauer) filter: `rp` dB of passband ripple and
    `rs` dB of stopband attenuation in the fewest poles of the four.
    `scipy.signal.ellip(order, rp, rs, wn, btype)`.

    The steepest transition an IIR filter of this order can have, paid for
    with ripple in both bands and the worst phase response of the four.
    Raises when `rp` and `rs` cannot both be met at any order.
    """
    _check_edge("ellip", btype, wn)
    return _to_transfer_function[dtype, order](
        _design(_ellipap(order, rp, rs), wn, 0.0, btype), ctx
    )


def ellip[
    dtype: DType, order: Int
](
    rp: Float64,
    rs: Float64,
    wn: Tuple[Float64, Float64],
    btype: StaticString = "bandpass",
    ctx: Optional[DeviceContext] = None,
) raises -> TransferFunction[dtype, 2 * order] where (
    dtype.is_floating_point() and order >= 1
):
    """A digital elliptic bandpass or bandstop filter at order
    `2 * order`. `scipy.signal.ellip(order, rp, rs, (low, high), btype)`."""
    _check_band("ellip", btype, wn)
    return _to_transfer_function[dtype, 2 * order](
        _design(_ellipap(order, rp, rs), wn[0], wn[1], btype), ctx
    )


def iirfilter[
    dtype: DType, order: Int
](
    wn: Float64,
    rp: Float64 = 1.0,
    rs: Float64 = 40.0,
    btype: StaticString = "lowpass",
    ftype: StaticString = "butter",
    ctx: Optional[DeviceContext] = None,
) raises -> TransferFunction[dtype, order] where (
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
    """
    _check_edge("iirfilter", btype, wn)
    return _to_transfer_function[dtype, order](
        _design(_prototype(ftype, order, rp, rs), wn, 0.0, btype), ctx
    )


def iirfilter[
    dtype: DType, order: Int
](
    wn: Tuple[Float64, Float64],
    rp: Float64 = 1.0,
    rs: Float64 = 40.0,
    btype: StaticString = "bandpass",
    ftype: StaticString = "butter",
    ctx: Optional[DeviceContext] = None,
) raises -> TransferFunction[dtype, 2 * order] where (
    dtype.is_floating_point() and order >= 1
):
    """The named-`ftype` front door for the band forms, at order
    `2 * order`. `scipy.signal.iirfilter(order, (low, high), rp, rs,
    btype, ftype=...)`."""
    _check_band("iirfilter", btype, wn)
    return _to_transfer_function[dtype, 2 * order](
        _design(_prototype(ftype, order, rp, rs), wn[0], wn[1], btype), ctx
    )


def freqz[
    dtype: DType, nb: Int, na: Int, worN: Int = 512, gpu: Bool = False
](
    mut b: Static[dtype, nb], mut a: Static[dtype, na]
) raises -> FrequencyResponse[dtype, worN] where (
    dtype.is_floating_point() and nb > 0 and na > 0 and worN > 0
):
    """The frequency response `H(e^{jw}) = B(e^{jw}) / A(e^{jw})` of the
    filter `(b, a)` at `worN` frequencies evenly spaced over `[0, pi)`.
    `scipy.signal.freqz(b, a, worN)`, with its default `whole=False` grid.

    One lane per frequency: both polynomials by complex Horner in
    `e^{-jw}`, then one complex division. `worN` is a parameter because it
    shapes the result; SciPy's default of 512 is kept.
    """
    var ctx = b.context()
    var w = List[Scalar[dtype]](capacity=worN)
    for k in range(worN):
        w.append(Scalar[dtype](_PI * Float64(k) / Float64(worN)))
    var grid = Static[dtype, worN](ctx, w^)
    var real = Static[dtype, worN]._uninitialized(ctx)
    var imag = Static[dtype, worN]._uninitialized(ctx)
    var ws = grid.view()
    var bs = b.view()
    var az = a.view()
    var rs = real.view()
    var ims = imag.view()

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
        var ni = Scalar[dtype](0)
        for step in range(1, nb):
            var tr = nr * er - ni * ei
            var ti = nr * ei + ni * er
            nr = tr + bs[Coord(nb - 1 - step)]
            ni = ti
        var dr = az[Coord(na - 1)]
        var di = Scalar[dtype](0)
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
    return FrequencyResponse[dtype, worN](grid^, real^, imag^)
