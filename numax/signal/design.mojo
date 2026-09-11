"""IIR filter design and frequency response over
`numax.core.array.Tensor`: `butter` and `freqz`.

**Tier 2.** A design is a few dozen complex numbers on the host --
prototype poles, a frequency warp, the bilinear transform, a polynomial
expansion -- and a table at the end, so `butter` runs on the host in
`Float64` and uploads `(b, a)` once, the way `firwin` does. `freqz` is
device work: one lane per frequency evaluating two polynomials on the unit
circle.

## The reversal this module records

`docs/parity.md` said filter *design* past a windowed sinc was out of
scope. It is not any more, and the reason it was is worth stating: the
design math needs complex poles and zeros, and a `dtype`-monomorphic
`Tensor` holds no `Complex`. But the design is scalar host arithmetic on a
handful of values and its *result* -- `b` and `a` -- is real, so nothing
about the `Tensor` tier stands in its way; the poles live in host
`Float64` pairs for the microseconds the design takes. `butter` is the
first, on `scipy.signal.iirfilter`'s route exactly: `buttap` prototype,
`lp2lp_zpk`/`lp2hp_zpk` at the warped edge, `bilinear_zpk` at `fs = 2`,
`zpk2tf`. Chebyshev, elliptic, bandpass and bandstop are follow-ups on the
same route rather than decisions.

## The MAX gate

Nothing: MAX has no filter design. **Extend.**
"""

from std.math import cos as _cos, sin as _sin, tan as _tan

from layout import Coord, coord_to_index_list
from max.algorithm.functional import elementwise
from max.gpu.host import DeviceContext

from ..core.array import Static

comptime _PI = 3.141592653589793


struct TransferFunction[dtype: DType, order: Int](Movable):
    """What `butter` returns: the numerator `b` and denominator `a` of a
    digital filter of the given `order`, each `order + 1` long, SciPy's
    `(b, a)` as a struct. Feed them to `lfilter`, `filtfilt` or `freqz`."""

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


def _complex_divide(
    ar: Float64, ai: Float64, br: Float64, bi: Float64
) -> Tuple[Float64, Float64]:
    var d = br * br + bi * bi
    return ((ar * br + ai * bi) / d, (ai * br - ar * bi) / d)


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

    SciPy's route to the digit: the analog prototype's poles on the unit
    circle, the edge pre-warped by `4 tan(pi wn / 2)` so the bilinear
    transform lands it where it was asked for, `lp2lp` or `lp2hp` to move
    the prototype there, the bilinear transform at `fs = 2`, and the
    zero-pole-gain form expanded to polynomials. Maximally flat in the
    passband, which is what the name promises and the only thing it does.

    `"lowpass"` or `"highpass"`; bandpass and bandstop need the
    order-doubling `lp2bp`/`lp2bs` and are not here yet. For an order above
    about eight, prefer running the design through second-order sections
    -- SciPy's `output="sos"` -- which this does not produce; the `(b, a)`
    of a long polynomial lose digits a cascade keeps. `wn` must lie in
    `(0, 1)`.
    """
    if not (btype == "lowpass" or btype == "highpass"):
        raise Error(
            "butter: unknown btype '",
            btype,
            (
                "'; expected 'lowpass' or 'highpass' (bandpass and bandstop are"
                " not provided)"
            ),
        )
    if wn <= 0 or wn >= 1:
        raise Error("butter: wn must lie strictly inside (0, 1)")

    # Prototype poles: -exp(i pi m / 2N) for m = -N+1, -N+3, ..., N-1.
    var pr = List[Float64](capacity=order)
    var pi = List[Float64](capacity=order)
    for j in range(order):
        var m = Float64(-order + 1 + 2 * j)
        var angle = _PI * m / Float64(2 * order)
        pr.append(-_cos(angle))
        pi.append(-_sin(angle))
    var gain = 1.0

    # Warp the edge and move the prototype to it.
    var warped = 4.0 * _tan(_PI * wn / 2.0)
    var zr = List[Float64]()
    var zi = List[Float64]()
    if btype == "lowpass":
        for j in range(order):
            pr[j] *= warped
            pi[j] *= warped
        for _ in range(order):
            gain *= warped
    else:
        # p -> wo / p, N zeros at the origin, gain / prod(-p).
        var prod_r = 1.0
        var prod_i = 0.0
        for j in range(order):
            var nr = -pr[j]
            var ni = -pi[j]
            var tr = prod_r * nr - prod_i * ni
            var ti = prod_r * ni + prod_i * nr
            prod_r = tr
            prod_i = ti
            var q = _complex_divide(warped, 0.0, pr[j], pi[j])
            pr[j] = q[0]
            pi[j] = q[1]
            zr.append(0.0)
            zi.append(0.0)
        var inv = _complex_divide(1.0, 0.0, prod_r, prod_i)
        gain *= inv[0]

    # Bilinear transform at fs = 2: s -> (4 + s) / (4 - s), the missing
    # zeros landing at -1, the gain by prod(4 - z) / prod(4 - p).
    var num_r = 1.0
    var num_i = 0.0
    for j in range(len(zr)):
        var tr = num_r * (4.0 - zr[j]) - num_i * (-zi[j])
        var ti = num_r * (-zi[j]) + num_i * (4.0 - zr[j])
        num_r = tr
        num_i = ti
        var q = _complex_divide(4.0 + zr[j], zi[j], 4.0 - zr[j], -zi[j])
        zr[j] = q[0]
        zi[j] = q[1]
    var den_r = 1.0
    var den_i = 0.0
    for j in range(order):
        var tr = den_r * (4.0 - pr[j]) - den_i * (-pi[j])
        var ti = den_r * (-pi[j]) + den_i * (4.0 - pr[j])
        den_r = tr
        den_i = ti
        var q = _complex_divide(4.0 + pr[j], pi[j], 4.0 - pr[j], -pi[j])
        pr[j] = q[0]
        pi[j] = q[1]
    for _ in range(order - len(zr)):
        zr.append(-1.0)
        zi.append(0.0)
    var ratio = _complex_divide(num_r, num_i, den_r, den_i)
    gain *= ratio[0]

    var numerator = _poly_from_roots(zr, zi)
    var denominator = _poly_from_roots(pr, pi)
    var b = List[Scalar[dtype]](capacity=order + 1)
    var a = List[Scalar[dtype]](capacity=order + 1)
    for j in range(order + 1):
        b.append(Scalar[dtype](gain * numerator[0][j]))
        a.append(Scalar[dtype](denominator[0][j]))
    var device = ctx.value() if ctx else DeviceContext(api="cpu")
    return TransferFunction[dtype, order](
        Static[dtype, order + 1](device, b^),
        Static[dtype, order + 1](device, a^),
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
        var er = Scalar[dtype](1)
        var ei = Scalar[dtype](0)
        # e^{-jw}: cos and sin at the working precision, once per lane.
        var c = _cos(angle)
        var s = -_sin(angle)
        er = c
        ei = s
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
