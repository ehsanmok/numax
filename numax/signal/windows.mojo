"""Window functions as `numax.core.array.Tensor` factories: `boxcar`,
`hann`, `hamming`, `blackman`, `bartlett`, `kaiser` and `get_window`, with
`scipy.signal.windows`' symmetric and periodic forms.

**Tier 2** by placement rather than by algorithm: a window is a table, so
each factory evaluates its formula on the host in `Float64` and uploads it
once, the way `numax.fft.fftfreq` does. `numax.signal.array` has the same
three cosine windows as compile-time tables inside a kernel body.

## Symmetric and periodic

`scipy.signal.windows.<name>(n)` is **symmetric** by default (`sym=True`):
the window is sampled so that its two ends match, and the first and last
values of Hann are exactly zero. `scipy.signal.get_window(name, n)` is
**periodic** by default (`fftbins=True`): the same formula sampled as if
for `n + 1` points with the last dropped, so that the window tiles without
a seam -- which is what spectral estimation over overlapping frames wants,
and why `welch` and `spectrogram` call `get_window`. Both forms are here
under SciPy's own switches, and the difference is one denominator: `n - 1`
for symmetric, `n` for periodic.

## The MAX gate

Nothing: MAX has no window functions. Tables, not kernels; nothing to
delegate and nothing worth a device launch to build.
"""

from std.math import cos as _cos, sqrt as _sqrt
from max.gpu.host import DeviceContext

from ..core.array import Static

comptime _TWO_PI = 6.283185307179586


def _context(ctx: Optional[DeviceContext]) raises -> DeviceContext:
    return ctx.value() if ctx else DeviceContext(api="cpu")


def _denominator(n: Int, sym: Bool) -> Float64:
    """`n - 1` for a symmetric window, `n` for a periodic one; `1` at
    `n == 1`, where every window is the single value `1`."""
    if n <= 1:
        return 1.0
    return Float64(n - 1) if sym else Float64(n)


def _cosine_window[
    dtype: DType, n: Int
](
    a0: Float64,
    a1: Float64,
    a2: Float64,
    sym: Bool,
    ctx: Optional[DeviceContext],
) raises -> Static[dtype, n]:
    """`a0 - a1 cos(2 pi i / d) + a2 cos(4 pi i / d)`, the generalized
    cosine window Hann, Hamming and Blackman are all instances of."""
    var d = _denominator(n, sym)
    var values = List[Scalar[dtype]](capacity=n)
    for i in range(n):
        var theta = _TWO_PI * Float64(i) / d
        values.append(
            Scalar[dtype](a0 - a1 * _cos(theta) + a2 * _cos(2.0 * theta))
        )
    return Static[dtype, n](_context(ctx), values^)


def boxcar[
    dtype: DType, n: Int
](sym: Bool = True, ctx: Optional[DeviceContext] = None) raises -> Static[
    dtype, n
] where (dtype.is_floating_point() and n > 0):
    """The rectangular window: all ones. `scipy.signal.windows.boxcar`.
    `sym` is accepted for uniformity and changes nothing."""
    _ = sym
    return Static[dtype, n](
        _context(ctx), List[Scalar[dtype]](length=n, fill=Scalar[dtype](1))
    )


def hann[
    dtype: DType, n: Int
](sym: Bool = True, ctx: Optional[DeviceContext] = None) raises -> Static[
    dtype, n
] where (dtype.is_floating_point() and n > 0):
    """The Hann window, `0.5 - 0.5 cos(2 pi i / d)`.
    `scipy.signal.windows.hann(n, sym)`.

    The default for spectral analysis: -31 dB sidelobes falling at 18
    dB/octave. Symmetric, the ends are exactly zero; periodic, the window
    tiles without a seam.
    """
    return _cosine_window[dtype, n](0.5, 0.5, 0.0, sym, ctx)


def hamming[
    dtype: DType, n: Int
](sym: Bool = True, ctx: Optional[DeviceContext] = None) raises -> Static[
    dtype, n
] where (dtype.is_floating_point() and n > 0):
    """The Hamming window, `0.54 - 0.46 cos(2 pi i / d)`.
    `scipy.signal.windows.hamming(n, sym)`. The classic `0.54/0.46`
    coefficients, matching NumPy and SciPy, rather than the exactly optimal
    `0.53836/0.46164`."""
    return _cosine_window[dtype, n](0.54, 0.46, 0.0, sym, ctx)


def blackman[
    dtype: DType, n: Int
](sym: Bool = True, ctx: Optional[DeviceContext] = None) raises -> Static[
    dtype, n
] where (dtype.is_floating_point() and n > 0):
    """The Blackman window, `0.42 - 0.5 cos(2 pi i / d) + 0.08 cos(4 pi i
    / d)`. `scipy.signal.windows.blackman(n, sym)`: -58 dB sidelobes for a
    main lobe half again as wide as Hann's."""
    return _cosine_window[dtype, n](0.42, 0.5, 0.08, sym, ctx)


def bartlett[
    dtype: DType, n: Int
](sym: Bool = True, ctx: Optional[DeviceContext] = None) raises -> Static[
    dtype, n
] where (dtype.is_floating_point() and n > 0):
    """The Bartlett (triangular, zero-ended) window, `1 - |2i/d - 1|`.
    `scipy.signal.windows.bartlett(n, sym)`."""
    var d = _denominator(n, sym)
    var values = List[Scalar[dtype]](capacity=n)
    for i in range(n):
        values.append(Scalar[dtype](1.0 - abs(2.0 * Float64(i) / d - 1.0)))
    return Static[dtype, n](_context(ctx), values^)


def _bessel_i0(x: Float64) -> Float64:
    """The modified Bessel function `I_0(x)` by its power series,
    `sum (x/2)^{2k} / (k!)^2`, in `Float64` on the host.

    Converges for every `x` and the terms fall off factorially, so fifty
    terms are exact to double precision for the `beta` a window uses (up to
    a few tens). `numax.special` has no `i0` yet; when it does, this is the
    one call to replace.
    """
    var half = x / 2.0
    var term = 1.0
    var total = 1.0
    for k in range(1, 60):
        term *= (half / Float64(k)) * (half / Float64(k))
        total += term
        if term < 1e-17 * total:
            break
    return total


def kaiser[
    dtype: DType, n: Int
](
    beta: Float64, sym: Bool = True, ctx: Optional[DeviceContext] = None
) raises -> Static[dtype, n] where (dtype.is_floating_point() and n > 0):
    """The Kaiser window, `I_0(beta sqrt(1 - (2i/d - 1)^2)) / I_0(beta)`.
    `scipy.signal.windows.kaiser(n, beta, sym)`.

    The one window with a knob: `beta` trades main-lobe width for sidelobe
    height continuously, `0` being the boxcar and `14` roughly the
    Blackman. `I_0` is evaluated by its power series on the host.
    """
    var d = _denominator(n, sym)
    var scale = 1.0 / _bessel_i0(beta)
    var values = List[Scalar[dtype]](capacity=n)
    for i in range(n):
        var ratio = 2.0 * Float64(i) / d - 1.0
        var inside = 1.0 - ratio * ratio
        if inside < 0:
            inside = 0.0
        values.append(Scalar[dtype](_bessel_i0(beta * _sqrt(inside)) * scale))
    return Static[dtype, n](_context(ctx), values^)


def get_window[
    dtype: DType, n: Int
](
    name: StaticString,
    beta: Float64 = 0.0,
    fftbins: Bool = True,
    ctx: Optional[DeviceContext] = None,
) raises -> Static[dtype, n] where (dtype.is_floating_point() and n > 0):
    """A window by name. `scipy.signal.get_window(window, n, fftbins)`.

    `"boxcar"`, `"hann"` (or `"hanning"`), `"hamming"`, `"blackman"`,
    `"bartlett"` and `"kaiser"`, the last taking `beta` where SciPy takes
    the tuple `("kaiser", beta)`. **Periodic by default** (`fftbins=True`),
    which is the opposite of the named factories' default and is SciPy's
    choice too: this is the spelling the spectral estimators use, and they
    want the window that tiles. An unknown name raises.
    """
    var sym = not fftbins
    if name == "boxcar":
        return boxcar[dtype, n](sym, ctx)
    if name == "hann" or name == "hanning":
        return hann[dtype, n](sym, ctx)
    if name == "hamming":
        return hamming[dtype, n](sym, ctx)
    if name == "blackman":
        return blackman[dtype, n](sym, ctx)
    if name == "bartlett":
        return bartlett[dtype, n](sym, ctx)
    if name == "kaiser":
        return kaiser[dtype, n](beta, sym, ctx)
    raise Error(
        "get_window: unknown window '",
        name,
        "'; expected boxcar, hann, hamming, blackman, bartlett or kaiser",
    )
