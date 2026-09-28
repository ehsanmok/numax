"""Window functions as `numax.core.tensor.Tensor` factories: `boxcar`,
`hann`, `hamming`, `blackman`, `bartlett`, `kaiser` and `get_window`, with
`scipy.signal.windows`' symmetric and periodic forms.

**Tier 2** by placement rather than by algorithm: a window is a table.
On a host context each factory evaluates its formula in `Float64`; on a
device context it evaluates the same formula at the tensor's dtype in one
`elementwise` launch, so nothing is uploaded -- `numax.core.tensor`'s
factories do the same, under the same gate (`_DEVICE_FILL`: an
accelerator build, not `float64`, which Metal cannot compile).
`numax.signal.array` has the same three cosine windows as compile-time
tables inside a kernel body.

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

Nothing: MAX has no window functions. An **extend**, and a small one:
each device fill is one launch over the window's index.
"""

from std.math import cos as _cos, sqrt as _sqrt
from layout import Coord, coord_to_index_list
from max.algorithm.functional import elementwise
from max.gpu.host import DeviceContext

from ..core.tensor import _DEVICE_FILL, Static

comptime _TWO_PI = 6.283185307179586


def _context(ctx: Optional[DeviceContext]) raises -> DeviceContext:
    return ctx.value() if ctx else DeviceContext(api="cpu")


def _denominator(n: Int, sym: Bool) -> Float64:
    """`n - 1` for a symmetric window, `n` for a periodic one; `1` at
    `n == 1`, where every window is the single value `1`."""
    if n <= 1:
        return 1.0
    return Float64(n - 1) if sym else Float64(n)


def _on_device[dtype: DType](ctx: Optional[DeviceContext]) raises -> Bool:
    """Whether a window for `ctx` fills on the device: `_DEVICE_FILL`
    allows it and the context is a GPU."""
    comptime if _DEVICE_FILL[dtype]:
        return Bool(ctx) and ctx.value().api() != "cpu"
    return False


def _device_window[
    dtype: DType, n: Int, //, kind: StaticString
](
    mut result: Static[dtype, n],
    d: Float64,
    a0: Float64 = 0,
    a1: Float64 = 0,
    a2: Float64 = 0,
) raises where dtype.is_floating_point():
    """Fill `result` on its device from each index `i`, at `dtype`:
    `"cosine"` is `a0 - a1 cos(2 pi i / d) + a2 cos(4 pi i / d)`,
    `"bartlett"` is `1 - |2i/d - 1|`, `"kaiser"` is `I_0(a0 sqrt(1 -
    (2i/d - 1)^2)) * a1` (`a0` the `beta`, `a1` the reciprocal of
    `I_0(beta)` from the host), and `"ones"` is `1`. One launch."""
    var ctx = result.context()
    var dst = result.tile()
    var dd = Scalar[dtype](d)
    var c0 = Scalar[dtype](a0)
    var c1 = Scalar[dtype](a1)
    var c2 = Scalar[dtype](a2)

    @always_inline
    def body[
        width: Int, alignment: Int = 1
    ](coord: Coord) {var dst, var dd, var c0, var c1, var c2}:
        var i = Scalar[dtype](coord_to_index_list(coord)[0])
        var value: Scalar[dtype]
        comptime if kind == "cosine":
            var theta = Scalar[dtype](_TWO_PI) * i / dd
            value = c0 - c1 * _cos(theta) + c2 * _cos(2 * theta)
        elif kind == "bartlett":
            value = 1 - abs(2 * i / dd - 1)
        elif kind == "kaiser":
            var ratio = 2 * i / dd - 1
            var inside = max(1 - ratio * ratio, Scalar[dtype](0))
            value = _bessel_i0(c0 * _sqrt(inside)) * c1
        else:
            value = Scalar[dtype](1)
        dst.store[1](coord, value)

    elementwise[simd_width=1, target="gpu"](body, Coord(n), ctx)
    ctx.synchronize()


def _cosine_window[
    dtype: DType, n: Int
](
    a0: Float64,
    a1: Float64,
    a2: Float64,
    sym: Bool,
    ctx: Optional[DeviceContext],
) raises -> Static[dtype, n] where dtype.is_floating_point():
    """`a0 - a1 cos(2 pi i / d) + a2 cos(4 pi i / d)`, the generalized
    cosine window Hann, Hamming and Blackman are all instances of."""
    var d = _denominator(n, sym)
    comptime if _DEVICE_FILL[dtype]:
        if _on_device[dtype](ctx):
            var result = Static[dtype, n]._uninitialized(ctx.value())
            _device_window["cosine"](result, d, a0, a1, a2)
            return result^
    var values = List[Scalar[dtype]](capacity=n)
    for i in range(n):
        var theta = _TWO_PI * Float64(i) / d
        values.append(
            Scalar[dtype](a0 - a1 * _cos(theta) + a2 * _cos(2.0 * theta))
        )
    return Static[dtype, n](values^, _context(ctx))


def boxcar[
    dtype: DType, n: Int
](sym: Bool = True, ctx: Optional[DeviceContext] = None) raises -> Static[
    dtype, n
] where (dtype.is_floating_point() and n > 0):
    """The rectangular window: all ones. `scipy.signal.windows.boxcar`.
    `sym` is accepted for uniformity and changes nothing."""
    _ = sym
    comptime if _DEVICE_FILL[dtype]:
        if _on_device[dtype](ctx):
            var result = Static[dtype, n]._uninitialized(ctx.value())
            _device_window["ones"](result, 1.0)
            return result^
    return Static[dtype, n](
        List[Scalar[dtype]](length=n, fill=Scalar[dtype](1)), _context(ctx)
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
    return _cosine_window[dtype=dtype, n=n](0.5, 0.5, 0.0, sym, ctx)


def hamming[
    dtype: DType, n: Int
](sym: Bool = True, ctx: Optional[DeviceContext] = None) raises -> Static[
    dtype, n
] where (dtype.is_floating_point() and n > 0):
    """The Hamming window, `0.54 - 0.46 cos(2 pi i / d)`.
    `scipy.signal.windows.hamming(n, sym)`. The classic `0.54/0.46`
    coefficients, matching NumPy and SciPy, rather than the exactly optimal
    `0.53836/0.46164`."""
    return _cosine_window[dtype=dtype, n=n](0.54, 0.46, 0.0, sym, ctx)


def blackman[
    dtype: DType, n: Int
](sym: Bool = True, ctx: Optional[DeviceContext] = None) raises -> Static[
    dtype, n
] where (dtype.is_floating_point() and n > 0):
    """The Blackman window, `0.42 - 0.5 cos(2 pi i / d) + 0.08 cos(4 pi i
    / d)`. `scipy.signal.windows.blackman(n, sym)`: -58 dB sidelobes for a
    main lobe half again as wide as Hann's."""
    return _cosine_window[dtype=dtype, n=n](0.42, 0.5, 0.08, sym, ctx)


def bartlett[
    dtype: DType, n: Int
](sym: Bool = True, ctx: Optional[DeviceContext] = None) raises -> Static[
    dtype, n
] where (dtype.is_floating_point() and n > 0):
    """The Bartlett (triangular, zero-ended) window, `1 - |2i/d - 1|`.
    `scipy.signal.windows.bartlett(n, sym)`."""
    var d = _denominator(n, sym)
    comptime if _DEVICE_FILL[dtype]:
        if _on_device[dtype](ctx):
            var result = Static[dtype, n]._uninitialized(ctx.value())
            _device_window["bartlett"](result, d)
            return result^
    var values = List[Scalar[dtype]](capacity=n)
    for i in range(n):
        values.append(Scalar[dtype](1.0 - abs(2.0 * Float64(i) / d - 1.0)))
    return Static[dtype, n](values^, _context(ctx))


def _bessel_i0[
    dtype: DType
](x: Scalar[dtype]) -> Scalar[dtype] where dtype.is_floating_point():
    """The modified Bessel function `I_0(x)` by its power series,
    `sum (x/2)^{2k} / (k!)^2`, at `dtype` -- `Float64` on the host, the
    tensor's dtype inside a device fill.

    Converges for every `x` and the terms fall off factorially, so fifty
    terms are exact to double precision for the `beta` a window uses (up to
    a few tens), and the loop stops once a term no longer moves the sum at
    `dtype`. `numax.special` has no `i0` yet; when it does, this is the
    one call to replace.
    """
    var half = x / 2
    var term = Scalar[dtype](1)
    var total = Scalar[dtype](1)
    for k in range(1, 60):
        var q = half / Scalar[dtype](k)
        term *= q * q
        var next = total + term
        if next == total:
            break
        total = next
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
    Blackman. `I_0` is evaluated by its power series, per sample on the
    device when `ctx` is one.
    """
    var d = _denominator(n, sym)
    var scale = 1.0 / _bessel_i0(beta)
    comptime if _DEVICE_FILL[dtype]:
        if _on_device[dtype](ctx):
            var result = Static[dtype, n]._uninitialized(ctx.value())
            _device_window["kaiser"](result, d, beta, scale)
            return result^
    var values = List[Scalar[dtype]](capacity=n)
    for i in range(n):
        var ratio = 2.0 * Float64(i) / d - 1.0
        var inside = 1.0 - ratio * ratio
        if inside < 0:
            inside = 0.0
        values.append(Scalar[dtype](_bessel_i0(beta * _sqrt(inside)) * scale))
    return Static[dtype, n](values^, _context(ctx))


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
        return boxcar[dtype=dtype, n=n](sym, ctx)
    if name == "hann" or name == "hanning":
        return hann[dtype=dtype, n=n](sym, ctx)
    if name == "hamming":
        return hamming[dtype=dtype, n=n](sym, ctx)
    if name == "blackman":
        return blackman[dtype=dtype, n=n](sym, ctx)
    if name == "bartlett":
        return bartlett[dtype=dtype, n=n](sym, ctx)
    if name == "kaiser":
        return kaiser[dtype=dtype, n=n](beta, sym, ctx)
    raise Error(
        "get_window: unknown window '",
        name,
        "'; expected boxcar, hann, hamming, blackman, bartlett or kaiser",
    )
