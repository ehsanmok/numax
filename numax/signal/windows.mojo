"""Window functions as `numax.core.tensor.Tensor` factories: `boxcar`,
`hann`, `hamming`, `blackman`, `bartlett`, `kaiser`, `tukey`, `gaussian`,
`flattop`, `nuttall`, `chebwin` and `get_window`, with
`scipy.signal.windows`' symmetric and periodic forms. Every one but
`chebwin` fills on the device in one launch when `ctx` is a GPU;
`chebwin` needs a transform of its Chebyshev samples, done on the host in
`Float64` at the window's parameter-sized length and uploaded.

**Tier 2** by placement rather than by algorithm: a window is a table.
On a host context each factory evaluates its formula in `Float64`; on a
device context it evaluates the same formula at the tensor's dtype in one
`elementwise` launch, so nothing is uploaded -- `numax.core.tensor`'s
factories do the same, under the same gate (`_DEVICE_FILL`: an
accelerator build, not `float64`, which Metal cannot compile).
`numax.signal`'s `Array` tier has the same three cosine windows as compile-time
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

from std.math import (
    acos as _acos,
    acosh as _acosh,
    cos as _cos,
    cosh as _cosh,
    exp as _exp,
    floor as _floor,
    sqrt as _sqrt,
)
from layout import Coord, coord_to_index_list
from max.algorithm.functional import elementwise
from max.gpu.host import DeviceContext

from ..core.tensor import _DEVICE_FILL, Static
from std.collections import Array
from ..core.numeric import FloatLike
from ._array.signal import blackman as _array_blackman
from ._array.signal import hamming as _array_hamming
from ._array.signal import hann as _array_hann

comptime _TWO_PI = 6.283185307179586
comptime _PI = 3.141592653589793


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
    a3: Float64 = 0,
    a4: Float64 = 0,
) raises where dtype.is_floating_point():
    """Fill `result` on its device from each index `i`, at `dtype`:
    `"cosine"` is `a0 - a1 cos(2 pi i / d) + a2 cos(4 pi i / d)`,
    `"bartlett"` is `1 - |2i/d - 1|`, `"kaiser"` is `I_0(a0 sqrt(1 -
    (2i/d - 1)^2)) * a1` (`a0` the `beta`, `a1` the reciprocal of
    `I_0(beta)` from the host), `"cosine5"` is the five-term
    `sum_k (-1)^k a_k cos(2 pi k i / d)`, `"tukey"` is the tapered cosine
    with `a0` the `alpha` and `a1` the taper width, `"gaussian"` is
    `exp(-((i - d/2) / a0)^2 / 2)`, and `"ones"` is `1`. One launch."""
    var ctx = result.context()
    var dst = result.tile()
    var dd = Scalar[dtype](d)
    var c0 = Scalar[dtype](a0)
    var c1 = Scalar[dtype](a1)
    var c2 = Scalar[dtype](a2)
    var c3 = Scalar[dtype](a3)
    var c4 = Scalar[dtype](a4)

    @always_inline
    def body[
        width: Int, alignment: Int = 1
    ](coord: Coord) {var dst, var dd, var c0, var c1, var c2, var c3, var c4}:
        var i = Scalar[dtype](coord_to_index_list(coord)[0])
        var value: Scalar[dtype]
        comptime if kind == "cosine":
            var theta = Scalar[dtype](_TWO_PI) * i / dd
            value = c0 - c1 * _cos(theta) + c2 * _cos(2 * theta)
        elif kind == "cosine5":
            var theta = Scalar[dtype](_TWO_PI) * i / dd
            value = (
                c0
                - c1 * _cos(theta)
                + c2 * _cos(2 * theta)
                - c3 * _cos(3 * theta)
                + c4 * _cos(4 * theta)
            )
        elif kind == "tukey":
            var pi = Scalar[dtype](_PI)
            if i <= c1:
                value = 0.5 * (1 + _cos(pi * (-1 + 2 * i / c0 / dd)))
            elif i >= dd - c1:
                value = 0.5 * (1 + _cos(pi * (-2 / c0 + 1 + 2 * i / c0 / dd)))
            else:
                value = Scalar[dtype](1)
        elif kind == "gaussian":
            var t = (i - dd / 2) / c0
            value = _exp(-0.5 * t * t)
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
    `sym` is accepted for uniformity and changes nothing.

    Parameters:
        dtype: Floating-point element type of the window.
        n: Window length, at least 1.

    Args:
        sym: Accepted for uniformity with the other windows; ignored.
        ctx: Device to allocate on and fill from; `None` uses the host.

    Returns:
        A length-`n` tensor of ones at `dtype` on `ctx`.

    Raises:
        If creating the host context or allocating the tensor fails.
    """
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

    Parameters:
        dtype: Floating-point element type of the window.
        n: Window length, at least 1.

    Args:
        sym: Symmetric window (ends match) when `True`; periodic (sampled
            for `n + 1` points, last dropped) when `False`.
        ctx: Device to allocate on and fill from; `None` uses the host.

    Returns:
        A length-`n` window tensor at `dtype` on `ctx`.

    Raises:
        If creating the host context or allocating the tensor fails.
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
    `0.53836/0.46164`.

    Parameters:
        dtype: Floating-point element type of the window.
        n: Window length, at least 1.

    Args:
        sym: Symmetric window (ends match) when `True`; periodic (sampled
            for `n + 1` points, last dropped) when `False`.
        ctx: Device to allocate on and fill from; `None` uses the host.

    Returns:
        A length-`n` window tensor at `dtype` on `ctx`.

    Raises:
        If creating the host context or allocating the tensor fails.
    """
    return _cosine_window[dtype=dtype, n=n](0.54, 0.46, 0.0, sym, ctx)


def blackman[
    dtype: DType, n: Int
](sym: Bool = True, ctx: Optional[DeviceContext] = None) raises -> Static[
    dtype, n
] where (dtype.is_floating_point() and n > 0):
    """The Blackman window, `0.42 - 0.5 cos(2 pi i / d) + 0.08 cos(4 pi i
    / d)`. `scipy.signal.windows.blackman(n, sym)`: -58 dB sidelobes for a
    main lobe half again as wide as Hann's.

    Parameters:
        dtype: Floating-point element type of the window.
        n: Window length, at least 1.

    Args:
        sym: Symmetric window (ends match) when `True`; periodic (sampled
            for `n + 1` points, last dropped) when `False`.
        ctx: Device to allocate on and fill from; `None` uses the host.

    Returns:
        A length-`n` window tensor at `dtype` on `ctx`.

    Raises:
        If creating the host context or allocating the tensor fails.
    """
    return _cosine_window[dtype=dtype, n=n](0.42, 0.5, 0.08, sym, ctx)


def bartlett[
    dtype: DType, n: Int
](sym: Bool = True, ctx: Optional[DeviceContext] = None) raises -> Static[
    dtype, n
] where (dtype.is_floating_point() and n > 0):
    """The Bartlett (triangular, zero-ended) window, `1 - |2i/d - 1|`.
    `scipy.signal.windows.bartlett(n, sym)`.

    Parameters:
        dtype: Floating-point element type of the window.
        n: Window length, at least 1.

    Args:
        sym: Symmetric window (ends match) when `True`; periodic (sampled
            for `n + 1` points, last dropped) when `False`.
        ctx: Device to allocate on and fill from; `None` uses the host.

    Returns:
        A length-`n` window tensor at `dtype` on `ctx`.

    Raises:
        If creating the host context or allocating the tensor fails.
    """
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

    Parameters:
        dtype: Floating-point element type of the window.
        n: Window length, at least 1.

    Args:
        beta: Shape parameter; `0` is the boxcar, larger values lower the
            sidelobes and widen the main lobe.
        sym: Symmetric window (ends match) when `True`; periodic (sampled
            for `n + 1` points, last dropped) when `False`.
        ctx: Device to allocate on and fill from; `None` uses the host.

    Returns:
        A length-`n` Kaiser window tensor at `dtype` on `ctx`.

    Raises:
        If creating the host context or allocating the tensor fails.
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


def _cosine5_window[
    dtype: DType, n: Int
](a: List[Float64], sym: Bool, ctx: Optional[DeviceContext]) raises -> Static[
    dtype, n
] where dtype.is_floating_point():
    """`sum_k (-1)^k a_k cos(2 pi k i / d)` over up to five terms,
    `scipy.signal.windows.general_cosine` at the coefficients given."""
    var c = List[Float64](length=5, fill=0.0)
    for k in range(len(a)):
        c[k] = a[k]
    var d = _denominator(n, sym)
    comptime if _DEVICE_FILL[dtype]:
        if _on_device[dtype](ctx):
            var result = Static[dtype, n]._uninitialized(ctx.value())
            _device_window["cosine5"](result, d, c[0], c[1], c[2], c[3], c[4])
            return result^
    var values = List[Scalar[dtype]](capacity=n)
    for i in range(n):
        var theta = _TWO_PI * Float64(i) / d
        var w = 0.0
        for k in range(5):
            w += (-1.0 if k % 2 == 1 else 1.0) * c[k] * _cos(Float64(k) * theta)
        values.append(Scalar[dtype](w))
    return Static[dtype, n](values^, _context(ctx))


def flattop[
    dtype: DType, n: Int
](sym: Bool = True, ctx: Optional[DeviceContext] = None) raises -> Static[
    dtype, n
] where (dtype.is_floating_point() and n > 0):
    """The flat-top window, a five-term cosine sum whose main lobe is flat
    to within `0.01 dB`, so a tone's amplitude reads correctly off any bin
    it falls in. `scipy.signal.windows.flattop(n, sym)`.

    SciPy's coefficients `0.21557895, 0.41663158, 0.277263158,
    0.083578947, 0.006947368`; the price of the flat top is the widest main
    lobe of the windows here.

    Parameters:
        dtype: Floating-point element type of the window.
        n: Window length, at least 1.

    Args:
        sym: Symmetric window when `True`; periodic when `False`.
        ctx: Device to allocate on and fill from; `None` uses the host.

    Returns:
        A length-`n` window tensor at `dtype` on `ctx`.

    Raises:
        If creating the host context or allocating the tensor fails.
    """
    return _cosine5_window[dtype=dtype, n=n](
        [0.21557895, 0.41663158, 0.277263158, 0.083578947, 0.006947368],
        sym,
        ctx,
    )


def nuttall[
    dtype: DType, n: Int
](sym: Bool = True, ctx: Optional[DeviceContext] = None) raises -> Static[
    dtype, n
] where (dtype.is_floating_point() and n > 0):
    """Nuttall's minimum four-term Blackman-Harris window, sidelobes near
    `-93 dB`. `scipy.signal.windows.nuttall(n, sym)`.

    Coefficients `0.3635819, 0.4891775, 0.1365995, 0.0106411`, SciPy's.

    Parameters:
        dtype: Floating-point element type of the window.
        n: Window length, at least 1.

    Args:
        sym: Symmetric window when `True`; periodic when `False`.
        ctx: Device to allocate on and fill from; `None` uses the host.

    Returns:
        A length-`n` window tensor at `dtype` on `ctx`.

    Raises:
        If creating the host context or allocating the tensor fails.
    """
    return _cosine5_window[dtype=dtype, n=n](
        [0.3635819, 0.4891775, 0.1365995, 0.0106411], sym, ctx
    )


def tukey[
    dtype: DType, n: Int
](
    alpha: Float64 = 0.5,
    sym: Bool = True,
    ctx: Optional[DeviceContext] = None,
) raises -> Static[dtype, n] where (dtype.is_floating_point() and n > 0):
    """The Tukey (tapered cosine) window: flat in the middle, a half-Hann
    taper over a fraction `alpha` of the length split between the ends.
    `scipy.signal.windows.tukey(n, alpha, sym)`.

    `alpha <= 0` is `boxcar` and `alpha >= 1` is `hann`, as in SciPy; in
    between, the taper is `floor(alpha (M - 1) / 2) + 1` samples at each
    end, `M` the (for a periodic window, extended) length.

    Parameters:
        dtype: Floating-point element type of the window.
        n: Window length, at least 1.

    Args:
        alpha: The tapered fraction, clamped to `[0, 1]` by the two limits.
        sym: Symmetric window when `True`; periodic when `False`.
        ctx: Device to allocate on and fill from; `None` uses the host.

    Returns:
        A length-`n` window tensor at `dtype` on `ctx`.

    Raises:
        If creating the host context or allocating the tensor fails.
    """
    if alpha <= 0:
        return boxcar[dtype=dtype, n=n](sym, ctx)
    if alpha >= 1:
        return hann[dtype=dtype, n=n](sym, ctx)
    var d = _denominator(n, sym)
    var width = _floor(alpha * d / 2.0)
    comptime if _DEVICE_FILL[dtype]:
        if _on_device[dtype](ctx):
            var result = Static[dtype, n]._uninitialized(ctx.value())
            _device_window["tukey"](result, d, alpha, width)
            return result^
    var values = List[Scalar[dtype]](capacity=n)
    for j in range(n):
        var i = Float64(j)
        var w: Float64
        if i <= width:
            w = 0.5 * (1.0 + _cos(_PI * (-1.0 + 2.0 * i / alpha / d)))
        elif i >= d - width:
            w = 0.5 * (
                1.0 + _cos(_PI * (-2.0 / alpha + 1.0 + 2.0 * i / alpha / d))
            )
        else:
            w = 1.0
        values.append(Scalar[dtype](w))
    return Static[dtype, n](values^, _context(ctx))


def gaussian[
    dtype: DType, n: Int
](
    std: Float64, sym: Bool = True, ctx: Optional[DeviceContext] = None
) raises -> Static[dtype, n] where (dtype.is_floating_point() and n > 0):
    """The Gaussian window `exp(-((i - (M-1)/2) / std)^2 / 2)`.
    `scipy.signal.windows.gaussian(n, std, sym)`.

    Named `gaussian` as SciPy's is; `numax.gaussian` at the root is the
    `FloatLike` kernel `exp(-x^2)`, so this one is reached as
    `numax.signal.gaussian`.

    Parameters:
        dtype: Floating-point element type of the window.
        n: Window length, at least 1.

    Args:
        std: The standard deviation in samples, positive.
        sym: Symmetric window when `True`; periodic when `False`.
        ctx: Device to allocate on and fill from; `None` uses the host.

    Returns:
        A length-`n` window tensor at `dtype` on `ctx`.

    Raises:
        If `std` is not positive, or creating the host context or
        allocating the tensor fails.
    """
    if std <= 0:
        raise Error("gaussian: std must be positive")
    var d = _denominator(n, sym)
    comptime if _DEVICE_FILL[dtype]:
        if _on_device[dtype](ctx):
            var result = Static[dtype, n]._uninitialized(ctx.value())
            _device_window["gaussian"](result, d, std)
            return result^
    var values = List[Scalar[dtype]](capacity=n)
    for i in range(n):
        var t = (Float64(i) - d / 2.0) / std
        values.append(Scalar[dtype](_exp(-0.5 * t * t)))
    return Static[dtype, n](values^, _context(ctx))


def chebwin[
    dtype: DType, n: Int
](
    at: Float64, sym: Bool = True, ctx: Optional[DeviceContext] = None
) raises -> Static[dtype, n] where (dtype.is_floating_point() and n > 0):
    """The Dolph-Chebyshev window: every sidelobe at exactly `-at` dB, the
    narrowest main lobe any window of this length can have at that level.
    `scipy.signal.windows.chebwin(n, at, sym)`.

    SciPy's construction: the Chebyshev polynomial of order `M - 1`
    sampled at `beta cos(pi k / M)`, with `cosh(acosh(10^(at/20)) / (M-1))`
    for `beta`, transformed, reordered and scaled to a peak of 1. The
    transform is a direct real DFT in `Float64` on the host -- the window
    is parameter-sized, as a filter design is -- and the result is uploaded
    once. SciPy warns below `45 dB`, where the window's end samples spike;
    that is the window's nature, not an error, so this does neither.

    Parameters:
        dtype: Floating-point element type of the window.
        n: Window length, at least 1.

    Args:
        at: The sidelobe attenuation in dB; its absolute value is used.
        sym: Symmetric window when `True`; periodic when `False`.
        ctx: Device to allocate on; `None` uses the host.

    Returns:
        A length-`n` window tensor at `dtype` on `ctx`.

    Raises:
        If creating the host context or allocating the tensor fails.
    """
    if n == 1:
        return boxcar[dtype=dtype, n=n](sym, ctx)
    var m = n if sym else n + 1
    var order = Float64(m - 1)
    var beta = _cosh(_acosh(10.0 ** (abs(at) / 20.0)) / order)
    var p = List[Float64](capacity=m)
    for k in range(m):
        var x = beta * _cos(_PI * Float64(k) / Float64(m))
        if x > 1:
            p.append(_cosh(order * _acosh(x)))
        elif x < -1:
            var sign = 1.0 if m % 2 == 1 else -1.0
            p.append(sign * _cosh(order * _acosh(-x)))
        else:
            p.append(_cos(order * _acos(x)))
    # `real(fft(p))`, with the half-sample shift `exp(i pi j / M)` folded
    # in for even `M`.
    var w = List[Float64](capacity=m)
    for k in range(m):
        var acc = 0.0
        for j in range(m):
            var shift = 0.0 if m % 2 == 1 else _PI * Float64(j) / Float64(m)
            acc += p[j] * _cos(
                shift - 2.0 * _PI * Float64(j * k % m) / Float64(m)
            )
        w.append(acc)
    var full = List[Float64](capacity=m)
    if m % 2 == 1:
        var half = (m + 1) // 2
        for i in range(half - 1, 0, -1):
            full.append(w[i])
        for i in range(half):
            full.append(w[i])
    else:
        var half = m // 2 + 1
        for i in range(half - 1, 0, -1):
            full.append(w[i])
        for i in range(1, half):
            full.append(w[i])
    var peak = full[0]
    for i in range(len(full)):
        peak = max(peak, full[i])
    var values = List[Scalar[dtype]](capacity=n)
    for i in range(n):
        values.append(Scalar[dtype](full[i] / peak))
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

    Parameters:
        dtype: Floating-point element type of the window.
        n: Window length, at least 1.

    Args:
        name: Window name: `"boxcar"`, `"hann"`, `"hanning"`, `"hamming"`,
            `"blackman"`, `"bartlett"` or `"kaiser"`.
        beta: Kaiser shape parameter; ignored by every other window.
        fftbins: Periodic window when `True` (the default), symmetric when
            `False`.
        ctx: Device to allocate on and fill from; `None` uses the host.

    Returns:
        The named length-`n` window tensor at `dtype` on `ctx`.

    Raises:
        If `name` is not a known window, or if allocation fails.
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
    if name == "flattop":
        return flattop[dtype=dtype, n=n](sym, ctx)
    if name == "nuttall":
        return nuttall[dtype=dtype, n=n](sym, ctx)
    raise Error(
        "get_window: unknown window '",
        name,
        (
            "'; expected boxcar, hann, hamming, blackman, bartlett, kaiser,"
            " flattop or nuttall"
        ),
    )


def blackman[T: FloatLike, n: Int]() -> Array[T, n]:
    """The `Array`-tier overload: one problem in registers, generic over
    the `FloatLike` conformer. The algorithm and its bound are documented
    at `numax.signal._array.signal.blackman`.

    Parameters:
        T: `FloatLike` conformer of each element.
        n: Window length.

    Returns:
        The length-`n` symmetric Blackman window as an `Array`.
    """
    return _array_blackman[T=T, n=n]()


def hamming[T: FloatLike, n: Int]() -> Array[T, n]:
    """The `Array`-tier overload: one problem in registers, generic over
    the `FloatLike` conformer. The algorithm and its bound are documented
    at `numax.signal._array.signal.hamming`.

    Parameters:
        T: `FloatLike` conformer of each element.
        n: Window length.

    Returns:
        The length-`n` symmetric Hamming window as an `Array`.
    """
    return _array_hamming[T=T, n=n]()


def hann[T: FloatLike, n: Int]() -> Array[T, n]:
    """The `Array`-tier overload: one problem in registers, generic over
    the `FloatLike` conformer. The algorithm and its bound are documented
    at `numax.signal._array.signal.hann`.

    Parameters:
        T: `FloatLike` conformer of each element.
        n: Window length.

    Returns:
        The length-`n` symmetric Hann window as an `Array`.
    """
    return _array_hann[T=T, n=n]()
