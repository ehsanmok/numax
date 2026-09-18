"""Spectral estimation over `numax.core.array.Tensor`: `periodogram`,
`welch`, `spectrogram`, `stft` and the analytic signal `hilbert`, with
`scipy.signal`'s conventions.

**This module is tier 2**, like the rest of `numax.signal` over `Tensor`,
and it is the part that most wants the `Tensor` tier: a spectral estimate
frames a long recording into many short segments and transforms every one
of them, which is exactly the batched shape `numax.fft`'s lane engine runs
as one set of launches. Nothing here touches the host between the framing
and the power, except to build the window.

## One framing, four estimators

Every estimator here is the same three steps with different bookkeeping.
Frame: one launch writes the `frames x nperseg` matrix of segments, each
segment gathered from the signal at its offset, mean-removed when the
estimator detrends, and multiplied by the window. Transform: `_dft` along
the rows -- `frames` transforms of length `nperseg` at once, radix-2 or
Bluestein as the segment length decides. Project: one launch takes the
one-sided spectrum and scales it -- power per unit frequency averaged over
frames for `welch`, per frame for `spectrogram`, the complex values for
`stft`. `periodogram` is `welch` with one frame the length of the signal.

## SciPy's conventions, kept

`scaling="density"` divides by `fs * sum(w^2)` and `"spectrum"` by
`sum(w)^2`; the one-sided spectrum doubles every bin but DC and, for an
even segment, Nyquist. `welch` and `spectrogram` remove each segment's
mean (`detrend="constant"`); `stft` does not, pads `nperseg / 2` zeros at
both ends (`boundary="zeros"`), extends to a whole number of steps
(`padded=True`) and scales by `sum(w)` (`scaling="spectrum"`). The windows
are `get_window`'s periodic ones, as SciPy's are. `noverlap` defaults to
`nperseg // 2` for `welch` and `stft` and `nperseg // 8` for
`spectrogram`, which are SciPy's defaults too. Frame counts and bin counts
are compile-time because they shape the result tensors; `noverlap`'s range
is a `comptime assert` rather than a `where` clause because its default is
`nperseg // 2`, which the `where` prover cannot evaluate.

## The MAX gate

Nothing: MAX has no spectral estimation, and its only transform is the
inverse real one `numax.fft` records. **Extend** on numax's own engine.

## What is not here

The `"psd"`/`"complex"` modes of `spectrogram` other than PSD,
`average="median"`, `nfft` padding and the `return_onesided=False`
two-sided spectra wait on a caller. `csd`, `coherence` and `istft` are
here now.
"""

from layout import Coord, coord_to_index_list
from max.algorithm.functional import elementwise
from max.gpu.host import DeviceContext

from ..core.array import Static, zeros
from ..fft.fft import Spectrum, _as_matrix, _dft, fft, ifft, irfft
from .windows import get_window


def _frame_count(n: Int, nperseg: Int, noverlap: Int) -> Int:
    """How many `nperseg`-long segments at hop `nperseg - noverlap` fit in
    `n` samples; evaluated at compile time for the result types."""
    return (n - nperseg) // (nperseg - noverlap) + 1


def _padded_length(n: Int, nperseg: Int, step: Int) -> Int:
    """`stft`'s extended length: `nperseg / 2` zeros at each end, then
    enough more to make the last frame land on a whole step."""
    var extended = n + 2 * (nperseg // 2)
    var remainder = (extended - nperseg) % step
    return extended if remainder == 0 else extended + step - remainder


def _framed[
    dtype: DType,
    n: Int,
    nperseg: Int,
    step: Int,
    frames: Int,
    gpu: Bool,
](
    mut x: Static[dtype, n],
    mut window: Static[dtype, nperseg],
    offset: Int,
    detrend: Bool,
) raises -> Spectrum[dtype, frames, nperseg]:
    """Frame, detrend and window, then transform every frame: the spectrum
    of segment `f` in row `f`.

    Segment `f` starts at sample `f * step - offset`; samples outside the
    signal read as zero, which is how `stft`'s boundary padding is spelled.
    Detrending needs each frame's mean before the frame is written, so it
    is one launch over the frames ahead of the launch over the elements.
    """
    var ctx = x.context()
    var xs = x.view()
    var means = Static[dtype, frames]._uninitialized(ctx)
    var ms = means.view()
    var shift = offset

    @always_inline
    def frame_mean[
        w: Int, alignment: Int = 1
    ](coord: Coord) {var xs, var ms, var shift}:
        var f = coord_to_index_list(coord)[0]
        var total = Scalar[dtype](0)
        for j in range(nperseg):
            var src = f * step + j - shift
            if src >= 0 and src < n:
                total += xs[Coord(src)]
        ms.store[1](Coord(f), total / Scalar[dtype](nperseg))

    if detrend:
        elementwise[simd_width=1, target="gpu" if gpu else "cpu"](
            frame_mean, Coord(frames), ctx
        )

    var re = Static[dtype, frames, nperseg]._uninitialized(ctx)
    var im = Static[dtype, frames, nperseg]._uninitialized(ctx)
    var rs = re.view()
    var ims = im.view()
    var ws = window.view()
    var remove_mean = detrend

    @always_inline
    def gather[
        w: Int, alignment: Int = 1
    ](coord: Coord) {
        var xs, var ms, var ws, var rs, var ims, var shift, var remove_mean
    }:
        var idx = coord_to_index_list(coord)
        var f = idx[0]
        var j = idx[1]
        var src = f * step + j - shift
        var sample = Scalar[dtype](0)
        if src >= 0 and src < n:
            sample = xs[Coord(src)]
        if remove_mean:
            sample -= ms[Coord(f)]
        rs.store[1](Coord(f, j), sample * ws[Coord(j)])
        ims.store[1](Coord(f, j), Scalar[dtype](0))

    elementwise[simd_width=1, target="gpu" if gpu else "cpu"](
        gather, Coord(frames, nperseg), ctx
    )

    var out_re = Static[dtype, frames, nperseg]._uninitialized(ctx)
    var out_im = Static[dtype, frames, nperseg]._uninitialized(ctx)
    _dft[dtype, frames, nperseg, gpu, False](
        _as_matrix[dtype, frames, nperseg](re),
        _as_matrix[dtype, frames, nperseg](im),
        _as_matrix[dtype, frames, nperseg](out_re),
        _as_matrix[dtype, frames, nperseg](out_im),
        ctx,
    )
    _ = means^
    _ = re^
    _ = im^
    return (out_re^, out_im^)


@always_inline
def _one_sided_factor(k: Int, nperseg: Int) -> Int:
    """SciPy's one-sided doubling: every bin but DC, and but Nyquist when
    the segment is even. An `Int`, cast to the working dtype at the call,
    because a `Float64` here would put a `double` inside a device kernel
    and Metal rejects that."""
    if k == 0:
        return 1
    if nperseg % 2 == 0 and k == nperseg // 2:
        return 1
    return 2


def _window_sums[
    dtype: DType, nperseg: Int
](mut window: Static[dtype, nperseg]) raises -> Tuple[Float64, Float64]:
    """`sum(w)` and `sum(w^2)`, the two normalizations."""
    var ws = window.to_host()
    var total = 0.0
    var squares = 0.0
    for j in range(nperseg):
        var v = Float64(ws[j])
        total += v
        squares += v * v
    return (total, squares)


def _frequencies[
    dtype: DType, nperseg: Int
](ctx: DeviceContext, fs: Float64) raises -> Static[dtype, nperseg // 2 + 1]:
    """`k * fs / nperseg` for the one-sided bins."""
    var values = List[Scalar[dtype]](capacity=nperseg // 2 + 1)
    for k in range(nperseg // 2 + 1):
        values.append(Scalar[dtype](Float64(k) * fs / Float64(nperseg)))
    return Static[dtype, nperseg // 2 + 1](ctx, values^)


def _times[
    dtype: DType, frames: Int
](ctx: DeviceContext, first: Float64, step: Int, fs: Float64) raises -> Static[
    dtype, frames
]:
    """`(first + f * step) / fs`: the centre of each frame in seconds."""
    var values = List[Scalar[dtype]](capacity=frames)
    for f in range(frames):
        values.append(Scalar[dtype]((first + Float64(f * step)) / fs))
    return Static[dtype, frames](ctx, values^)


struct Periodogram[dtype: DType, keep: Int](Movable):
    """What `periodogram` and `welch` return: the one-sided frequency grid
    and the power spectral density on it, `scipy.signal.welch`'s `(f,
    Pxx)` as a struct rather than a tuple -- a `Tuple` of two `Tensor`s
    cannot be destructured in Mojo 1.0."""

    var frequencies: Static[Self.dtype, Self.keep]
    var power: Static[Self.dtype, Self.keep]

    def __init__(
        out self,
        var frequencies: Static[Self.dtype, Self.keep],
        var power: Static[Self.dtype, Self.keep],
    ):
        self.frequencies = frequencies^
        self.power = power^


struct Spectrogram[dtype: DType, keep: Int, frames: Int](Movable):
    """What `spectrogram` returns: `scipy.signal.spectrogram`'s `(f, t,
    Sxx)`, with `power` in SciPy's `(frequencies, times)` orientation."""

    var frequencies: Static[Self.dtype, Self.keep]
    var times: Static[Self.dtype, Self.frames]
    var power: Static[Self.dtype, Self.keep, Self.frames]

    def __init__(
        out self,
        var frequencies: Static[Self.dtype, Self.keep],
        var times: Static[Self.dtype, Self.frames],
        var power: Static[Self.dtype, Self.keep, Self.frames],
    ):
        self.frequencies = frequencies^
        self.times = times^
        self.power = power^


struct STFT[dtype: DType, keep: Int, frames: Int](Movable):
    """What `stft` returns: `scipy.signal.stft`'s `(f, t, Zxx)`, the complex
    `Zxx` as a real/imaginary pair in `(frequencies, times)` orientation,
    since a `dtype`-monomorphic tensor holds no `Complex`."""

    var frequencies: Static[Self.dtype, Self.keep]
    var times: Static[Self.dtype, Self.frames]
    var real: Static[Self.dtype, Self.keep, Self.frames]
    var imag: Static[Self.dtype, Self.keep, Self.frames]

    def __init__(
        out self,
        var frequencies: Static[Self.dtype, Self.keep],
        var times: Static[Self.dtype, Self.frames],
        var real: Static[Self.dtype, Self.keep, Self.frames],
        var imag: Static[Self.dtype, Self.keep, Self.frames],
    ):
        self.frequencies = frequencies^
        self.times = times^
        self.real = real^
        self.imag = imag^


def _averaged_power[
    dtype: DType, frames: Int, nperseg: Int, gpu: Bool
](
    var spectra: Spectrum[dtype, frames, nperseg], scale: Float64
) raises -> Static[dtype, nperseg // 2 + 1]:
    """`scale * mean_f |X[f, k]|^2`, one-sided, one lane per bin."""
    comptime keep = nperseg // 2 + 1
    var ctx = spectra[0].context()
    var out = Static[dtype, keep]._uninitialized(ctx)
    # The two halves of a `Spectrum` share the tuple's origin, so their
    # tracked views read as aliasing when a body captures both; erase to
    # `MutAnyOrigin`, which is the type the kernels take anyway. The owner
    # outlives every launch below.
    var re = spectra[0].view().as_unsafe_any_origin()
    var im = spectra[1].view().as_unsafe_any_origin()
    var ys = out.view()
    var factor = Scalar[dtype](scale / Float64(frames))

    @always_inline
    def lane[
        w: Int, alignment: Int = 1
    ](coord: Coord) {var re, var im, var ys, var factor}:
        var k = coord_to_index_list(coord)[0]
        var total = Scalar[dtype](0)
        for f in range(frames):
            var a = re[Coord(f, k)]
            var b = im[Coord(f, k)]
            total += a * a + b * b
        var doubled = Scalar[dtype](_one_sided_factor(k, nperseg))
        ys.store[1](Coord(k), total * factor * doubled)

    elementwise[simd_width=1, target="gpu" if gpu else "cpu"](
        lane, Coord(keep), ctx
    )
    ctx.synchronize()
    _ = spectra^
    return out^


def periodogram[
    dtype: DType, n: Int, gpu: Bool = False
](
    mut x: Static[dtype, n],
    fs: Float64 = 1.0,
    window: StaticString = "boxcar",
    scaling: StaticString = "density",
) raises -> Periodogram[dtype, n // 2 + 1] where (
    dtype.is_floating_point() and n > 1 and n > 0
):
    """The power spectral density of `x` from one transform of the whole
    signal, mean removed, windowed by `window` (a rectangle by default).
    `scipy.signal.periodogram(x, fs, window, detrend="constant",
    scaling)`.

    `scaling="density"` is power per unit frequency (`V^2/Hz`, dividing by
    `fs * sum(w^2)`); `"spectrum"` is power per bin (dividing by
    `sum(w)^2`). One frame of `welch` -- the estimator with the least bias
    and the most variance, which averaging over frames trades the other
    way.
    """
    if not (scaling == "density" or scaling == "spectrum"):
        raise Error(
            "periodogram: unknown scaling '",
            scaling,
            "'; expected 'density' or 'spectrum'",
        )
    var ctx = x.context()
    var win = get_window[dtype, n](window, ctx=ctx)
    var sums = _window_sums(win)
    var scale = 1.0 / (fs * sums[1]) if scaling == "density" else 1.0 / (
        sums[0] * sums[0]
    )
    var spectra = _framed[dtype, n, n, 1, 1, gpu](x, win, 0, True)
    var power = _averaged_power[dtype, 1, n, gpu](spectra^, scale)
    return Periodogram[dtype, n // 2 + 1](
        _frequencies[dtype, n](ctx, fs), power^
    )


def welch[
    dtype: DType,
    n: Int,
    nperseg: Int,
    noverlap: Int = nperseg // 2,
    gpu: Bool = False,
](
    mut x: Static[dtype, n],
    fs: Float64 = 1.0,
    window: StaticString = "hann",
    detrend: Bool = True,
    scaling: StaticString = "density",
) raises -> Periodogram[dtype, nperseg // 2 + 1] where (
    dtype.is_floating_point() and n >= nperseg and nperseg > 1 and nperseg > 0
):
    """Welch's power spectral density: the signal in `nperseg`-long frames
    overlapping by `noverlap`, each mean-removed (`detrend`), windowed and
    transformed, the periodograms averaged. `scipy.signal.welch(x, fs,
    window, nperseg, noverlap, detrend, scaling)`.

    Hann and half overlap by default, SciPy's choice and the usual one: the
    window's sidelobes keep leakage down and half overlap recovers the
    samples the window's taper discounts. `frames` transforms as one batch
    on the lane engine; `noverlap` is a parameter because the frame count
    shapes the work. `scaling` as for `periodogram`.
    """
    if not (scaling == "density" or scaling == "spectrum"):
        raise Error(
            "welch: unknown scaling '",
            scaling,
            "'; expected 'density' or 'spectrum'",
        )
    comptime assert (
        noverlap >= 0 and noverlap < nperseg
    ), "noverlap must lie in [0, nperseg)"
    comptime step = nperseg - noverlap
    comptime frames = _frame_count(n, nperseg, noverlap)
    var ctx = x.context()
    var win = get_window[dtype, nperseg](window, ctx=ctx)
    var sums = _window_sums(win)
    var scale = 1.0 / (fs * sums[1]) if scaling == "density" else 1.0 / (
        sums[0] * sums[0]
    )
    var spectra = _framed[dtype, n, nperseg, step, frames, gpu](
        x, win, 0, detrend
    )
    var power = _averaged_power[dtype, frames, nperseg, gpu](spectra^, scale)
    return Periodogram[dtype, nperseg // 2 + 1](
        _frequencies[dtype, nperseg](ctx, fs), power^
    )


def spectrogram[
    dtype: DType,
    n: Int,
    nperseg: Int,
    noverlap: Int = nperseg // 8,
    gpu: Bool = False,
](
    mut x: Static[dtype, n],
    fs: Float64 = 1.0,
    window: StaticString = "hann",
    detrend: Bool = True,
    scaling: StaticString = "density",
) raises -> Spectrogram[
    dtype, nperseg // 2 + 1, _frame_count(n, nperseg, noverlap)
] where (
    dtype.is_floating_point() and n >= nperseg and nperseg > 1 and nperseg > 0
):
    """The power spectral density of every frame, unaveraged: `welch`'s
    frames laid out in time. `scipy.signal.spectrogram(x, fs, window,
    nperseg, noverlap, detrend, scaling, mode="psd")`.

    `power[k, f]` is bin `k` of frame `f`, whose centre is `times[f] =
    (nperseg / 2 + f * step) / fs`. SciPy's default `noverlap` here is
    `nperseg // 8`, an eighth rather than `welch`'s half, and its default
    window is a Tukey taper this module does not have -- pass `"hann"` (the
    default here) or any `get_window` name.
    """
    if not (scaling == "density" or scaling == "spectrum"):
        raise Error(
            "spectrogram: unknown scaling '",
            scaling,
            "'; expected 'density' or 'spectrum'",
        )
    comptime assert (
        noverlap >= 0 and noverlap < nperseg
    ), "noverlap must lie in [0, nperseg)"
    comptime step = nperseg - noverlap
    comptime frames = _frame_count(n, nperseg, noverlap)
    comptime keep = nperseg // 2 + 1
    var ctx = x.context()
    var win = get_window[dtype, nperseg](window, ctx=ctx)
    var sums = _window_sums(win)
    var scale = 1.0 / (fs * sums[1]) if scaling == "density" else 1.0 / (
        sums[0] * sums[0]
    )
    var spectra = _framed[dtype, n, nperseg, step, frames, gpu](
        x, win, 0, detrend
    )

    var power = Static[dtype, keep, frames]._uninitialized(ctx)
    # The two halves of a `Spectrum` share the tuple's origin, so their
    # tracked views read as aliasing when a body captures both; erase to
    # `MutAnyOrigin`, which is the type the kernels take anyway. The owner
    # outlives every launch below.
    var re = spectra[0].view().as_unsafe_any_origin()
    var im = spectra[1].view().as_unsafe_any_origin()
    var ps = power.view()
    var factor = Scalar[dtype](scale)

    @always_inline
    def lane[
        w: Int, alignment: Int = 1
    ](coord: Coord) {var re, var im, var ps, var factor}:
        var idx = coord_to_index_list(coord)
        var k = idx[0]
        var f = idx[1]
        var a = re[Coord(f, k)]
        var b = im[Coord(f, k)]
        var doubled = Scalar[dtype](_one_sided_factor(k, nperseg))
        ps.store[1](Coord(k, f), (a * a + b * b) * factor * doubled)

    elementwise[simd_width=1, target="gpu" if gpu else "cpu"](
        lane, Coord(keep, frames), ctx
    )
    ctx.synchronize()
    _ = spectra^
    return Spectrogram[dtype, keep, frames](
        _frequencies[dtype, nperseg](ctx, fs),
        _times[dtype, frames](ctx, Float64(nperseg // 2), step, fs),
        power^,
    )


def stft[
    dtype: DType,
    n: Int,
    nperseg: Int,
    noverlap: Int = nperseg // 2,
    gpu: Bool = False,
](
    mut x: Static[dtype, n], fs: Float64 = 1.0, window: StaticString = "hann"
) raises -> STFT[
    dtype,
    nperseg // 2 + 1,
    _frame_count(
        _padded_length(n, nperseg, nperseg - noverlap), nperseg, noverlap
    ),
] where (
    dtype.is_floating_point() and n >= 1 and nperseg > 1 and nperseg > 0
):
    """The short-time Fourier transform: the complex one-sided spectrum of
    every frame, in `(frequencies, times)` orientation.
    `scipy.signal.stft(x, fs, window, nperseg, noverlap, boundary="zeros",
    padded=True, scaling="spectrum")`.

    SciPy's legacy `stft` defaults exactly: `nperseg // 2` zeros are added
    at both ends so the first frame is centred on the first sample, the
    signal is extended to a whole number of hops, no detrending, and every
    value is divided by `sum(w)` so a pure tone reads its amplitude. The
    time of frame `f` is `f * step / fs`. Every frame is one row of one
    batched transform.
    """
    comptime assert (
        noverlap >= 0 and noverlap < nperseg
    ), "noverlap must lie in [0, nperseg)"
    comptime step = nperseg - noverlap
    comptime extended = _padded_length(n, nperseg, step)
    comptime frames = _frame_count(extended, nperseg, noverlap)
    comptime keep = nperseg // 2 + 1
    var ctx = x.context()
    var win = get_window[dtype, nperseg](window, ctx=ctx)
    var sums = _window_sums(win)
    var spectra = _framed[dtype, n, nperseg, step, frames, gpu](
        x, win, nperseg // 2, False
    )

    var real = Static[dtype, keep, frames]._uninitialized(ctx)
    var imag = Static[dtype, keep, frames]._uninitialized(ctx)
    # The two halves of a `Spectrum` share the tuple's origin, so their
    # tracked views read as aliasing when a body captures both; erase to
    # `MutAnyOrigin`, which is the type the kernels take anyway. The owner
    # outlives every launch below.
    var re = spectra[0].view().as_unsafe_any_origin()
    var im = spectra[1].view().as_unsafe_any_origin()
    var rs = real.view()
    var ims = imag.view()
    var factor = Scalar[dtype](1.0 / sums[0])

    @always_inline
    def lane[
        w: Int, alignment: Int = 1
    ](coord: Coord) {var re, var im, var rs, var ims, var factor}:
        var idx = coord_to_index_list(coord)
        var k = idx[0]
        var f = idx[1]
        rs.store[1](Coord(k, f), re[Coord(f, k)] * factor)
        ims.store[1](Coord(k, f), im[Coord(f, k)] * factor)

    elementwise[simd_width=1, target="gpu" if gpu else "cpu"](
        lane, Coord(keep, frames), ctx
    )
    ctx.synchronize()
    _ = spectra^
    return STFT[dtype, keep, frames](
        _frequencies[dtype, nperseg](ctx, fs),
        _times[dtype, frames](ctx, 0.0, step, fs),
        real^,
        imag^,
    )


struct CrossSpectrum[dtype: DType, keep: Int](Movable):
    """What `csd` returns: `scipy.signal.csd`'s `(f, Pxy)`, with the
    complex `Pxy` as a real/imaginary pair because a `dtype`-monomorphic
    tensor holds no `Complex` -- the same split `STFT` makes.

    Unlike `Periodogram`'s `power`, this is genuinely complex: the phase of
    `Pxy[k]` is the average phase lead of `x` over `y` at that frequency,
    which is the reason to compute a cross spectrum at all.
    """

    var frequencies: Static[Self.dtype, Self.keep]
    var real: Static[Self.dtype, Self.keep]
    var imag: Static[Self.dtype, Self.keep]

    def __init__(
        out self,
        var frequencies: Static[Self.dtype, Self.keep],
        var real: Static[Self.dtype, Self.keep],
        var imag: Static[Self.dtype, Self.keep],
    ):
        self.frequencies = frequencies^
        self.real = real^
        self.imag = imag^


def csd[
    dtype: DType,
    n: Int,
    nperseg: Int,
    noverlap: Int = nperseg // 2,
    gpu: Bool = False,
](
    mut x: Static[dtype, n],
    mut y: Static[dtype, n],
    fs: Float64 = 1.0,
    window: StaticString = "hann",
    detrend: Bool = True,
    scaling: StaticString = "density",
) raises -> CrossSpectrum[dtype, nperseg // 2 + 1] where (
    dtype.is_floating_point() and n >= nperseg and nperseg > 1 and nperseg > 0
):
    """The cross power spectral density of `x` and `y` by Welch's method.
    `scipy.signal.csd(x, y, fs, window, nperseg, noverlap, detrend,
    scaling)`.

    `welch` with `X conj(Y)` in place of `|X|^2`, so every default and
    every scaling convention is that routine's, and `csd(x, x)` is
    `welch(x)` with a zero imaginary part -- an identity the tests check
    rather than an approximation.

    Both signals must be the same length, which is the type's job here.
    SciPy broadcasts unequal lengths by truncating; that silent truncation
    is worth not having.
    """
    if not (scaling == "density" or scaling == "spectrum"):
        raise Error(
            "csd: unknown scaling '",
            scaling,
            "'; expected 'density' or 'spectrum'",
        )
    comptime assert (
        noverlap >= 0 and noverlap < nperseg
    ), "noverlap must lie in [0, nperseg)"
    comptime step = nperseg - noverlap
    comptime frames = _frame_count(n, nperseg, noverlap)
    var ctx = x.context()
    var win = get_window[dtype, nperseg](window, ctx=ctx)
    var sums = _window_sums(win)
    var scale = 1.0 / (fs * sums[1]) if scaling == "density" else 1.0 / (
        sums[0] * sums[0]
    )
    comptime keep = nperseg // 2 + 1
    var xf = _framed[dtype, n, nperseg, step, frames, gpu](x, win, 0, detrend)
    var yf = _framed[dtype, n, nperseg, step, frames, gpu](y, win, 0, detrend)

    # `scale * mean_f X[f, k] conj(Y[f, k])`, one lane per bin. Inline
    # rather than a helper returning a `Spectrum`: moving one tensor out of
    # a `Tuple` element is not something Mojo 1.0 allows, and the two
    # destinations have to be built here anyway.
    var out_re = Static[dtype, keep]._uninitialized(ctx)
    var out_im = Static[dtype, keep]._uninitialized(ctx)
    # The two halves of a `Spectrum` share the tuple's origin, so their
    # tracked views read as aliasing when a body captures both; erase to
    # `MutAnyOrigin`, which is the type the kernels take anyway. The owner
    # outlives every launch below.
    var xr = xf[0].view().as_unsafe_any_origin()
    var xi = xf[1].view().as_unsafe_any_origin()
    var yr = yf[0].view().as_unsafe_any_origin()
    var yi = yf[1].view().as_unsafe_any_origin()
    var rs = out_re.view()
    var ims = out_im.view()
    var factor = Scalar[dtype](scale / Float64(frames))

    @always_inline
    def lane[
        w: Int, alignment: Int = 1
    ](coord: Coord) {
        var xr, var xi, var yr, var yi, var rs, var ims, var factor
    }:
        var k = coord_to_index_list(coord)[0]
        var acc_re = Scalar[dtype](0)
        var acc_im = Scalar[dtype](0)
        for f in range(frames):
            var a = xr[Coord(f, k)]
            var b = xi[Coord(f, k)]
            var c = yr[Coord(f, k)]
            var d = yi[Coord(f, k)]
            # (a + bi) * conj(c + di) = (ac + bd) + (bc - ad)i
            acc_re += a * c + b * d
            acc_im += b * c - a * d
        var doubled = Scalar[dtype](_one_sided_factor(k, nperseg))
        rs.store[1](Coord(k), acc_re * factor * doubled)
        ims.store[1](Coord(k), acc_im * factor * doubled)

    elementwise[simd_width=1, target="gpu" if gpu else "cpu"](
        lane, Coord(keep), ctx
    )
    ctx.synchronize()
    _ = xf^
    _ = yf^
    return CrossSpectrum[dtype, keep](
        _frequencies[dtype, nperseg](ctx, fs), out_re^, out_im^
    )


def coherence[
    dtype: DType,
    n: Int,
    nperseg: Int,
    noverlap: Int = nperseg // 2,
    gpu: Bool = False,
](
    mut x: Static[dtype, n],
    mut y: Static[dtype, n],
    fs: Float64 = 1.0,
    window: StaticString = "hann",
    detrend: Bool = True,
) raises -> Periodogram[dtype, nperseg // 2 + 1] where (
    dtype.is_floating_point() and n >= nperseg and nperseg > 1 and nperseg > 0
):
    """The magnitude-squared coherence of `x` and `y`:
    `|Pxy|^2 / (Pxx Pyy)`, in `[0, 1]`. `scipy.signal.coherence(x, y, fs,
    window, nperseg, noverlap, detrend)`.

    How much of `y` at each frequency is linearly explained by `x`. Read it
    as a correlation per bin rather than as a power: `1` means a fixed gain
    and phase between the two at that frequency, `0` means none.

    Returns a `Periodogram`, whose `power` field holds the coherence rather
    than a power -- reusing the struct because its shape is exactly
    `(frequencies, one value per bin)` and a second identical type would
    say nothing.

    **The scaling cancels.** `Pxy`, `Pxx` and `Pyy` all carry the same
    factor, and the ratio drops it, which is why there is no `scaling`
    parameter here when `csd` and `welch` both have one.

    One frame gives a coherence of exactly `1` at every bin -- with a
    single periodogram the ratio is an identity, not an estimate -- so
    `nperseg` well below `n` is the point, as it is in SciPy.
    """
    comptime assert (
        noverlap >= 0 and noverlap < nperseg
    ), "noverlap must lie in [0, nperseg)"
    comptime keep = nperseg // 2 + 1
    comptime step = nperseg - noverlap
    comptime frames = _frame_count(n, nperseg, noverlap)
    var ctx = x.context()
    var win = get_window[dtype, nperseg](window, ctx=ctx)

    # Two framings, not four: all three spectra come out of one lane pass,
    # since the ratio needs them at the same bin at the same time anyway.
    var xf = _framed[dtype, n, nperseg, step, frames, gpu](x, win, 0, detrend)
    var yf = _framed[dtype, n, nperseg, step, frames, gpu](y, win, 0, detrend)

    var out = Static[dtype, keep]._uninitialized(ctx)
    # The two halves of a `Spectrum` share the tuple's origin, so their
    # tracked views read as aliasing when a body captures both; erase to
    # `MutAnyOrigin`, which is the type the kernels take anyway. The owner
    # outlives every launch below.
    var xr = xf[0].view().as_unsafe_any_origin()
    var xi = xf[1].view().as_unsafe_any_origin()
    var yr = yf[0].view().as_unsafe_any_origin()
    var yi = yf[1].view().as_unsafe_any_origin()
    var cs = out.view()

    @always_inline
    def lane[
        w: Int, alignment: Int = 1
    ](coord: Coord) {var xr, var xi, var yr, var yi, var cs}:
        var k = coord_to_index_list(coord)[0]
        var cross_re = Scalar[dtype](0)
        var cross_im = Scalar[dtype](0)
        var pxx = Scalar[dtype](0)
        var pyy = Scalar[dtype](0)
        for f in range(frames):
            var a = xr[Coord(f, k)]
            var b = xi[Coord(f, k)]
            var c = yr[Coord(f, k)]
            var d = yi[Coord(f, k)]
            cross_re += a * c + b * d
            cross_im += b * c - a * d
            pxx += a * a + b * b
            pyy += c * c + d * d
        # Every scaling factor, including the one-sided doubling and the
        # 1/frames, is common to the three and cancels in the ratio.
        var numerator = cross_re * cross_re + cross_im * cross_im
        var denominator = pxx * pyy
        # A bin with no power in either signal has no coherence to report;
        # SciPy leaves a NaN there, and zero is the reading that composes.
        var value = Scalar[dtype](0)
        if denominator > Scalar[dtype](0):
            value = numerator / denominator
        cs.store[1](Coord(k), value)

    elementwise[simd_width=1, target="gpu" if gpu else "cpu"](
        lane, Coord(keep), ctx
    )
    ctx.synchronize()
    _ = xf^
    _ = yf^

    return Periodogram[dtype, keep](_frequencies[dtype, nperseg](ctx, fs), out^)


def istft[
    dtype: DType,
    keep: Int,
    frames: Int,
    nperseg: Int,
    noverlap: Int,
    gpu: Bool = False,
](
    mut spectra: STFT[dtype, keep, frames], window: StaticString = "hann"
) raises -> Static[
    dtype,
    nperseg + (frames - 1) * (nperseg - noverlap) - 2 * (nperseg // 2),
] where (
    dtype.is_floating_point()
    and nperseg > 1
    # `get_window`'s own clause, restated: the prover does no arithmetic on
    # evidence, so `nperseg > 1` does not discharge `nperseg > 0`.
    and nperseg > 0
    and noverlap >= 0
):
    """The inverse short-time Fourier transform, by windowed overlap-add.
    `scipy.signal.istft(Zxx, window=window, nperseg, noverlap,
    boundary=True)`.

    `stft`'s inverse, and the round trip recovers the signal: each column
    is scaled back by `sum(w)`, inverted with `irfft`, multiplied by the
    window again, accumulated at its hop position, and finally divided by
    the summed squared window at each sample -- the least-squares
    reconstruction, which is what makes the result exact wherever the
    window satisfies COLA rather than only approximate.

    The `nperseg // 2` boundary samples `stft` prepends and appends are cut
    off, so `istft(stft(x))` lines up with `x` from index zero. The result
    can still be *longer* than the original, because `stft` extends the
    signal to a whole number of hops and there is nothing in `Zxx` saying
    where the original ended; SciPy has the same property. Take the first
    `n` samples.

    `window` must be the one `stft` used -- the spectra do not record it.
    **Tier 2.**

    **`nperseg` and `noverlap` are explicit parameters**, so a round trip
    reads `istft[nperseg=8, noverlap=4](stft[dtype, n, 8](x))`. They are
    not defaulted from `keep`, and that is forced rather than chosen: the
    `keep` a caller receives is `stft`'s *unevaluated* `nperseg // 2 + 1`,
    and the `where` prover cannot fold `//` any more than it can fold `%`,
    so every clause derived from `keep` is undischargeable. `irfft` takes
    its `n` for the same reason. The agreement between the two is a
    `comptime assert` below.

    `ponytail:` `frames` separate `irfft` launches, one per column, where
    the forward direction transforms every frame as one batch on the lane
    engine. The upgrade is a batched inverse of the same shape
    `_framed` uses forward; at the frame counts a spectrogram has, the
    launches dominate above a few hundred frames.
    """
    # Asserted rather than constrained in the `where` clause: `keep` and
    # `frames` reach a caller as `stft`'s unevaluated `nperseg // 2 + 1`
    # and `_frame_count(...)`, and the prover folds neither a `//` nor a
    # `def` call (`findings.mdc`).
    comptime assert frames >= 1, "istft: frames must be at least 1"
    comptime assert (
        keep == nperseg // 2 + 1
    ), "istft: nperseg disagrees with the spectrum's bin count"
    comptime assert noverlap < nperseg, "noverlap must lie in [0, nperseg)"
    comptime step = nperseg - noverlap
    comptime padded = nperseg + (frames - 1) * step
    comptime trim = nperseg // 2
    comptime out_n = padded - 2 * trim

    var ctx = spectra.frequencies.context()
    var win = get_window[dtype, nperseg](window, ctx=ctx)
    var sums = _window_sums(win)
    var taps = win.to_host()

    var re = spectra.real.to_host()
    var im = spectra.imag.to_host()

    var acc = List[Float64](length=padded, fill=0.0)
    var weight = List[Float64](length=padded, fill=0.0)
    var gain = sums[0]

    for f in range(frames):
        # Column `f` of the spectrum, with `stft`'s 1/sum(w) undone.
        var col_re = List[Scalar[dtype]](capacity=keep)
        var col_im = List[Scalar[dtype]](capacity=keep)
        for k in range(keep):
            col_re.append(re[k * frames + f] * Scalar[dtype](gain))
            col_im.append(im[k * frames + f] * Scalar[dtype](gain))
        var column: Spectrum[dtype, keep] = (
            Static[dtype, keep](ctx, col_re^),
            Static[dtype, keep](ctx, col_im^),
        )
        var frame = irfft[dtype, keep, gpu, nperseg](column^).to_host()

        var base = f * step
        for j in range(nperseg):
            var w = Float64(taps[j])
            acc[base + j] += Float64(frame[j]) * w
            weight[base + j] += w * w

    var values = List[Scalar[dtype]](capacity=out_n)
    for i in range(out_n):
        var at = i + trim
        var w = weight[at]
        # A sample no window covered cannot be reconstructed; zero rather
        # than a division by zero, which is what SciPy's masked divide does.
        values.append(Scalar[dtype](acc[at] / w if w > 0.0 else 0.0))
    return Static[dtype, out_n](ctx, values^)


def hilbert[
    dtype: DType, n: Int, gpu: Bool = False
](mut x: Static[dtype, n]) raises -> Spectrum[dtype, n] where (
    dtype.is_floating_point() and n > 0
):
    """The analytic signal of `x`: real part `x`, imaginary part its
    Hilbert transform. `scipy.signal.hilbert(x)`, as a real/imaginary pair.

    Transform, zero the negative frequencies and double the positive ones
    (DC and, for even `n`, Nyquist kept at one), transform back -- SciPy's
    construction, in one launch between two transforms. The envelope is the
    pair's magnitude and the instantaneous phase its angle, which is what
    the analytic signal is for.
    """
    var ctx = x.context()
    var spectrum = fft[dtype, n, gpu](
        Spectrum[dtype, n](
            Static[dtype, n](ctx, x.to_host()), zeros[dtype, n](ctx)
        )
    )
    # The two halves of a `Spectrum` share the tuple's origin, so their
    # tracked views read as aliasing when a body captures both; erase to
    # `MutAnyOrigin`, which is the type the kernels take anyway. The owner
    # outlives every launch below.
    var re = spectrum[0].view().as_unsafe_any_origin()
    var im = spectrum[1].view().as_unsafe_any_origin()

    @always_inline
    def weight[w: Int, alignment: Int = 1](coord: Coord) {var re, var im}:
        var k = coord_to_index_list(coord)[0]
        var h = Scalar[dtype](0)
        if k == 0 or (n % 2 == 0 and k == n // 2):
            h = Scalar[dtype](1)
        elif k < (n + 1) // 2:
            h = Scalar[dtype](2)
        re.store[1](Coord(k), re[Coord(k)] * h)
        im.store[1](Coord(k), im[Coord(k)] * h)

    elementwise[simd_width=1, target="gpu" if gpu else "cpu"](
        weight, Coord(n), ctx
    )
    ctx.synchronize()
    return ifft[dtype, n, gpu](spectrum^)
