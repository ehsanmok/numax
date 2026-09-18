"""Filtering over `numax.core.array.Tensor`: `lfilter`, `lfilter_zi`,
`filtfilt`, `sosfilt`, `medfilt`, `detrend`, `savgol_filter`, `resample`
and the `firwin` design, with `scipy.signal`'s signatures and semantics.

**This module is tier 2**, and it is the one place in `numax.signal` over
`Tensor` where *host-side* is the honest label rather than a placement:
the recursive filters -- `lfilter`, `filtfilt`, `sosfilt` -- are a
sequential recurrence, each output depending on the last, with no GEMM to
feed and no independent lanes to launch. They run on the host, the way
`numax.linalg.banded` does and for the same reason, and their docstrings
say so. `numax.signal.array.lfilter` is the tier-1 sibling: the same
recurrence per SIMD lane over a register-resident frame, which is the
shape a recurrence *can* parallelize in.

Host-side is not the same as off-tensor, and the difference is most of
what these three cost. The recurrence reads and writes through
`buffer.map_to_host()` -- the accessor `Tensor.to_host` and
`Tensor.copy_from_host` use internally, and the only one correct on both a
CPU and a GPU context (`numax/core/array.mojo`'s docstring says why a raw
`unsafe_ptr()` is not). Reaching for it directly is legitimate *inside*
numax and nowhere else; what it buys is that no signal is ever copied into
a `List` on the way in or out. Coefficients and delay state are the only
`List`s, and they are `max(len(b), len(a))` long -- the filter order, not
the signal length. Arithmetic is at `dtype`, where SciPy also does it, so
a `float32` filter now costs `float32` work and carries `float32` error
rather than silently widening.

The passes are in place where in place is correct. Transposed direct form
II reads `x[i]` before it stores `y[i]`, so `filtfilt` filters one
extension buffer forwards and then backwards over itself -- a backward
pass being an index direction, not a reversed copy -- and `sosfilt` chains
every section through the destination buffer. `filtfilt` used to hold four
full-length `List[Float64]`s; it now holds one `List[Scalar[dtype]]`.

The rest is device work. `medfilt` and `savgol_filter` are one
`elementwise` launch each (a window per lane), `detrend` is two reductions
and one launch, and `resample` is two transforms around a host reshuffle
of the spectrum, `numax.linalg.solve_circulant`'s shape.

## The MAX gate

Nothing to delegate to: MAX has no IIR filter, median filter, detrend,
Savitzky-Golay or FIR design -- the neural-network convolution
`numax/signal/convolution.mojo` records is the closest thing and is not
these. **Extend** throughout.

## What is not here

`lfilter`'s `zi`/`zf` state is used internally by `filtfilt` and not
exposed; `sosfiltfilt`, `lfiltic`, `deconvolve` and `resample_poly` wait
on a caller. `decimate` is here, in SciPy's `ftype="fir"` form. IIR *design* -- `butter`, `cheby1`,
`iirfilter` -- is `design.mojo`'s, one module over.
"""

from std.math import cos as _cos, sin as _sin

from layout import Coord, coord_to_index_list
from max.algorithm.functional import elementwise
from max.gpu.host import DeviceContext

from ..core.array import Static, arange, zeros
from ..core.ops import subtract
from ..fft.fft import Spectrum, fft, ifft
from ..linalg.blas import dot
from ..stats.statistics import mean
from .windows import get_window

comptime _PI = 3.141592653589793


def _as_float64[dtype: DType](values: List[Scalar[dtype]]) -> List[Float64]:
    var out = List[Float64](capacity=len(values))
    for i in range(len(values)):
        out.append(Float64(values[i]))
    return out^


def _upload[
    dtype: DType, n: Int
](ctx: DeviceContext, values: List[Float64]) raises -> Static[dtype, n]:
    var out = List[Scalar[dtype]](capacity=n)
    for i in range(n):
        out.append(Scalar[dtype](values[i]))
    return Static[dtype, n](ctx, out^)


# ---------------------------------------------------------------------------
# The recurrence, on the host
# ---------------------------------------------------------------------------


def _normalized[
    dtype: DType
](b: List[Scalar[dtype]], a: List[Scalar[dtype]]) raises -> Tuple[
    List[Scalar[dtype]], List[Scalar[dtype]]
]:
    """`b` and `a` divided by `a[0]` and zero-padded to a common length,
    which is what the transposed direct form II below assumes.

    Both lists are `max(len(a), len(b))` long -- the filter order, a
    handful of values, never the signal length.
    """
    if a[0] == 0:
        raise Error("lfilter: a[0] must be non-zero")
    var taps = max(len(a), len(b))
    var bb = List[Scalar[dtype]](length=taps, fill=0)
    var aa = List[Scalar[dtype]](length=taps, fill=0)
    for i in range(len(b)):
        bb[i] = b[i] / a[0]
    for i in range(len(a)):
        aa[i] = a[i] / a[0]
    return (bb^, aa^)


@always_inline
def _loose[
    dtype: DType, origin: MutOrigin, //
](p: Pointer[Scalar[dtype], origin]) -> Pointer[Scalar[dtype], MutAnyOrigin]:
    """`p` with its origin erased, so two of them may name the same memory.

    `_recurrence` below is safe in place, but Mojo checks aliasing through
    the origin embedded in a value: passing a `List`'s own
    `Pointer[..., origin_of(ext)]` as both `src` and `dst` is rejected with
    `aliasing values passed mutably`. Erasing the origin is the documented
    way through (`.cursor/rules/findings.mdc`), and it is exactly as
    unchecked as it sounds -- which is why only this module's three
    recurrences use it, each over a buffer they own.
    """
    return rebind[Pointer[Scalar[dtype], MutAnyOrigin]](p)


def _recurrence[
    dtype: DType
](
    b: List[Scalar[dtype]],
    a: List[Scalar[dtype]],
    mut state: List[Scalar[dtype]],
    src: Pointer[Scalar[dtype], MutAnyOrigin],
    dst: Pointer[Scalar[dtype], MutAnyOrigin],
    count: Int,
    reverse: Bool = False,
):
    """SciPy's `lfilter` recurrence over host memory, at `dtype`.

    Transposed direct form II with `b` and `a` already normalized to the
    same length, `state` the `taps - 1` delay values in and out:
    `y[i] = b[0] x[i] + z[0]`, then `z[k] = b[k+1] x[i] + z[k+1] - a[k+1]
    y[i]`, the last `z` without a `z[k+1]`. One pass, sequential by
    construction.

    **`src` and `dst` may be the same pointer.** `y[i]` is read out of
    `x[i]` and the state before anything is stored, so a pass is safe in
    place -- which is what lets `filtfilt` run its backward pass over the
    buffer its forward pass just filled, and `sosfilt` chain its sections
    through one buffer, with no reversed or per-section copy.

    `reverse` walks the buffer from the end. A backward pass is then an
    index direction rather than a materialized reversal.
    """
    var taps = len(b)
    # The three `List`s are walked through their own pointers: the inner
    # loop runs `taps - 2` times per sample, so a bounds check per access is
    # the difference between this and SciPy's C loop.
    var bp = b.unsafe_ptr()
    var ap = a.unsafe_ptr()
    var sp = state.unsafe_ptr()
    var b0 = bp.unsafe_load[width=1](0)
    for step in range(count):
        var i = count - 1 - step if reverse else step
        var xi = src.unsafe_load[width=1](i)
        var yi = b0 * xi + (
            sp.unsafe_load[width=1](0) if taps > 1 else Scalar[dtype](0)
        )
        for k in range(taps - 2):
            sp.unsafe_store(
                k,
                bp.unsafe_load[width=1](k + 1) * xi
                + sp.unsafe_load[width=1](k + 1)
                - ap.unsafe_load[width=1](k + 1) * yi,
            )
        if taps > 1:
            sp.unsafe_store(
                taps - 2,
                bp.unsafe_load[width=1](taps - 1) * xi
                - ap.unsafe_load[width=1](taps - 1) * yi,
            )
        dst.unsafe_store(i, yi)


def _zi_host[
    dtype: DType
](b: List[Scalar[dtype]], a: List[Scalar[dtype]]) raises -> List[Scalar[dtype]]:
    """`lfilter_zi` on normalized, equal-length `b` and `a`: `z[k] =
    sum_{j > k} (b[j] - y_inf a[j])` with `y_inf = sum(b) / sum(a)`, the
    state that makes a step input come out flat."""
    var n = len(b)
    var sum_a = Scalar[dtype](0)
    var sum_b = Scalar[dtype](0)
    for i in range(n):
        sum_a += a[i]
        sum_b += b[i]
    if sum_a == 0:
        raise Error("lfilter_zi: the filter has a pole at z = 1")
    var y_inf = sum_b / sum_a
    var zi = List[Scalar[dtype]](length=n - 1, fill=0)
    var running = Scalar[dtype](0)
    for step in range(n - 1):
        var j = n - 1 - step
        running += b[j] - y_inf * a[j]
        zi[j - 1] = running
    return zi^


def lfilter[
    dtype: DType, nb: Int, na: Int, n: Int
](
    mut b: Static[dtype, nb], mut a: Static[dtype, na], mut x: Static[dtype, n]
) raises -> Static[dtype, n] where (
    dtype.is_floating_point() and nb > 0 and na > 0 and n > 0
):
    """Apply the difference equation `a[0] y[i] = sum_j b[j] x[i-j] - sum_j
    a[j] y[i-j]` to `x`, from rest. `scipy.signal.lfilter(b, a, x)`.

    The recursive filter a convolution cannot express: `na == 1` is a
    convolution, anything longer feeds its own past output back. Transposed
    direct form II, SciPy's structure, so the arithmetic and its rounding
    match SciPy's to the digit; `a[0]` is divided out rather than assumed
    to be one, and a zero `a[0]` raises.

    **Host-side**, for the reason the module docstring gives: a recurrence
    is sequential. It is host-side without being *off*-tensor, though --
    the pass reads `x` and writes the result through the mapping
    `numax.core.array`'s own `to_host`/`copy_from_host` use, so nothing is
    copied into a `List` on the way in or out. Arithmetic is at `dtype`,
    which is also where SciPy computes it.
    `numax.signal.array.lfilter` is the tier-1 form that runs per SIMD lane
    inside a kernel.
    """
    var norm = _normalized(b.to_host(), a.to_host())
    var state = List[Scalar[dtype]](length=max(len(norm[0]) - 1, 1), fill=0)
    var out = Static[dtype, n]._uninitialized(x.context())
    # `_uninitialized` is sound here because the loop below writes every one
    # of the `n` elements before anything reads them, and the mapping flushes
    # on scope exit -- `copy_from_host` is the same write, through a `List`.
    with x.buffer.map_to_host() as src:
        with out.buffer.map_to_host() as dst:
            _recurrence(
                norm[0],
                norm[1],
                state,
                _loose(src.unsafe_ptr()),
                _loose(dst.unsafe_ptr()),
                n,
            )
    return out^


def lfilter_zi[
    dtype: DType, nb: Int, na: Int
](mut b: Static[dtype, nb], mut a: Static[dtype, na]) raises -> Static[
    dtype, (nb if nb > na else na) - 1
] where (
    dtype.is_floating_point() and nb > 0 and na > 0 and (nb > 1 or na > 1)
):
    """The initial state at which `lfilter` responds to a unit step with no
    transient. `scipy.signal.lfilter_zi(b, a)`.

    `max(len(b), len(a)) - 1` values: with `y_inf = sum(b) / sum(a)` the
    step's steady state, `z[k] = sum_{j > k} (b[j] - y_inf a[j])` after
    normalizing by `a[0]`. Scale it by the first sample of the signal to
    start a filter "already settled", which is what `filtfilt` does.
    Raises when `sum(a) == 0`, a pole on the unit circle. Computed at
    `dtype`, like the recurrence it feeds.
    """
    var norm = _normalized(b.to_host(), a.to_host())
    var zi = _zi_host(norm[0], norm[1])
    return Static[dtype, (nb if nb > na else na) - 1](b.context(), zi^)


def filtfilt[
    dtype: DType, nb: Int, na: Int, n: Int
](
    mut b: Static[dtype, nb],
    mut a: Static[dtype, na],
    mut x: Static[dtype, n],
    padlen: Optional[Int] = None,
) raises -> Static[dtype, n] where (
    dtype.is_floating_point() and nb > 0 and na > 0 and n > 0
):
    """Forward-backward filtering: `lfilter` forwards, then backwards over
    the result, so the phase response cancels and the magnitude response
    is applied twice. `scipy.signal.filtfilt(b, a, x, padtype="odd",
    padlen)`.

    SciPy's `"pad"` method exactly: the signal is extended at both ends by
    `padlen` samples (default `3 * max(len(a), len(b))`) with its odd
    reflection, each pass starts from `lfilter_zi` scaled by its first
    sample, and the extension is cut away afterwards. Odd reflection is
    what keeps the end transients out of the result; the `"gust"` method is
    not provided. `x` must be longer than `padlen`, as SciPy requires.

    Host-side, twice the cost of `lfilter` plus the padding -- but over
    **one** extended buffer: the forward pass runs in place, and the
    backward pass runs in place over the same buffer walking from the end,
    so no reversed copy is ever built. The extension is the only allocation,
    and its length is a run-time `n + 2 padlen`, which is why it is a
    `List[Scalar[dtype]]` rather than a `Static`.
    """
    var norm = _normalized(b.to_host(), a.to_host())
    var edge = padlen.value() if padlen else 3 * max(nb, na)
    if n <= edge:
        raise Error(
            "filtfilt: the signal length must be greater than padlen ", edge
        )

    # Odd extension: `2 x[0] - x[edge..1]`, `x`, `2 x[n-1] - x[n-2..n-1-edge]`.
    var ext = List[Scalar[dtype]](capacity=n + 2 * edge)
    with x.buffer.map_to_host() as src:
        var first = src[0]
        var last = src[n - 1]
        for i in range(edge):
            ext.append(2 * first - src[edge - i])
        for i in range(n):
            ext.append(src[i])
        for i in range(edge):
            ext.append(2 * last - src[n - 2 - i])
    var m = len(ext)

    var zi = _zi_host(norm[0], norm[1])
    var order = len(zi)
    var state = List[Scalar[dtype]](length=max(order, 1), fill=0)
    for k in range(order):
        state[k] = zi[k] * ext[0]
    var forward = _loose(ext.unsafe_ptr())
    _recurrence(norm[0], norm[1], state, forward, forward, m)

    # The backward pass over the same buffer. Walking from the end is the
    # reversal, so index `j` after it holds what a reversed-copy pass would
    # have left at `m - 1 - j`, and the window to keep is `[edge, edge + n)`
    # either way.
    for k in range(order):
        state[k] = zi[k] * ext[m - 1]
    var backward = _loose(ext.unsafe_ptr())
    _recurrence(norm[0], norm[1], state, backward, backward, m, True)

    var out = Static[dtype, n]._uninitialized(x.context())
    with out.buffer.map_to_host() as dst:
        for i in range(n):
            dst[i] = ext[edge + i]
    return out^


def sosfilt[
    dtype: DType, sections: Int, n: Int
](
    mut sos: Static[dtype, sections, 6], mut x: Static[dtype, n]
) raises -> Static[dtype, n] where (
    dtype.is_floating_point() and sections > 0 and n > 0
):
    """Filter `x` through a cascade of second-order sections, from rest.
    `scipy.signal.sosfilt(sos, x)`.

    Each row of `sos` is `[b0, b1, b2, a0, a1, a2]`, SciPy's layout, and
    the sections are applied in order, each a transposed direct form II
    biquad. The cascade is the numerically sound way to run a high-order
    IIR filter -- the coefficients of a single long `a` polynomial lose
    digits that a product of quadratics keeps -- which is why
    `scipy.signal.butter(..., output="sos")` exists. Host-side, like
    `lfilter`, and for the same reason.

    The cascade runs in **one** buffer: `x` is copied into the result once
    and each section filters that buffer in place, so `sections` passes
    cost one allocation rather than `sections` intermediate signals.
    """
    var table = sos.to_host()
    var out = Static[dtype, n]._uninitialized(x.context())
    with x.buffer.map_to_host() as src:
        with out.buffer.map_to_host() as dst:
            for i in range(n):
                dst[i] = src[i]
            var buffer = _loose(dst.unsafe_ptr())
            for s in range(sections):
                var b = List[Scalar[dtype]](capacity=3)
                var a = List[Scalar[dtype]](capacity=3)
                for k in range(3):
                    b.append(table[s * 6 + k])
                    a.append(table[s * 6 + 3 + k])
                var norm = _normalized(b^, a^)
                var state = List[Scalar[dtype]](length=2, fill=0)
                _recurrence(norm[0], norm[1], state, buffer, buffer, n)
    return out^


# ---------------------------------------------------------------------------
# Windowed passes, on the device
# ---------------------------------------------------------------------------


def medfilt[
    dtype: DType, n: Int, kernel_size: Int = 3, gpu: Bool = False
](mut x: Static[dtype, n]) raises -> Static[dtype, n] where (
    dtype.is_floating_point() and n > 0 and kernel_size > 0
):
    """The running median over a window of `kernel_size` samples, zero
    padded at both ends. `scipy.signal.medfilt(x, kernel_size)`.

    One launch, one lane per output: the lane gathers its window (zeros
    past the ends, as SciPy pads) and insertion-sorts it in registers,
    which for the window sizes a median filter uses is faster than anything
    cleverer. `kernel_size` is a compile-time parameter because the
    register array's length is; SciPy requires it odd and so does this.
    """
    comptime assert kernel_size % 2 == 1, "medfilt: kernel_size must be odd"
    comptime half = kernel_size // 2
    var ctx = x.context()
    var out = Static[dtype, n]._uninitialized(ctx)
    var xs = x.view()
    var ys = out.view()

    @always_inline
    def lane[w: Int, alignment: Int = 1](coord: Coord) {var xs, var ys}:
        var i = coord_to_index_list(coord)[0]
        var window = InlineArray[Scalar[dtype], kernel_size](fill=0)
        for j in range(kernel_size):
            var src = i - half + j
            window[j] = xs[Coord(src)] if (src >= 0 and src < n) else Scalar[
                dtype
            ](0)
        # Insertion sort; `kernel_size` is small and compile-time.
        for j in range(1, kernel_size):
            var value = window[j]
            var k = j
            while k > 0 and window[k - 1] > value:
                window[k] = window[k - 1]
                k -= 1
            window[k] = value
        ys.store[1](Coord(i), window[half])

    elementwise[simd_width=1, target="gpu" if gpu else "cpu"](
        lane, Coord(n), ctx
    )
    ctx.synchronize()
    return out^


def detrend[
    dtype: DType, n: Int, gpu: Bool = False
](mut x: Static[dtype, n], type: StaticString = "linear") raises -> Static[
    dtype, n
] where (dtype.is_floating_point() and n > 0):
    """`x` with its mean (`type="constant"`) or its least-squares line
    (`type="linear"`, the default) removed. `scipy.signal.detrend(x,
    type=type)`.

    The line comes from the closed-form normal equations over the sample
    index: two reductions (`mean`, and `dot` against the index) and one
    launch to subtract, all device-resident. An unknown `type` raises.
    """
    if not (type == "linear" or type == "constant"):
        raise Error(
            "detrend: unknown type '",
            type,
            "'; expected 'linear' or 'constant'",
        )
    var ctx = x.context()
    var mean_y = Float64(mean[gpu=gpu](x))
    if type == "constant":
        return subtract(x, Scalar[dtype](mean_y))

    var index = arange[n, dtype](ctx=ctx)
    var mean_i = Float64(n - 1) / 2.0
    var sxx = Float64(n) * (Float64(n) * Float64(n) - 1.0) / 12.0
    var sxy = Float64(dot[gpu=gpu](index, x)) - Float64(n) * mean_i * mean_y
    var slope = sxy / sxx if sxx > 0 else 0.0
    var intercept = mean_y - slope * mean_i

    var out = Static[dtype, n]._uninitialized(ctx)
    var xs = x.view()
    var ys = out.view()
    var m = Scalar[dtype](slope)
    var c = Scalar[dtype](intercept)

    @always_inline
    def remove[
        w: Int, alignment: Int = 1
    ](coord: Coord) {var xs, var ys, var m, var c}:
        var i = coord_to_index_list(coord)[0]
        ys.store[1](Coord(i), xs[Coord(i)] - (c + m * Scalar[dtype](i)))

    elementwise[simd_width=1, target="gpu" if gpu else "cpu"](
        remove, Coord(n), ctx
    )
    ctx.synchronize()
    _ = index^
    return out^


def _solve_small(mut a: List[Float64], mut b: List[Float64], n: Int) raises:
    """Gaussian elimination with partial pivoting on an `n x n` row-major
    `a`, solution left in `b`. For the `(polyorder + 1)`-sized normal
    equations of a Savitzky-Golay fit and nothing larger."""
    for col in range(n):
        var pivot = col
        for r in range(col + 1, n):
            if abs(a[r * n + col]) > abs(a[pivot * n + col]):
                pivot = r
        if a[pivot * n + col] == 0:
            raise Error("savgol_filter: singular fit")
        if pivot != col:
            for c in range(n):
                var t = a[col * n + c]
                a[col * n + c] = a[pivot * n + c]
                a[pivot * n + c] = t
            var tb = b[col]
            b[col] = b[pivot]
            b[pivot] = tb
        for r in range(col + 1, n):
            var f = a[r * n + col] / a[col * n + col]
            for c in range(col, n):
                a[r * n + c] -= f * a[col * n + c]
            b[r] -= f * b[col]
    for step in range(n):
        var r = n - 1 - step
        var total = b[r]
        for c in range(r + 1, n):
            total -= a[r * n + c] * b[c]
        b[r] = total / a[r * n + r]


def _polyfit_host(
    xs: List[Float64], ys: List[Float64], degree: Int
) raises -> List[Float64]:
    """Least-squares polynomial coefficients, ascending, by the normal
    equations -- fine at the degrees a smoothing filter uses."""
    var p = degree + 1
    var ata = List[Float64](length=p * p, fill=0.0)
    var atb = List[Float64](length=p, fill=0.0)
    for i in range(len(xs)):
        var powers = List[Float64](capacity=p)
        var v = 1.0
        for _ in range(p):
            powers.append(v)
            v *= xs[i]
        for r in range(p):
            atb[r] += powers[r] * ys[i]
            for c in range(p):
                ata[r * p + c] += powers[r] * powers[c]
    _solve_small(ata, atb, p)
    return atb^


def _polyval_derivative(
    coefficients: List[Float64], deriv: Int, at: Float64
) -> Float64:
    """The `deriv`-th derivative of the ascending polynomial at `at`."""
    var total = 0.0
    var power = 1.0
    for k in range(deriv, len(coefficients)):
        var factor = 1.0
        for j in range(k - deriv + 1, k + 1):
            factor *= Float64(j)
        total += coefficients[k] * factor * power
        power *= at
    return total


def savgol_filter[
    dtype: DType,
    n: Int,
    window_length: Int,
    polyorder: Int,
    deriv: Int = 0,
    gpu: Bool = False,
](
    mut x: Static[dtype, n],
    delta: Float64 = 1.0,
    mode: StaticString = "interp",
    cval: Float64 = 0.0,
) raises -> Static[dtype, n] where (
    dtype.is_floating_point()
    and n > 0
    and window_length > 0
    and polyorder >= 0
    and polyorder < window_length
    and deriv >= 0
):
    """The Savitzky-Golay filter: at each sample, the `deriv`-th derivative
    of the degree-`polyorder` polynomial fitted by least squares to the
    `window_length` samples around it. `scipy.signal.savgol_filter(x,
    window_length, polyorder, deriv, delta, mode, cval)`.

    The fit is linear in the data, so it is one set of `window_length`
    coefficients applied as a correlation, and those are built once on the
    host from the normal equations. The pass is one launch; the edges
    follow SciPy's `mode`: `"interp"` (the default) fits a polynomial to
    the first and last `window_length` samples and evaluates it there,
    `"nearest"`, `"mirror"`, `"wrap"` and `"constant"` (with `cval`) extend
    the signal as `scipy.ndimage` does. `delta` is the sample spacing the
    derivative is taken against. Sizes are compile-time because the
    coefficient array's is; `window_length` must be odd (a `comptime
    assert`, since the `where` prover cannot evaluate `%`), `polyorder`
    less than it, and for `"interp"` no longer than the signal.
    """
    if not (
        mode == "interp"
        or mode == "nearest"
        or mode == "mirror"
        or mode == "wrap"
        or mode == "constant"
    ):
        raise Error(
            "savgol_filter: unknown mode '",
            mode,
            "'; expected interp, nearest, mirror, wrap or constant",
        )
    comptime assert (
        window_length % 2 == 1
    ), "savgol_filter: window_length must be odd"
    comptime half = window_length // 2
    comptime p = polyorder + 1
    var ctx = x.context()

    # The correlation coefficients: row `deriv` of `(V^T V)^{-1} V^T` over
    # the centred positions, times `deriv! / delta^deriv`.
    var positions = List[Float64](capacity=window_length)
    for j in range(window_length):
        positions.append(Float64(j - half))
    var gram = List[Float64](length=p * p, fill=0.0)
    var unit = List[Float64](length=p, fill=0.0)
    for j in range(window_length):
        var v = 1.0
        var powers = List[Float64](capacity=p)
        for _ in range(p):
            powers.append(v)
            v *= positions[j]
        for r in range(p):
            for c in range(p):
                gram[r * p + c] += powers[r] * powers[c]
    var factorial = 1.0
    for j in range(1, deriv + 1):
        factorial *= Float64(j)
    var scale = factorial
    for _ in range(deriv):
        scale /= delta
    if deriv < p:
        unit[deriv] = 1.0
        _solve_small(gram, unit, p)
    var coefficients = List[Float64](capacity=window_length)
    for j in range(window_length):
        var total = 0.0
        var v = 1.0
        for k in range(p):
            total += unit[k] * v
            v *= positions[j]
        coefficients.append(total * scale)
    var taps = _upload[dtype, window_length](ctx, coefficients)

    # `"interp"`: the edge values from SciPy's polynomial fits to the first
    # and last `window_length` samples, uploaded for the edge lanes to read.
    var edge_values = List[Float64](
        length=2 * half if half > 0 else 1, fill=0.0
    )
    var interp = mode == "interp"
    if interp:
        if window_length > n:
            raise Error(
                "savgol_filter: with mode 'interp', window_length must not"
                " exceed the signal length"
            )
        var xs = _as_float64(x.to_host())
        var grid = List[Float64](capacity=window_length)
        for j in range(window_length):
            grid.append(Float64(j))
        var left = List[Float64](capacity=window_length)
        var right = List[Float64](capacity=window_length)
        for j in range(window_length):
            left.append(xs[j])
            right.append(xs[n - window_length + j])
        var left_fit = _polyfit_host(grid, left, polyorder)
        var right_fit = _polyfit_host(grid, right, polyorder)
        var inv_delta = 1.0
        for _ in range(deriv):
            inv_delta /= delta
        for i in range(half):
            edge_values[i] = (
                _polyval_derivative(left_fit, deriv, Float64(i)) * inv_delta
            )
            edge_values[half + i] = (
                _polyval_derivative(
                    right_fit, deriv, Float64(window_length - half + i)
                )
                * inv_delta
            )
    var edges = _upload[dtype, 2 * half if half > 0 else 1](ctx, edge_values)

    var out = Static[dtype, n]._uninitialized(ctx)
    var xs_view = x.view()
    var cs = taps.view()
    var es = edges.view()
    var ys = out.view()
    var mode_code = 0 if mode == "interp" else (
        1 if mode
        == "nearest" else (
            2 if mode == "mirror" else (3 if mode == "wrap" else 4)
        )
    )
    var fill = Scalar[dtype](cval)

    @always_inline
    def lane[
        w: Int, alignment: Int = 1
    ](coord: Coord) {
        var xs_view, var cs, var es, var ys, var mode_code, var fill
    }:
        var i = coord_to_index_list(coord)[0]
        var value: Scalar[dtype]
        if mode_code == 0 and i < half:
            value = es[Coord(i)]
        elif mode_code == 0 and i >= n - half:
            value = es[Coord(half + i - (n - half))]
        else:
            var total = Scalar[dtype](0)
            for j in range(window_length):
                var src = i - half + j
                var sample: Scalar[dtype]
                if src >= 0 and src < n:
                    sample = xs_view[Coord(src)]
                elif mode_code == 1:
                    sample = xs_view[Coord(0 if src < 0 else n - 1)]
                elif mode_code == 2:
                    var reflected = -src if src < 0 else 2 * (n - 1) - src
                    sample = xs_view[Coord(reflected)]
                elif mode_code == 3:
                    sample = xs_view[Coord(((src % n) + n) % n)]
                else:
                    sample = fill
                total += cs[Coord(j)] * sample
            value = total
        ys.store[1](Coord(i), value)

    elementwise[simd_width=1, target="gpu" if gpu else "cpu"](
        lane, Coord(n), ctx
    )
    ctx.synchronize()
    _ = taps^
    _ = edges^
    return out^


# ---------------------------------------------------------------------------
# Resampling and FIR design
# ---------------------------------------------------------------------------


def resample[
    dtype: DType, n: Int, num: Int, gpu: Bool = False
](mut x: Static[dtype, n]) raises -> Static[dtype, num] where (
    dtype.is_floating_point() and n > 0 and num > 0
):
    """`x` resampled to `num` samples by the Fourier method: transform,
    keep or zero-pad the spectrum to `num` bins, transform back.
    `scipy.signal.resample(x, num)`.

    SciPy's handling of the Nyquist bin exactly: when the shorter length is
    even, its unpaired bin is halved and mirrored on upsampling and folded
    onto its partner on downsampling, so a real input gives a real output
    and a pure tone below the new Nyquist survives unchanged. Assumes the
    signal is periodic, as the method does; a signal that is not will ring
    at the ends.

    Two transforms on the device around a host reshuffle of the spectrum,
    which is `O(n)` against their `O(n log n)`; the spelling
    `numax.linalg.solve_circulant` uses.
    """
    var ctx = x.context()
    var spectrum = fft[dtype, n, gpu](
        Spectrum[dtype, n](
            Static[dtype, n](ctx, x.to_host()), zeros[dtype, n](ctx)
        )
    )
    var re = _as_float64(spectrum[0].to_host())
    var im = _as_float64(spectrum[1].to_host())
    _ = spectrum^

    comptime m = min(num, n)
    comptime m2 = m // 2 + 1
    var yre = List[Float64](length=num, fill=0.0)
    var yim = List[Float64](length=num, fill=0.0)
    for k in range(m2):
        yre[k] = re[k]
        yim[k] = im[k]
    # The negative frequencies, aligned to the ends of both spectra.
    for step in range(m - m2):
        yre[num - 1 - step] = re[n - 1 - step]
        yim[num - 1 - step] = im[n - 1 - step]
    comptime if m % 2 == 0:
        comptime if num < n:
            yre[num - m // 2] += re[n - m // 2]
            yim[num - m // 2] += im[n - m // 2]
        elif n < num:
            yre[m // 2] /= 2.0
            yim[m // 2] /= 2.0
            yre[num - m // 2] = yre[m // 2]
            yim[num - m // 2] = yim[m // 2]
    var gain = Float64(num) / Float64(n)
    for k in range(num):
        yre[k] *= gain
        yim[k] *= gain

    var back = ifft[dtype, num, gpu](
        Spectrum[dtype, num](
            _upload[dtype, num](ctx, yre), _upload[dtype, num](ctx, yim)
        )
    )
    var out = Static[dtype, num](ctx, back[0].to_host())
    _ = back^
    return out^


def _sinc(x: Float64) -> Float64:
    """`sin(pi x) / (pi x)`, `1` at zero -- NumPy's normalized `sinc`."""
    if x == 0:
        return 1.0
    var angle = _PI * x
    return _sin(angle) / angle


def firwin[
    dtype: DType, numtaps: Int
](
    cutoff: List[Float64],
    pass_zero: Bool = True,
    window: StaticString = "hamming",
    scale: Bool = True,
    ctx: Optional[DeviceContext] = None,
) raises -> Static[dtype, numtaps] where (
    dtype.is_floating_point() and numtaps > 0
):
    """FIR taps by the window method for a lowpass, highpass, bandpass,
    bandstop or multiband response. `scipy.signal.firwin(numtaps, cutoff,
    window, pass_zero, scale)` with the cutoffs as a fraction of Nyquist.

    `cutoff` lists the band edges in ascending order; `pass_zero` says
    whether the band starting at zero frequency passes. One edge with
    `pass_zero=True` is a lowpass, with `pass_zero=False` a highpass; two
    edges are a bandstop or a bandpass respectively, and so on. SciPy's
    construction: the ideal response is a sum of sinc differences over the
    pass bands, tapered by the symmetric `window` (Hamming by default) and,
    with `scale`, normalized to unit gain at the centre of the first pass
    band. A design that must pass Nyquist -- a highpass, or a bandpass whose
    top edge is 1 -- needs an odd `numtaps`, as SciPy insists, since an
    even-length symmetric filter has a zero there.

    Host arithmetic, uploaded once; a design is a table, not a kernel. The
    `Array` tier's `firwin` is the lowpass case of this.
    """
    var edges = len(cutoff)
    var pass_nyquist = ((edges % 2) == 1) != pass_zero
    if pass_nyquist and numtaps % 2 == 0:
        raise Error(
            "firwin: a filter with even numtaps cannot pass Nyquist; use an"
            " odd numtaps"
        )
    var bands = List[Float64]()
    if pass_zero:
        bands.append(0.0)
    for i in range(edges):
        if cutoff[i] <= 0 or cutoff[i] >= 1:
            raise Error("firwin: cutoffs must lie strictly inside (0, 1)")
        bands.append(cutoff[i])
    if pass_nyquist:
        bands.append(1.0)

    var alpha = 0.5 * Float64(numtaps - 1)
    var taps = List[Float64](length=numtaps, fill=0.0)
    for pair in range(len(bands) // 2):
        var left = bands[2 * pair]
        var right = bands[2 * pair + 1]
        for i in range(numtaps):
            var m = Float64(i) - alpha
            taps[i] += right * _sinc(right * m) - left * _sinc(left * m)

    var device = ctx.value() if ctx else DeviceContext(api="cpu")
    var win = _as_float64(
        get_window[dtype, numtaps](window, fftbins=False, ctx=device).to_host()
    )
    for i in range(numtaps):
        taps[i] *= win[i]

    if scale:
        var left = bands[0]
        var right = bands[1]
        var centre = 0.0 if left == 0 else (
            1.0 if right == 1 else 0.5 * (left + right)
        )
        var total = 0.0
        for i in range(numtaps):
            total += taps[i] * _cos(_PI * (Float64(i) - alpha) * centre)
        for i in range(numtaps):
            taps[i] /= total
    return _upload[dtype, numtaps](device, taps)


def decimate[
    dtype: DType, n: Int, q: Int, numtaps: Int = 20 * q + 1
](mut x: Static[dtype, n]) raises -> Static[dtype, (n + q - 1) // q] where (
    dtype.is_floating_point() and n > 0 and q >= 2 and numtaps > 0
):
    """Downsample `x` by the integer factor `q`, low-passing first.
    `scipy.signal.decimate(x, q, ftype="fir", zero_phase=True)`.

    The anti-alias filter is the point: taking every `q`-th sample of a
    signal with energy above the new Nyquist folds that energy down onto
    the bands below it, and no later step can separate the two again. So
    this is `firwin` at a cutoff of `1 / q`, applied with `filtfilt` so
    the phase response cancels, and only then the stride.

    `q` is a parameter because the output length `ceil(n / q)` is part of
    the type. `numtaps` follows SciPy's FIR default, `20 * q + 1`, odd so
    the taps are symmetric about a sample.

    SciPy's default is `ftype="iir"` -- an order-8 Chebyshev type I -- and
    this is its `ftype="fir"` branch. The FIR route is chosen because
    `filtfilt` over an order-8 IIR at `zero_phase=True` runs the recurrence
    twice and its transient handling has more ways to go wrong, where a
    symmetric FIR is exactly linear phase by construction. A caller wanting
    the IIR spelling composes it: `cheby1[dtype, 8](0.8 / q, ...)` then
    `filtfilt` then the stride.

    **Tier 2**, since `filtfilt` is.
    """
    var ctx = x.context()
    var cutoff = List[Float64](capacity=1)
    cutoff.append(1.0 / Float64(q))
    var taps = firwin[dtype, numtaps](cutoff^, True, ctx=ctx)
    var unit = List[Scalar[dtype]](capacity=1)
    unit.append(Scalar[dtype](1))
    var denominator = Static[dtype, 1](ctx, unit^)

    # SciPy's own padlen for this call, `3 * (len(b) // 2)`, rather than
    # `filtfilt`'s default `3 * max(len(a), len(b))`. The smaller pad is
    # what lets a signal only a few times longer than the taps be
    # decimated at all: at `q = 2` the default would demand 124 samples
    # where this needs 61.
    var smoothed = filtfilt(taps, denominator, x, 3 * (numtaps // 2))
    var host = smoothed.to_host()

    comptime out_n = (n + q - 1) // q
    var values = List[Scalar[dtype]](capacity=out_n)
    for i in range(out_n):
        values.append(host[i * q])
    return Static[dtype, out_n](ctx, values^)
