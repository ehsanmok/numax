"""Filtering over `numax.core.tensor.Tensor`: `lfilter`, `lfilter_zi`,
`filtfilt`, `sosfilt`, `medfilt`, `detrend`, `savgol_filter`, `resample`
and the `firwin` design, with `scipy.signal`'s signatures and semantics.

**This module is tier 2.** The recursive filters -- `lfilter`,
`filtfilt`, `sosfilt` -- are a sequential recurrence, each output depending
on the last. On the host they run that recurrence directly. At `gpu=True`
they run it block-parallel (`_iir_device`): the state-space form lets each
block be filtered from a zero state independently, a short sequential
carry fixes up the block boundaries, and a correction launch adds each
block's true initial state back, so the `O(n)` work is on the device and
only an `O(n / 128)` carry of `K x K` products is sequential.
`numax.signal.lfilter` is the tier-1 sibling: the same recurrence
per SIMD lane over a register-resident frame.

On the host, host-side is not the same as off-tensor, and the difference
is most of what these three cost. The recurrence reads and writes through
`buffer.map_to_host()` -- the accessor `Tensor.to_host` and
`Tensor.copy_from_host` use internally, and the only one correct on both a
CPU and a GPU context (`numax/core/tensor.mojo`'s docstring says why a raw
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
and one launch, and `resample` is two transforms around a reshuffle of the
spectrum that runs on the device at `gpu=True`.

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

from std.collections import Array
from std.math import cos as _cos, sin as _sin

from layout import Coord, coord_to_index_list
from max.algorithm.functional import elementwise
from max.gpu.host import DeviceContext

from layout.tile_layout import row_major

from ..core._drive import _check_device, _notice, _require_contiguous
from ..core.tensorlike import TensorLike, TensorView, dim, is_row_major
from ..core.tensor import (
    _axis_gather,
    _canonical,
    _dyn_shape,
    _same_order,
    Dynamic,
    Static,
    arange_n,
    asarray,
    zeros,
)
from ..core.ops import subtract
from ..fft.fft import Spectrum, fft, ifft
from ..linalg.blas import dot
from ..stats.statistics import mean
from .windows import get_window
from ..core.numeric import FloatLike
from ._array.signal import firwin as _array_firwin
from ._array.signal import lfilter as _array_lfilter

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
    return Static[dtype, n](out^, ctx)


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


def _read_ptr[
    dtype: DType, T: TensorLike
](x: T, mut staged: List[Scalar[dtype]]) raises -> Pointer[
    Scalar[dtype], MutAnyOrigin
]:
    """A host pointer to `x`'s elements, without a copy where none is needed.

    On a CPU context the storage itself is host memory and the pointer is
    the tensor's own; the `mut` cast adds nothing the recurrence uses, it
    only reads. Off the host the elements are staged into `staged`, which
    the caller keeps alive for as long as the pointer is used. This is what
    `map_to_host` gave the old `Tensor`-typed signature for free; a
    `TensorLike` has no buffer to map, and copying a million-sample
    recording to filter it is a cost the old spelling did not pay.
    """
    if x.on_host():
        return (
            x.tile_as[dtype]()
            .ptr.unsafe_mut_cast[True]()
            .unsafe_origin_cast[MutAnyOrigin]()
        )
    staged = x.to_host[dtype]()
    return _loose(staged.unsafe_ptr())


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


comptime _IIR_BLOCK = 128
"""Samples per block of the device recurrence: long enough that the
sequential carry across blocks is short, short enough that the powers of
the companion matrix in the correction table stay tame for a stable
filter."""


def _iir_device[
    dtype: DType
](
    b: List[Scalar[dtype]],
    a: List[Scalar[dtype]],
    src: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    dst: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    n: Int,
    zi: List[Scalar[dtype]],
    scale_zi: Bool,
    reverse: Bool,
    ctx: DeviceContext,
) raises:
    """`_recurrence` on the device, block-parallel. `src` and `dst` are
    device pointers and may be the same.

    With `z' = A z + B x`, `y = z[0] + b0 x` the transposed direct form II
    in state-space form, the signal splits into blocks of `_IIR_BLOCK`:
    one thread per block filters it from a zero state and records the
    state it ends in; one thread then carries the true initial state
    across the blocks, `S[j+1] = A^L S[j] + f[j]`; and every sample adds
    its block's initial state through `e0 A^t`, the response to a state
    after `t` steps. Three launches, the only sequential one `O(n / L)`
    steps of a `K x K` product. The tables (`e0 A^t` for `t < L`, and
    `A^L`) come from the filter's handful of coefficients, built on the
    host in `Float64`. `scale_zi` multiplies the initial state by the first
    sample the pass reads, `filtfilt`'s steady-state start. `reverse` walks
    from the end. The answer matches the host recurrence to rounding for a
    stable filter; a filter with poles very near the unit circle loses
    accuracy in the `A^t` table first.
    """
    var taps = len(b)
    var order = taps - 1
    comptime L = _IIR_BLOCK
    if n == 0:
        return
    var blocks = (n + L - 1) // L
    var coef = List[Scalar[dtype]](capacity=2 * order + 1)
    for k in range(taps):
        coef.append(b[k])
    for k in range(1, taps):
        coef.append(a[k])
    var width = max(order, 1)
    # Host tables: the companion matrix, its rows `e0 A^t` and `A^L`.
    var g = List[Scalar[dtype]](length=L * width, fill=0)
    var al = List[Scalar[dtype]](length=width * width, fill=0)
    if order > 0:
        var mat = List[Float64](length=order * order, fill=0.0)
        for k in range(order):
            mat[k * order] = -Float64(a[k + 1])
            if k + 1 < order:
                mat[k * order + k + 1] = 1.0
        var row = List[Float64](length=order, fill=0.0)
        row[0] = 1.0
        for t in range(L):
            for k in range(order):
                g[t * order + k] = Scalar[dtype](row[k])
            var nxt = List[Float64](length=order, fill=0.0)
            for m in range(order):
                var acc = 0.0
                for k in range(order):
                    acc += row[k] * mat[k * order + m]
                nxt[m] = acc
            row = nxt^
        var power = List[Float64](length=order * order, fill=0.0)
        for k in range(order):
            power[k * order + k] = 1.0
        for _ in range(L):
            var nxt = List[Float64](length=order * order, fill=0.0)
            for r in range(order):
                for c in range(order):
                    var acc = 0.0
                    for k in range(order):
                        acc += power[r * order + k] * mat[k * order + c]
                    nxt[r * order + c] = acc
            power = nxt^
        for k in range(order * order):
            al[k] = Scalar[dtype](power[k])
    var initial = List[Scalar[dtype]](length=width, fill=0)
    for k in range(len(zi)):
        initial[k] = zi[k]
    var coef_d = asarray(coef^, ctx)
    var g_d = asarray(g^, ctx)
    var al_d = asarray(al^, ctx)
    var zi_d = asarray(initial^, ctx)
    var finals = Dynamic[dtype, 1](
        row_major(_dyn_shape[1](blocks * width)), ctx
    )
    var inits = Dynamic[dtype, 1](row_major(_dyn_shape[1](blocks * width)), ctx)
    var cp = _device_ptr[dtype](coef_d)
    var gp = _device_ptr[dtype](g_d)
    var ap = _device_ptr[dtype](al_d)
    var zp = _device_ptr[dtype](zi_d)
    var fp = _device_ptr[dtype](finals)
    var ip = _device_ptr[dtype](inits)

    # The initial state goes in first: in place, the zero-state pass below
    # overwrites the first sample `scale_zi` multiplies by.
    @always_inline
    def seed[
        w: Int, alignment: Int = 1
    ](coord: Coord) {
        var src, var zp, var ip, var order, var n, var reverse, var scale_zi
    }:
        var first = src[unsafe_offset=n - 1 if reverse else 0]
        for k in range(order):
            var value = zp[unsafe_offset=k]
            if scale_zi:
                value = value * first
            ip[unsafe_offset=k] = value

    if order > 0:
        elementwise[simd_width=1, target="gpu"](seed, Coord(1), ctx)

    @always_inline
    def zero_state[
        w: Int, alignment: Int = 1
    ](coord: Coord) {
        var src,
        var dst,
        var cp,
        var fp,
        var order,
        var width,
        var n,
        var reverse,
    }:
        var j = coord_to_index_list(coord)[0]
        var z = fp + j * width
        for k in range(order):
            z[unsafe_offset=k] = 0
        var stop = min(n, (j + 1) * L)
        for s in range(j * L, stop):
            var i = n - 1 - s if reverse else s
            var xi = src[unsafe_offset=i]
            var yi = cp[unsafe_offset=0] * xi
            if order > 0:
                yi += z[unsafe_offset=0]
            for k in range(order - 1):
                z[unsafe_offset=k] = (
                    cp[unsafe_offset=k + 1] * xi
                    + z[unsafe_offset=k + 1]
                    - cp[unsafe_offset=order + k + 1] * yi
                )
            if order > 0:
                z[unsafe_offset=order - 1] = (
                    cp[unsafe_offset=order] * xi
                    - cp[unsafe_offset=2 * order] * yi
                )
            dst[unsafe_offset=i] = yi

    elementwise[simd_width=1, target="gpu"](zero_state, Coord(blocks), ctx)
    if order == 0:
        ctx.synchronize()
        return

    @always_inline
    def carry[
        w: Int, alignment: Int = 1
    ](coord: Coord) {var ap, var fp, var ip, var order, var blocks}:
        for j in range(blocks - 1):
            for k in range(order):
                var acc = fp[unsafe_offset=j * order + k]
                for m in range(order):
                    acc += (
                        ap[unsafe_offset=k * order + m]
                        * ip[unsafe_offset=j * order + m]
                    )
                ip[unsafe_offset=(j + 1) * order + k] = acc

    elementwise[simd_width=1, target="gpu"](carry, Coord(1), ctx)

    @always_inline
    def correct[
        w: Int, alignment: Int = 1
    ](coord: Coord) {var dst, var gp, var ip, var order, var n, var reverse}:
        var s = coord_to_index_list(coord)[0]
        var j = s // L
        var t = s % L
        var i = n - 1 - s if reverse else s
        var acc = Scalar[dtype](0)
        for k in range(order):
            acc += (
                gp[unsafe_offset=t * order + k]
                * ip[unsafe_offset=j * order + k]
            )
        dst[unsafe_offset=i] = dst[unsafe_offset=i] + acc

    elementwise[simd_width=1, target="gpu"](correct, Coord(n), ctx)
    ctx.synchronize()
    _ = coef_d^
    _ = g_d^
    _ = al_d^
    _ = zi_d^
    _ = finals^
    _ = inits^


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
    A: TensorLike,
    B: TensorLike,
    C: TensorLike,
    gpu: Bool = False,
](b: A, a: B, x: C) raises -> Static[A.dtype, dim[C, 0]] where (
    (
        A.dtype.is_floating_point()
        and dim[A, 0] > 0
        and dim[B, 0] > 0
        and dim[C, 0] > 0
    )
    and A.LayoutType.rank == 1
    and A.LayoutType.all_dims_known
    and B.dtype == A.dtype
    and B.LayoutType.rank == 1
    and B.LayoutType.all_dims_known
    and C.dtype == A.dtype
    and C.LayoutType.rank == 1
    and C.LayoutType.all_dims_known
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
    `numax.core.tensor`'s own `to_host`/`copy_from_host` use, so nothing is
    copied into a `List` on the way in or out. Arithmetic is at `A.dtype`,
    which is also where SciPy computes it.
    `numax.signal.lfilter` is the tier-1 form that runs per SIMD lane
    inside a kernel.

    Parameters:
        A: Rank-1 static-length floating-point tensor type of `b`.
        B: Rank-1 static-length tensor type of `a`, same dtype.
        C: Rank-1 static-length tensor type of `x`, same dtype.
        gpu: Run the block-parallel device recurrence when `True` and `x`
            is on a device; a residency mismatch falls back to the host
            with a notice.

    Args:
        b: Numerator (feed-forward) coefficients.
        a: Denominator (feedback) coefficients; `a[0]` must be nonzero.
        x: Signal to filter.

    Returns:
        The filtered signal, the same length as `x`, on `x`'s device.

    Raises:
        If `a[0]` is zero, the fallback policy is `"raise"` on a residency
        mismatch, or a device operation fails.
    """
    comptime na = dim[B, 0]
    comptime n = dim[C, 0]
    var norm = _normalized(b.to_host(), a.to_host[A.dtype]())
    if _check_device[C, gpu](x):
        comptime if gpu:
            _require_contiguous(x)
            var result = Static[A.dtype, n]._uninitialized(x.context())
            _iir_device(
                norm[0],
                norm[1],
                _device_ptr[A.dtype](x),
                _device_ptr[A.dtype](result),
                n,
                List[Scalar[A.dtype]](),
                False,
                False,
                x.context(),
            )
            return result^
    else:
        _notice[gpu]("lfilter")
    var state = List[Scalar[A.dtype]](length=max(len(norm[0]) - 1, 1), fill=0)
    var out = Static[A.dtype, n]._uninitialized(x.context())
    # `_uninitialized` is sound here because the loop below writes every one
    # of the `n` elements before anything reads them, and the mapping flushes
    # on scope exit -- `copy_from_host` is the same write, through a `List`.
    var staged = List[Scalar[A.dtype]]()
    var src = _read_ptr[A.dtype](x, staged)
    with out._buffer.map_to_host() as dst:
        _recurrence(
            norm[0],
            norm[1],
            state,
            src,
            _loose(dst.unsafe_ptr()),
            n,
        )
    return out^


def lfilter_zi[
    A: TensorLike,
    B: TensorLike,
](b: A, a: B) raises -> Static[
    A.dtype, (dim[A, 0] if dim[A, 0] > dim[B, 0] else dim[B, 0]) - 1
] where (
    (
        A.dtype.is_floating_point()
        and dim[A, 0] > 0
        and dim[B, 0] > 0
        and (dim[A, 0] > 1 or dim[B, 0] > 1)
    )
    and A.LayoutType.rank == 1
    and A.LayoutType.all_dims_known
    and B.dtype == A.dtype
    and B.LayoutType.rank == 1
    and B.LayoutType.all_dims_known
    and is_row_major[A]
    and is_row_major[B]
):
    """The initial state at which `lfilter` responds to a unit step with no
    transient. `scipy.signal.lfilter_zi(b, a)`.

    `max(len(b), len(a)) - 1` values: with `y_inf = sum(b) / sum(a)` the
    step's steady state, `z[k] = sum_{j > k} (b[j] - y_inf a[j])` after
    normalizing by `a[0]`. Scale it by the first sample of the signal to
    start a filter "already settled", which is what `filtfilt` does.
    Raises when `sum(a) == 0`, a pole on the unit circle. Computed at
    `A.dtype`, like the recurrence it feeds.
    """
    comptime nb = dim[A, 0]
    comptime na = dim[B, 0]
    var norm = _normalized(b.to_host(), a.to_host[A.dtype]())
    var zi = _zi_host(norm[0], norm[1])
    return Static[A.dtype, (nb if nb > na else na) - 1](zi^, b.context())


def _device_ptr[
    dtype: DType, T: TensorLike
](x: T) -> UnsafePointer[Scalar[dtype], MutAnyOrigin]:
    """`x`'s device pointer at `dtype`, erased for a recurrence that reads it
    and, in place, writes it."""
    return (
        x.tile()
        .ptr.unsafe_bitcast[Scalar[dtype]]()
        .unsafe_mut_cast[True]()
        .unsafe_origin_cast[MutAnyOrigin]()
    )


def _filtfilt_device[
    n: Int, dtype: DType, T: TensorLike
](
    b: List[Scalar[dtype]], a: List[Scalar[dtype]], x: T, edge: Int
) raises -> Static[dtype, n]:
    """`filtfilt` on the device: one launch builds the odd extension, the
    device recurrence runs forward and then backward over it in place, each
    started at `lfilter_zi` times the first sample it reads, and a window
    copies out the middle."""
    _require_contiguous(x)
    var ctx = x.context()
    var m = n + 2 * edge
    var ext = Dynamic[dtype, 1]._uninitialized(ctx, row_major(_dyn_shape[1](m)))
    var xp = _device_ptr[dtype](x)
    var ep = _device_ptr[dtype](ext)

    @always_inline
    def extend[
        w: Int, alignment: Int = 1
    ](coord: Coord) {var xp, var ep, var edge}:
        var j = coord_to_index_list(coord)[0]
        var value: Scalar[dtype]
        if j < edge:
            value = 2 * xp[unsafe_offset=0] - xp[unsafe_offset=edge - j]
        elif j < edge + n:
            value = xp[unsafe_offset=j - edge]
        else:
            value = (
                2 * xp[unsafe_offset=n - 1]
                - xp[unsafe_offset=n - 2 - (j - edge - n)]
            )
        ep[unsafe_offset=j] = value

    elementwise[simd_width=1, target="gpu"](extend, Coord(m), ctx)
    var zi = _zi_host(b, a)
    _iir_device(b, a, ep, ep, m, zi, True, False, ctx)
    _iir_device(b, a, ep, ep, m, zi, True, True, ctx)
    return _axis_gather["offset"](
        ext, Static[dtype, n]._static_layout(), 0, edge
    )


def filtfilt[
    A: TensorLike,
    B: TensorLike,
    C: TensorLike,
    gpu: Bool = False,
](b: A, a: B, x: C, padlen: Optional[Int] = None) raises -> Static[
    A.dtype, dim[C, 0]
] where (
    (
        A.dtype.is_floating_point()
        and dim[A, 0] > 0
        and dim[B, 0] > 0
        and dim[C, 0] > 0
    )
    and A.LayoutType.rank == 1
    and A.LayoutType.all_dims_known
    and B.dtype == A.dtype
    and B.LayoutType.rank == 1
    and B.LayoutType.all_dims_known
    and C.dtype == A.dtype
    and C.LayoutType.rank == 1
    and C.LayoutType.all_dims_known
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
    `List[Scalar[A.dtype]]` rather than a `Static`.

    Parameters:
        A: Rank-1 static-length floating-point tensor type of `b`.
        B: Rank-1 static-length tensor type of `a`, same dtype.
        C: Rank-1 static-length tensor type of `x`, same dtype.
        gpu: Run the block-parallel device recurrence when `True` and `x`
            is on a device; a residency mismatch falls back to the host
            with a notice.

    Args:
        b: Numerator (feed-forward) coefficients.
        a: Denominator (feedback) coefficients; `a[0]` must be nonzero.
        x: Signal to filter, longer than `padlen`.
        padlen: Odd-extension length at each end; `None` means
            `3 * max(len(a), len(b))`.

    Returns:
        The zero-phase filtered signal, the same length as `x`.

    Raises:
        If `x` is not longer than `padlen`, `a[0]` is zero, the fallback
        policy is `"raise"` on a residency mismatch, or a device operation
        fails.
    """
    comptime nb = dim[A, 0]
    comptime na = dim[B, 0]
    comptime n = dim[C, 0]
    var norm = _normalized(b.to_host(), a.to_host[A.dtype]())
    var edge = padlen.value() if padlen else 3 * max(nb, na)
    if n <= edge:
        raise Error(
            "filtfilt: the signal length must be greater than padlen ", edge
        )
    if _check_device[C, gpu](x):
        comptime if gpu:
            return _filtfilt_device[n](norm[0], norm[1], x, edge)
    else:
        _notice[gpu]("filtfilt")

    # Odd extension: `2 x[0] - x[edge..1]`, `x`, `2 x[n-1] - x[n-2..n-1-edge]`.
    var ext = List[Scalar[A.dtype]](capacity=n + 2 * edge)
    var staged = List[Scalar[A.dtype]]()
    var src = _read_ptr[A.dtype](x, staged)
    var first = src[unsafe_offset=0]
    var last = src[unsafe_offset=n - 1]
    for i in range(edge):
        ext.append(2 * first - src[unsafe_offset=edge - i])
    for i in range(n):
        ext.append(src[unsafe_offset=i])
    for i in range(edge):
        ext.append(2 * last - src[unsafe_offset=n - 2 - i])
    var m = len(ext)

    var zi = _zi_host(norm[0], norm[1])
    var order = len(zi)
    var state = List[Scalar[A.dtype]](length=max(order, 1), fill=0)
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

    var out = Static[A.dtype, n]._uninitialized(x.context())
    with out._buffer.map_to_host() as dst:
        for i in range(n):
            dst[i] = ext[edge + i]
    return out^


def sosfilt[
    A: TensorLike,
    B: TensorLike,
    gpu: Bool = False,
](sos: A, x: B) raises -> Static[A.dtype, dim[B, 0]] where (
    (A.dtype.is_floating_point() and dim[B, 0] > 0)
    and A.LayoutType.rank == 2
    and A.LayoutType.all_dims_known
    and dim[A, 1] == 6
    and B.dtype == A.dtype
    and B.LayoutType.rank == 1
    and B.LayoutType.all_dims_known
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

    Parameters:
        A: Rank-2 `(sections, 6)` static floating-point tensor type of
            `sos`.
        B: Rank-1 static-length tensor type of `x`, same dtype.
        gpu: Run each section's block-parallel device recurrence when
            `True` and `x` is on a device; a residency mismatch falls back
            to the host with a notice.

    Args:
        sos: Second-order sections, one `[b0, b1, b2, a0, a1, a2]` row
            each, applied top to bottom.
        x: Signal to filter.

    Returns:
        The filtered signal, the same length as `x`.

    Raises:
        If a section's `a0` is zero, the fallback policy is `"raise"` on a
        residency mismatch, or a device operation fails.
    """
    comptime sections = dim[A, 0]
    comptime n = dim[B, 0]
    var table = sos.to_host()
    if _check_device[B, gpu](x):
        comptime if gpu:
            # Each section filters the buffer in place on the device.
            var result = _same_order(
                _canonical[n, dtype=A.dtype](x),
                Static[A.dtype, n]._static_layout(),
            )
            var rp = _device_ptr[A.dtype](result)
            for s in range(sections):
                var b = List[Scalar[A.dtype]](capacity=3)
                var a = List[Scalar[A.dtype]](capacity=3)
                for k in range(3):
                    b.append(table[s * 6 + k])
                    a.append(table[s * 6 + 3 + k])
                var norm = _normalized(b^, a^)
                _iir_device(
                    norm[0],
                    norm[1],
                    rp,
                    rp,
                    n,
                    List[Scalar[A.dtype]](),
                    False,
                    False,
                    x.context(),
                )
            return result^
    else:
        _notice[gpu]("sosfilt")
    var out = Static[A.dtype, n]._uninitialized(x.context())
    var staged = List[Scalar[A.dtype]]()
    var src = _read_ptr[A.dtype](x, staged)
    with out._buffer.map_to_host() as dst:
        for i in range(n):
            dst[i] = src[unsafe_offset=i]
        var buffer = _loose(dst.unsafe_ptr())
        for s in range(sections):
            var b = List[Scalar[A.dtype]](capacity=3)
            var a = List[Scalar[A.dtype]](capacity=3)
            for k in range(3):
                b.append(table[s * 6 + k])
                a.append(table[s * 6 + 3 + k])
            var norm = _normalized(b^, a^)
            var state = List[Scalar[A.dtype]](length=2, fill=0)
            _recurrence(norm[0], norm[1], state, buffer, buffer, n)
    return out^


def _sections_of[
    dtype: DType
](table: List[Scalar[dtype]], sections: Int) raises -> List[
    Tuple[List[Scalar[dtype]], List[Scalar[dtype]]]
]:
    """Each row of a `(sections, 6)` table as its normalized `(b, a)`."""
    var out = List[Tuple[List[Scalar[dtype]], List[Scalar[dtype]]]](
        capacity=sections
    )
    for s in range(sections):
        var b = List[Scalar[dtype]](capacity=3)
        var a = List[Scalar[dtype]](capacity=3)
        for k in range(3):
            b.append(table[s * 6 + k])
            a.append(table[s * 6 + 3 + k])
        out.append(_normalized(b^, a^))
    return out^


def sosfilt_zi[
    T: TensorLike
](sos: T) raises -> Static[T.dtype, dim[T, 0], 2] where (
    T.dtype.is_floating_point()
    and T.LayoutType.rank == 2
    and T.LayoutType.all_dims_known
    and dim[T, 1] == 6
):
    """The initial state at which `sosfilt` responds to a unit step with
    no transient. `scipy.signal.sosfilt_zi(sos)`.

    Section by section, `lfilter_zi` of the section scaled by the product
    of the DC gains `sum(b) / sum(a)` of the sections before it -- the
    step level that section sees in steady state -- as SciPy builds it.
    Scale it by the signal's first sample to start the cascade settled.

    Parameters:
        T: The tensor type of `sos`, `(sections, 6)`, floating-point.

    Args:
        sos: The sections, one `[b0, b1, b2, a0, a1, a2]` row each.

    Returns:
        A `(sections, 2)` tensor of initial states on `sos`'s device.

    Raises:
        If a section's `a0` is zero, or a section has a pole at `z = 1`.
    """
    comptime sections = dim[T, 0]
    var rows = _sections_of(sos.to_host(), sections)
    var zi = List[Scalar[T.dtype]](capacity=sections * 2)
    var scale = Scalar[T.dtype](1)
    for s in range(sections):
        var bq = rows[s][0].copy()
        var aq = rows[s][1].copy()
        var z = _zi_host(bq, aq)
        zi.append(scale * z[0])
        zi.append(scale * z[1])
        var sum_b = bq[0] + bq[1] + bq[2]
        var sum_a = aq[0] + aq[1] + aq[2]
        scale *= sum_b / sum_a
    return Static[T.dtype, sections, 2](zi^, sos.context())


def sosfiltfilt[
    A: TensorLike,
    B: TensorLike,
    gpu: Bool = False,
](sos: A, x: B, padlen: Optional[Int] = None) raises -> Static[
    A.dtype, dim[B, 0]
] where (
    (A.dtype.is_floating_point() and dim[B, 0] > 0)
    and A.LayoutType.rank == 2
    and A.LayoutType.all_dims_known
    and dim[A, 1] == 6
    and B.dtype == A.dtype
    and B.LayoutType.rank == 1
    and B.LayoutType.all_dims_known
):
    """Forward-backward filtering through second-order sections, so the
    phase cancels and the magnitude response is applied twice.
    `scipy.signal.sosfiltfilt(sos, x, padtype="odd", padlen)`.

    `filtfilt`'s scheme with the cascade in place of one long recurrence --
    the stable way to run it at a high order: the signal is odd-extended by
    `padlen` samples at each end (default `3 * (2 sections + 1 - k)`,
    `k` the sections whose `b2` and `a2` both vanish, SciPy's count), every
    section runs forward over the extension and then backward, each started
    from its `lfilter_zi` scaled by the first sample it reads -- which is
    `sosfilt_zi(sos)` scaled by the signal's first sample, since each
    section's first output is its DC gain times that sample -- and the
    extension is cut away. On the device that is one extension launch, the
    block-parallel recurrence per section in each direction, and a window.

    Parameters:
        A: The tensor type of `sos`, `(sections, 6)`, floating-point.
        B: The tensor type of `x`, rank 1 with a static length, same dtype.
        gpu: Run on `x`'s device when `True`; a residency mismatch falls
            back to the host with a notice.

    Args:
        sos: The sections, one `[b0, b1, b2, a0, a1, a2]` row each.
        x: The signal, longer than `padlen`.
        padlen: The odd-extension length at each end; `None` is SciPy's
            default.

    Returns:
        The zero-phase filtered signal, the same length as `x`.

    Raises:
        If `x` is not longer than `padlen`, a section's `a0` is zero, the
        fallback policy is `"raise"` on a residency mismatch, or a device
        operation fails.
    """
    comptime sections = dim[A, 0]
    comptime n = dim[B, 0]
    comptime dtype = A.dtype
    var table = sos.to_host()
    var rows = _sections_of(table, sections)
    var trivial_b = 0
    var trivial_a = 0
    for s in range(sections):
        if table[s * 6 + 2] == 0:
            trivial_b += 1
        if table[s * 6 + 5] == 0:
            trivial_a += 1
    var ntaps = 2 * sections + 1 - min(trivial_b, trivial_a)
    var edge = padlen.value() if padlen else 3 * ntaps
    if n <= edge:
        raise Error(
            "sosfiltfilt: the signal length must be greater than padlen ",
            edge,
        )
    var m = n + 2 * edge
    if _check_device[B, gpu](x):
        comptime if gpu:
            _require_contiguous(x)
            var ctx = x.context()
            var ext = Dynamic[dtype, 1]._uninitialized(
                ctx, row_major(_dyn_shape[1](m))
            )
            var xp = _device_ptr[dtype](x)
            var ep = _device_ptr[dtype](ext)

            @always_inline
            def extend[
                w: Int, alignment: Int = 1
            ](coord: Coord) {var xp, var ep, var edge}:
                var j = coord_to_index_list(coord)[0]
                var value: Scalar[dtype]
                if j < edge:
                    value = 2 * xp[unsafe_offset=0] - xp[unsafe_offset=edge - j]
                elif j < edge + n:
                    value = xp[unsafe_offset=j - edge]
                else:
                    value = (
                        2 * xp[unsafe_offset=n - 1]
                        - xp[unsafe_offset=n - 2 - (j - edge - n)]
                    )
                ep[unsafe_offset=j] = value

            elementwise[simd_width=1, target="gpu"](extend, Coord(m), ctx)
            for backward in range(2):
                for s in range(sections):
                    var bq = rows[s][0].copy()
                    var aq = rows[s][1].copy()
                    var zi = _zi_host(bq, aq)
                    _iir_device(bq, aq, ep, ep, m, zi, True, backward == 1, ctx)
            return _axis_gather["offset"](
                ext, Static[dtype, n]._static_layout(), 0, edge
            )
    else:
        _notice[gpu]("sosfiltfilt")
    var ext = List[Scalar[dtype]](capacity=m)
    var staged = List[Scalar[dtype]]()
    var src = _read_ptr[dtype](x, staged)
    var first = src[unsafe_offset=0]
    var last = src[unsafe_offset=n - 1]
    for i in range(edge):
        ext.append(2 * first - src[unsafe_offset=edge - i])
    for i in range(n):
        ext.append(src[unsafe_offset=i])
    for i in range(edge):
        ext.append(2 * last - src[unsafe_offset=n - 2 - i])
    for backward in range(2):
        for s in range(sections):
            var bq = rows[s][0].copy()
            var aq = rows[s][1].copy()
            var zi = _zi_host(bq, aq)
            var start = ext[m - 1] if backward == 1 else ext[0]
            var state = List[Scalar[dtype]](length=2, fill=0)
            state[0] = zi[0] * start
            state[1] = zi[1] * start
            var buffer = _loose(ext.unsafe_ptr())
            _recurrence(bq, aq, state, buffer, buffer, m, backward == 1)
    var out = Static[dtype, n]._uninitialized(x.context())
    with out._buffer.map_to_host() as dst:
        for i in range(n):
            dst[i] = ext[edge + i]
    return out^


# ---------------------------------------------------------------------------
# Windowed passes, on the device
# ---------------------------------------------------------------------------


def medfilt[
    T: TensorLike,
    kernel_size: Int = 3,
    gpu: Bool = False,
](x: T) raises -> Static[T.dtype, dim[T, 0]] where (
    (T.dtype.is_floating_point() and dim[T, 0] > 0 and kernel_size > 0)
    and T.LayoutType.rank == 1
    and T.LayoutType.all_dims_known
):
    """The running median over a window of `kernel_size` samples, zero
    padded at both ends. `scipy.signal.medfilt(x, kernel_size)`.

    One launch, one lane per output: the lane gathers its window (zeros
    past the ends, as SciPy pads) and insertion-sorts it in registers,
    which for the window sizes a median filter uses is faster than anything
    cleverer. `kernel_size` is a compile-time parameter because the
    register array's length is; SciPy requires it odd and so does this.

    Parameters:
        T: Rank-1 static-length floating-point tensor type.
        kernel_size: Odd window length, 3 by default.
        gpu: Launch on the device holding `x` when `True`, on the host
            when `False`.

    Args:
        x: Signal to filter.

    Returns:
        The running median, the same length as `x`, on `x`'s device.

    Raises:
        If the launch or allocation on `x`'s device fails.
    """
    comptime n = dim[T, 0]
    comptime assert kernel_size % 2 == 1, "medfilt: kernel_size must be odd"
    comptime half = kernel_size // 2
    var ctx = x.context()
    var out = Static[T.dtype, n]._uninitialized(ctx)
    var xs = x.tile()
    var ys = out.tile()

    @always_inline
    def lane[w: Int, alignment: Int = 1](coord: Coord) {var xs, var ys}:
        var i = coord_to_index_list(coord)[0]
        var window = Array[Scalar[T.dtype], kernel_size](fill=0)
        for j in range(kernel_size):
            var src = i - half + j
            window[j] = xs[Coord(src)] if (src >= 0 and src < n) else Scalar[
                T.dtype
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
    T: TensorLike,
    gpu: Bool = False,
](x: T, type: StaticString = "linear") raises -> Static[
    T.dtype, dim[T, 0]
] where (
    (T.dtype.is_floating_point() and dim[T, 0] > 0)
    and T.LayoutType.rank == 1
    and T.LayoutType.all_dims_known
    and is_row_major[T]
):
    """`x` with its mean (`type="constant"`) or its least-squares line
    (`type="linear"`, the default) removed. `scipy.signal.detrend(x,
    type=type)`.

    The line comes from the closed-form normal equations over the sample
    index: two reductions (`mean`, and `dot` against the index) and one
    launch to subtract, all device-resident. An unknown `type` raises.

    Parameters:
        T: Rank-1, row-major, static-length floating-point tensor type.
        gpu: Run the reductions and the subtraction on the device holding
            `x` when `True`, on the host when `False`.

    Args:
        x: Signal to detrend.
        type: `"linear"` to remove the least-squares line, `"constant"` to
            remove the mean.

    Returns:
        `x` minus its trend, the same length, on `x`'s device.

    Raises:
        If `type` is not `"linear"` or `"constant"`, or a device operation
        fails.
    """
    comptime n = dim[T, 0]
    if not (type == "linear" or type == "constant"):
        raise Error(
            "detrend: unknown type '",
            type,
            "'; expected 'linear' or 'constant'",
        )
    var ctx = x.context()
    var mean_y = Float64(mean[gpu=gpu](x))
    if type == "constant":
        return subtract(_canonical[n](x), Scalar[T.dtype](mean_y))

    var index = arange_n[n, T.dtype](ctx=ctx)
    var mean_i = Float64(n - 1) / 2.0
    var sxx = Float64(n) * (Float64(n) * Float64(n) - 1.0) / 12.0
    var sxy = Float64(dot[gpu=gpu](index, x)) - Float64(n) * mean_i * mean_y
    var slope = sxy / sxx if sxx > 0 else 0.0
    var intercept = mean_y - slope * mean_i

    var out = Static[T.dtype, n]._uninitialized(ctx)
    var xs = x.tile()
    var ys = out.tile()
    var m = Scalar[T.dtype](slope)
    var c = Scalar[T.dtype](intercept)

    @always_inline
    def remove[
        w: Int, alignment: Int = 1
    ](coord: Coord) {var xs, var ys, var m, var c}:
        var i = coord_to_index_list(coord)[0]
        ys.store[1](Coord(i), xs[Coord(i)] - (c + m * Scalar[T.dtype](i)))

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
    T: TensorLike,
    window_length: Int,
    polyorder: Int,
    deriv: Int = 0,
    gpu: Bool = False,
](
    x: T,
    delta: Float64 = 1.0,
    mode: StaticString = "interp",
    cval: Float64 = 0.0,
) raises -> Static[T.dtype, dim[T, 0]] where (
    (
        T.dtype.is_floating_point()
        and dim[T, 0] > 0
        and window_length > 0
        and polyorder >= 0
        and polyorder < window_length
        and deriv >= 0
    )
    and T.LayoutType.rank == 1
    and T.LayoutType.all_dims_known
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

    Parameters:
        T: Rank-1 static-length floating-point tensor type.
        window_length: Odd number of samples in each fit.
        polyorder: Degree of the fitted polynomial, below `window_length`.
        deriv: Order of the derivative to return; `0` smooths.
        gpu: Launch the correlation on the device holding `x` when `True`,
            on the host when `False`.

    Args:
        x: Signal to filter.
        delta: Sample spacing the derivative is taken against.
        mode: Edge handling: `"interp"`, `"nearest"`, `"mirror"`, `"wrap"`
            or `"constant"`.
        cval: Fill value past the ends for `mode="constant"`.

    Returns:
        The filtered signal (or its derivative), the same length as `x`.

    Raises:
        If `mode` is unknown, `window_length` exceeds the signal under
        `"interp"`, a fit is singular, or a device operation fails.
    """
    comptime n = dim[T, 0]
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
    # the centered positions, times `deriv! / delta^deriv`.
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
    var taps = _upload[dtype=T.dtype, n=window_length](ctx, coefficients)

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
        var grid = List[Float64](capacity=window_length)
        for j in range(window_length):
            grid.append(Float64(j))
        var left = List[Float64](capacity=window_length)
        var right = List[Float64](capacity=window_length)
        # Only the two edge windows are fitted, so a device signal sends
        # back those `2 * window_length` samples rather than all of itself.
        var head: List[Scalar[T.dtype]]
        var tail: List[Scalar[T.dtype]]
        comptime if gpu:
            if not x.on_host():
                head = _axis_gather["offset"](
                    x, Static[T.dtype, window_length]._static_layout(), 0, 0
                ).to_host()
                tail = _axis_gather["offset"](
                    x,
                    Static[T.dtype, window_length]._static_layout(),
                    0,
                    n - window_length,
                ).to_host()
            else:
                var all = x.to_host()
                head = List[Scalar[T.dtype]](capacity=window_length)
                tail = List[Scalar[T.dtype]](capacity=window_length)
                for j in range(window_length):
                    head.append(all[j])
                    tail.append(all[n - window_length + j])
        else:
            var all = x.to_host()
            head = List[Scalar[T.dtype]](capacity=window_length)
            tail = List[Scalar[T.dtype]](capacity=window_length)
            for j in range(window_length):
                head.append(all[j])
                tail.append(all[n - window_length + j])
        for j in range(window_length):
            left.append(Float64(head[j]))
            right.append(Float64(tail[j]))
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
    var edges = _upload[dtype=T.dtype, n=2 * half if half > 0 else 1](
        ctx, edge_values
    )

    var out = Static[T.dtype, n]._uninitialized(ctx)
    var xs_view = x.tile()
    var cs = taps.tile()
    var es = edges.tile()
    var ys = out.tile()
    var mode_code = 0 if mode == "interp" else (
        1 if mode
        == "nearest" else (
            2 if mode == "mirror" else (3 if mode == "wrap" else 4)
        )
    )
    var fill = Scalar[T.dtype](cval)

    @always_inline
    def lane[
        w: Int, alignment: Int = 1
    ](coord: Coord) {
        var xs_view, var cs, var es, var ys, var mode_code, var fill
    }:
        var i = coord_to_index_list(coord)[0]
        var value: Scalar[T.dtype]
        if mode_code == 0 and i < half:
            value = es[Coord(i)]
        elif mode_code == 0 and i >= n - half:
            value = es[Coord(half + i - (n - half))]
        else:
            var total = Scalar[T.dtype](0)
            for j in range(window_length):
                var src = i - half + j
                var sample: Scalar[T.dtype]
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


def _resample_device[
    n: Int, num: Int, dtype: DType
](var spectrum: Spectrum[dtype, n], ctx: DeviceContext) raises -> Static[
    dtype, num
] where (dtype.is_floating_point() and n > 0 and num > 0):
    """`resample`'s spectrum reshuffle on the device: one launch builds the
    `num`-bin spectrum from the `n`-bin one by the host rule -- the leading
    `m // 2 + 1` bins, the trailing negative frequencies, SciPy's Nyquist
    fold or split, and the `num / n` gain -- then the inverse transform,
    whose real half is the answer."""
    comptime m = min(num, n)
    comptime m2 = m // 2 + 1
    var yre = Static[dtype, num]._uninitialized(ctx)
    var yim = Static[dtype, num]._uninitialized(ctx)
    var re = spectrum[0].tile().as_unsafe_any_origin()
    var im = spectrum[1].tile().as_unsafe_any_origin()
    var ore = yre.tile()
    var oim = yim.tile()
    var gain = Scalar[dtype](Float64(num) / Float64(n))

    @always_inline
    def reshuffle[
        width: Int, alignment: Int = 1
    ](coord: Coord) {var re, var im, var ore, var oim, var gain}:
        var k = coord_to_index_list(coord)[0]
        var r = Scalar[dtype](0)
        var i = Scalar[dtype](0)
        if k < m2:
            r = re[Coord(k)][0]
            i = im[Coord(k)][0]
        elif k >= num - (m - m2):
            r = re[Coord(n - num + k)][0]
            i = im[Coord(n - num + k)][0]
        comptime if m % 2 == 0:
            comptime if num < n:
                if k == num - m // 2:
                    r += re[Coord(n - m // 2)][0]
                    i += im[Coord(n - m // 2)][0]
            elif n < num:
                if k == m // 2 or k == num - m // 2:
                    r = re[Coord(m // 2)][0] / 2
                    i = im[Coord(m // 2)][0] / 2
        ore.store[1](coord, r * gain)
        oim.store[1](coord, i * gain)

    elementwise[simd_width=1, target="gpu"](reshuffle, Coord(num), ctx)
    ctx.synchronize()
    _ = spectrum^
    var back = ifft[gpu=True](Spectrum[dtype, num](yre^, yim^))
    return _same_order(back[0], Static[dtype, num]._static_layout())


def resample[
    T: TensorLike,
    num: Int,
    gpu: Bool = False,
](x: T) raises -> Static[T.dtype, num] where (
    (T.dtype.is_floating_point() and dim[T, 0] > 0 and num > 0)
    and T.LayoutType.rank == 1
    and T.LayoutType.all_dims_known
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

    Two transforms around a reshuffle of the spectrum, `O(n)` against their
    `O(n log n)`. At `gpu=True` the reshuffle is one device launch too, so
    nothing crosses to the host; on the host it is a loop.

    Parameters:
        T: Rank-1 static-length floating-point tensor type.
        num: Number of output samples.
        gpu: Run both transforms, and the spectrum reshuffle when `x` is
            on a device, on the device when `True`.

    Args:
        x: Signal to resample, treated as periodic.

    Returns:
        The length-`num` resampled signal on `x`'s device.

    Raises:
        If a transform or device operation fails.
    """
    comptime n = dim[T, 0]
    var ctx = x.context()
    var spectrum = fft[gpu=gpu](
        Spectrum[T.dtype, n](
            _same_order(x, Static[T.dtype, n]._static_layout()),
            zeros[T.dtype, n](ctx),
        )
    )
    comptime m = min(num, n)
    comptime m2 = m // 2 + 1
    comptime if gpu:
        if not x.on_host():
            return _resample_device[n, num](spectrum^, ctx)
    var re = _as_float64(spectrum[0].to_host())
    var im = _as_float64(spectrum[1].to_host())
    _ = spectrum^

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

    var back = ifft[gpu=gpu](
        Spectrum[T.dtype, num](
            _upload[dtype=T.dtype, n=num](ctx, yre),
            _upload[dtype=T.dtype, n=num](ctx, yim),
        )
    )
    var out = Static[T.dtype, num](back[0].to_host(), ctx)
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
    with `scale`, normalized to unit gain at the center of the first pass
    band. A design that must pass Nyquist -- a highpass, or a bandpass whose
    top edge is 1 -- needs an odd `numtaps`, as SciPy insists, since an
    even-length symmetric filter has a zero there.

    Host arithmetic, uploaded once; a design is a table, not a kernel. The
    `Array` tier's `firwin` is the lowpass case of this.

    Parameters:
        dtype: Floating-point element type of the taps.
        numtaps: Filter length; odd when the response passes Nyquist.

    Args:
        cutoff: Ascending band edges as fractions of Nyquist, each strictly
            inside `(0, 1)`.
        pass_zero: Whether the band starting at zero frequency passes.
        window: `get_window` name of the symmetric taper.
        scale: Normalize to unit gain at the center of the first pass band.
        ctx: Device to upload the taps to; `None` uses the host.

    Returns:
        The length-`numtaps` FIR coefficients on `ctx`.

    Raises:
        If a cutoff lies outside `(0, 1)`, an even `numtaps` must pass
        Nyquist, `window` is unknown, or allocation fails.
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
        get_window[dtype=dtype, n=numtaps](
            window, fftbins=False, ctx=device
        ).to_host()
    )
    for i in range(numtaps):
        taps[i] *= win[i]

    if scale:
        var left = bands[0]
        var right = bands[1]
        var center = 0.0 if left == 0 else (
            1.0 if right == 1 else 0.5 * (left + right)
        )
        var total = 0.0
        for i in range(numtaps):
            total += taps[i] * _cos(_PI * (Float64(i) - alpha) * center)
        for i in range(numtaps):
            taps[i] /= total
    return _upload[dtype=dtype, n=numtaps](device, taps)


def decimate[
    T: TensorLike,
    q: Int,
    numtaps: Int = 20 * q + 1,
    gpu: Bool = False,
](x: T) raises -> Static[T.dtype, (dim[T, 0] + q - 1) // q] where (
    (T.dtype.is_floating_point() and dim[T, 0] > 0 and q >= 2 and numtaps > 0)
    and T.LayoutType.rank == 1
    and T.LayoutType.all_dims_known
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
    the IIR spelling composes it: `cheby1[dtype=T.dtype, order=8](0.8 / q, ...)` then
    `filtfilt` then the stride.

    **Tier 2**, since `filtfilt` is.
    """
    comptime n = dim[T, 0]
    var ctx = x.context()
    var cutoff = List[Float64](capacity=1)
    cutoff.append(1.0 / Float64(q))
    var taps = firwin[dtype=T.dtype, numtaps=numtaps](cutoff^, True, ctx=ctx)
    var unit = List[Scalar[T.dtype]](capacity=1)
    unit.append(Scalar[T.dtype](1))
    var denominator = Static[T.dtype, 1](unit^, ctx)

    # SciPy's own padlen for this call, `3 * (len(b) // 2)`, rather than
    # `filtfilt`'s default `3 * max(len(a), len(b))`. The smaller pad is
    # what lets a signal only a few times longer than the taps be
    # decimated at all: at `q = 2` the default would demand 124 samples
    # where this needs 61.
    var smoothed = filtfilt[gpu=gpu](taps, denominator, x, 3 * (numtaps // 2))
    comptime out_n = (n + q - 1) // q
    comptime if gpu:
        if not smoothed.on_host():
            return _axis_gather["stride"](
                smoothed, Static[T.dtype, out_n]._static_layout(), 0, q
            )
    var host = smoothed.to_host()

    var values = List[Scalar[T.dtype]](capacity=out_n)
    for i in range(out_n):
        values.append(host[i * q])
    return Static[T.dtype, out_n](values^, ctx)


def firwin[T: FloatLike, n: Int](cutoff: T) -> Array[T, n]:
    """The `Array`-tier overload: one problem in registers, generic over
    the `FloatLike` conformer. The algorithm and its bound are documented
    at `numax.signal._array.signal.firwin`.

    Parameters:
        T: `FloatLike` conformer of the cutoff and taps.
        n: Number of taps.

    Args:
        cutoff: Lowpass cutoff as a fraction of Nyquist.

    Returns:
        The `n` Hamming-windowed lowpass taps, scaled to unit gain at DC.
    """
    return _array_firwin[T=T, n=n](cutoff)


def lfilter[
    T: FloatLike, n: Int, nb: Int, na: Int
](b: Array[T, nb], a: Array[T, na], x: Array[T, n]) -> Array[T, n]:
    """The `Array`-tier overload: one problem in registers, generic over
    the `FloatLike` conformer. The algorithm and its bound are documented
    at `numax.signal._array.signal.lfilter`.

    Parameters:
        T: `FloatLike` conformer of every element.
        n: Signal length.
        nb: Number of numerator coefficients.
        na: Number of denominator coefficients.

    Args:
        b: Numerator (feed-forward) coefficients.
        a: Denominator (feedback) coefficients; `a[0]` is divided out.
        x: Signal to filter, from a zero initial state.

    Returns:
        The length-`n` filtered signal.
    """
    return _array_lfilter[T=T, n=n, nb=nb, na=na](b, a, x)
