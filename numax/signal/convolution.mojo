"""Linear convolution and correlation over `numax.core.tensor.Tensor`:
`convolve`, `correlate` and `fftconvolve`, in NumPy's three modes.

**This module is tier 2**, like the rest of `numax.signal` over `Tensor`:
host-driven, device-resident, `Plain`-only. `numax.signal`'s `Array` tier has the
`FloatLike` tier that differentiates at `Dual` and runs per SIMD lane, for
the register-resident sizes; this one is for a recording against a filter.

## The MAX gate

MAX ships convolution, and the gate was run against it rather than past
it. `nn.conv.conv_gpu` takes `TileTensor`s, a `DeviceContext`, spatial
rank 1 to 3 and *asymmetric* padding, so a 1-D signal convolution is
expressible in principle: the signal as an `[1, W, 1]` NHWC image, the
taps as an `[R, 1, 1]` filter, `padding = (k-1, k-1)` for `full`. It is
not delegated to, for two reasons that are the operator's shape rather
than a missing feature. It is GPU-only, and its CPU sibling
`conv_nhwc_direct` wants the filter pre-packed by `pack_filter` -- two
code paths for one operation, where numax's rule is one kernel and a
`gpu: Bool`. And it tiles over channels and filters, of which a signal has
one each, so the arithmetic it is tuned for is exactly the arithmetic this
operation does not have. What it does not ship at all is the
transform-domain route, so `fftconvolve` is numax's on `numax.fft` either
way. The direct form is therefore written here as one `elementwise`
launch -- one output lane per element, a dot product over the overlap --
which is the shape the operation actually is, and `docs/parity.md` records
the operator MAX has and why this module is not it. **Extend.**

## Modes

`MODE_FULL` (the default), `MODE_SAME` and `MODE_VALID`, as `numpy.convolve` defines them
and with its lengths: `m + k - 1`, `max(m, k)` centered, and `max(m, k) -
min(m, k) + 1`. `mode` is a compile-time parameter because the length is
part of the return type. `correlate` is `convolve` with the second
argument read backwards, in the same loop, so its modes and offsets are
`numpy.correlate`'s exactly -- including the offset convention `numax.signal`'s `Array` tier
documents.

## When to reach for `fftconvolve`

The direct sum is `O(m k)`; the transform route is `O(n log n)` at `n =
next_fast_len(m + k - 1)` and costs three transforms plus two padding
passes. For a short kernel the direct sum wins, for a long one the
transform does, and the crossover is a measurement rather than a rule.
Measured on an M3 Pro at `float32`, it is near `k = 100` at `m = 4096` and
near `k = 300` at `m = 65536`; `bench/bench_signal.mojo` is the harness
and `docs/performance.md` carries the table, which is the thing to consult
rather than these two numbers, since the crossover moves with every change
to `numax.fft`'s engine -- it halved when the engine went to a fused first
block and radix-4. The two agree to
rounding on every input, which the tests check, so a caller can switch
without changing the answer.
"""

from layout import Coord, coord_to_index_list
from max.algorithm.functional import elementwise

from ..core.tensorlike import TensorLike, TensorView, dim, is_row_major
from ..core.tensor import Static
from ..fft.fft import Spectrum, _rfft, irfft, next_fast_len
from std.collections import Array
from ..core.numeric import FloatLike
from ._array.signal import _convolve_len
from ._array.signal import convolve as _array_convolve
from ._array.signal import correlate as _array_correlate

comptime MODE_FULL = 0
"""`mode`: every overlap, `m + k - 1` outputs. The default."""
comptime MODE_SAME = 1
"""`mode`: the central `max(m, k)` outputs, as long as the longer input."""
comptime MODE_VALID = 2
"""`mode`: only the outputs where the two inputs overlap completely,
`max(m, k) - min(m, k) + 1` of them."""


def _out_len(m: Int, k: Int, mode: Int) -> Int:
    """The output length of a `mode` convolution of lengths `m` and `k`,
    evaluated at compile time so it can appear in a return type."""
    if mode == MODE_FULL:
        return m + k - 1
    if mode == MODE_SAME:
        return max(m, k)
    return max(m, k) - min(m, k) + 1


def _offset(m: Int, k: Int, mode: Int) -> Int:
    """The index into the `full` result at which `mode`'s output starts:
    NumPy centers `same` on the longer input and `valid` where the shorter
    one first fits entirely."""
    if mode == MODE_FULL:
        return 0
    if mode == MODE_SAME:
        return (min(m, k) - 1) // 2
    return min(m, k) - 1


def _direct[
    A: TensorLike,
    B: TensorLike,
    mode: Int,
    gpu: Bool,
    reverse: Bool,
](a: A, b: B) raises -> Static[
    A.dtype, _out_len(dim[A, 0], dim[B, 0], mode)
] where (
    A.LayoutType.rank == 1
    and A.LayoutType.all_dims_known
    and B.dtype == A.dtype
    and B.LayoutType.rank == 1
    and B.LayoutType.all_dims_known
):
    """The direct sum, one lane per output: `out[o] = sum_j a[j] *
    b[i - j]` over the overlap at `i = o + offset`, with `b` read
    backwards when `reverse` -- which is what turns it into
    `correlate`. The loop bounds are integer arithmetic on the lane
    index, so the kernel has no data-dependent branch."""
    comptime m = dim[A, 0]
    comptime k = dim[B, 0]
    comptime out_n = _out_len(m, k, mode)
    comptime offset = _offset(m, k, mode)
    var ctx = a.context()
    var out = Static[A.dtype, out_n]._uninitialized(ctx)
    var xs = a.tile()
    var taps = b.tile_as[A.dtype]()
    var ys = out.tile()

    @always_inline
    def lane[
        w: Int, alignment: Int = 1
    ](coord: Coord) {var xs, var taps, var ys}:
        var o = coord_to_index_list(coord)[0]
        var i = o + offset
        var lo = 0 if i < k else i - k + 1
        var hi = i if i < m else m - 1
        var total = Scalar[A.dtype](0)
        for j in range(lo, hi + 1):
            comptime if reverse:
                total += xs[Coord(j)] * taps[Coord(k - 1 - i + j)]
            else:
                total += xs[Coord(j)] * taps[Coord(i - j)]
        ys.store[1](Coord(o), total)

    elementwise[simd_width=1, target="gpu" if gpu else "cpu"](
        lane, Coord(out_n), ctx
    )
    ctx.synchronize()
    return out^


def convolve[
    A: TensorLike,
    B: TensorLike,
    mode: Int = MODE_FULL,
    gpu: Bool = False,
](a: A, b: B) raises -> Static[
    A.dtype, _out_len(dim[A, 0], dim[B, 0], mode)
] where (
    (
        A.dtype.is_floating_point()
        and dim[A, 0] > 0
        and dim[B, 0] > 0
        and (mode == MODE_FULL or mode == MODE_SAME or mode == MODE_VALID)
    )
    and A.LayoutType.rank == 1
    and A.LayoutType.all_dims_known
    and B.dtype == A.dtype
    and B.LayoutType.rank == 1
    and B.LayoutType.all_dims_known
):
    """The linear convolution of `a` and `b`. `numpy.convolve(a, b, mode)`.

    `out[i] = sum_j a[j] * b[i - j]` over the `j` where both indices are
    in range, in `mode` `MODE_FULL` (default), `MODE_SAME` or `MODE_VALID`; the module
    docstring has the lengths and offsets. Linear, not circular: nothing
    wraps, which is what `numax.fft.circular_convolve` does instead.

    One `elementwise` launch, `O(m k)`; `fftconvolve` is the same answer by
    the transform route for a long kernel. Both inputs are borrowed, since
    the same taps are usually applied to many signals.

    Parameters:
        A: The `TensorLike` type of `a`, rank 1 with a static length `m`.
        B: The `TensorLike` type of `b`, rank 1 with a static length `k`
            and the same dtype as `A`.
        mode: `MODE_FULL`, `MODE_SAME` or `MODE_VALID`, which fixes the
            output length.
        gpu: Whether the launches target the GPU rather than the CPU; the
            inputs must live on the matching device.

    Args:
        a: The signal, of length `m`.
        b: The kernel, of length `k`.

    Returns:
        A new `Static` tensor on `a`'s device holding the `mode` slice of the
        convolution: `m + k - 1`, `max(m, k)` or `max(m, k) - min(m, k) + 1`
        elements.

    Raises:
        If allocating the output or launching the kernel on `a`'s device
        fails.
    """
    comptime m = dim[A, 0]
    comptime k = dim[B, 0]
    return _direct[mode=mode, gpu=gpu, reverse=False](a, b)


def correlate[
    A: TensorLike,
    B: TensorLike,
    mode: Int = MODE_FULL,
    gpu: Bool = False,
](a: A, b: B) raises -> Static[
    A.dtype, _out_len(dim[A, 0], dim[B, 0], mode)
] where (
    (
        A.dtype.is_floating_point()
        and dim[A, 0] > 0
        and dim[B, 0] > 0
        and (mode == MODE_FULL or mode == MODE_SAME or mode == MODE_VALID)
    )
    and A.LayoutType.rank == 1
    and A.LayoutType.all_dims_known
    and B.dtype == A.dtype
    and B.LayoutType.rank == 1
    and B.LayoutType.all_dims_known
):
    """The cross-correlation of `a` and `b`. `numpy.correlate(a, b, mode)`,
    and `scipy.signal.correlate` for real input.

    `convolve(a, reversed(b))`, computed by indexing `b` backwards in the
    same loop rather than reversing it. The zero-lag term of the `full`
    result sits at index `k - 1`, not `0` -- NumPy's convention, and the
    one thing about correlation a caller has to know.

    Parameters:
        A: The `TensorLike` type of `a`, rank 1 with a static length `m`.
        B: The `TensorLike` type of `b`, rank 1 with a static length `k`
            and the same dtype as `A`.
        mode: `MODE_FULL`, `MODE_SAME` or `MODE_VALID`, which fixes the
            output length.
        gpu: Whether the launches target the GPU rather than the CPU; the
            inputs must live on the matching device.

    Args:
        a: The signal, of length `m`.
        b: The template correlated against `a`, of length `k`.

    Returns:
        A new `Static` tensor on `a`'s device holding the `mode` slice of the
        correlation, with the zero-lag term of `full` at index `k - 1`.

    Raises:
        If allocating the output or launching the kernel on `a`'s device
        fails.
    """
    comptime m = dim[A, 0]
    comptime k = dim[B, 0]
    return _direct[mode=mode, gpu=gpu, reverse=True](a, b)


def fftconvolve[
    A: TensorLike,
    B: TensorLike,
    mode: Int = MODE_FULL,
    gpu: Bool = False,
](a: A, b: B) raises -> Static[
    A.dtype, _out_len(dim[A, 0], dim[B, 0], mode)
] where (
    (
        A.dtype.is_floating_point()
        and dim[A, 0] > 0
        and dim[B, 0] > 0
        and (mode == MODE_FULL or mode == MODE_SAME or mode == MODE_VALID)
    )
    and A.LayoutType.rank == 1
    and A.LayoutType.all_dims_known
    and B.dtype == A.dtype
    and B.LayoutType.rank == 1
    and B.LayoutType.all_dims_known
):
    """`convolve` by the transform route: zero-pad both inputs to
    `next_fast_len(m + k - 1)`, multiply their `rfft`s, `irfft`, and take
    the `mode` slice. `scipy.signal.fftconvolve(a, b, mode)` for real 1-D
    input.

    Padding to at least `m + k - 1` is what makes the circular convolution
    the transform computes equal the linear one: no wrapped term lands on a
    real output. The extra padding to a power of two keeps every transform
    on the radix-2 path. Three transforms, two padding gathers, one complex
    product and one slice, all device-resident; `O(n log n)` where the
    direct sum is `O(m k)`, so this is the one to call when `k` is long.
    Agrees with `convolve` to rounding, which the tests check.

    Parameters:
        A: The `TensorLike` type of `a`, rank 1 with a static length `m`.
        B: The `TensorLike` type of `b`, rank 1 with a static length `k`
            and the same dtype as `A`.
        mode: `MODE_FULL`, `MODE_SAME` or `MODE_VALID`, which fixes the
            output length.
        gpu: Whether the launches target the GPU rather than the CPU; the
            inputs must live on the matching device.

    Args:
        a: The signal, of length `m`.
        b: The kernel, of length `k`.

    Returns:
        A new `Static` tensor on `a`'s device holding the `mode` slice of the
        linear convolution, equal to `convolve(a, b)` to rounding.

    Raises:
        If allocating a buffer, a transform, or a kernel launch on `a`'s
        device fails.
    """
    comptime m = dim[A, 0]
    comptime k = dim[B, 0]
    comptime out_n = _out_len(m, k, mode)
    comptime offset = _offset(m, k, mode)
    comptime n = next_fast_len(m + k - 1)
    comptime keep = n // 2 + 1
    var ctx = a.context()

    var padded_a = Static[A.dtype, n]._uninitialized(ctx)
    var padded_b = Static[A.dtype, n]._uninitialized(ctx)
    var xs = a.tile()
    var taps = b.tile_as[A.dtype]()
    # `pad` captures views of the two padded buffers, and the buffers are
    # consumed by `_rfft` right after the launch; a tracked origin would
    # hold them past that point, so erase it. The launch has completed by
    # the time they are consumed.
    var pa = padded_a.tile().as_unsafe_any_origin()
    var pb = padded_b.tile().as_unsafe_any_origin()

    @always_inline
    def pad[
        w: Int, alignment: Int = 1
    ](coord: Coord) {var xs, var taps, var pa, var pb}:
        var i = coord_to_index_list(coord)[0]
        pa.store[1](Coord(i), xs[Coord(i)] if i < m else Scalar[A.dtype](0))
        pb.store[1](Coord(i), taps[Coord(i)] if i < k else Scalar[A.dtype](0))

    elementwise[simd_width=1, target="gpu" if gpu else "cpu"](
        pad, Coord(n), ctx
    )

    var spectrum_a = _rfft[gpu=gpu](padded_a^)
    var spectrum_b = _rfft[gpu=gpu](padded_b^)
    var product_re = Static[A.dtype, keep]._uninitialized(ctx)
    var product_im = Static[A.dtype, keep]._uninitialized(ctx)
    # The two halves of a `Spectrum` share the tuple's origin, so their
    # tracked views read as aliasing when a body captures both; erase to
    # `MutAnyOrigin`, which is the type the kernels take anyway. The owner
    # outlives every launch below.
    var ar = spectrum_a[0].tile().as_unsafe_any_origin()
    var ai = spectrum_a[1].tile().as_unsafe_any_origin()
    var br = spectrum_b[0].tile().as_unsafe_any_origin()
    var bi = spectrum_b[1].tile().as_unsafe_any_origin()
    var pr = product_re.tile()
    var pi = product_im.tile()

    @always_inline
    def multiply[
        w: Int, alignment: Int = 1
    ](coord: Coord) {var ar, var ai, var br, var bi, var pr, var pi}:
        var c = Coord(coord_to_index_list(coord)[0])
        pr.store[1](c, ar[c] * br[c] - ai[c] * bi[c])
        pi.store[1](c, ar[c] * bi[c] + ai[c] * br[c])

    elementwise[simd_width=1, target="gpu" if gpu else "cpu"](
        multiply, Coord(keep), ctx
    )
    ctx.synchronize()
    _ = spectrum_a^
    _ = spectrum_b^

    var product: Spectrum[A.dtype, keep] = (product_re^, product_im^)
    var circular = irfft[gpu=gpu, n=n](product^)

    var out = Static[A.dtype, out_n]._uninitialized(ctx)
    var src = circular.tile()
    var dst = out.tile()

    @always_inline
    def slice[w: Int, alignment: Int = 1](coord: Coord) {var src, var dst}:
        var o = coord_to_index_list(coord)[0]
        dst.store[1](Coord(o), src[Coord(o + offset)])

    elementwise[simd_width=1, target="gpu" if gpu else "cpu"](
        slice, Coord(out_n), ctx
    )
    ctx.synchronize()
    _ = circular^
    return out^


def oaconvolve[
    A: TensorLike,
    B: TensorLike,
    mode: Int = MODE_FULL,
    gpu: Bool = False,
](a: A, b: B) raises -> Static[
    A.dtype, _out_len(dim[A, 0], dim[B, 0], mode)
] where (
    (
        A.dtype.is_floating_point()
        and dim[A, 0] > 0
        and dim[B, 0] > 0
        and (mode == MODE_FULL or mode == MODE_SAME or mode == MODE_VALID)
    )
    and A.LayoutType.rank == 1
    and A.LayoutType.all_dims_known
    and B.dtype == A.dtype
    and B.LayoutType.rank == 1
    and B.LayoutType.all_dims_known
):
    """`convolve` by overlap-add. `scipy.signal.oaconvolve(a, b, mode)`.

    Element for element the same answer as `fftconvolve` -- SciPy's
    `oaconvolve` is not a different convolution, it is the same one
    computed in blocks -- so the name ships and the two are checked
    against each other.

    `ponytail:` this *is* `fftconvolve`. The block decomposition is the
    whole point of SciPy's version and it is not here: the gain is memory,
    `O(k + block)` of transform workspace against `O(m + k)`, plus a
    shorter transform when `m` is far longer than `k`. The upgrade is to
    split `a` into blocks of about `8k`, transform each against one
    pre-transformed `b`, and add the tails -- worth doing when a caller
    convolves a long recording against a short kernel and the padded
    length stops fitting comfortably in memory. Until then the answers
    agree and only the peak footprint differs.
    """
    comptime m = dim[A, 0]
    comptime k = dim[B, 0]
    return fftconvolve[mode=mode, gpu=gpu](a, b)


def convolve[
    T: FloatLike, m: Int, k: Int, mode: Int = MODE_FULL
](a: Array[T, m], b: Array[T, k]) -> Array[
    T, _convolve_len[m, k, mode]()
] where (mode == MODE_FULL or mode == MODE_SAME):
    """The `Array`-tier overload: one problem in registers, generic over
    the `FloatLike` conformer. The algorithm and its bound are documented
    at `numax.signal._array.signal.convolve`.

    Parameters:
        T: The `FloatLike` conformer of the elements.
        m: The length of `a`.
        k: The length of `b`.
        mode: `MODE_FULL` or `MODE_SAME`; `MODE_VALID` is not offered at this
            tier.

    Args:
        a: The signal, of length `m`.
        b: The kernel, of length `k`.

    Returns:
        The `mode` slice of the linear convolution: `m + k - 1` elements for
        `MODE_FULL`, `m` (the length of `a`) for `MODE_SAME`.
    """
    return _array_convolve[T=T, m=m, k=k, mode=mode](a, b)


def correlate[
    T: FloatLike, m: Int, k: Int
](a: Array[T, m], b: Array[T, k]) -> Array[T, m + k - 1]:
    """The `Array`-tier overload: one problem in registers, generic over
    the `FloatLike` conformer. The algorithm and its bound are documented
    at `numax.signal._array.signal.correlate`.

    Parameters:
        T: The `FloatLike` conformer of the elements.
        m: The length of `a`.
        k: The length of `b`.

    Args:
        a: The signal, of length `m`.
        b: The template correlated against `a`, of length `k`.

    Returns:
        The full cross-correlation, `m + k - 1` elements with the zero-lag
        term at index `k - 1`.
    """
    return _array_correlate[T=T, m=m, k=k](a, b)
