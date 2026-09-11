"""Linear convolution and correlation over `numax.core.array.Tensor`:
`convolve`, `correlate` and `fftconvolve`, in NumPy's three modes.

**This module is tier 2**, like the rest of `numax.signal` over `Tensor`:
host-driven, device-resident, `Plain`-only. `numax.signal.array` has the
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

`full` (the default), `same` and `valid`, as `numpy.convolve` defines them
and with its lengths: `m + k - 1`, `max(m, k)` centred, and `max(m, k) -
min(m, k) + 1`. `mode` is a compile-time parameter because the length is
part of the return type. `correlate` is `convolve` with the second
argument read backwards, in the same loop, so its modes and offsets are
`numpy.correlate`'s exactly -- including the offset convention `numax.signal.array`
documents.

## When to reach for `fftconvolve`

The direct sum is `O(m k)`; the transform route is `O(n log n)` at `n =
next_fast_len(m + k - 1)` and costs three transforms plus two padding
passes. For a short kernel the direct sum wins, for a long one the
transform does, and the crossover is a measurement rather than a rule;
`docs/performance.md` is where it belongs once measured. The two agree to
rounding on every input, which the tests check, so a caller can switch
without changing the answer.
"""

from layout import Coord, coord_to_index_list
from max.algorithm.functional import elementwise

from ..core.array import Static
from ..fft.fft import Spectrum, _rfft, irfft, next_fast_len

comptime full = 0
"""`mode`: every overlap, `m + k - 1` outputs. The default."""
comptime same = 1
"""`mode`: the central `max(m, k)` outputs, as long as the longer input."""
comptime valid = 2
"""`mode`: only the outputs where the two inputs overlap completely,
`max(m, k) - min(m, k) + 1` of them."""


def _out_len(m: Int, k: Int, mode: Int) -> Int:
    """The output length of a `mode` convolution of lengths `m` and `k`,
    evaluated at compile time so it can appear in a return type."""
    if mode == full:
        return m + k - 1
    if mode == same:
        return max(m, k)
    return max(m, k) - min(m, k) + 1


def _offset(m: Int, k: Int, mode: Int) -> Int:
    """The index into the `full` result at which `mode`'s output starts:
    NumPy centres `same` on the longer input and `valid` where the shorter
    one first fits entirely."""
    if mode == full:
        return 0
    if mode == same:
        return (min(m, k) - 1) // 2
    return min(m, k) - 1


def _direct[
    dtype: DType, m: Int, k: Int, mode: Int, gpu: Bool, reverse: Bool
](mut a: Static[dtype, m], mut b: Static[dtype, k]) raises -> Static[
    dtype, _out_len(m, k, mode)
]:
    """The direct sum, one lane per output: `out[o] = sum_j a[j] *
    b[i - j]` over the overlap at `i = o + offset`, with `b` read
    backwards when `reverse` -- which is what turns it into
    `correlate`. The loop bounds are integer arithmetic on the lane
    index, so the kernel has no data-dependent branch."""
    comptime out_n = _out_len(m, k, mode)
    comptime offset = _offset(m, k, mode)
    var ctx = a.context()
    var out = Static[dtype, out_n]._uninitialized(ctx)
    var xs = a.view()
    var taps = b.view()
    var ys = out.view()

    @always_inline
    def lane[
        w: Int, alignment: Int = 1
    ](coord: Coord) {var xs, var taps, var ys}:
        var o = coord_to_index_list(coord)[0]
        var i = o + offset
        var lo = 0 if i < k else i - k + 1
        var hi = i if i < m else m - 1
        var total = Scalar[dtype](0)
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
    dtype: DType, m: Int, k: Int, mode: Int = full, gpu: Bool = False
](mut a: Static[dtype, m], mut b: Static[dtype, k]) raises -> Static[
    dtype, _out_len(m, k, mode)
] where (
    dtype.is_floating_point()
    and m > 0
    and k > 0
    and (mode == full or mode == same or mode == valid)
):
    """The linear convolution of `a` and `b`. `numpy.convolve(a, b, mode)`.

    `out[i] = sum_j a[j] * b[i - j]` over the `j` where both indices are
    in range, in `mode` `full` (default), `same` or `valid`; the module
    docstring has the lengths and offsets. Linear, not circular: nothing
    wraps, which is what `numax.fft.array.circular_convolve` does instead.

    One `elementwise` launch, `O(m k)`; `fftconvolve` is the same answer by
    the transform route for a long kernel. Both inputs are borrowed, since
    the same taps are usually applied to many signals.
    """
    return _direct[dtype, m, k, mode, gpu, False](a, b)


def correlate[
    dtype: DType, m: Int, k: Int, mode: Int = full, gpu: Bool = False
](mut a: Static[dtype, m], mut b: Static[dtype, k]) raises -> Static[
    dtype, _out_len(m, k, mode)
] where (
    dtype.is_floating_point()
    and m > 0
    and k > 0
    and (mode == full or mode == same or mode == valid)
):
    """The cross-correlation of `a` and `b`. `numpy.correlate(a, b, mode)`,
    and `scipy.signal.correlate` for real input.

    `convolve(a, reversed(b))`, computed by indexing `b` backwards in the
    same loop rather than reversing it. The zero-lag term of the `full`
    result sits at index `k - 1`, not `0` -- NumPy's convention, and the
    one thing about correlation a caller has to know.
    """
    return _direct[dtype, m, k, mode, gpu, True](a, b)


def fftconvolve[
    dtype: DType, m: Int, k: Int, mode: Int = full, gpu: Bool = False
](mut a: Static[dtype, m], mut b: Static[dtype, k]) raises -> Static[
    dtype, _out_len(m, k, mode)
] where (
    dtype.is_floating_point()
    and m > 0
    and k > 0
    and (mode == full or mode == same or mode == valid)
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
    """
    comptime out_n = _out_len(m, k, mode)
    comptime offset = _offset(m, k, mode)
    comptime n = next_fast_len(m + k - 1)
    comptime keep = n // 2 + 1
    var ctx = a.context()

    var padded_a = Static[dtype, n]._uninitialized(ctx)
    var padded_b = Static[dtype, n]._uninitialized(ctx)
    var xs = a.view()
    var taps = b.view()
    var pa = padded_a.view()
    var pb = padded_b.view()

    @always_inline
    def pad[
        w: Int, alignment: Int = 1
    ](coord: Coord) {var xs, var taps, var pa, var pb}:
        var i = coord_to_index_list(coord)[0]
        pa.store[1](Coord(i), xs[Coord(i)] if i < m else Scalar[dtype](0))
        pb.store[1](Coord(i), taps[Coord(i)] if i < k else Scalar[dtype](0))

    elementwise[simd_width=1, target="gpu" if gpu else "cpu"](
        pad, Coord(n), ctx
    )

    var spectrum_a = _rfft[dtype, n, gpu](padded_a^)
    var spectrum_b = _rfft[dtype, n, gpu](padded_b^)
    var product_re = Static[dtype, keep]._uninitialized(ctx)
    var product_im = Static[dtype, keep]._uninitialized(ctx)
    var ar = spectrum_a[0].view()
    var ai = spectrum_a[1].view()
    var br = spectrum_b[0].view()
    var bi = spectrum_b[1].view()
    var pr = product_re.view()
    var pi = product_im.view()

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

    var product: Spectrum[dtype, keep] = (product_re^, product_im^)
    var circular = irfft[dtype, keep, gpu, n](product^)

    var out = Static[dtype, out_n]._uninitialized(ctx)
    var src = circular.view()
    var dst = out.view()

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
