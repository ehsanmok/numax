"""Discrete Fourier transforms over `numax.core.array.Tensor`.

**This module is tier 2.** The stage loop runs on the host and each stage is
a device kernel, so nothing here is launchable *inside* a kernel body the
way the `FloatLike` tier is. `numax.fft.array` is that tier, and it is the
one that differentiates; this one is `Plain`-only and exists for the sizes
an `Array` cannot hold. The two are cross-referenced rather than ranked:
call the `Array` tier for a 64-point transform inside a per-lane kernel,
call this one for a four-million-point spectrogram.

## The MAX gate

MAX ships **no forward transform at all**. Its only one is `nn.irfft`:
inverse, real, last-axis, and NVIDIA-only, a thin wrapper over the private
`_cufft` package, so there is nothing on Metal or AMD either. This module
is therefore an **extend** in the contract's sense -- the gap is filled in
MAX's own idiom, `gpu: Bool` selecting `target`, device-resident
throughout -- rather than a delegation.

## Complex over a dtype-monomorphic tensor

A `Tensor` owns a `DeviceBuffer[dtype]` and `DType` is MAX's closed enum of
machine scalars, so `Complex[T]` cannot live in one; that is the same
structural fact that keeps every `FloatLike` conformer out of this tier.
The transform therefore travels as **a pair of real tensors**, real and
imaginary, rather than as one interleaved tensor with a trailing axis of 2.
The pair is what `rfft` naturally produces, and it keeps every butterfly's
reads unit-stride where interleaving would make them stride-2 gathers.

## The algorithm, and the launch count

Bit-reversal permutation, then `log2(n)` radix-2 Cooley-Tukey stages, each
one `elementwise` launch of `n/2` butterflies. `log2(n) + 1` launches, and
the data never touches the host in between -- the same device-residency
rule the blocked factorizations follow.

The stages run **in place**, which is safe rather than lucky: butterfly `t`
of a stage touches exactly the pair `(i, i + half)`, and those pairs are
disjoint across `t`, so no two threads of a launch address the same
element. That is what makes the permutation the only extra pass. Stockham
autosort would fold the permutation into the stages and save that one
launch of the `log2(n) + 1`; it is not written, because the permutation is
a pure gather and the stages are where the arithmetic is.

Twiddles come from a table of `n/2` entries built once per transform,
`W[q] = exp(-2*pi*i*q/n)`, which every stage indexes with a stride: a stage
of span `s` reads `W[pos * (n/s)]`. Two reasons, and neither is the obvious
one. The table is evaluated on the host in `Float64` and rounded once into
`dtype`, so at `float32` it is *more* accurate than computing the angle in
the working precision inside the kernel. And Metal has no `float64`
transcendentals at all -- `air.sin.f64` and `air.cos.f64` are rejected
outright -- so a kernel that computes its own angles in `Float64` does not
compile for a device at any `dtype`. The inverse transform conjugates this
same table rather than building a second one.

`gpu=True` is `float32` on Apple silicon, which is Metal's limit rather than
this module's: Metal rejects `double` loads outright, so a `float64` tensor
does not compile for it in any kernel numax writes (`findings.mdc` records
the same rejection for the Bessel recurrences). CUDA has no such
restriction.

**Power-of-two only, and structurally so:** `n` is checked in a `where`
clause, so a size that is not a power of two is a compile error rather than
a run-time raise. Bluestein's chirp-z and mixed radix are **out of scope,
not missing** -- they are a different algorithm with a different error
bound, and the `Array` tier makes the same choice for the same reason.
"""

from std.math import cos as _cos, sin as _sin

from layout import Coord, coord_to_index_list
from max.algorithm.functional import elementwise
from max.gpu.host import DeviceContext

from ..core.array import Static, zeros

comptime _TWO_PI = 6.283185307179586

comptime Spectrum[dtype: DType, n: Int] = Tuple[
    Static[dtype, n], Static[dtype, n]
]
"""A complex sequence over `Tensor`, as `(real, imaginary)`.

One value rather than two arguments so the transforms compose --
`ifft(fft(x))` type-checks, where a pair of `mut` arguments read as aliasing
each other when they come out of the same tuple. `numax` still owns exactly
one tensor type; this is a pair of them, not a new one.
"""


def _log2_exact(n: Int) -> Int:
    """`log2(n)` for a power-of-two `n`, at compile time. The stage count."""
    var bits = 0
    var rest = n
    while rest > 1:
        rest >>= 1
        bits += 1
    return bits


def _reverse_bits(value: Int, bits: Int) -> Int:
    """`value`'s low `bits` bits, reversed. The permutation the first pass
    applies."""
    var out = 0
    var rest = value
    for _ in range(bits):
        out = (out << 1) | (rest & 1)
        rest >>= 1
    return out


def _transform[
    dtype: DType, n: Int, gpu: Bool, inverse: Bool
](var x: Spectrum[dtype, n]) raises -> Spectrum[dtype, n]:
    """The shared engine: permute, then `log2(n)` in-place butterfly stages.

    `inverse` flips the sign of every twiddle angle and divides the result
    by `n`, which is the only difference between the two directions and the
    reason `ifft` is not a second implementation.
    """
    comptime bits = _log2_exact(n)
    comptime half_n = n // 2
    var ctx = x[0].context()

    # `W[q] = exp(-2*pi*i*q/n)`, in Float64 and rounded once. Stage `span`
    # reads `W[pos * (n // span)]`, so one table serves every stage.
    var twiddle_re = List[Scalar[dtype]](capacity=half_n)
    var twiddle_im = List[Scalar[dtype]](capacity=half_n)
    for q in range(half_n):
        var angle = -_TWO_PI * Float64(q) / Float64(n)
        twiddle_re.append(Scalar[dtype](_cos(angle)))
        twiddle_im.append(Scalar[dtype](_sin(angle)))
    var wr_all = Static[dtype, half_n](ctx, twiddle_re^)
    var wi_all = Static[dtype, half_n](ctx, twiddle_im^)

    var re = Static[dtype, n]._uninitialized(ctx)
    var im = Static[dtype, n]._uninitialized(ctx)

    var src_re = x[0].view()
    var src_im = x[1].view()
    var dst_re = re.view()
    var dst_im = im.view()

    @always_inline
    def permute[
        w: Int, alignment: Int = 1
    ](coord: Coord) {var src_re, var src_im, var dst_re, var dst_im}:
        var i = coord_to_index_list(coord)[0]
        var j = _reverse_bits(i, bits)
        dst_re.store[1](Coord(i), src_re[Coord(j)])
        dst_im.store[1](Coord(i), src_im[Coord(j)])

    elementwise[simd_width=1, target="gpu" if gpu else "cpu"](
        permute, Coord(n), ctx
    )

    # One launch per stage, `half_n` butterflies each. In place: butterfly
    # `t` owns the pair `(i, i + half)` and those are disjoint across `t`.
    comptime for stage in range(bits):
        comptime half = 1 << stage
        comptime span = half << 1
        comptime stride = n // span
        var bre = re.view()
        var bim = im.view()
        var twr = wr_all.view()
        var twi = wi_all.view()

        @always_inline
        def butterfly[
            w: Int, alignment: Int = 1
        ](coord: Coord) {var bre, var bim, var twr, var twi}:
            var t = coord_to_index_list(coord)[0]
            var block = t // half
            var pos = t % half
            var i = block * span + pos
            var j = i + half

            var q = pos * stride
            var wr = twr[Coord(q)]
            var wi = twi[Coord(q)]
            comptime if inverse:
                wi = -wi

            var ur = bre[Coord(i)]
            var ui = bim[Coord(i)]
            var vr = bre[Coord(j)]
            var vi = bim[Coord(j)]
            var tr = vr * wr - vi * wi
            var ti = vr * wi + vi * wr

            bre.store[1](Coord(i), ur + tr)
            bim.store[1](Coord(i), ui + ti)
            bre.store[1](Coord(j), ur - tr)
            bim.store[1](Coord(j), ui - ti)

        elementwise[simd_width=1, target="gpu" if gpu else "cpu"](
            butterfly, Coord(half_n), ctx
        )

    comptime if inverse:
        var nre = re.view()
        var nim = im.view()
        comptime scale = 1.0 / Float64(n)

        @always_inline
        def normalize[
            w: Int, alignment: Int = 1
        ](coord: Coord) {var nre, var nim}:
            var i = coord_to_index_list(coord)[0]
            nre.store[1](Coord(i), nre[Coord(i)] * Scalar[dtype](scale))
            nim.store[1](Coord(i), nim[Coord(i)] * Scalar[dtype](scale))

        elementwise[simd_width=1, target="gpu" if gpu else "cpu"](
            normalize, Coord(n), ctx
        )

    ctx.synchronize()

    # `view()` erases the origin, so neither `x` nor the twiddle tables are
    # kept alive by the views the kernels read through.
    _ = x^
    _ = wr_all^
    _ = wi_all^

    return (re^, im^)


def fft[
    dtype: DType, n: Int, gpu: Bool = False
](var x: Spectrum[dtype, n]) raises -> Spectrum[
    dtype, n
] where dtype.is_floating_point() and (n > 0 and (n & (n - 1)) == 0):
    """The forward transform of the complex sequence `x`,
    unnormalized. `numpy.fft.fft`, returned as a real/imaginary pair.

    `X[k] = sum_j x[j] * exp(-2*pi*i*j*k/n)` -- NumPy's and SciPy's sign
    convention. `n` must be a power of two, which the `where` clause makes a
    compile error rather than a run-time check.

    `numax.fft.array.fft` is the sibling that differentiates and runs inside
    a kernel body, at register-resident sizes.
    """
    return _transform[dtype, n, gpu, False](x^)


def ifft[
    dtype: DType, n: Int, gpu: Bool = False
](var x: Spectrum[dtype, n]) raises -> Spectrum[
    dtype, n
] where dtype.is_floating_point() and (n > 0 and (n & (n - 1)) == 0):
    """The inverse transform of `x`, normalized by `1/n`.
    `numpy.fft.ifft`.

    The forward engine with the twiddle angles negated and a scaling pass,
    so `ifft(fft(x))` returns `x` to rounding.
    """
    return _transform[dtype, n, gpu, True](x^)


def rfft[
    dtype: DType, n: Int, gpu: Bool = False
](var x: Static[dtype, n]) raises -> Spectrum[
    dtype, n // 2 + 1
] where dtype.is_floating_point() and (n > 0 and (n & (n - 1)) == 0):
    """The forward transform of a **real** sequence, returning the half
    spectrum `X[0..n/2]`. `numpy.fft.rfft`.

    The second half is the conjugate mirror of the first for real input, so
    returning it would be returning known information.

    This embeds the input as complex with a zero imaginary part and
    truncates the result, which does about twice the arithmetic the
    half-length-plus-post-pass trick would. That is the same choice
    `numax.fft.array` documents, and for the same reason: it is one code
    path rather than two, and the transform is memory-bound at the sizes
    this tier is for. Specializing it is a later commit, not a missing
    feature.
    """
    comptime keep = n // 2 + 1
    var ctx = x.context()
    var imag = zeros[dtype, n](ctx)
    var full = _transform[dtype, n, gpu, False]((x^, imag^))

    var re = Static[dtype, keep]._uninitialized(ctx)
    var im = Static[dtype, keep]._uninitialized(ctx)
    var fre = full[0].view()
    var fim = full[1].view()
    var hre = re.view()
    var him = im.view()

    @always_inline
    def truncate[
        w: Int, alignment: Int = 1
    ](coord: Coord) {var fre, var fim, var hre, var him}:
        var i = coord_to_index_list(coord)[0]
        hre.store[1](Coord(i), fre[Coord(i)])
        him.store[1](Coord(i), fim[Coord(i)])

    elementwise[simd_width=1, target="gpu" if gpu else "cpu"](
        truncate, Coord(keep), ctx
    )
    ctx.synchronize()

    # `view()` erases the origin, so `full` is not kept alive by `fre`/`fim`
    # and its buffers would be freed while `truncate` still reads them.
    _ = full^

    return (re^, im^)


def fftfreq[
    dtype: DType, n: Int
](
    spacing: Scalar[dtype] = 1, ctx: Optional[DeviceContext] = None
) raises -> Static[dtype, n] where dtype.is_floating_point():
    """The frequency grid `fft` output sits on. `numpy.fft.fftfreq`.

    `[0, 1, ..., n/2-1, -n/2, ..., -1] / (n * spacing)` -- the second half
    is negative, which is what `fftshift` reorders.
    """
    var values = List[Scalar[dtype]](capacity=n)
    var denominator = Scalar[dtype](n) * spacing
    for i in range(n):
        var index = i if i < (n + 1) // 2 else i - n
        values.append(Scalar[dtype](index) / denominator)
    return Static[dtype, n](
        ctx.value() if ctx else DeviceContext(api="cpu"), values^
    )


def rfftfreq[
    dtype: DType, n: Int
](
    spacing: Scalar[dtype] = 1, ctx: Optional[DeviceContext] = None
) raises -> Static[dtype, n // 2 + 1] where dtype.is_floating_point():
    """The frequency grid `rfft` output sits on: `[0, 1, ..., n/2] / (n *
    spacing)`, all non-negative. `numpy.fft.rfftfreq`."""
    comptime keep = n // 2 + 1
    var values = List[Scalar[dtype]](capacity=keep)
    var denominator = Scalar[dtype](n) * spacing
    for i in range(keep):
        values.append(Scalar[dtype](i) / denominator)
    return Static[dtype, keep](
        ctx.value() if ctx else DeviceContext(api="cpu"), values^
    )
