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

## One engine, over lanes

Every transform here is `batch` independent transforms of length `n`,
lane `b` of a rank-2 view into lane `b` of another. A 1-D `fft` is that
engine at `batch = 1`; `fft2` is the engine along the rows and then, through
a zero-copy `TileTensor.transpose()` of the same buffer, along the columns,
so the row-column decomposition costs no transpose pass and no second set
of kernels. The rank-2 view is built over the tensor's own pointer with a
`row_major[rows, cols]()` layout -- the construction `numax.core.tensor`'s
runtime `map` uses to flatten, run the other way -- so no `Tensor` is ever
reshaped or copied to get there.

## The algorithm, and the launch count

Bit-reversal permutation, then `log2(n)` radix-2 Cooley-Tukey stages, each
one `elementwise` launch of `batch * n/2` butterflies. `log2(n) + 1`
launches, and the data never touches the host in between -- the same
device-residency rule the blocked factorizations follow.

The stages run **in place**, which is safe rather than lucky: butterfly `t`
of a stage touches exactly the pair `(i, i + half)` of its own lane, and
those pairs are disjoint across `t`, so no two threads of a launch address
the same element. That is what makes the permutation the only extra pass.
Stockham autosort would fold the permutation into the stages and save that
one launch of the `log2(n) + 1`; it is not written, because the permutation
is a pure gather and the stages are where the arithmetic is.

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

## Any length: Bluestein

The radix-2 engine needs a power of two. Every other length goes through
Bluestein's chirp-z identity, `jk = (j^2 + k^2 - (k-j)^2) / 2`, which turns
a length-`n` DFT into one circular convolution of length `m =
next_fast_len(2n - 1)` -- a power of two, so the radix-2 engine runs it:
pre-multiply by the chirp `exp(-i*pi*j^2/n)`, three length-`m` transforms
(the chirp's, the data's, and the inverse of their product), post-multiply
by the chirp again. `_dft` picks the path from `n` at compile time, so the
public transforms take any `n > 0` and a caller never names the algorithm.

The cost is honest and stated: a length-`n` Bluestein transform is three
radix-2 transforms of a length between `2n` and `4n`, plus three
`elementwise` passes, so it runs roughly six to twelve times slower than a
power-of-two transform of length `n` and its rounding error is that of the
length-`m` engine with two extra complex products -- about twice a
power-of-two transform's. `next_fast_len` exists so a caller who can pad
does. The chirp angles are formed from `j^2 mod 2n` on the host in
`Float64`, so they never lose digits to a large `j^2`.

The `Array` tier stays power-of-two: Bluestein triples the register
footprint, which is the one resource that tier is built around.
"""

from std.math import cos as _cos, sin as _sin

from layout import Coord, TileTensor, coord_to_index_list
from layout.tile_layout import TensorLayout, row_major
from max.algorithm.functional import elementwise
from max.gpu.host import DeviceContext

from ..core.array import Static, Tensor, zeros

comptime _TWO_PI = 6.283185307179586
comptime _PI = 3.141592653589793

comptime Spectrum[dtype: DType, *dims: Int] = Tuple[
    Static[dtype, *dims], Static[dtype, *dims]
]
"""A complex array over `Tensor`, as `(real, imaginary)`.

One value rather than two arguments so the transforms compose --
`ifft(fft(x))` type-checks, where a pair of `mut` arguments read as aliasing
each other when they come out of the same tuple. `numax` still owns exactly
one tensor type; this is a pair of them, not a new one.

The shape is `Static`'s own: `Spectrum[dtype, n]` is the sequence `fft`
takes and `Spectrum[dtype, rows, cols]` the image `fft2` does.
"""

comptime _Lanes[dtype: DType, LayoutType: TensorLayout] = TileTensor[
    dtype, LayoutType, MutAnyOrigin
]
"""What the engine transforms along: a rank-2 view whose `[b, i]` is
element `i` of lane `b`.

Any rank-2 layout, so a row-major view walks a matrix's rows and its
`transpose()` walks the columns of the same memory, and the kernels never
know which."""


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


def _is_power_of_two(n: Int) -> Bool:
    return n > 0 and (n & (n - 1)) == 0


def _next_power_of_two(n: Int) -> Int:
    """The smallest power of two `>= n`; `1` for `n <= 1`."""
    var m = 1
    while m < n:
        m <<= 1
    return m


def next_fast_len(n: Int) -> Int:
    """The smallest length `>= n` this module transforms at full speed --
    the next power of two. `scipy.fft.next_fast_len`, for numax's engine.

    SciPy's answer is the next 2-3-5-7-11-smooth number because pocketfft
    has a radix for each of those primes; this engine has radix 2 and
    Bluestein, so its fast lengths are the powers of two and every other
    length costs three transforms of `next_fast_len(2n - 1)`. Same contract
    -- "pad to this and the transform is cheapest" -- and a different set,
    which is why the name is kept and the divergence is recorded in
    `docs/parity.md`. Zero-padding changes the frequency grid (`fftfreq`)
    but not what the spectrum says about the signal.

    Usable at compile time: `comptime m = next_fast_len(1000)` is `1024`.
    """
    return _next_power_of_two(n)


def _as_matrix[
    dtype: DType, rows: Int, cols: Int, LayoutType: TensorLayout
](mut t: Tensor[dtype, LayoutType]) -> _Lanes[
    dtype, type_of(row_major[rows, cols]())
]:
    """`t`'s storage seen as a `rows x cols` row-major matrix, with no copy.

    The tensor's elements are contiguous and row-major, so a rank-2 layout
    over the same pointer addresses the same memory in the same order --
    the construction the runtime `map` in `numax.core.tensor` uses to
    flatten, run the other way. `rows * cols` must equal `t`'s element
    count; every caller here passes `t`'s own shape or `(1, n)`.

    Valid only while `t` is alive, like `view()`; the origin is erased.
    """
    var v: _Lanes[dtype, type_of(row_major[rows, cols]())] = TileTensor(
        t.view().ptr_at_offset(Coord(0)), row_major[rows, cols]()
    )
    return v


def _radix2[
    dtype: DType,
    batch: Int,
    n: Int,
    gpu: Bool,
    inverse: Bool,
    SrcLayout: TensorLayout,
    DstLayout: TensorLayout,
](
    src_re: _Lanes[dtype, SrcLayout],
    src_im: _Lanes[dtype, SrcLayout],
    dst_re: _Lanes[dtype, DstLayout],
    dst_im: _Lanes[dtype, DstLayout],
    ctx: DeviceContext,
) raises:
    """The engine: `batch` transforms of length `n`, lane `b` of `src` into
    lane `b` of `dst`. Permute, then `log2(n)` in-place butterfly stages
    over `dst`.

    `inverse` flips the sign of every twiddle angle and divides the result
    by `n`, which is the only difference between the two directions and the
    reason `ifft` is not a second implementation. `n` must be a power of
    two; the public entry points' `where` clauses guarantee it, so nothing
    here checks.
    """
    comptime bits = _log2_exact(n)
    comptime half_n = n // 2
    # A length-1 transform has no stages and no twiddles; the table still
    # has to exist, since a zero-length buffer cannot be allocated.
    comptime table = half_n if half_n > 0 else 1

    # `W[q] = exp(-2*pi*i*q/n)`, in Float64 and rounded once. Stage `span`
    # reads `W[pos * (n // span)]`, so one table serves every stage.
    var twiddle_re = List[Scalar[dtype]](capacity=table)
    var twiddle_im = List[Scalar[dtype]](capacity=table)
    for q in range(table):
        var angle = -_TWO_PI * Float64(q) / Float64(n)
        twiddle_re.append(Scalar[dtype](_cos(angle)))
        twiddle_im.append(Scalar[dtype](_sin(angle)))
    var wr_all = Static[dtype, table](ctx, twiddle_re^)
    var wi_all = Static[dtype, table](ctx, twiddle_im^)

    var sre = src_re
    var sim = src_im
    var dre = dst_re
    var dim = dst_im

    @always_inline
    def permute[
        w: Int, alignment: Int = 1
    ](coord: Coord) {var sre, var sim, var dre, var dim}:
        var idx = coord_to_index_list(coord)
        var b = idx[0]
        var i = idx[1]
        var j = _reverse_bits(i, bits)
        dre.store[1](Coord(b, i), sre[Coord(b, j)])
        dim.store[1](Coord(b, i), sim[Coord(b, j)])

    elementwise[simd_width=1, target="gpu" if gpu else "cpu"](
        permute, Coord(batch, n), ctx
    )

    # One launch per stage, `batch * half_n` butterflies each. In place:
    # butterfly `t` of lane `b` owns the pair `(i, i + half)` of that lane
    # and those are disjoint across `t`.
    comptime for stage in range(bits):
        comptime half = 1 << stage
        comptime span = half << 1
        comptime stride = n // span
        var bre = dst_re
        var bim = dst_im
        var twr = wr_all.view()
        var twi = wi_all.view()

        @always_inline
        def butterfly[
            w: Int, alignment: Int = 1
        ](coord: Coord) {var bre, var bim, var twr, var twi}:
            var idx = coord_to_index_list(coord)
            var b = idx[0]
            var t = idx[1]
            var block = t // half
            var pos = t % half
            var i = block * span + pos
            var j = i + half

            var q = pos * stride
            var wr = twr[Coord(q)]
            var wi = twi[Coord(q)]
            comptime if inverse:
                wi = -wi

            var ur = bre[Coord(b, i)]
            var ui = bim[Coord(b, i)]
            var vr = bre[Coord(b, j)]
            var vi = bim[Coord(b, j)]
            var tr = vr * wr - vi * wi
            var ti = vr * wi + vi * wr

            bre.store[1](Coord(b, i), ur + tr)
            bim.store[1](Coord(b, i), ui + ti)
            bre.store[1](Coord(b, j), ur - tr)
            bim.store[1](Coord(b, j), ui - ti)

        elementwise[simd_width=1, target="gpu" if gpu else "cpu"](
            butterfly, Coord(batch, half_n), ctx
        )

    comptime if inverse:
        var nre = dst_re
        var nim = dst_im
        comptime scale = 1.0 / Float64(n)

        @always_inline
        def normalize[
            w: Int, alignment: Int = 1
        ](coord: Coord) {var nre, var nim}:
            var idx = coord_to_index_list(coord)
            var c = Coord(idx[0], idx[1])
            nre.store[1](c, nre[c] * Scalar[dtype](scale))
            nim.store[1](c, nim[c] * Scalar[dtype](scale))

        elementwise[simd_width=1, target="gpu" if gpu else "cpu"](
            normalize, Coord(batch, n), ctx
        )

    ctx.synchronize()

    # `view()` erases the origin, so the twiddle tables are not kept alive
    # by the views the stages read through.
    _ = wr_all^
    _ = wi_all^


def _bluestein[
    dtype: DType,
    batch: Int,
    n: Int,
    gpu: Bool,
    inverse: Bool,
    SrcLayout: TensorLayout,
    DstLayout: TensorLayout,
](
    src_re: _Lanes[dtype, SrcLayout],
    src_im: _Lanes[dtype, SrcLayout],
    dst_re: _Lanes[dtype, DstLayout],
    dst_im: _Lanes[dtype, DstLayout],
    ctx: DeviceContext,
) raises:
    """Bluestein's chirp-z: `batch` length-`n` DFTs at any `n`, each as one
    circular convolution of length `m = next_fast_len(2n - 1)` run by
    `_radix2`. Same lanes contract as `_radix2`, any `n > 0`.

    With `w[j] = exp(s*i*pi*j^2/n)` (`s = -1` forward, `+1` inverse),
    `X[k] = w[k] * sum_j (x[j] w[j]) conj(w[k-j])`. The sum is a linear
    convolution over `k - j` in `(-n, n)`, which a circular one of length
    `m >= 2n - 1` reproduces exactly once `conj(w)` is wrapped -- `h[j] =
    h[m-j] = conj(w[j])` -- so no term aliases onto another. The
    convolution itself is `ifft(fft(a) * fft(h))` through the radix-2
    engine, whose own `1/m` is the convolution's normalization; the DFT's
    `1/n` for the inverse direction is applied in the last pass.

    Three `elementwise` passes (spread, multiply, collect) around three
    length-`m` transforms. `h`'s transform is recomputed per call rather
    than cached; at `O(m log m)` against the data's own transforms it is a
    fixed third of the work, and a cache would be the module's first piece
    of mutable global state.
    """
    comptime m = _next_power_of_two(2 * n - 1)
    comptime sign = 1.0 if inverse else -1.0

    # The chirp, with `j^2` reduced mod `2n` first: `exp(i*pi*j^2/n)` has
    # that period in `j^2`, and the reduced angle keeps every digit where
    # `j^2` itself would not past a few thousand.
    var wr = List[Scalar[dtype]](capacity=n)
    var wi = List[Scalar[dtype]](capacity=n)
    var hr = List[Scalar[dtype]](length=m, fill=Scalar[dtype](0))
    var hi = List[Scalar[dtype]](length=m, fill=Scalar[dtype](0))
    for j in range(n):
        var angle = sign * _PI * Float64((j * j) % (2 * n)) / Float64(n)
        var c = Scalar[dtype](_cos(angle))
        var d = Scalar[dtype](_sin(angle))
        wr.append(c)
        wi.append(d)
        # `h = conj(w)`, wrapped so `h[-j]` sits at `m - j`.
        hr[j] = c
        hi[j] = -d
        if j > 0:
            hr[m - j] = c
            hi[m - j] = -d
    var chirp_re = Static[dtype, n](ctx, wr^)
    var chirp_im = Static[dtype, n](ctx, wi^)
    var h_re = Static[dtype, m](ctx, hr^)
    var h_im = Static[dtype, m](ctx, hi^)

    # `H = fft(h)`, once per call.
    var big_h_re = Static[dtype, m]._uninitialized(ctx)
    var big_h_im = Static[dtype, m]._uninitialized(ctx)
    _radix2[dtype, 1, m, gpu, False](
        _as_matrix[dtype, 1, m](h_re),
        _as_matrix[dtype, 1, m](h_im),
        _as_matrix[dtype, 1, m](big_h_re),
        _as_matrix[dtype, 1, m](big_h_im),
        ctx,
    )

    # `a[b, j] = x[b, j] * w[j]` for `j < n`, zero to `m`.
    var a_re = Static[dtype, batch, m]._uninitialized(ctx)
    var a_im = Static[dtype, batch, m]._uninitialized(ctx)
    var sre = src_re
    var sim = src_im
    var are = a_re.view()
    var aim = a_im.view()
    var cre = chirp_re.view()
    var cim = chirp_im.view()

    @always_inline
    def spread[
        w: Int, alignment: Int = 1
    ](coord: Coord) {var sre, var sim, var are, var aim, var cre, var cim}:
        var idx = coord_to_index_list(coord)
        var b = idx[0]
        var j = idx[1]
        if j < n:
            var xr = sre[Coord(b, j)]
            var xi = sim[Coord(b, j)]
            var wr = cre[Coord(j)]
            var wi = cim[Coord(j)]
            are.store[1](Coord(b, j), xr * wr - xi * wi)
            aim.store[1](Coord(b, j), xr * wi + xi * wr)
        else:
            are.store[1](Coord(b, j), Scalar[dtype](0))
            aim.store[1](Coord(b, j), Scalar[dtype](0))

    elementwise[simd_width=1, target="gpu" if gpu else "cpu"](
        spread, Coord(batch, m), ctx
    )

    # `A = fft(a)`, then `A *= H` in place, then `a = ifft(A)` back into
    # `a`'s buffers, which are free once `A` exists.
    var big_re = Static[dtype, batch, m]._uninitialized(ctx)
    var big_im = Static[dtype, batch, m]._uninitialized(ctx)
    _radix2[dtype, batch, m, gpu, False](
        _as_matrix[dtype, batch, m](a_re),
        _as_matrix[dtype, batch, m](a_im),
        _as_matrix[dtype, batch, m](big_re),
        _as_matrix[dtype, batch, m](big_im),
        ctx,
    )
    var bre = big_re.view()
    var bim = big_im.view()
    var hre = big_h_re.view()
    var him = big_h_im.view()

    @always_inline
    def multiply[
        w: Int, alignment: Int = 1
    ](coord: Coord) {var bre, var bim, var hre, var him}:
        var idx = coord_to_index_list(coord)
        var c = Coord(idx[0], idx[1])
        var ar = bre[c]
        var ai = bim[c]
        var gr = hre[Coord(idx[1])]
        var gi = him[Coord(idx[1])]
        bre.store[1](c, ar * gr - ai * gi)
        bim.store[1](c, ar * gi + ai * gr)

    elementwise[simd_width=1, target="gpu" if gpu else "cpu"](
        multiply, Coord(batch, m), ctx
    )
    _radix2[dtype, batch, m, gpu, True](
        _as_matrix[dtype, batch, m](big_re),
        _as_matrix[dtype, batch, m](big_im),
        _as_matrix[dtype, batch, m](a_re),
        _as_matrix[dtype, batch, m](a_im),
        ctx,
    )

    # `X[k] = a[k] * w[k]`, and the inverse direction's `1/n`.
    comptime scale = 1.0 / Float64(n) if inverse else 1.0
    var dre = dst_re
    var dim = dst_im

    @always_inline
    def collect[
        w: Int, alignment: Int = 1
    ](coord: Coord) {var are, var aim, var cre, var cim, var dre, var dim}:
        var idx = coord_to_index_list(coord)
        var b = idx[0]
        var k = idx[1]
        var ar = are[Coord(b, k)]
        var ai = aim[Coord(b, k)]
        var wr = cre[Coord(k)]
        var wi = cim[Coord(k)]
        dre.store[1](Coord(b, k), (ar * wr - ai * wi) * Scalar[dtype](scale))
        dim.store[1](Coord(b, k), (ar * wi + ai * wr) * Scalar[dtype](scale))

    elementwise[simd_width=1, target="gpu" if gpu else "cpu"](
        collect, Coord(batch, n), ctx
    )
    ctx.synchronize()

    # Every table and workspace was read through an origin-erased view.
    _ = chirp_re^
    _ = chirp_im^
    _ = h_re^
    _ = h_im^
    _ = big_h_re^
    _ = big_h_im^
    _ = a_re^
    _ = a_im^
    _ = big_re^
    _ = big_im^


def _dft[
    dtype: DType,
    batch: Int,
    n: Int,
    gpu: Bool,
    inverse: Bool,
    SrcLayout: TensorLayout,
    DstLayout: TensorLayout,
](
    src_re: _Lanes[dtype, SrcLayout],
    src_im: _Lanes[dtype, SrcLayout],
    dst_re: _Lanes[dtype, DstLayout],
    dst_im: _Lanes[dtype, DstLayout],
    ctx: DeviceContext,
) raises:
    """`batch` length-`n` DFTs along the lanes of `src`, into `dst`. The one
    place the algorithm is chosen, from `n` at compile time: radix-2 at a
    power of two, Bluestein otherwise. Every public transform comes here."""
    comptime if _is_power_of_two(n):
        _radix2[dtype, batch, n, gpu, inverse](
            src_re, src_im, dst_re, dst_im, ctx
        )
    else:
        _bluestein[dtype, batch, n, gpu, inverse](
            src_re, src_im, dst_re, dst_im, ctx
        )


def _dft1[
    dtype: DType, n: Int, gpu: Bool, inverse: Bool
](var x: Spectrum[dtype, n]) raises -> Spectrum[dtype, n]:
    """A single length-`n` transform: the engine at `batch = 1`."""
    var ctx = x[0].context()
    var re = Static[dtype, n]._uninitialized(ctx)
    var im = Static[dtype, n]._uninitialized(ctx)
    _dft[dtype, 1, n, gpu, inverse](
        _as_matrix[dtype, 1, n](x[0]),
        _as_matrix[dtype, 1, n](x[1]),
        _as_matrix[dtype, 1, n](re),
        _as_matrix[dtype, 1, n](im),
        ctx,
    )
    _ = x^
    return (re^, im^)


def _dft2[
    dtype: DType, rows: Int, cols: Int, gpu: Bool, inverse: Bool
](var x: Spectrum[dtype, rows, cols]) raises -> Spectrum[dtype, rows, cols]:
    """The 2-D transform: the engine along every row, then along every
    column of the result through a transposed view of the same buffer."""
    var ctx = x[0].context()
    var mid_re = Static[dtype, rows, cols]._uninitialized(ctx)
    var mid_im = Static[dtype, rows, cols]._uninitialized(ctx)
    _dft[dtype, rows, cols, gpu, inverse](
        _as_matrix[dtype, rows, cols](x[0]),
        _as_matrix[dtype, rows, cols](x[1]),
        _as_matrix[dtype, rows, cols](mid_re),
        _as_matrix[dtype, rows, cols](mid_im),
        ctx,
    )
    var re = Static[dtype, rows, cols]._uninitialized(ctx)
    var im = Static[dtype, rows, cols]._uninitialized(ctx)
    _dft[dtype, cols, rows, gpu, inverse](
        _as_matrix[dtype, rows, cols](mid_re).transpose(),
        _as_matrix[dtype, rows, cols](mid_im).transpose(),
        _as_matrix[dtype, rows, cols](re).transpose(),
        _as_matrix[dtype, rows, cols](im).transpose(),
        ctx,
    )
    _ = x^
    _ = mid_re^
    _ = mid_im^
    return (re^, im^)


def fft[
    dtype: DType, n: Int, gpu: Bool = False
](var x: Spectrum[dtype, n]) raises -> Spectrum[dtype, n] where (
    dtype.is_floating_point() and n > 0
):
    """The forward transform of the complex sequence `x`,
    unnormalized. `numpy.fft.fft`, returned as a real/imaginary pair.

    `X[k] = sum_j x[j] * exp(-2*pi*i*j*k/n)` -- NumPy's and SciPy's sign
    convention. Any `n > 0`: a power of two runs the radix-2 engine
    directly and every other length goes through Bluestein, at roughly the
    cost of three power-of-two transforms of length `next_fast_len(2n - 1)`
    -- the module docstring has the accounting, and `next_fast_len` is the
    length to pad to when padding is an option.

    `numax.fft.array.fft` is the sibling that differentiates and runs inside
    a kernel body, at register-resident sizes.
    """
    return _dft1[dtype, n, gpu, False](x^)


def ifft[
    dtype: DType, n: Int, gpu: Bool = False
](var x: Spectrum[dtype, n]) raises -> Spectrum[dtype, n] where (
    dtype.is_floating_point() and n > 0
):
    """The inverse transform of `x`, normalized by `1/n`.
    `numpy.fft.ifft`.

    The forward engine with the twiddle angles negated and a scaling pass,
    so `ifft(fft(x))` returns `x` to rounding.
    """
    return _dft1[dtype, n, gpu, True](x^)


def rfft[
    dtype: DType, n: Int, gpu: Bool = False
](var x: Static[dtype, n]) raises -> Spectrum[dtype, n // 2 + 1] where (
    dtype.is_floating_point() and n > 0
):
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
    var full = _dft1[dtype, n, gpu, False]((x^, imag^))

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


def irfft[
    dtype: DType, keep: Int, gpu: Bool = False, n: Int = 2 * (keep - 1)
](var x: Spectrum[dtype, keep]) raises -> Static[
    dtype, n
] where dtype.is_floating_point():
    """The inverse of `rfft`: the real sequence of length `n` whose half
    spectrum is `x`. `numpy.fft.irfft(x, n)`.

    `n` cannot be read off the half spectrum -- `n // 2 + 1` is the same for
    `n` and `n - 1` -- so it is a parameter with NumPy's default, the even
    length `2 * (keep - 1)`; an odd-length signal is recovered with
    `irfft[n=7](x)`, as NumPy's `n=` argument does it. An `n` the half
    spectrum could not have come from is still a compile error, but through
    `comptime assert` in the body rather than a `where` clause: `keep`
    arrives from `rfft`'s return type as the unevaluated expression `n // 2
    + 1`, and the `where` prover cannot evaluate `//`.

    The missing half is rebuilt by conjugate symmetry, `X[n-k] = conj(X[k])`,
    in one gather kernel rather than stored; the inverse engine then runs
    on the full spectrum and the real half of its result is the answer.
    For a spectrum that really is conjugate-symmetric the imaginary half is
    zero to rounding, and dropping it is what makes `irfft(rfft(x)) == x`.
    Handed a half spectrum that is not, this returns the transform of its
    symmetrized version, silently, as NumPy does -- the imaginary parts of
    bins `0` and `n/2` are the information that has nowhere to go.
    """
    comptime assert n == 2 * keep - 2 or n == 2 * keep - 1, (
        "irfft: a half spectrum of `keep` bins comes from a signal of length"
        " 2 * keep - 2 or 2 * keep - 1"
    )
    var ctx = x[0].context()

    var full_re = Static[dtype, n]._uninitialized(ctx)
    var full_im = Static[dtype, n]._uninitialized(ctx)
    var hre = x[0].view()
    var him = x[1].view()
    var fre = full_re.view()
    var fim = full_im.view()

    @always_inline
    def mirror[
        w: Int, alignment: Int = 1
    ](coord: Coord) {var hre, var him, var fre, var fim}:
        var i = coord_to_index_list(coord)[0]
        var src = i if i < keep else n - i
        var sign = Scalar[dtype](1) if i < keep else Scalar[dtype](-1)
        fre.store[1](Coord(i), hre[Coord(src)])
        fim.store[1](Coord(i), sign * him[Coord(src)])

    elementwise[simd_width=1, target="gpu" if gpu else "cpu"](
        mirror, Coord(n), ctx
    )

    var re = Static[dtype, n]._uninitialized(ctx)
    var im = Static[dtype, n]._uninitialized(ctx)
    _dft[dtype, 1, n, gpu, True](
        _as_matrix[dtype, 1, n](full_re),
        _as_matrix[dtype, 1, n](full_im),
        _as_matrix[dtype, 1, n](re),
        _as_matrix[dtype, 1, n](im),
        ctx,
    )

    # `mirror` read `x` through origin-erased views; keep it alive past the
    # launch that read it, the way every driver here pins its inputs.
    _ = x^
    _ = full_re^
    _ = full_im^
    _ = im^
    return re^


def fft2[
    dtype: DType, rows: Int, cols: Int, gpu: Bool = False
](var x: Spectrum[dtype, rows, cols]) raises -> Spectrum[
    dtype, rows, cols
] where (dtype.is_floating_point() and rows > 0 and cols > 0):
    """The 2-D transform of a `rows x cols` complex image, unnormalized.
    `numpy.fft.fft2`.

    Row-column decomposition: every row is transformed, then every column
    of the result. The 2-D DFT separates exactly, so this is the definition
    evaluated in the cheaper order -- `rows + cols` transforms rather than
    one of length `rows * cols` -- not an approximation. Rectangular, where
    `numax.fft.array.fft2` is square only: the column pass runs the same
    engine over a transposed view of the same buffer, so a second extent
    costs a second twiddle table and nothing else. Either extent may be any
    length; each axis picks radix-2 or Bluestein on its own.
    """
    return _dft2[dtype, rows, cols, gpu, False](x^)


def ifft2[
    dtype: DType, rows: Int, cols: Int, gpu: Bool = False
](var x: Spectrum[dtype, rows, cols]) raises -> Spectrum[
    dtype, rows, cols
] where (dtype.is_floating_point() and rows > 0 and cols > 0):
    """The inverse of `fft2`, normalized by `1/(rows * cols)`.
    `numpy.fft.ifft2`.

    The inverse engine along both axes; each pass contributes its own
    `1/extent` and the two compose to the full normalization, so
    `ifft2(fft2(x))` returns `x` to rounding.
    """
    return _dft2[dtype, rows, cols, gpu, True](x^)


def rfft2[
    dtype: DType, rows: Int, cols: Int, gpu: Bool = False
](var x: Static[dtype, rows, cols]) raises -> Spectrum[
    dtype, rows, cols // 2 + 1
] where (dtype.is_floating_point() and rows > 0 and cols > 0):
    """The 2-D transform of a **real** image, keeping the half spectrum
    along the last axis: `rows x (cols/2 + 1)`. `numpy.fft.rfft2`.

    `rfft` along every row, then a full `fft` down every column of the
    kept half -- NumPy's order, and the one that does the least work:
    the row pass is real-input so its second half is redundant, and
    dropping it *before* the column pass halves that pass too. The column
    transforms are complex, so nothing further is dropped; the result is
    exactly the first `cols/2 + 1` columns of `fft2` on the same image.
    """
    comptime keep = cols // 2 + 1
    var ctx = x.context()
    var imag = zeros[dtype, rows, cols](ctx)

    var full_re = Static[dtype, rows, cols]._uninitialized(ctx)
    var full_im = Static[dtype, rows, cols]._uninitialized(ctx)
    _dft[dtype, rows, cols, gpu, False](
        _as_matrix[dtype, rows, cols](x),
        _as_matrix[dtype, rows, cols](imag),
        _as_matrix[dtype, rows, cols](full_re),
        _as_matrix[dtype, rows, cols](full_im),
        ctx,
    )

    var half_re = Static[dtype, rows, keep]._uninitialized(ctx)
    var half_im = Static[dtype, rows, keep]._uninitialized(ctx)
    var fre = full_re.view()
    var fim = full_im.view()
    var hre = half_re.view()
    var him = half_im.view()

    @always_inline
    def truncate[
        w: Int, alignment: Int = 1
    ](coord: Coord) {var fre, var fim, var hre, var him}:
        var idx = coord_to_index_list(coord)
        var c = Coord(idx[0], idx[1])
        hre.store[1](c, fre[c])
        him.store[1](c, fim[c])

    elementwise[simd_width=1, target="gpu" if gpu else "cpu"](
        truncate, Coord(rows, keep), ctx
    )

    var re = Static[dtype, rows, keep]._uninitialized(ctx)
    var im = Static[dtype, rows, keep]._uninitialized(ctx)
    _dft[dtype, keep, rows, gpu, False](
        _as_matrix[dtype, rows, keep](half_re).transpose(),
        _as_matrix[dtype, rows, keep](half_im).transpose(),
        _as_matrix[dtype, rows, keep](re).transpose(),
        _as_matrix[dtype, rows, keep](im).transpose(),
        ctx,
    )

    _ = x^
    _ = imag^
    _ = full_re^
    _ = full_im^
    _ = half_re^
    _ = half_im^
    return (re^, im^)


def _rolled[
    dtype: DType, n: Int, gpu: Bool
](var x: Static[dtype, n], offset: Int) raises -> Static[dtype, n]:
    """`out[i] = x[(i + offset) % n]`: the cyclic shift both `fftshift`s
    are, one gather launch."""
    var ctx = x.context()
    var out = Static[dtype, n]._uninitialized(ctx)
    var src = x.view()
    var dst = out.view()
    var shift = offset

    @always_inline
    def gather[
        w: Int, alignment: Int = 1
    ](coord: Coord) {var src, var dst, var shift}:
        var i = coord_to_index_list(coord)[0]
        dst.store[1](Coord(i), src[Coord((i + shift) % n)])

    elementwise[simd_width=1, target="gpu" if gpu else "cpu"](
        gather, Coord(n), ctx
    )
    ctx.synchronize()
    _ = x^
    return out^


def _rolled2[
    dtype: DType, rows: Int, cols: Int, gpu: Bool
](
    var x: Static[dtype, rows, cols], row_offset: Int, col_offset: Int
) raises -> Static[dtype, rows, cols]:
    """The rank-2 `_rolled`: both axes shifted cyclically in one gather."""
    var ctx = x.context()
    var out = Static[dtype, rows, cols]._uninitialized(ctx)
    var src = x.view()
    var dst = out.view()
    var di = row_offset
    var dj = col_offset

    @always_inline
    def gather[
        w: Int, alignment: Int = 1
    ](coord: Coord) {var src, var dst, var di, var dj}:
        var idx = coord_to_index_list(coord)
        var i = idx[0]
        var j = idx[1]
        dst.store[1](Coord(i, j), src[Coord((i + di) % rows, (j + dj) % cols)])

    elementwise[simd_width=1, target="gpu" if gpu else "cpu"](
        gather, Coord(rows, cols), ctx
    )
    ctx.synchronize()
    _ = x^
    return out^


def fftshift[
    dtype: DType, n: Int, gpu: Bool = False
](var x: Static[dtype, n]) raises -> Static[dtype, n] where n > 0:
    """`x` rotated so the zero-frequency bin sits in the middle, which is
    how a spectrum is plotted. `numpy.fft.fftshift`.

    Takes one real tensor, not a `Spectrum`, because that is what NumPy's
    takes too: the canonical call is `fftshift(fftfreq(n))`, and a
    spectrum's two halves are shifted by calling this on each. Bin `k`
    lands at `(k + n // 2) % n`, so for odd `n` this and `ifftshift` are
    different rotations and only `ifftshift` undoes it.
    """
    return _rolled[dtype, n, gpu](x^, (n + 1) // 2)


def fftshift[
    dtype: DType, rows: Int, cols: Int, gpu: Bool = False
](var x: Static[dtype, rows, cols]) raises -> Static[dtype, rows, cols] where (
    rows > 0 and cols > 0
):
    """`fftshift` over both axes of a matrix -- NumPy's default for a 2-D
    input, so `fftshift(fft2(image))` puts DC at the centre pixel."""
    return _rolled2[dtype, rows, cols, gpu](
        x^, (rows + 1) // 2, (cols + 1) // 2
    )


def ifftshift[
    dtype: DType, n: Int, gpu: Bool = False
](var x: Static[dtype, n]) raises -> Static[dtype, n] where n > 0:
    """Undo `fftshift`: the zero-frequency bin back to index 0.
    `numpy.fft.ifftshift`.

    For even `n` the same rotation as `fftshift`; for odd `n` it is the
    other one, which is why both names exist.
    """
    return _rolled[dtype, n, gpu](x^, n // 2)


def ifftshift[
    dtype: DType, rows: Int, cols: Int, gpu: Bool = False
](var x: Static[dtype, rows, cols]) raises -> Static[dtype, rows, cols] where (
    rows > 0 and cols > 0
):
    """`ifftshift` over both axes of a matrix."""
    return _rolled2[dtype, rows, cols, gpu](x^, rows // 2, cols // 2)


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
