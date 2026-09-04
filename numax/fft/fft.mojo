"""Radix-2 Cooley-Tukey FFT over `Complex[Inner]`, at a compile-time size.

The transform is `X[k] = sum_j x[j] * exp(-2*pi*i*j*k/n)` -- the standard
sign convention, matching NumPy's `fft` and SciPy's.

Size is a compile-time parameter, and it's given as `log2n` rather than `n`
so that "must be a power of two" is structural instead of a constraint the
compiler would have to check (`numax.integrate.simpson` takes a panel
count for the same reason). A caller wanting a 64-point transform writes
`fft[Inner, 6]`.

Everything here inherits `numax`'s usual properties from `Complex`, and one
of them is unusual for an FFT: this one differentiates. `fft` over
`Complex[Dual[Plain[...]]]` returns the transform *and* its derivative with
respect to whatever the input was seeded on, with no adjoint rule written
anywhere -- the butterfly is built from `Complex` arithmetic, and `Complex`
is built from `FloatLike` arithmetic, which `Dual` already knows how to
differentiate. It also runs inside a single GPU thread, since the loop
structure depends only on `n`.

Two things this is deliberately not:

- **Not a large-transform FFT.** The data lives in an `Array[Complex[Inner],
  n]`, which is a register/stack object, and the twiddle table is built by a
  `comptime for` over `n/2` entries, so compile time and register pressure
  both grow with `n`. This is sized for the transforms that appear *inside*
  a per-element kernel -- 16, 64, 256 points -- not for a 4-million-point
  spectrogram. The environment's `_cufft` binding covers that case on CUDA
  hardware, and nothing covers it on Metal.
- **Not radix-4, split-radix, or real-input-specialized.** A real-input
  transform of length `n` can be done as a complex transform of length
  `n/2`, and radix-4 halves the number of twiddle multiplies; neither is
  implemented, and both are ordinary continuations of what's here rather
  than redesigns.
"""

from std.collections import Array
from std.math import cos, sin

from ..core.complex import Complex
from ..core.numeric import FloatLike

comptime _TWO_PI = 6.283185307179586


def _reverse_bits(index: Int, bits: Int) -> Int:
    """`index` with its low `bits` bits reversed.

    A scalar `Int` computation on a loop counter, identical in every SIMD
    lane and every GPU thread, so the data-independent-control-flow rule is
    satisfied -- this is the same kind of index arithmetic as
    `numax.special.legendre`'s recurrence bound, not a per-lane branch.
    """
    var out = 0
    var rest = index
    for _ in range(bits):
        out = (out << 1) | (rest & 1)
        rest = rest >> 1
    return out


def _twiddles[
    T: FloatLike, log2n: Int
]() -> Array[Complex[T], (1 << log2n) // 2]:
    """`exp(-2*pi*i*k/n)` for `k` in `[0, n/2)`.

    Built by a `comptime for` so each angle's `cos`/`sin` is evaluated by
    the compiler in `Float64` and reaches the generated code as a
    `dtype`-native literal. A runtime loop would need `Float64(k)` from a
    runtime `Int`, which emits an int64-to-double conversion Metal rejects
    outright (see `numax.special.orthopoly`'s module docstring). This is also why
    the table is not stored as `Array[Float64, ...]` and narrowed later,
    the way `Compensated.exp`'s coefficients once were.
    """
    comptime n = 1 << log2n
    var out = Array[Complex[T], n // 2](fill=Complex[T].one())
    comptime for k in range(n // 2):
        comptime angle = -_TWO_PI * Float64(k) / Float64(n)
        comptime c = cos(angle)
        comptime s = sin(angle)
        out[k] = Complex[T](T.constant(c), T.constant(s))
    return out^


def _bit_reversed[
    T: FloatLike, log2n: Int
](x: Array[Complex[T], 1 << log2n]) -> Array[Complex[T], 1 << log2n]:
    comptime n = 1 << log2n
    var out = Array[Complex[T], n](fill=Complex[T].constant(0.0))
    for i in range(n):
        out[_reverse_bits(i, log2n)] = x[i].copy()
    return out^


def _butterflies[
    T: FloatLike, log2n: Int
](
    mut values: Array[Complex[T], 1 << log2n],
    tw: Array[Complex[T], (1 << log2n) // 2],
):
    """The `log2n` decimation-in-time stages, in place on bit-reversed input.

    `stride` is how far apart this stage's twiddles sit in the shared
    full-size table, which is what lets one table serve every stage.
    """
    comptime n = 1 << log2n
    for stage in range(log2n):
        var half = 1 << stage
        var length = half * 2
        var stride = n // length
        for base in range(0, n, length):
            for j in range(half):
                var top = values[base + j].copy()
                var bottom = values[base + j + half] * tw[j * stride]
                values[base + j] = top + bottom
                values[base + j + half] = top - bottom


def fft[
    T: FloatLike, log2n: Int
](x: Array[Complex[T], 1 << log2n]) -> Array[Complex[T], 1 << log2n]:
    """The forward transform of `x`, unnormalized.

    `X[k] = sum_j x[j] * exp(-2*pi*i*j*k/n)`, for `n = 2^log2n`.
    """
    var values = _bit_reversed[T, log2n](x)
    var tw = _twiddles[T, log2n]()
    _butterflies[T, log2n](values, tw)
    return values^


def ifft[
    T: FloatLike, log2n: Int
](x: Array[Complex[T], 1 << log2n]) -> Array[Complex[T], 1 << log2n]:
    """The inverse transform of `x`, normalized by `1/n`.

    `ifft(fft(x)) == x` to rounding. Implemented as conjugate-forward-
    conjugate rather than with a second twiddle table of opposite sign,
    since a conjugation is two negations and a table is `n/2` constants.
    """
    comptime n = 1 << log2n
    var scale = T.constant(1.0 / Float64(n))
    var conjugated = Array[Complex[T], n](fill=Complex[T].constant(0.0))
    for i in range(n):
        conjugated[i] = Complex[T](x[i].re.copy(), -x[i].im)

    var spectrum = fft[T, log2n](conjugated)
    var out = Array[Complex[T], n](fill=Complex[T].constant(0.0))
    for i in range(n):
        out[i] = Complex[T](spectrum[i].re * scale, -(spectrum[i].im * scale))
    return out^


def circular_convolve[
    T: FloatLike, log2n: Int
](a: Array[Complex[T], 1 << log2n], b: Array[Complex[T], 1 << log2n]) -> Array[
    Complex[T], 1 << log2n
]:
    """`ifft(fft(a) * fft(b))` -- circular convolution via the frequency
    domain.

    Circular, not linear: index arithmetic wraps modulo `n`, so a caller
    wanting a linear convolution of two `m`-point sequences pads both to
    `n >= 2*m` first. At the sizes this module targets the direct `O(n^2)`
    convolution is often faster; this exists because it's the identity the
    FFT is for, and because it differentiates like everything else here.
    """
    comptime n = 1 << log2n
    var fa = fft[T, log2n](a)
    var fb = fft[T, log2n](b)
    var product = Array[Complex[T], n](fill=Complex[T].constant(0.0))
    for i in range(n):
        product[i] = fa[i] * fb[i]
    return ifft[T, log2n](product)


def rfft[
    T: FloatLike, log2n: Int
](x: Array[T, 1 << log2n]) -> Array[Complex[T], (1 << log2n) // 2 + 1]:
    """The forward transform of a *real* sequence, returning only the
    non-redundant half of the spectrum.

    A real input's spectrum is conjugate-symmetric -- `X[n-k] ==
    conj(X[k])` -- so the second half carries no information the first does
    not. `rfft` returns bins `0 .. n/2` inclusive: `n/2 + 1` values, of
    which bin 0 (DC) and bin `n/2` (Nyquist) are purely real for a real
    input.

    Implemented by embedding the real sequence as complex with zero
    imaginary parts and calling `fft`, then truncating. That does about
    twice the arithmetic a dedicated real-input algorithm would (the
    standard trick packs the even and odd samples into one half-length
    complex transform), and it is the right trade here: at the sizes this
    module targets -- transforms small enough to live in registers inside a
    per-element kernel -- correctness and one code path matter more than a
    factor of two, and the packing trick needs a post-processing pass whose
    twiddle table would double the compile-time constants.
    """
    comptime n = 1 << log2n
    var embedded = Array[Complex[T], n](fill=Complex[T].constant(0.0))
    for i in range(n):
        embedded[i] = Complex[T](x[i].copy(), T.constant(0.0))

    var full = fft[T, log2n](embedded)
    var out = Array[Complex[T], n // 2 + 1](fill=Complex[T].constant(0.0))
    for k in range(n // 2 + 1):
        out[k] = full[k].copy()
    return out^


def irfft[
    T: FloatLike, log2n: Int
](spectrum: Array[Complex[T], (1 << log2n) // 2 + 1]) -> Array[T, 1 << log2n]:
    """The inverse of `rfft`: rebuild the real sequence from its half
    spectrum.

    The missing half is reconstructed by conjugate symmetry rather than
    stored, which is the whole point of the `rfft` layout. Only the real
    part of the inverse transform is returned; for a spectrum that really
    is conjugate-symmetric the imaginary part is zero to rounding, and
    discarding it is what makes `irfft(rfft(x)) == x`.

    A caller who hands this an arbitrary (non-symmetric) half spectrum gets
    the transform of its symmetrized version, silently -- there is nothing
    to check against without a branch, and the operation is still
    well-defined.
    """
    comptime n = 1 << log2n
    var full = Array[Complex[T], n](fill=Complex[T].constant(0.0))
    for k in range(n // 2 + 1):
        full[k] = spectrum[k].copy()
    for k in range(n // 2 + 1, n):
        var mirrored = spectrum[n - k].copy()
        full[k] = Complex[T](mirrored.re.copy(), -mirrored.im)

    var inverted = ifft[T, log2n](full)
    var out = Array[T, n](fill=T.constant(0.0))
    for i in range(n):
        out[i] = inverted[i].re.copy()
    return out^


def fftfreq[T: FloatLike, log2n: Int](spacing: T) -> Array[T, 1 << log2n]:
    """The frequency of each `fft` output bin, in cycles per unit of
    `spacing`.

    Matches `numpy.fft.fftfreq`'s layout exactly, including its sign
    convention: bins `0 .. n/2 - 1` are the non-negative frequencies
    `k / (n*spacing)`, and bins `n/2 .. n-1` are the negative ones
    `(k - n) / (n*spacing)`. The Nyquist bin `n/2` is therefore reported as
    *negative*, which looks wrong and is what NumPy does -- for even `n`
    that bin is genuinely ambiguous (`+f_nyq` and `-f_nyq` alias), and
    matching NumPy matters more than picking a side.

    `spacing` is the sample interval, so pass `1/sample_rate`.
    """
    comptime n = 1 << log2n
    var out = Array[T, n](fill=T.constant(0.0))
    var scale = T.one() / (T.constant(Float64(n)) * spacing)
    for k in range(n):
        var index = k if k < n // 2 else k - n
        out[k] = T.constant(Float64(index)) * scale
    return out^


def rfftfreq[
    T: FloatLike, log2n: Int
](spacing: T) -> Array[T, (1 << log2n) // 2 + 1]:
    """The frequency of each `rfft` output bin. All non-negative, matching
    `numpy.fft.rfftfreq`: `k / (n*spacing)` for `k` in `0 .. n/2`."""
    comptime n = 1 << log2n
    var out = Array[T, n // 2 + 1](fill=T.constant(0.0))
    var scale = T.one() / (T.constant(Float64(n)) * spacing)
    for k in range(n // 2 + 1):
        out[k] = T.constant(Float64(k)) * scale
    return out^


def fftshift[
    T: FloatLike, log2n: Int
](x: Array[Complex[T], 1 << log2n]) -> Array[Complex[T], 1 << log2n]:
    """Rotate a spectrum so the zero frequency sits in the middle, which is
    how a spectrum is usually plotted. The inverse of itself for even `n`,
    which is the only `n` this module has."""
    comptime n = 1 << log2n
    var out = Array[Complex[T], n](fill=Complex[T].constant(0.0))
    for k in range(n):
        out[(k + n // 2) % n] = x[k].copy()
    return out^


def ifftshift[
    T: FloatLike, log2n: Int
](x: Array[Complex[T], 1 << log2n]) -> Array[Complex[T], 1 << log2n]:
    """Undo `fftshift`: move the zero frequency from the middle back to
    index 0.

    For the even `n` this module is limited to, this is the same rotation
    `fftshift` applies, so the two agree elementwise. It exists under its
    own name because `numpy.fft` has both and a caller undoing a shift
    should not have to know that the two coincide here -- the day a
    rectangular or odd-length transform arrives, they stop coinciding and
    this is the one that stays correct.
    """
    comptime n = 1 << log2n
    var out = Array[Complex[T], n](fill=Complex[T].constant(0.0))
    for k in range(n):
        out[k] = x[(k + n // 2) % n].copy()
    return out^


def fft2[
    T: FloatLike, log2n: Int
](x: Array[Complex[T], (1 << log2n) * (1 << log2n)]) -> Array[
    Complex[T], (1 << log2n) * (1 << log2n)
]:
    """The two-dimensional transform of a square `n x n` array, row-major.

    Row-column decomposition: transform every row, then every column. The
    2-D DFT separates exactly, so this is not an approximation -- it is the
    definition, evaluated in the cheaper order (`2n` transforms of length
    `n` rather than one of length `n**2`).

    Square only, and `n` a power of two, because `fft` is. A rectangular
    transform would need two `log2n` parameters and two twiddle tables;
    nothing in `numax` needs one yet.
    """
    comptime n = 1 << log2n
    var out = Array[Complex[T], n * n](fill=Complex[T].constant(0.0))

    var row = Array[Complex[T], n](fill=Complex[T].constant(0.0))
    for i in range(n):
        for j in range(n):
            row[j] = x[i * n + j].copy()
        var transformed = fft[T, log2n](row)
        for j in range(n):
            out[i * n + j] = transformed[j].copy()

    var column = Array[Complex[T], n](fill=Complex[T].constant(0.0))
    for j in range(n):
        for i in range(n):
            column[i] = out[i * n + j].copy()
        var transformed = fft[T, log2n](column)
        for i in range(n):
            out[i * n + j] = transformed[i].copy()

    return out^


def ifft2[
    T: FloatLike, log2n: Int
](x: Array[Complex[T], (1 << log2n) * (1 << log2n)]) -> Array[
    Complex[T], (1 << log2n) * (1 << log2n)
]:
    """The inverse of `fft2`, normalized by `1/n**2`. Row-column like the
    forward transform, using `ifft` on each pass -- so each pass
    contributes its own `1/n` and the two compose to `1/n**2`."""
    comptime n = 1 << log2n
    var out = Array[Complex[T], n * n](fill=Complex[T].constant(0.0))

    var row = Array[Complex[T], n](fill=Complex[T].constant(0.0))
    for i in range(n):
        for j in range(n):
            row[j] = x[i * n + j].copy()
        var transformed = ifft[T, log2n](row)
        for j in range(n):
            out[i * n + j] = transformed[j].copy()

    var column = Array[Complex[T], n](fill=Complex[T].constant(0.0))
    for j in range(n):
        for i in range(n):
            column[i] = out[i * n + j].copy()
        var transformed = ifft[T, log2n](column)
        for i in range(n):
            out[i * n + j] = transformed[i].copy()

    return out^
