"""Radix-2 Cooley-Tukey FFT over `Complex[Inner]`, at a compile-time size.

The transform is `X[k] = sum_j x[j] * exp(-2*pi*i*j*k/n)` -- the standard
sign convention, matching NumPy's `fft` and SciPy's.

Size is a compile-time parameter, and it's given as `log2n` rather than `n`
so that "must be a power of two" is structural instead of a constraint the
compiler would have to check (`numax.quadrature.simpson` takes a panel
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

from .complex import Complex
from .numeric import FloatLike

comptime _TWO_PI = 6.283185307179586


def _reverse_bits(index: Int, bits: Int) -> Int:
    """`index` with its low `bits` bits reversed.

    A scalar `Int` computation on a loop counter, identical in every SIMD
    lane and every GPU thread, so the data-independent-control-flow rule is
    satisfied -- this is the same kind of index arithmetic as
    `numax.legendre`'s recurrence bound, not a per-lane branch.
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
    outright (see `numax.orthopoly`'s module docstring). This is also why
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
                values[base + j + half] = top + (-bottom)


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
