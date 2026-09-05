"""Linear convolution, correlation, filtering, and window functions.

**Tier 1.** Everything here has a fixed shape and a fixed amount of work
determined at compile time, no branching per SIMD lane, so it runs inside a
GPU thread like the rest of the `FloatLike` surface.

MAX ships convolution, but not this convolution: `nn.conv` is the neural
network operator -- NHWC/NDHWC layouts, filter packing, batching, stride and
dilation. What is missing is the signal-processing kind, `numpy.convolve`
and `scipy.signal.correlate` over a one-dimensional sequence, which is what
this module is.

## Linear, not circular

`numax.fft.circular_convolve` wraps modulo `n`. These do not: the output of
convolving an `m`-point sequence with a `k`-point one is `m + k - 1` points
long, and every output index is a genuine sum over the overlap. That length
is a compile-time consequence of the input lengths, which is what keeps this
tier 1.

The direct `O(m*k)` sum is what is implemented, deliberately. At the sizes
this module targets -- register-resident sequences inside a per-element
kernel -- it beats going through the frequency domain, which needs both
inputs zero-padded to a power of two at least `m + k - 1` long, two forward
transforms, a pointwise product, and an inverse. `circular_convolve` is
there for callers who are already in the frequency domain for other reasons.

## Correlation is convolution with one argument reversed

`correlate(a, b)` equals `convolve(a, reversed(b))`, and rather than
implement the reversal as a separate pass, `correlate` indexes `b` backwards
in the same loop. The distinction that actually matters to a caller is the
*offset* convention, which is documented per function: `numpy.correlate`'s
`full` mode and `numpy.convolve`'s agree in length but not in what index
zero means.
"""

from std.collections import Array
from std.math import cos as _cos_f64, pi as _PI

from ..core.numeric import FloatLike, guard_nonzero


comptime full = 0
"""`mode` for `convolve`: the full `m + k - 1` convolution. The default."""
comptime same = 1
"""`mode` for `convolve`: the central `m` points, output as long as `a`."""


def _convolve_len[m: Int, k: Int, mode: Int]() -> Int:
    """`convolve`'s output length for `mode`, evaluated at compile time so
    it can appear in the return type."""
    return m + k - 1 if mode == full else m


def convolve[
    T: FloatLike, m: Int, k: Int, mode: Int = full
](a: Array[T, m], b: Array[T, k]) -> Array[
    T, _convolve_len[m, k, mode]()
] where (mode == full or mode == same):
    """Linear convolution of `a` and `b`. `numpy.convolve(a, b, mode)`.

    `out[i] = sum_j a[j] * b[i - j]`, over the `j` where both indices are
    in range.

    | `mode` | Output |
    |---|---|
    | `full` (default) | Length `m + k - 1`, every overlap |
    | `same` | The central `m` points, so the output is as long as `a` |

    Both lengths are compile-time functions of the input lengths, which is
    why `mode` is a parameter rather than an argument -- the return type
    depends on it. `numpy.convolve`'s third mode, `"valid"`, is not
    provided; nothing in numax needs it yet.

    For `same`, the centre offset is `(k - 1) // 2`, matching NumPy for
    both odd and even `k`. For an even-length kernel the centre is
    ambiguous and NumPy picks the earlier of the two; this does the same
    rather than choosing differently and being subtly incompatible.

    The loop bounds are computed from `i` rather than guarded by an `if`
    inside the sum, so no lane-dependent branch appears: `j` runs over
    `max(0, i - k + 1) .. min(i, m - 1)`, which is an ordinary integer
    range on a *loop index*, not on any `T` value.
    """
    comptime n = m + k - 1
    comptime out_n = _convolve_len[m, k, mode]()
    comptime offset = 0 if mode == full else (k - 1) // 2
    var out = Array[T, out_n](fill=T.constant(0.0))
    for o in range(out_n):
        var i = o + offset
        var total = T.constant(0.0)
        var lo = 0 if i < k else i - k + 1
        var hi = i if i < m else m - 1
        for j in range(lo, hi + 1):
            total = total + a[j] * b[i - j]
        out[o] = total^
    _ = n
    return out^


def correlate[
    T: FloatLike, m: Int, k: Int
](a: Array[T, m], b: Array[T, k]) -> Array[T, m + k - 1]:
    """Cross-correlation of `a` and `b`: `numpy.correlate(a, b, "full")`.

    `out[i] = sum_j a[j] * b[j - i + k - 1]`, the sliding inner product of
    `a` against `b` at every overlap. Equivalently `convolve(a,
    reversed(b))`, which is how it is computed -- `b` is simply indexed
    backwards, with no reversal pass.

    Note the offset convention: at `i = k - 1` the two sequences are
    aligned at their starts, so *that* is the zero-lag term, not `i = 0`.
    This matches `numpy.correlate(a, b, "full")` exactly.
    """
    comptime n = m + k - 1
    var out = Array[T, n](fill=T.constant(0.0))
    for i in range(n):
        var total = T.constant(0.0)
        var lo = 0 if i < k else i - k + 1
        var hi = i if i < m else m - 1
        for j in range(lo, hi + 1):
            total = total + a[j] * b[k - 1 - i + j]
        out[i] = total^
    return out^


def _cosine_window[
    T: FloatLike, n: Int, a0: Float64, a1: Float64, a2: Float64
]() -> Array[T, n]:
    """The generalized cosine window `a0 - a1*cos(2πi/(n-1)) +
    a2*cos(4πi/(n-1))`, which Hann, Hamming and Blackman are all special
    cases of.

    The cosines are evaluated in `Float64` at compile time and embedded as
    `T.constant`s. That is not just an optimization: a runtime
    `T.constant(Float64(i))` would need an int64-to-double conversion,
    which Metal rejects (see `numax.special.orthopoly`'s module docstring), so the
    `comptime for` is what keeps windows usable inside a GPU kernel.
    """
    var out = Array[T, n](fill=T.constant(0.0))
    comptime for i in range(n):
        comptime denominator = Float64(n - 1) if n > 1 else 1.0
        comptime theta = 2.0 * _PI * Float64(i) / denominator
        comptime value = a0 - a1 * _cos_f64(theta) + a2 * _cos_f64(2.0 * theta)
        out[i] = T.constant(value)
    return out^


def hann[T: FloatLike, n: Int]() -> Array[T, n]:
    """The Hann (raised-cosine) window, `0.5 - 0.5*cos(2πi/(n-1))`.

    Symmetric (`sym=True` in `scipy.signal.get_window` terms), so the
    first and last samples are exactly zero. The default choice for
    spectral analysis: -31 dB sidelobes falling off at 18 dB/octave.
    """
    return _cosine_window[T, n, 0.5, 0.5, 0.0]()


def hamming[T: FloatLike, n: Int]() -> Array[T, n]:
    """The Hamming window, `0.54 - 0.46*cos(2πi/(n-1))`.

    Same shape as Hann with a pedestal, which trades a slightly wider main
    lobe for a much lower first sidelobe (-41 dB). Uses the classic
    0.54/0.46 coefficients rather than the exactly-optimal 0.53836/0.46164,
    matching `numpy.hamming` and `scipy.signal.hamming`.
    """
    return _cosine_window[T, n, 0.54, 0.46, 0.0]()


def blackman[T: FloatLike, n: Int]() -> Array[T, n]:
    """The Blackman window, `0.42 - 0.5*cos(2πi/(n-1)) +
    0.08*cos(4πi/(n-1))`.

    Three cosine terms instead of two: -58 dB sidelobes, at the cost of a
    main lobe half again as wide as Hann's. The one to reach for when
    leakage from a strong nearby tone is the problem.
    """
    return _cosine_window[T, n, 0.42, 0.5, 0.08]()


def apply_window[
    T: FloatLike, n: Int
](x: Array[T, n], w: Array[T, n]) -> Array[T, n]:
    """Multiply a sequence by a window, elementwise.

    Trivial, and worth a name anyway: windowing before a transform is the
    step most easily forgotten, and `apply_window(x, hann[T, n]())` reads
    as the intent where a bare loop reads as arithmetic.
    """
    var out = Array[T, n](fill=T.constant(0.0))
    for i in range(n):
        out[i] = x[i] * w[i]
    return out^


def lfilter[
    T: FloatLike, n: Int, nb: Int, na: Int
](b: Array[T, nb], a: Array[T, na], x: Array[T, n]) -> Array[T, n]:
    """Apply the difference equation `a[0]*y[i] = sum(b[j]*x[i-j]) -
    sum(a[j]*y[i-j])`. `scipy.signal.lfilter`.

    This is what `convolve` cannot express. A convolution is the `na == 1`
    case -- output depends on the input alone -- while a recursive filter
    feeds its own past output back, which is how an infinite impulse
    response fits in a handful of coefficients. A one-pole lowpass is
    `b = [alpha]`, `a = [1, alpha - 1]`, and no finite convolution kernel
    equals it.

    Still tier 1: the recursion runs exactly `n` times over compile-time
    lengths, and the initial state is zero rather than a data-dependent
    warm-up. Every lane of a SIMD `T` filters its own signal with the same
    coefficients.

    `a[0]` is divided by rather than assumed to be 1, matching SciPy. It is
    floored away from zero, so a caller who passes `a[0] = 0` gets a finite
    answer instead of NaN -- and a wrong one, since that is not a filter.

    `zi`/`zf` initial and final state are not supported, so filtering a
    signal in blocks restarts the recursion at each block boundary. Filter
    the whole sequence in one call, or accept the transient.
    """
    var out = Array[T, n](fill=T.constant(0.0))
    var lead = guard_nonzero(a[0], T.constant(1e-30))

    for i in range(n):
        var total = T.constant(0.0)
        for j in range(nb):
            if j <= i:
                total = total + b[j] * x[i - j]
        for j in range(1, na):
            if j <= i:
                total = total - a[j] * out[i - j]
        out[i] = total / lead
    return out^


def firwin[T: FloatLike, n: Int](cutoff: Float64) -> Array[T, n]:
    """A lowpass FIR filter of `n` taps by the window method.
    `scipy.signal.firwin`, with the default Hamming window.

    `cutoff` is in the same units SciPy uses: a fraction of the Nyquist
    frequency, so `0.25` passes the lower quarter of the representable
    band. The taps are a sinc -- the ideal brick wall's impulse response --
    truncated to `n` points and tapered by a Hamming window, then scaled
    for unit gain at DC so a constant input passes through unchanged.

    An odd `n` puts a tap at the sinc's center and gives a filter with no
    fractional-sample delay, which is why filter designs are usually
    specified that way. An even `n` works and delays by half a sample.

    Windowing is what makes this usable at all: truncating a sinc without
    it leaves Gibbs ringing that no amount of extra taps removes, since the
    overshoot converges to a fixed fraction rather than to zero.

    Only lowpass. Highpass, bandpass and bandstop are the same taps
    combined differently (a spectral inversion, or a difference of two
    lowpass designs), which is a continuation rather than a redesign.
    """
    var window = hamming[T, n]()
    var center = Float64(n - 1) / 2.0

    var taps = Array[T, n](fill=T.constant(0.0))
    var total = T.constant(0.0)
    for i in range(n):
        var offset = cutoff * (Float64(i) - center)
        # sinc(0) is 1, and the floored denominator delivers exactly that:
        # `sin(eps)/eps` rounds to 1 rather than dividing by zero.
        var angle = guard_nonzero(T.constant(_PI * offset), T.constant(1e-30))
        taps[i] = angle.sin() / angle * window[i]
        total = total + taps[i]

    var gain = guard_nonzero(total, T.constant(1e-30))
    for i in range(n):
        taps[i] = taps[i] / gain
    return taps^
