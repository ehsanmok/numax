"""`scipy.signal` over a `Tensor`: design a filter, run it, look at the spectrum.

One signal carries the whole file: two tones at 50 Hz and 220 Hz sampled
at 1 kHz, with a slow drift and some deterministic jitter on top. Each
step below does one thing to it and prints the evidence that it worked.

**Three shapes of work, and the library treats them differently**, which
is the thing worth taking away:

- `convolve`, `medfilt`, `savgol_filter` and the windows are one
  `elementwise` launch over the output. Every lane is independent, so
  they are the parts that go to a device unchanged.
- `fftconvolve`, `welch` and the spectral estimators are `numax.fft`
  underneath: a batch of transforms, device-resident across
  `log2(n) + 1` stages. Whether the transform beats the direct sum
  depends on the kernel length, and `docs/performance.md` measures the
  crossover on this machine rather than guessing at it.
- `lfilter` and `filtfilt` are recurrences. Sample `k` needs sample
  `k - 1`, so there is neither a GEMM to send the work to nor
  independent lanes to spread it over; they run on the host in
  `float64` and `numax/signal/filters.mojo` says so at the top. That is
  a property of the algorithm, not a gap in the port.

IIR design used to be out of scope here -- it needs complex poles, and a
`Tensor` is monomorphic in a `DType`. `butter` exists because the design
math is a few dozen host `Float64` pairs and the answer is real, which
`docs/parity.md` records as a reversal rather than a quiet addition.

Run: `pixi run example-signal-processing`
"""

from std.math import sin

from max.gpu.host import DeviceContext

from numax.core.array import Static
from numax.signal import (
    butter,
    convolve,
    fftconvolve,
    filtfilt,
    find_peaks,
    firwin,
    freqz,
    get_window,
    lfilter,
    medfilt,
    savgol_filter,
    valid,
    welch,
)

comptime dtype = DType.float64
comptime n = 1024
comptime fs = 1000.0
comptime taps = 33


def two_tones(ctx: DeviceContext) raises -> Static[dtype, n]:
    """50 Hz plus a third of a 220 Hz, on a slow drift, with a hashed
    wobble so nothing here is exactly band-limited."""
    var values = List[Scalar[dtype]](capacity=n)
    for i in range(n):
        var t = Float64(i) / fs
        var h = (i * 2654435761 + 12345) % 16777216
        var jitter = 0.05 * (Float64(h) / 16777216.0 - 0.5)
        values.append(
            Scalar[dtype](
                sin(2.0 * 3.141592653589793 * 50.0 * t)
                + 0.33 * sin(2.0 * 3.141592653589793 * 220.0 * t)
                + 0.4 * t
                + jitter
            )
        )
    return Static[dtype, n](ctx, values^)


def main() raises:
    var ctx = DeviceContext(api="cpu")

    print("Windows")
    print("-------")

    # SciPy's symmetric and periodic forms both, `fftbins=True` being the
    # periodic one a transform wants.
    var window = get_window[dtype, 8]("hann", fftbins=True, ctx=ctx)
    var w = window.to_host()
    print("  get_window('hann', 8)")
    print("   ", w[0], w[1], w[2], w[3], w[4], w[5], w[6], w[7])

    print()
    print("FIR design and two ways to apply it")
    print("-----------------------------------")

    # firwin(numtaps, cutoff) -- the cutoff is in units of Nyquist, so
    # 0.2 is 100 Hz here: the 50 Hz tone passes, the 220 Hz does not.
    var lowpass = firwin[dtype, taps]([0.2], ctx=ctx)
    var coefficients = lowpass.to_host()
    print("  firwin(33, 0.2) centre tap =", coefficients[taps // 2])

    # Route one: the direct convolution, one launch of dot products.
    var signal_a = two_tones(ctx)
    var kernel_a = firwin[dtype, taps]([0.2], ctx=ctx)
    var direct = convolve[dtype, n, taps, mode=valid](signal_a, kernel_a)

    # Route two: the same answer through three transforms. Which one is
    # faster is a measurement, not a rule -- `bench_signal.mojo` runs the
    # sweep and `docs/performance.md` reports where they cross.
    var signal_b = two_tones(ctx)
    var kernel_b = firwin[dtype, taps]([0.2], ctx=ctx)
    var transformed = fftconvolve[dtype, n, taps, mode=valid](
        signal_b, kernel_b
    )

    var d = direct.to_host()
    var t = transformed.to_host()
    var worst = Float64(0)
    for i in range(n - taps + 1):
        var diff = abs(Float64(d[i]) - Float64(t[i]))
        if diff > worst:
            worst = diff
    print("  max |convolve - fftconvolve| =", worst)
    print("    the same answer; only the cost differs")

    # lfilter is the recurrence form of the same FIR, `a = [1]`.
    var signal_c = two_tones(ctx)
    var kernel_c = firwin[dtype, taps]([0.2], ctx=ctx)
    var unity = Static[dtype, 1](ctx, [Scalar[dtype](1)])
    var filtered = lfilter[dtype, taps, 1, n](kernel_c, unity, signal_c)
    var f = filtered.to_host()
    print("  lfilter(b, [1], x)[512] =", f[512])

    print()
    print("IIR design, and the zero-phase pass")
    print("-----------------------------------")

    # butter(4, 0.1) -- fourth order, cutoff at 50 Hz.
    var design = butter[dtype, 4](0.1, ctx=ctx)
    var b_host = design.b.to_host()
    var a_host = design.a.to_host()
    print("  butter(4, 0.1)")
    print("    b0 =", b_host[0], " a1 =", a_host[1])

    # freqz reports the response those coefficients actually have.
    var b_for_response = Static[dtype, 5](ctx, design.b.to_host())
    var a_for_response = Static[dtype, 5](ctx, design.a.to_host())
    var response = freqz[dtype, 5, 5, 8](b_for_response, a_for_response)
    # `FrequencyResponse` carries the complex `H` as two real tensors, the
    # answer a `dtype`-monomorphic tensor forces and the one `eigvals` and
    # `numax.fft` give to the same constraint.
    var re = response.real.to_host()
    var im = response.imag.to_host()
    var at_dc = (Float64(re[0]) ** 2 + Float64(im[0]) ** 2) ** 0.5
    var at_top = (Float64(re[7]) ** 2 + Float64(im[7]) ** 2) ** 0.5
    print("    |H| at DC =", at_dc, " near Nyquist =", at_top)

    # filtfilt runs the filter forwards and backwards, which squares the
    # magnitude response and cancels the phase entirely -- the reason to
    # reach for it over lfilter when the data is not causal.
    var signal_d = two_tones(ctx)
    var b_again = Static[dtype, 5](ctx, design.b.to_host())
    var a_again = Static[dtype, 5](ctx, design.a.to_host())
    var zero_phase = filtfilt[dtype, 5, 5, n](b_again, a_again, signal_d)
    var z = zero_phase.to_host()
    print("  filtfilt(b, a, x)[512] =", z[512])

    print()
    print("Nonlinear and smoothing filters")
    print("-------------------------------")

    var signal_e = two_tones(ctx)
    var median_filtered = medfilt[dtype, n, 5](signal_e)
    print("  medfilt(x, 5)[512]        =", median_filtered.to_host()[512])

    var signal_f = two_tones(ctx)
    var smoothed = savgol_filter[dtype, n, 11, 3](signal_f)
    print("  savgol_filter(x, 11, 3)[512] =", smoothed.to_host()[512])
    print("    a least-squares cubic per window, one launch over the lot")

    print()
    print("Spectrum")
    print("--------")

    # Welch's method: frame the signal, window each frame, transform, and
    # average the periodograms. 1024 samples at nperseg=256 is seven
    # half-overlapping frames, transformed as one batch.
    var signal_g = two_tones(ctx)
    var estimate = welch[dtype, n, 256](signal_g, fs=fs)
    var power = estimate.power.to_host()
    var frequencies = estimate.frequencies.to_host()
    var loudest = 0
    for i in range(129):
        if Float64(power[i]) > Float64(power[loudest]):
            loudest = i
    print("  welch(x, nperseg=256) peak at", frequencies[loudest], "Hz")
    print("    the 50 Hz tone, which is the loud one")

    # Both tones should show up as peaks in that spectrum. `find_peaks`
    # returns the bin indices, and `welch` hands back the frequency axis
    # they index into.
    var power_copy = Static[dtype, 129](ctx, estimate.power.to_host())
    var located = find_peaks(power_copy, height=0.001)
    print("  find_peaks(power, height=0.001) at:")
    for i in range(len(located)):
        print("    ", frequencies[located[i]], "Hz")
