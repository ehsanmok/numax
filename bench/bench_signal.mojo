"""How fast is the `Tensor` tier of `numax.signal` on this CPU, and where
does `fftconvolve` overtake `convolve`?

Two tables:

1. **Convolution** -- `convolve` (the direct `O(m k)` sum, one
   `elementwise` over the output) against `fftconvolve` (three transforms
   through `numax.fft` at `next_fast_len(m + k - 1)`) for a signal of
   `m = 4096` and `m = 65536` and kernels from 8 to 2048 taps. The
   `numax/signal/convolution.mojo` docstring says the crossover "is a
   measurement rather than a rule" and points here; this is the
   measurement. The error column is `max |direct - fft|`, so the two rows
   at each `(m, k)` are also a check on each other.
2. **Filters and spectra** -- `lfilter` with a 32-tap FIR, `filtfilt`
   with a fourth-order Butterworth, `medfilt` at kernel 5, `savgol_filter`
   at window 11 / order 3, and `welch` at `nperseg = 256`, all at
   `n = 2^20`. `lfilter`, `filtfilt` and `sosfilt` are sequential
   recurrences with no GEMM to feed (the module docstring says why they
   diverge), so this is what a scalar host loop costs against SciPy's
   C one; `medfilt` and `savgol_filter` are `elementwise` and `welch` is
   `numax.fft` on `nperseg`-long segments.

Microseconds per call throughout -- these are linear-work operations, so
no GFLOP/s column; the `bench/scipy/signal.py` baseline prints the same
columns from `scipy.signal`. `float32`, for the reason `bench_linalg.mojo`
gives: the GPU siblings of these kernels cannot be anything else, and a
CPU table should measure the same computation.

CPU only. Run with `pixi run bench-signal`.
"""

from max.gpu.host import DeviceContext
from std.benchmark import keep, run
from std.math import sin

from numax.core.array import Static
from numax.signal import (
    butter,
    convolve,
    fftconvolve,
    filtfilt,
    firwin,
    lfilter,
    medfilt,
    savgol_filter,
    welch,
)

comptime dtype = DType.float32
comptime warmup_iters = 2
comptime budget_secs = 1.0


def _signal_entry(i: Int) -> Float64:
    """A sine with a deterministic hash on top, so the data is neither
    smooth enough to be trivial nor different between runs."""
    var h = (i * 2654435761 + 12345) % 16777216
    return sin(Float64(i) * 0.01) + 0.1 * (Float64(h) / 16777216.0 - 0.5)


def _signal[n: Int](ctx: DeviceContext) raises -> Static[dtype, n]:
    var values = List[Scalar[dtype]](capacity=n)
    for i in range(n):
        values.append(Scalar[dtype](_signal_entry(i)))
    return Static[dtype, n](ctx, values^)


def _kernel[k: Int](ctx: DeviceContext) raises -> Static[dtype, k]:
    """A normalized triangular kernel."""
    var values = List[Scalar[dtype]](capacity=k)
    var total = Float64(0)
    for i in range(k):
        var t = 1.0 - abs(2.0 * Float64(i) / Float64(max(k - 1, 1)) - 1.0)
        total += t + 0.05
    for i in range(k):
        var t = 1.0 - abs(2.0 * Float64(i) / Float64(max(k - 1, 1)) - 1.0)
        values.append(Scalar[dtype]((t + 0.05) / total))
    return Static[dtype, k](ctx, values^)


def _row(name: String, n: Int, k: Int, ns: Float64, err: Float64):
    print(name, "\t", n, "\t", k, "\t", ns / 1e3, "\t", err)


def bench_convolution[
    m: Int, k: Int
](ctx: DeviceContext) raises where m > 0 and k > 0:
    var x = _signal[m](ctx)
    var h = _kernel[k](ctx)

    def direct() raises {mut x, mut h}:
        var y = convolve[dtype, m, k](x, h)
        keep(y.buffer.unsafe_ptr())

    def transform() raises {mut x, mut h}:
        var y = fftconvolve[dtype, m, k](x, h)
        keep(y.buffer.unsafe_ptr())

    var direct_ns = (
        run(
            direct, num_warmup_iters=warmup_iters, max_runtime_secs=budget_secs
        ).mean()
        * 1e9
    )
    var fft_ns = (
        run(
            transform,
            num_warmup_iters=warmup_iters,
            max_runtime_secs=budget_secs,
        ).mean()
        * 1e9
    )

    var a = convolve[dtype, m, k](x, h).to_host()
    var b = fftconvolve[dtype, m, k](x, h).to_host()
    var worst = Float64(0)
    for i in range(m + k - 1):
        var diff = abs(Float64(a[i]) - Float64(b[i]))
        if diff > worst:
            worst = diff
    _row("convolve", m, k, direct_ns, worst)
    _row("fftconvolve", m, k, fft_ns, worst)


def bench_filters[n: Int](ctx: DeviceContext) raises where n > 0 and n >= 256:
    var x = _signal[n](ctx)

    # `lfilter` with a 32-tap lowpass FIR: `a = [1]`.
    comptime taps = 32
    var fir = firwin[dtype, taps]([0.2], ctx=ctx)
    var one = Static[dtype, 1](ctx, [Scalar[dtype](1)])

    def fir_work() raises {mut fir, mut one, mut x}:
        var y = lfilter[dtype, taps, 1, n](fir, one, x)
        keep(y.buffer.unsafe_ptr())

    var fir_ns = (
        run(
            fir_work,
            num_warmup_iters=warmup_iters,
            max_runtime_secs=budget_secs,
        ).mean()
        * 1e9
    )
    _row("lfilter fir32", n, taps, fir_ns, 0.0)

    # `filtfilt` with a fourth-order Butterworth lowpass at 0.1 Nyquist.
    var tf = butter[dtype, 4](0.1, ctx=ctx)

    def iir_work() raises {mut tf, mut x}:
        var y = filtfilt[dtype, 5, 5, n](tf.b, tf.a, x)
        keep(y.buffer.unsafe_ptr())

    var iir_ns = (
        run(
            iir_work,
            num_warmup_iters=warmup_iters,
            max_runtime_secs=budget_secs,
        ).mean()
        * 1e9
    )
    _row("filtfilt butter4", n, 4, iir_ns, 0.0)

    def med_work() raises {mut x}:
        var y = medfilt[dtype, n, 5](x)
        keep(y.buffer.unsafe_ptr())

    var med_ns = (
        run(
            med_work,
            num_warmup_iters=warmup_iters,
            max_runtime_secs=budget_secs,
        ).mean()
        * 1e9
    )
    _row("medfilt", n, 5, med_ns, 0.0)

    def sg_work() raises {mut x}:
        var y = savgol_filter[dtype, n, 11, 3](x)
        keep(y.buffer.unsafe_ptr())

    var sg_ns = (
        run(
            sg_work, num_warmup_iters=warmup_iters, max_runtime_secs=budget_secs
        ).mean()
        * 1e9
    )
    _row("savgol_filter w11 p3", n, 11, sg_ns, 0.0)

    def welch_work() raises {mut x}:
        var p = welch[dtype, n, 256](x)
        keep(p.power.buffer.unsafe_ptr())

    var welch_ns = (
        run(
            welch_work,
            num_warmup_iters=warmup_iters,
            max_runtime_secs=budget_secs,
        ).mean()
        * 1e9
    )
    _row("welch nperseg=256", n, 256, welch_ns, 0.0)


def main() raises:
    var ctx = DeviceContext(api="cpu")
    print("dtype =", dtype, " target = cpu")

    print()
    print("Convolution (us is per call; error is max |direct - fft|)")
    print("op\tm\tk\tus\tmax |diff|")
    bench_convolution[4096, 8](ctx)
    bench_convolution[4096, 32](ctx)
    bench_convolution[4096, 128](ctx)
    bench_convolution[4096, 512](ctx)
    bench_convolution[4096, 2048](ctx)
    bench_convolution[65536, 8](ctx)
    bench_convolution[65536, 32](ctx)
    bench_convolution[65536, 128](ctx)
    bench_convolution[65536, 512](ctx)
    bench_convolution[65536, 2048](ctx)

    print()
    print("Filters and spectra (us is per call)")
    print("op\tn\tk\tus\t-")
    bench_filters[1 << 20](ctx)
