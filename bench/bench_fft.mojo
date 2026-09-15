"""How fast is the `Tensor` tier of `numax.fft` on this CPU, and how many
kernel launches does one transform cost?

`numax.fft`'s engine is Cooley-Tukey in three kinds of launch -- one fused
launch that does the bit-reversal gather and the first six stages in
registers, one radix-4 launch per remaining pair of stages, and a radix-2
launch when an odd stage is left. The `launches` column is that count,
`1 + ceil((log2(n) - 6) / 2)`, computed the same way the engine computes
it; it is the number this file exists to put beside a wall-clock figure,
because on a CPU every launch is a thread-pool dispatch over butterflies
of a few flops each and the dispatch is a real share of the cost.

Two tables:

1. **1-D transforms** -- `fft` (complex in, complex out), `rfft` (real in,
   half spectrum out) and `irfft` (the way back) at `2^10` through `2^20`.
2. **2-D** -- `fft2` at `512 x 512`, which is 512 row transforms and 512
   column transforms through a transposed view of the same buffer.

Every public transform here consumes its input, so a timed call has to
build one; the `input` row at each size is that construction alone -- a
`List` copy and a buffer fill -- and is included in the three rows above
it. Subtract it for the transform's own cost.

Microseconds per call throughout. `float32`, for the reason
`bench_signal.mojo` gives: the GPU sibling of this engine cannot be
anything else, and a CPU table should measure the same computation. The
data is a sine with a deterministic hash on top, the same generator
`bench_signal.mojo` uses, so the two files' numbers are about the same
signal. `bench/scipy/fft_transforms.py` prints the same columns from
`scipy.fft`.

CPU only. Run with `pixi run bench-fft`.
"""

from max.gpu.host import DeviceContext
from std.benchmark import keep, run
from std.math import sin

from numax.core.array import Static
from numax.fft import fft, fft2, irfft, rfft
from numax.fft.fft import _FUSED_STAGES, _log2_exact

comptime dtype = DType.float32
comptime warmup_iters = 2
comptime budget_secs = 1.0


def _launches(n: Int) -> Int:
    """What `_radix2` costs in `elementwise` dispatches at length `n`: the
    fused launch, then one per remaining pair of stages, then one more if
    an odd stage is left."""
    var bits = _log2_exact(n)
    var fused = bits if bits < _FUSED_STAGES else _FUSED_STAGES
    var rest = bits - fused
    return 1 + (rest + 1) // 2


def _signal_entry(i: Int) -> Float64:
    """A sine with a deterministic hash on top, so the data is neither
    smooth enough to be trivial nor different between runs."""
    var h = (i * 2654435761 + 12345) % 16777216
    return sin(Float64(i) * 0.01) + 0.1 * (Float64(h) / 16777216.0 - 0.5)


def _values[n: Int]() -> List[Scalar[dtype]]:
    var values = List[Scalar[dtype]](capacity=n)
    for i in range(n):
        values.append(Scalar[dtype](_signal_entry(i)))
    return values^


def _row(name: String, n: Int, launches: Int, ns: Float64):
    print(name, "\t", n, "\t", launches, "\t", ns / 1e3)


def bench_transforms[n: Int](ctx: DeviceContext) raises where n > 0:
    comptime keep_bins = n // 2 + 1
    var real = _values[n]()
    var imag = List[Scalar[dtype]](length=n, fill=Scalar[dtype](0))
    var spectrum = rfft[dtype, n](Static[dtype, n](ctx, real.copy()))
    var half_re = spectrum[0].to_host()
    var half_im = spectrum[1].to_host()

    def build() raises {mut real, mut imag, var ctx}:
        var x = Static[dtype, n](ctx, real.copy())
        var y = Static[dtype, n](ctx, imag.copy())
        keep(x.buffer.unsafe_ptr())
        keep(y.buffer.unsafe_ptr())

    def forward() raises {mut real, mut imag, var ctx}:
        var out = fft[dtype, n](
            (
                Static[dtype, n](ctx, real.copy()),
                Static[dtype, n](ctx, imag.copy()),
            )
        )
        keep(out[0].buffer.unsafe_ptr())

    def real_forward() raises {mut real, var ctx}:
        var out = rfft[dtype, n](Static[dtype, n](ctx, real.copy()))
        keep(out[0].buffer.unsafe_ptr())

    def real_inverse() raises {mut half_re, mut half_im, var ctx}:
        var out = irfft[dtype, keep_bins, False, n](
            (
                Static[dtype, keep_bins](ctx, half_re.copy()),
                Static[dtype, keep_bins](ctx, half_im.copy()),
            )
        )
        keep(out.buffer.unsafe_ptr())

    var count = _launches(n)
    _row(
        "input",
        n,
        0,
        run(
            build, num_warmup_iters=warmup_iters, max_runtime_secs=budget_secs
        ).mean()
        * 1e9,
    )
    _row(
        "fft",
        n,
        count,
        run(
            forward, num_warmup_iters=warmup_iters, max_runtime_secs=budget_secs
        ).mean()
        * 1e9,
    )
    _row(
        "rfft",
        n,
        count,
        run(
            real_forward,
            num_warmup_iters=warmup_iters,
            max_runtime_secs=budget_secs,
        ).mean()
        * 1e9,
    )
    _row(
        "irfft",
        n,
        count,
        run(
            real_inverse,
            num_warmup_iters=warmup_iters,
            max_runtime_secs=budget_secs,
        ).mean()
        * 1e9,
    )


def bench_fft2[
    rows: Int, cols: Int
](ctx: DeviceContext) raises where rows > 0 and cols > 0:
    var real = _values[rows * cols]()
    var imag = List[Scalar[dtype]](length=rows * cols, fill=Scalar[dtype](0))

    def build() raises {mut real, mut imag, var ctx}:
        var x = Static[dtype, rows, cols](ctx, real.copy())
        var y = Static[dtype, rows, cols](ctx, imag.copy())
        keep(x.buffer.unsafe_ptr())
        keep(y.buffer.unsafe_ptr())

    def plane() raises {mut real, mut imag, var ctx}:
        var out = fft2[dtype, rows, cols](
            (
                Static[dtype, rows, cols](ctx, real.copy()),
                Static[dtype, rows, cols](ctx, imag.copy()),
            )
        )
        keep(out[0].buffer.unsafe_ptr())

    _row(
        "input",
        rows * cols,
        0,
        run(
            build, num_warmup_iters=warmup_iters, max_runtime_secs=budget_secs
        ).mean()
        * 1e9,
    )
    _row(
        "fft2",
        rows * cols,
        _launches(rows) + _launches(cols),
        run(
            plane, num_warmup_iters=warmup_iters, max_runtime_secs=budget_secs
        ).mean()
        * 1e9,
    )


def main() raises:
    var ctx = DeviceContext(api="cpu")
    print("dtype =", dtype, " target = cpu  fused stages =", _FUSED_STAGES)

    print()
    print("1-D transforms (us is per call, input construction included)")
    print("op\tn\tlaunches\tus")
    bench_transforms[1 << 10](ctx)
    bench_transforms[1 << 12](ctx)
    bench_transforms[1 << 14](ctx)
    bench_transforms[1 << 16](ctx)
    bench_transforms[1 << 18](ctx)
    bench_transforms[1 << 20](ctx)

    print()
    print("2-D transform, 512 x 512 (us is per call)")
    print("op\tn\tlaunches\tus")
    bench_fft2[512, 512](ctx)
