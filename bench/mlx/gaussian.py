"""Apple MLX CPU and GPU baseline for the same gaussian(x) = exp(-x^2) sweep
`../bench_tensor_map_gpu.mojo` runs, at the same sizes -- see ../README.md.

MLX is the closest Python-ecosystem match to what numax's own GPU path is
doing: one array library, one API, a `stream=mx.cpu`/`stream=mx.gpu` switch
to pick the backend, running on the same Metal GPU `numax`'s
`bench_tensor_map_gpu.mojo` targets via `DeviceContext`. Unlike NumPy, this
gets a real CPU-vs-GPU crossover comparison, not just a CPU number.

MLX is lazy and asynchronous by default: an op returns immediately with an
unevaluated array, and work only happens (on whichever stream it was issued
to) when a result is actually read. `mx.eval(...)` forces that, and is what
this benchmark times -- without it, the "GPU" timing would just be
measuring how fast MLX can build a graph node, not how fast the GPU runs.

Run: pixi run bench-mlx (from the repo root; Apple Silicon only)
"""

import time

import mlx.core as mx

DTYPE = mx.float32
SIZES = [1 << 16, 1 << 18, 1 << 20, 1 << 22, 1 << 24, 1 << 26]
WARMUP_ITERS = 3
TIMED_ITERS = 10


def gaussian(x: mx.array, stream) -> mx.array:
    return mx.exp(-(x * x), stream=stream)


def bench_stream(
    xs: mx.array, stream, label: str, n: int, amortize_eval: bool = False
) -> mx.array:
    """Time `gaussian` on `stream`, forcing evaluation per call or once total.

    `amortize_eval=False` calls `mx.eval` inside every timed iteration, so
    each one is a full round trip -- per-call latency. `amortize_eval=True`
    builds all `TIMED_ITERS` graph nodes and evaluates them in a single
    `mx.eval`, letting the work pipeline -- steady-state throughput. Both
    shapes are reported because they differ substantially on the GPU stream,
    and because `../bench_tensor_map_gpu.mojo` and `../torch/gaussian.py`
    both report both, so all three are comparable at either shape rather
    than only by accident.

    One caveat specific to MLX, visible in the numbers: the amortized branch
    has to keep all `TIMED_ITERS` outputs alive at once, because MLX is lazy
    and an unreferenced graph node is simply never evaluated -- forcing only
    the last one would time a single kernel, not ten. At 67M elements that is
    ten live 268MB buffers, and the memory pressure makes amortized *slower*
    than per-call there, which is the opposite of what pipelining does for
    `numax` and PyTorch (whose loops rebind one output and free the previous).
    So MLX's per-call column is its representative number at large sizes, and
    its amortized column should not be read as "MLX pipelines badly" -- it is
    a consequence of how this benchmark has to force evaluation.
    """
    for _ in range(WARMUP_ITERS):
        mx.eval(gaussian(xs, stream))

    t0 = time.perf_counter()
    if amortize_eval:
        pending = [gaussian(xs, stream) for _ in range(TIMED_ITERS)]
        mx.eval(pending)
        ys = pending[-1]
    else:
        for _ in range(TIMED_ITERS):
            ys = gaussian(xs, stream)
            mx.eval(ys)
    elapsed_ns = (time.perf_counter() - t0) * 1e9

    avg_ms = elapsed_ns / TIMED_ITERS / 1e6
    elems_per_sec_m = n / (avg_ms / 1e3) / 1e6
    print(f"  {label}={avg_ms:8.4f} ms  ({elems_per_sec_m:9.1f} M elem/s)", end="")
    return ys


def bench_at_size(n: int) -> None:
    xs = mx.arange(n, dtype=DTYPE) * mx.array(0.0001, dtype=DTYPE) - mx.array(
        50.0, dtype=DTYPE
    )
    mx.eval(xs)

    print(f"n={n:>9} ", end="")
    ys_cpu = bench_stream(xs, mx.cpu, "CPU", n)
    ys_gpu = bench_stream(xs, mx.gpu, "GPU per-call", n)
    bench_stream(xs, mx.gpu, "GPU amortized", n, amortize_eval=True)

    ratio = float(mx.mean(mx.abs(ys_cpu.astype(mx.float32) - ys_gpu.astype(mx.float32))))
    print(f"  |CPU-GPU| mean diff={ratio:.2e}")


def main() -> None:
    print(f"MLX  dtype=float32  default device={mx.default_device()}")
    for n in SIZES:
        bench_at_size(n)


if __name__ == "__main__":
    main()
