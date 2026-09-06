"""CuPy baseline for the same gaussian(x) = exp(-x^2) sweep the other
benchmarks in ../ run, at the same sizes -- see ../README.md.

CuPy is the NVIDIA-only member of this set: it has no CPU backend, so unlike
../torch/gaussian.py there are no CPU columns here, and the file skips
itself with a printed note where there is no CUDA device.

Three variants run, because "CuPy" is not one number:

- **eager**: `cupy.exp(-(x * x))` as written, which dispatches three
  elementwise kernels and materializes two temporaries.
- **fused**: the same expression under `cupy.fuse()`, which compiles it into
  a single kernel. This is CuPy's answer to `torch.compile` for elementwise
  work, and the honest comparison against a `numax` `step`, which is one
  fused kernel by construction (composition happens inside `step`, before
  any tensor walk).
- **kernel**: a hand-written `cupy.ElementwiseKernel`, the ceiling for this
  expression in CuPy. It separates CuPy's dispatch overhead from what the
  device can actually do, the way ../bench_gpu_roofline.mojo does for
  `numax`.

Both synchronize placements are reported, matching ../torch/gaussian.py:
**per-call** synchronizes inside every timed iteration (latency for one
call), **amortized** synchronizes once after all of them (steady-state
throughput). The two differ by more than 2x at small sizes, so a table
mixing one library's per-call number with another's amortized number is
measuring the methodology rather than the libraries.

Run: pixi run -e bench-python bench-cupy (from the repo root)
"""

import time

import numpy as np

try:
    import cupy
except ImportError as exc:  # pragma: no cover - environment-dependent
    raise SystemExit(f"cupy not importable: {exc}")

SIZES = [1 << 16, 1 << 18, 1 << 20, 1 << 22, 1 << 24, 1 << 26]
WARMUP_ITERS = 3
TIMED_ITERS = 10


def gaussian(x):
    return cupy.exp(-(x * x))


gaussian_fused = cupy.fuse()(gaussian)

gaussian_kernel = cupy.ElementwiseKernel(
    "float32 x", "float32 y", "y = expf(-(x * x))", "gaussian_kernel"
)


def make_input(n: int):
    return cupy.arange(n, dtype=cupy.float32) * 0.0001 - 50.0


def bench_fn(fn, xs, label: str, n: int, amortize_sync: bool = True):
    for _ in range(WARMUP_ITERS):
        fn(xs)
    cupy.cuda.Device().synchronize()

    t0 = time.perf_counter()
    if amortize_sync:
        for _ in range(TIMED_ITERS):
            ys = fn(xs)
        cupy.cuda.Device().synchronize()
    else:
        for _ in range(TIMED_ITERS):
            ys = fn(xs)
            cupy.cuda.Device().synchronize()
    elapsed_ns = (time.perf_counter() - t0) * 1e9

    avg_ms = elapsed_ns / TIMED_ITERS / 1e6
    elems_per_sec_m = n / (avg_ms / 1e3) / 1e6
    print(f"  {label}={avg_ms:8.4f} ms ({elems_per_sec_m:8.1f} M elem/s)", end="")
    return ys


def check(xs, ys) -> None:
    reference = np.exp(-(cupy.asnumpy(xs).astype(np.float64) ** 2))
    max_err = float(np.abs(cupy.asnumpy(ys).astype(np.float64) - reference).max())
    if max_err > 1e-5:
        print(f"  WARNING: max |cupy - f64 reference| = {max_err}")


def bench_at_size(n: int) -> None:
    xs = make_input(n)

    print(f"n={n:>9} ", end="")
    bench_fn(gaussian, xs, "eager-amort", n)
    bench_fn(gaussian_fused, xs, "fused-amort", n)
    ys = bench_fn(gaussian_kernel, xs, "kernel-amort", n)
    print()

    print(f"{'':>11}", end="")
    bench_fn(gaussian, xs, "eager-percall", n, amortize_sync=False)
    bench_fn(gaussian_fused, xs, "fused-percall", n, amortize_sync=False)
    bench_fn(gaussian_kernel, xs, "kernel-percall", n, amortize_sync=False)
    print()

    check(xs, ys)


def main() -> None:
    if cupy.cuda.runtime.getDeviceCount() == 0:
        print("no CUDA device visible to cupy -- skipping")
        return
    name = cupy.cuda.runtime.getDeviceProperties(0)["name"].decode()
    print(f"CuPy {cupy.__version__}  dtype=float32  GPU={name} (cuda)")

    for n in SIZES:
        bench_at_size(n)


if __name__ == "__main__":
    main()
