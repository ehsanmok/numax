"""NumPy CPU baseline for the same gaussian(x) = exp(-x^2) sweep the other
benchmarks in ../ run, at the same sizes, dtype, input values, and
warmup/timed-iteration counts -- see ../README.md for why those specific
numbers were picked and how to read the combined results.

NumPy has no GPU path of its own (that's what ../mlx/ is for), so this only
reports a CPU number, dispatched through whatever BLAS/vectorized loop
NumPy's C backend uses for ufuncs -- not calling into any of numax's,
thermite's, or torch's SIMD code.

Run: pixi run bench-numpy (from the repo root)
"""

import time

import numpy as np

DTYPE = np.float32
SIZES = [1 << 16, 1 << 18, 1 << 20, 1 << 22, 1 << 24, 1 << 26]
WARMUP_ITERS = 3
TIMED_ITERS = 10


def gaussian(x: np.ndarray) -> np.ndarray:
    return np.exp(-(x * x))


def bench_at_size(n: int) -> None:
    xs = (np.arange(n, dtype=DTYPE) * DTYPE(0.0001) - DTYPE(50.0)).astype(DTYPE)

    for _ in range(WARMUP_ITERS):
        gaussian(xs)

    t0 = time.perf_counter()
    for _ in range(TIMED_ITERS):
        ys = gaussian(xs)
    elapsed_ns = (time.perf_counter() - t0) * 1e9

    avg_ms = elapsed_ns / TIMED_ITERS / 1e6
    elems_per_sec_m = n / (avg_ms / 1e3) / 1e6
    print(f"n={n:>9}  CPU={avg_ms:8.4f} ms  ({elems_per_sec_m:9.1f} M elem/s)")

    reference = np.exp(-(xs.astype(np.float64) ** 2))
    max_err = float(np.max(np.abs(ys.astype(np.float64) - reference)))
    if max_err > 1e-5:
        print(f"  WARNING: max |numpy - f64 reference| = {max_err}")


def main() -> None:
    print(f"NumPy {np.__version__}  dtype={DTYPE.__name__}")
    for n in SIZES:
        bench_at_size(n)


if __name__ == "__main__":
    main()
