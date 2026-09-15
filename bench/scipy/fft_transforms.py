"""SciPy CPU baseline for the transform sweep ../bench_fft.mojo runs.

Same signal (`sin(0.01 i)` plus a hashed jitter), same sizes, same
`float32`, same columns, so the two outputs read side by side.
`scipy.fft.fft`, `rfft`, `irfft` and `fft2` are pocketfft, which has a
radix for 2, 3, 4, 5, 7 and 11 and runs the whole transform in one C call
-- there is no launch count to report on that side, so the column is `-`.

The file is not `fft.py`: the task runs with `bench/scipy` as its working
directory, which puts it first on `sys.path`, and a module there that
shadows one SciPy imports is how every baseline in this directory stops
importing SciPy at all (`signal_processing.py` carries the long version of
that note).

Run: pixi run -e bench-python bench-scipy-fft (from the repo root)
"""

import time

import numpy as np
from scipy import fft as sfft

DTYPE = np.float32
CDTYPE = np.complex64
WARMUP_ITERS = 2
BUDGET_SECS = 1.0


def hashed(n: int, salt: int = 12345) -> np.ndarray:
    i = np.arange(n, dtype=np.int64)
    return ((i * 2654435761 + salt) % 16777216) / 16777216.0 - 0.5


def make_signal(n: int) -> np.ndarray:
    i = np.arange(n, dtype=np.float64)
    return (np.sin(i * 0.01) + 0.1 * hashed(n)).astype(DTYPE)


def time_call(fn) -> float:
    """Average nanoseconds per call within a one-second budget, after
    warmup -- the Mojo harness's `run(max_runtime_secs=1)`."""
    for _ in range(WARMUP_ITERS):
        fn()
    count = 0
    t0 = time.perf_counter()
    while True:
        fn()
        count += 1
        elapsed = time.perf_counter() - t0
        if elapsed >= BUDGET_SECS and count >= 3:
            break
    return elapsed * 1e9 / count


def row(name: str, n: int, launches: str, ns: float) -> None:
    print(f"{name}\t{n}\t{launches}\t{ns / 1e3:.3f}")


def bench_transforms(n: int) -> None:
    real = make_signal(n)
    complex_in = real.astype(CDTYPE)
    half = sfft.rfft(real)
    row("fft", n, "-", time_call(lambda: sfft.fft(complex_in)))
    row("rfft", n, "-", time_call(lambda: sfft.rfft(real)))
    row("irfft", n, "-", time_call(lambda: sfft.irfft(half, n)))


def bench_fft2(rows: int, cols: int) -> None:
    plane = make_signal(rows * cols).reshape(rows, cols).astype(CDTYPE)
    row("fft2", rows * cols, "-", time_call(lambda: sfft.fft2(plane)))


def main() -> None:
    import scipy

    print(
        f"SciPy {scipy.__version__}  NumPy {np.__version__}"
        f"  dtype={DTYPE.__name__}  target=cpu"
    )
    print()
    print("1-D transforms (us is per call)")
    print("op\tn\tlaunches\tus")
    for power in (10, 12, 14, 16, 18, 20):
        bench_transforms(1 << power)

    print()
    print("2-D transform, 512 x 512 (us is per call)")
    print("op\tn\tlaunches\tus")
    bench_fft2(512, 512)


if __name__ == "__main__":
    main()
