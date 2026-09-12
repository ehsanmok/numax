"""SciPy CPU baseline for the signal sweep ../bench_signal.mojo runs.

Same signal (`sin(0.01 i)` plus a hashed jitter), same normalized
triangular kernels, same sizes, same `float32`, same columns, so the two
outputs read side by side. `scipy.signal.convolve(method="direct")` and
`fftconvolve` are the two convolution routes numax's `convolve` and
`fftconvolve` are named after; `lfilter`, `filtfilt`, `medfilt`,
`savgol_filter` and `welch` are the C implementations behind the same
names.

The file is not `signal.py`: the task runs with `bench/scipy` as its
working directory, which puts it first on `sys.path`, and a `signal.py`
there shadows the standard library's `signal` inside `subprocess`, which
SciPy imports before anything else -- every baseline in this directory
then fails to import SciPy at all.

Run: pixi run -e bench-python bench-scipy-signal (from the repo root)
"""

import time

import numpy as np
from scipy import signal

DTYPE = np.float32
WARMUP_ITERS = 2
BUDGET_SECS = 1.0


def hashed(n: int, salt: int = 12345) -> np.ndarray:
    i = np.arange(n, dtype=np.int64)
    return ((i * 2654435761 + salt) % 16777216) / 16777216.0 - 0.5


def make_signal(n: int) -> np.ndarray:
    i = np.arange(n, dtype=np.float64)
    return (np.sin(i * 0.01) + 0.1 * hashed(n)).astype(DTYPE)


def kernel(k: int) -> np.ndarray:
    i = np.arange(k, dtype=np.float64)
    t = 1.0 - np.abs(2.0 * i / max(k - 1, 1) - 1.0) + 0.05
    return (t / t.sum()).astype(DTYPE)


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


def row(name: str, n: int, k: int, ns: float, err: float) -> None:
    print(f"{name}\t{n}\t{k}\t{ns / 1e3:.3f}\t{err}")


def bench_convolution(m: int, k: int) -> None:
    x = make_signal(m)
    h = kernel(k)
    direct_ns = time_call(lambda: signal.convolve(x, h, mode="full", method="direct"))
    fft_ns = time_call(lambda: signal.fftconvolve(x, h, mode="full"))
    a = signal.convolve(x, h, mode="full", method="direct").astype(np.float64)
    b = signal.fftconvolve(x, h, mode="full").astype(np.float64)
    worst = float(np.max(np.abs(a - b)))
    row("convolve", m, k, direct_ns, worst)
    row("fftconvolve", m, k, fft_ns, worst)


def bench_filters(n: int) -> None:
    x = make_signal(n)
    fir = signal.firwin(32, 0.2).astype(DTYPE)
    one = np.ones(1, dtype=DTYPE)
    row("lfilter fir32", n, 32, time_call(lambda: signal.lfilter(fir, one, x)), 0.0)
    b, a = signal.butter(4, 0.1)
    b = b.astype(DTYPE)
    a = a.astype(DTYPE)
    row("filtfilt butter4", n, 4, time_call(lambda: signal.filtfilt(b, a, x)), 0.0)
    row("medfilt", n, 5, time_call(lambda: signal.medfilt(x, 5)), 0.0)
    row(
        "savgol_filter w11 p3",
        n,
        11,
        time_call(lambda: signal.savgol_filter(x, 11, 3)),
        0.0,
    )
    row(
        "welch nperseg=256",
        n,
        256,
        time_call(lambda: signal.welch(x, nperseg=256)),
        0.0,
    )


def main() -> None:
    import scipy

    print(f"SciPy {scipy.__version__}  NumPy {np.__version__}  dtype={DTYPE.__name__}  target=cpu")
    print()
    print("Convolution (us is per call; error is max |direct - fft|)")
    print("op\tm\tk\tus\tmax |diff|")
    for m in (4096, 65536):
        for k in (8, 32, 128, 512, 2048):
            bench_convolution(m, k)
    print()
    print("Filters and spectra (us is per call)")
    print("op\tn\tk\tus\t-")
    bench_filters(1 << 20)


if __name__ == "__main__":
    main()
