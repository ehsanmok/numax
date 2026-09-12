"""NumPy/SciPy CPU baseline for the statistics sweep ../bench_stats.mojo
runs.

Same hashed data on `[-3, 3)`, same sizes, same `float32`, same columns.
`scipy.stats.norm.cdf`, `numpy.histogram`, `numpy.quantile`, `numpy.cov`
and `numpy.corrcoef` are the routines numax's names come from.

Run: pixi run -e bench-python bench-scipy-stats (from the repo root)
"""

import time

import numpy as np
from scipy import stats

DTYPE = np.float32
WARMUP_ITERS = 2
BUDGET_SECS = 1.0


def entry(i: np.ndarray) -> np.ndarray:
    return 6.0 * (((i * 2654435761 + 12345) % 16777216) / 16777216.0 - 0.5)


def data(n: int) -> np.ndarray:
    return entry(np.arange(n, dtype=np.int64)).astype(DTYPE)


def matrix(rows: int, n: int) -> np.ndarray:
    i = np.arange(n, dtype=np.int64)
    out = np.empty((rows, n), dtype=DTYPE)
    for r in range(rows):
        out[r] = entry(i) + 0.5 * r * entry(i + r)
    return out


def time_call(fn) -> float:
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


def row(name: str, n: int, ns: float, err: float) -> None:
    print(f"{name}\t{n}\t{ns / 1e3:.3f}\t{err}")


def band_row(name: str, n: int, ns: float, nbytes: int, err: float) -> None:
    print(f"{name}\t{n}\t{ns / 1e3:.3f}\t{nbytes / (ns / 1e9) / 1e9:.2f}\t{err}")


def bench_norm_cdf(n: int) -> None:
    x = data(n)
    ns = time_call(lambda: stats.norm.cdf(x))
    p = stats.norm.cdf(x).astype(np.float64)
    want = stats.norm.cdf(x[::997].astype(np.float64))
    band_row("norm.cdf", n, ns, 2 * n * 4, float(np.max(np.abs(p[::997] - want))))


def bench_histogram(n: int) -> None:
    x = data(n)
    ns = time_call(lambda: np.histogram(x, bins=64, range=(-3.0, 3.0)))
    counts, _ = np.histogram(x, bins=64, range=(-3.0, 3.0))
    row("histogram bins=64", n, ns, abs(float(counts.sum()) - n))


def bench_quantile(n: int) -> None:
    x = data(n)
    ns = time_call(lambda: np.quantile(x, 0.5))
    row("quantile q=0.5", n, ns, abs(float(np.quantile(x, 0.5))))


def bench_cov(rows: int, n: int) -> None:
    m = matrix(rows, n)
    cov_ns = time_call(lambda: np.cov(m))
    corr_ns = time_call(lambda: np.corrcoef(m))
    want = np.cov(m.astype(np.float64))[0, 1]
    row(f"cov {rows} x n", n, cov_ns, abs(float(np.cov(m)[0, 1]) - float(want)))
    row(f"corrcoef {rows} x n", n, corr_ns, abs(float(np.corrcoef(m)[0, 0]) - 1.0))


def main() -> None:
    import scipy

    print(f"SciPy {scipy.__version__}  NumPy {np.__version__}  dtype={DTYPE.__name__}  target=cpu")
    print()
    print("Elementwise (us is per call, GB/s over one read and one write)")
    print("op\tn\tus\tGB/s\tmax |error|")
    bench_norm_cdf(1 << 24)
    print()
    print("Counting, selection, correlation (us is per call)")
    print("op\tn\tus\terror")
    bench_histogram(1 << 24)
    bench_quantile(1 << 24)
    bench_cov(8, 1 << 20)


if __name__ == "__main__":
    main()
