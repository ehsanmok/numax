"""NumPy/SciPy CPU baseline for the interpolation sweep
../bench_interpolate.mojo runs.

Same jittered knots on `[0, 10]`, same sampled functions (`x^2` for the
linear rows, `sin` for the spline rows), same hashed query points, same
`float32`, same columns. `numpy.interp` is what numax's `interp` is named
after; `scipy.interpolate.CubicSpline` with its default not-a-knot ends is
what `CubicSpline` is.

Run: pixi run -e bench-python bench-scipy-interpolate (from the repo root)
"""

import time

import numpy as np
from scipy.interpolate import CubicSpline

DTYPE = np.float32
WARMUP_ITERS = 2
BUDGET_SECS = 1.0
KNOT_SPAN = 10.0


def knots(n: int) -> np.ndarray:
    i = np.arange(n, dtype=np.int64)
    jitter = 0.3 * (((i * 2654435761 + 12345) % 16777216) / 16777216.0 - 0.5)
    t = i.astype(np.float64) + jitter
    t[0] = 0.0
    t[-1] = float(n - 1)
    return (t * KNOT_SPAN / (n - 1)).astype(DTYPE)


def queries(m: int) -> np.ndarray:
    i = np.arange(m, dtype=np.int64)
    h = ((i * 2654435761 + 97) % 16777216) / 16777216.0
    return (0.01 + 0.98 * KNOT_SPAN * h).astype(DTYPE)


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


def row(name: str, n: int, m: int, ns: float, err: float) -> None:
    print(f"{name}\t{n}\t{m}\t{ns / 1e3:.3f}\t{err}")


def bench_interp(n: int, m: int) -> None:
    xp = knots(n)
    fp = (xp.astype(np.float64) ** 2).astype(DTYPE)
    x = queries(m)
    ns = time_call(lambda: np.interp(x, xp, fp))
    y = np.interp(x, xp, fp).astype(np.float64)
    row("interp", n, m, ns, float(np.max(np.abs(y - x.astype(np.float64) ** 2))))


def bench_spline_eval(n: int, m: int) -> None:
    xp = knots(n)
    fp = np.sin(xp.astype(np.float64)).astype(DTYPE)
    spline = CubicSpline(xp, fp)
    x = queries(m)
    ns = time_call(lambda: spline(x))
    y = spline(x).astype(np.float64)
    row("CubicSpline eval", n, m, ns, float(np.max(np.abs(y - np.sin(x.astype(np.float64))))))


def bench_spline_build(n: int) -> None:
    xp = knots(n)
    fp = np.sin(xp.astype(np.float64)).astype(DTYPE)
    row("CubicSpline build", n, 0, time_call(lambda: CubicSpline(xp, fp)), 0.0)


def main() -> None:
    import scipy

    print(f"SciPy {scipy.__version__}  NumPy {np.__version__}  dtype={DTYPE.__name__}  target=cpu")
    print()
    print("Interpolation (us is per call; error is against the sampled function)")
    print("op\tn knots\tm queries\tus\tmax |error|")
    bench_interp(1024, 1 << 20)
    bench_spline_eval(1024, 1 << 20)
    bench_spline_build(1024)
    bench_spline_build(4096)


if __name__ == "__main__":
    main()
