"""SciPy/LAPACK CPU baseline for the linalg sweep ../bench_linalg.mojo runs.

Same matrices, same sizes, same dtype, same flop counts, same residual
definitions -- the two files are meant to be read side by side, so anything
that would change a number is spelled the same way in both:

- `float32`, because the GPU half of this comparison cannot be anything
  else (`linalg.matmul` does not compile for GPU at `float64`, since its
  GEMV path reduces through `warp.shuffle`), and a `float64` CPU row
  beside a `float32` GPU row
  would be two different computations.
- The SPD matrix is `1/(1 + |i - j|)` off the diagonal and `n` on it; the
  general matrix is `((37i + 11j) mod 17) / 16 - 0.5` off the diagonal and
  `n` on it. Both diagonally dominant, so neither factorization is
  measuring a conditioning problem.
- Iteration counts scale with the cubic work, floored at three, exactly as
  the Mojo harness does.

What this baseline *is* is reference LAPACK: `cholesky` is `potrf`,
`lu_factor` is `getrf`, `solve` is `gesv`, `qr` is `geqrf` plus `orgqr`,
and the BLAS-1 calls are `sdot`/`snrm2`/`sasum`/`saxpy` from
`scipy.linalg.blas` rather than NumPy ufuncs, so the comparison is against
the routine numax's overload is named after. Which BLAS is underneath
depends on the wheel -- printed in the header, since it decides the whole
table.

Run: pixi run -e bench-python bench-scipy-linalg (from the repo root)
"""

import time

import numpy as np
import scipy.linalg as sla
from scipy.linalg import blas

DTYPE = np.float32
FACTOR_SIZES = [128, 256, 512, 1024]
BLAS_SIZES = [1 << 16, 1 << 20, 1 << 24, 1 << 26]
WARMUP_ITERS = 2


def iters(work: int) -> int:
    """Iteration count scaled to keep each measurement around a tenth of a
    second, floored at three -- the Mojo harness's `_iters`."""
    return max(3, min(200, 400_000_000 // work))


def spd(n: int) -> np.ndarray:
    i = np.arange(n).reshape(-1, 1)
    j = np.arange(n).reshape(1, -1)
    a = 1.0 / (1.0 + np.abs(i - j))
    np.fill_diagonal(a, float(n))
    return np.ascontiguousarray(a, dtype=DTYPE)


def general(n: int) -> np.ndarray:
    i = np.arange(n).reshape(-1, 1)
    j = np.arange(n).reshape(1, -1)
    a = ((i * 37 + j * 11) % 17) * 0.0625 - 0.5
    np.fill_diagonal(a, float(n))
    return np.ascontiguousarray(a, dtype=DTYPE)


def ramp(n: int, salt: int) -> np.ndarray:
    i = np.arange(n)
    return np.ascontiguousarray(((i * 37 + salt * 11) % 17) - 8.0, dtype=DTYPE)


def row(name: str, n: int, ns: float, flops: float, resid: float) -> None:
    print(f"{name}\t{n}\t{ns / 1e6:.4f}\t{flops / (ns / 1e9) / 1e9:.1f}\t{resid}")


def band_row(name: str, n: int, ns: float, nbytes: int, err: float) -> None:
    print(f"{name}\t{n}\t{ns / 1e3:.3f}\t{nbytes / (ns / 1e9) / 1e9:.2f}\t{err}")


def time_call(fn, count: int) -> float:
    """Average nanoseconds per call over `count` calls, after warmup."""
    for _ in range(WARMUP_ITERS):
        fn()
    t0 = time.perf_counter()
    for _ in range(count):
        fn()
    return (time.perf_counter() - t0) * 1e9 / count


def bench_gemm(n: int) -> None:
    a = general(n)
    b = general(n) + DTYPE(1.0)
    ns = time_call(lambda: a @ b, iters(2 * n**3))
    want = a.astype(np.float64) @ b.astype(np.float64)
    resid = float(np.max(np.abs((a @ b).astype(np.float64) - want)))
    row("matmul (ceiling)", n, ns, 2.0 * n**3, resid)


def bench_cholesky(n: int) -> None:
    a = spd(n)
    ns = time_call(lambda: sla.cholesky(a, lower=True), iters(n**3 // 3))
    lower = sla.cholesky(a, lower=True)
    resid = float(
        np.max(np.abs(lower.astype(np.float64) @ lower.T.astype(np.float64) - a))
    )
    row("cholesky", n, ns, n**3 / 3.0, resid)


def bench_lu(n: int) -> None:
    a = general(n)
    ns = time_call(lambda: sla.lu_factor(a), iters(2 * n**3 // 3))
    lu, piv = sla.lu_factor(a)
    b = ramp(n, 3)
    x = sla.lu_solve((lu, piv), b)
    resid = float(np.max(np.abs(a.astype(np.float64) @ x - b)))
    row("lu_factor", n, ns, 2.0 * n**3 / 3.0, resid)


def bench_solve(n: int) -> None:
    a = general(n)
    b = ramp(n, 5)
    ns = time_call(lambda: sla.solve(a, b), iters(2 * n**3 // 3))
    x = sla.solve(a, b)
    resid = float(np.max(np.abs(a.astype(np.float64) @ x - b)))
    row("solve", n, ns, 2.0 * n**3 / 3.0 + 2.0 * n**2, resid)


def bench_qr(n: int) -> None:
    a = general(n)
    # `mode="economic"` matches numax's `TensorQR`: `Q` is `m x n`, not
    # `m x m`, and for a square matrix the two agree anyway.
    ns = time_call(lambda: sla.qr(a, mode="economic"), iters(4 * n**3 // 3))
    q, r = sla.qr(a, mode="economic")
    resid = float(np.max(np.abs(q.astype(np.float64) @ r.astype(np.float64) - a)))
    row("qr", n, ns, 2.0 * n**2 * (n - n / 3.0), resid)


def bench_blas1(n: int) -> None:
    x = ramp(n, 1)
    y = ramp(n, 2)
    count = iters(n)

    dot_ns = time_call(lambda: blas.sdot(x, y), count)
    nrm2_ns = time_call(lambda: blas.snrm2(x), count)
    asum_ns = time_call(lambda: blas.sasum(x), count)
    axpy_ns = time_call(lambda: blas.saxpy(x, y, a=2.5), count)

    xd = x.astype(np.float64)
    yd = y.astype(np.float64)
    want_dot = float(np.dot(xd, yd))
    want_dot_abs = float(np.sum(np.abs(xd * yd)))
    want_sq = float(np.dot(xd, xd))
    want_nrm2 = want_sq**0.5
    want_asum = float(np.sum(np.abs(xd)))

    # Error relative to `sum |x_i y_i|`, the quantity that bounds a
    # floating-point sum -- the same normalization the Mojo harness uses.
    band_row(
        "dot",
        n,
        dot_ns,
        2 * n * 4,
        abs(float(blas.sdot(x, y)) - want_dot) / want_dot_abs,
    )
    band_row(
        "nrm2", n, nrm2_ns, n * 4, abs(float(blas.snrm2(x)) - want_nrm2) / want_nrm2
    )
    band_row(
        "asum", n, asum_ns, n * 4, abs(float(blas.sasum(x)) - want_asum) / want_asum
    )
    band_row(
        "axpy",
        n,
        axpy_ns,
        3 * n * 4,
        float(np.max(np.abs(blas.saxpy(x, y, a=2.5).astype(np.float64) - (2.5 * xd + yd)))),
    )


def main() -> None:
    import scipy

    config = np.__config__.show(mode="dicts")
    blas_name = config.get("Build Dependencies", {}).get("blas", {}).get("name", "?")
    print(
        f"SciPy {scipy.__version__}  NumPy {np.__version__}  "
        f"BLAS={blas_name}  dtype={DTYPE.__name__}  target=cpu"
    )

    print()
    print("Factorizations (ms is per call, GFLOP/s from the LAPACK count)")
    print("op\tn\tms\tGFLOP/s\tmax |residual|")
    for n in FACTOR_SIZES:
        bench_gemm(n)
    for n in FACTOR_SIZES:
        bench_cholesky(n)
    for n in FACTOR_SIZES:
        bench_lu(n)
    for n in FACTOR_SIZES:
        bench_solve(n)
    for n in FACTOR_SIZES:
        bench_qr(n)

    print()
    print("BLAS-1 (us is per call, GB/s over the traffic the op must move)")
    print("op\tn\tus\tGB/s\terror")
    for n in BLAS_SIZES:
        bench_blas1(n)


if __name__ == "__main__":
    main()
