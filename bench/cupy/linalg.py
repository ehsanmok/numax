"""CuPy CUDA baseline for the linalg sweep ../bench_linalg_gpu.mojo runs.

Same matrices, sizes, dtype, flop counts and residual definitions as the
Mojo GPU harness, ../scipy/linalg.py and ../torch/linalg.py -- see the
SciPy file for the list.

CuPy is the closest baseline in *shape* to what numax does: a thin typed
array over cuSOLVER/cuBLAS, no framework around it. Two notes:

- The synchronize is inside the timed loop
  (`cupy.cuda.runtime.deviceSynchronize()`), so every row is one full
  launch-through-completion round trip, as in the Mojo GPU harness.
- `cupyx.scipy.linalg.lu_factor` is the `getrf` equivalent. `cupy.linalg`
  itself has no `lu_factor`, which is why this file imports from two
  places.

Run: pixi run -e bench-python bench-cupy-linalg (from the repo root)
"""

import ctypes
import glob
import os
import time


def _preload_cuda12_math_libs() -> None:
    """Put the pip-installed CUDA 12 math libraries on the loader's path
    before CuPy asks for them.

    `cupy-cuda12x` carries only its own kernels and dlopens
    `libcublas.so.12`/`libcusolver.so.11` by soname, expecting a system
    CUDA 12 toolkit. This box's system CUDA is 13, so the import dies with
    `libcublas.so.12: cannot open shared object file` even though the
    matching wheels (`nvidia-cublas-cu12` and friends, pinned in
    `pixi.toml`) are installed -- they live under
    `site-packages/nvidia/*/lib`, which is not a loader directory. PyTorch
    solves this by preloading them itself; CuPy does not, so this does it
    for CuPy. `nvjitlink` comes first because `cusparse` links against it.

    Harmless when the libraries are absent or already resolvable: the loads
    are individually guarded and the whole thing is a no-op without the
    `nvidia` namespace package.
    """
    try:
        import nvidia
    except ImportError:
        return

    # A namespace package has no `__file__`, so take its search paths.
    roots = list(getattr(nvidia, "__path__", []))
    for root in roots:
        _load_from(root)


def _load_from(root: str) -> None:
    for package in ("nvjitlink", "cublas", "cusparse", "cusolver"):
        for so in sorted(
            glob.glob(os.path.join(root, package, "lib", "*.so.1*"))
        ):
            try:
                ctypes.CDLL(so, mode=ctypes.RTLD_GLOBAL)
            except OSError:
                pass


_preload_cuda12_math_libs()

import cupy as cp  # noqa: E402
import cupyx.scipy.linalg as cpx_linalg  # noqa: E402

DTYPE = cp.float32
FACTOR_SIZES = [128, 256, 512, 1024]
BLAS_SIZES = [1 << 16, 1 << 20, 1 << 24, 1 << 26]
WARMUP_ITERS = 2


def sync() -> None:
    cp.cuda.runtime.deviceSynchronize()


def iters(work: int) -> int:
    return max(3, min(200, 400_000_000 // work))


def spd(n: int) -> cp.ndarray:
    i = cp.arange(n).reshape(-1, 1)
    j = cp.arange(n).reshape(1, -1)
    a = (1.0 / (1.0 + cp.abs(i - j))).astype(DTYPE)
    cp.fill_diagonal(a, DTYPE(n))
    return cp.ascontiguousarray(a)


def general(n: int) -> cp.ndarray:
    i = cp.arange(n).reshape(-1, 1)
    j = cp.arange(n).reshape(1, -1)
    a = (((i * 37 + j * 11) % 17) * 0.0625 - 0.5).astype(DTYPE)
    cp.fill_diagonal(a, DTYPE(n))
    return cp.ascontiguousarray(a)


def ramp(n: int, salt: int) -> cp.ndarray:
    i = cp.arange(n)
    return cp.ascontiguousarray((((i * 37 + salt * 11) % 17) - 8.0).astype(DTYPE))


def row(name: str, n: int, ns: float, flops: float, resid: float) -> None:
    print(f"{name}\t{n}\t{ns / 1e6:.4f}\t{flops / (ns / 1e9) / 1e9:.1f}\t{resid}")


def band_row(name: str, n: int, ns: float, nbytes: int, err: float) -> None:
    print(f"{name}\t{n}\t{ns / 1e3:.3f}\t{nbytes / (ns / 1e9) / 1e9:.2f}\t{err}")


def time_call(fn, count: int) -> float:
    for _ in range(WARMUP_ITERS):
        fn()
    sync()
    t0 = time.perf_counter()
    for _ in range(count):
        fn()
        sync()
    return (time.perf_counter() - t0) * 1e9 / count


def max_abs(a: cp.ndarray) -> float:
    return float(cp.abs(a).max())


def bench_gemm(n: int) -> None:
    a = general(n)
    b = general(n) + DTYPE(1.0)
    ns = time_call(lambda: a @ b, iters(2 * n**3))
    want = a.astype(cp.float64) @ b.astype(cp.float64)
    row("matmul (ceiling)", n, ns, 2.0 * n**3, max_abs((a @ b).astype(cp.float64) - want))


def bench_cholesky(n: int) -> None:
    a = spd(n)
    ns = time_call(lambda: cp.linalg.cholesky(a), iters(n**3 // 3))
    lower = cp.linalg.cholesky(a).astype(cp.float64)
    row("cholesky", n, ns, n**3 / 3.0, max_abs(lower @ lower.T - a.astype(cp.float64)))


def bench_lu(n: int) -> None:
    a = general(n)
    ns = time_call(lambda: cpx_linalg.lu_factor(a), iters(2 * n**3 // 3))
    lu, piv = cpx_linalg.lu_factor(a)
    b = ramp(n, 3)
    x = cpx_linalg.lu_solve((lu, piv), b).astype(cp.float64)
    resid = max_abs(a.astype(cp.float64) @ x - b.astype(cp.float64))
    row("lu_factor", n, ns, 2.0 * n**3 / 3.0, resid)


def bench_solve(n: int) -> None:
    a = general(n)
    b = ramp(n, 5)
    ns = time_call(lambda: cp.linalg.solve(a, b), iters(2 * n**3 // 3))
    x = cp.linalg.solve(a, b).astype(cp.float64)
    resid = max_abs(a.astype(cp.float64) @ x - b.astype(cp.float64))
    row("solve", n, ns, 2.0 * n**3 / 3.0 + 2.0 * n**2, resid)


def bench_qr(n: int) -> None:
    a = general(n)
    ns = time_call(lambda: cp.linalg.qr(a, mode="reduced"), iters(4 * n**3 // 3))
    q, r = cp.linalg.qr(a, mode="reduced")
    resid = max_abs(
        q.astype(cp.float64) @ r.astype(cp.float64) - a.astype(cp.float64)
    )
    row("qr", n, ns, 2.0 * n**2 * (n - n / 3.0), resid)


def bench_blas1(n: int) -> None:
    x = ramp(n, 1)
    y = ramp(n, 2)
    count = iters(n)

    dot_ns = time_call(lambda: cp.dot(x, y), count)
    nrm2_ns = time_call(lambda: cp.linalg.norm(x), count)
    asum_ns = time_call(lambda: cp.abs(x).sum(), count)
    axpy_ns = time_call(lambda: DTYPE(2.5) * x + y, count)

    xd = x.astype(cp.float64)
    yd = y.astype(cp.float64)
    want_dot = float(cp.dot(xd, yd))
    want_dot_abs = float(cp.abs(xd * yd).sum())
    want_sq = float(cp.dot(xd, xd))
    want_nrm2 = want_sq**0.5
    want_asum = float(cp.abs(xd).sum())

    got_dot = float(cp.dot(x, y))
    got_nrm2 = float(cp.linalg.norm(x))
    got_asum = float(cp.abs(x).sum())
    got_axpy = (DTYPE(2.5) * x + y).astype(cp.float64)

    band_row("dot", n, dot_ns, 2 * n * 4, abs(got_dot - want_dot) / want_dot_abs)
    band_row("nrm2", n, nrm2_ns, n * 4, abs(got_nrm2 - want_nrm2) / want_nrm2)
    band_row("asum", n, asum_ns, n * 4, abs(got_asum - want_asum) / want_asum)
    band_row("axpy", n, axpy_ns, 3 * n * 4, max_abs(got_axpy - (2.5 * xd + yd)))


def main() -> None:
    props = cp.cuda.runtime.getDeviceProperties(0)
    print(
        f"CuPy {cp.__version__}  device={props['name'].decode()}  "
        f"dtype=float32"
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
