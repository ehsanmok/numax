"""PyTorch CUDA baseline for the linalg sweep ../bench_linalg_gpu.mojo runs.

Same matrices, sizes, dtype, flop counts and residual definitions as the
Mojo GPU harness and as ../scipy/linalg.py -- see the SciPy file for the
list, since all three deliberately spell them identically.

Two methodological points, both of which change the numbers:

- **The synchronize is inside the timed loop.** `torch.linalg.cholesky` on
  CUDA returns as soon as the work is queued, so timing without a
  `torch.cuda.synchronize()` per iteration measures dispatch and nothing
  else. numax's GPU harness synchronizes per call for the same reason, so
  both report launch-through-completion latency.
- **TF32 is left at PyTorch's default.** On Ampere that means `matmul` may
  use TF32 tensor cores while the factorizations stay `float32`, which is
  also what MAX's `matmul` does. Printed in the header either way, because
  a TF32 GEMM row is not comparable to an FP32 GEMM row and the flag is
  the only way to tell which one this is.

The factorizations here are cuSOLVER underneath, which is the comparison
that matters: numax's blocked `Tensor` tier against a vendor library, not
against another portable implementation.

Run: pixi run -e bench-python bench-torch-linalg (from the repo root)
"""

import time

import numpy as np
import torch

DTYPE = torch.float32
FACTOR_SIZES = [128, 256, 512, 1024]
BLAS_SIZES = [1 << 16, 1 << 20, 1 << 24, 1 << 26]
WARMUP_ITERS = 2


def iters(work: int) -> int:
    return max(3, min(200, 400_000_000 // work))


def spd(n: int, device: str) -> torch.Tensor:
    i = torch.arange(n, device=device).reshape(-1, 1)
    j = torch.arange(n, device=device).reshape(1, -1)
    a = 1.0 / (1.0 + (i - j).abs().to(DTYPE))
    a.fill_diagonal_(float(n))
    return a.contiguous()


def general(n: int, device: str) -> torch.Tensor:
    i = torch.arange(n, device=device).reshape(-1, 1)
    j = torch.arange(n, device=device).reshape(1, -1)
    a = (((i * 37 + j * 11) % 17).to(DTYPE)) * 0.0625 - 0.5
    a.fill_diagonal_(float(n))
    return a.contiguous()


def ramp(n: int, salt: int, device: str) -> torch.Tensor:
    i = torch.arange(n, device=device)
    return (((i * 37 + salt * 11) % 17).to(DTYPE) - 8.0).contiguous()


def row(name: str, n: int, ns: float, flops: float, resid: float) -> None:
    print(f"{name}\t{n}\t{ns / 1e6:.4f}\t{flops / (ns / 1e9) / 1e9:.1f}\t{resid}")


def band_row(name: str, n: int, ns: float, nbytes: int, err: float) -> None:
    print(f"{name}\t{n}\t{ns / 1e3:.3f}\t{nbytes / (ns / 1e9) / 1e9:.2f}\t{err}")


def time_call(fn, count: int) -> float:
    """Average nanoseconds per call, synchronizing inside every iteration so
    each one is a full launch-through-completion round trip."""
    for _ in range(WARMUP_ITERS):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(count):
        fn()
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) * 1e9 / count


def max_abs(t: torch.Tensor) -> float:
    return float(t.abs().max().item())


def bench_gemm(n: int, device: str) -> None:
    a = general(n, device)
    b = general(n, device) + 1.0
    ns = time_call(lambda: a @ b, iters(2 * n**3))
    want = a.double() @ b.double()
    row("matmul (ceiling)", n, ns, 2.0 * n**3, max_abs((a @ b).double() - want))


def bench_cholesky(n: int, device: str) -> None:
    a = spd(n, device)
    ns = time_call(lambda: torch.linalg.cholesky(a), iters(n**3 // 3))
    lower = torch.linalg.cholesky(a)
    resid = max_abs(lower.double() @ lower.double().T - a.double())
    row("cholesky", n, ns, n**3 / 3.0, resid)


def bench_lu(n: int, device: str) -> None:
    a = general(n, device)
    ns = time_call(lambda: torch.linalg.lu_factor(a), iters(2 * n**3 // 3))
    lu, piv = torch.linalg.lu_factor(a)
    b = ramp(n, 3, device).reshape(-1, 1)
    x = torch.linalg.lu_solve(lu, piv, b)
    resid = max_abs(a.double() @ x.double() - b.double())
    row("lu_factor", n, ns, 2.0 * n**3 / 3.0, resid)


def bench_solve(n: int, device: str) -> None:
    a = general(n, device)
    b = ramp(n, 5, device)
    ns = time_call(lambda: torch.linalg.solve(a, b), iters(2 * n**3 // 3))
    x = torch.linalg.solve(a, b)
    resid = max_abs(a.double() @ x.double() - b.double())
    row("solve", n, ns, 2.0 * n**3 / 3.0 + 2.0 * n**2, resid)


def bench_qr(n: int, device: str) -> None:
    a = general(n, device)
    ns = time_call(lambda: torch.linalg.qr(a, mode="reduced"), iters(4 * n**3 // 3))
    q, r = torch.linalg.qr(a, mode="reduced")
    resid = max_abs(q.double() @ r.double() - a.double())
    row("qr", n, ns, 2.0 * n**2 * (n - n / 3.0), resid)


def bench_blas1(n: int, device: str) -> None:
    x = ramp(n, 1, device)
    y = ramp(n, 2, device)
    count = iters(n)

    dot_ns = time_call(lambda: torch.dot(x, y), count)
    nrm2_ns = time_call(lambda: torch.linalg.vector_norm(x), count)
    asum_ns = time_call(lambda: x.abs().sum(), count)
    axpy_ns = time_call(lambda: torch.add(y, x, alpha=2.5), count)

    xd = x.double()
    yd = y.double()
    want_dot = float(torch.dot(xd, yd).item())
    want_dot_abs = float((xd * yd).abs().sum().item())
    want_sq = float(torch.dot(xd, xd).item())
    want_nrm2 = want_sq**0.5
    want_asum = float(xd.abs().sum().item())

    got_dot = float(torch.dot(x, y).item())
    got_nrm2 = float(torch.linalg.vector_norm(x).item())
    got_asum = float(x.abs().sum().item())
    got_axpy = torch.add(y, x, alpha=2.5).double()

    band_row("dot", n, dot_ns, 2 * n * 4, abs(got_dot - want_dot) / want_dot_abs)
    band_row("nrm2", n, nrm2_ns, n * 4, abs(got_nrm2 - want_nrm2) / want_nrm2)
    band_row("asum", n, asum_ns, n * 4, abs(got_asum - want_asum) / want_asum)
    band_row("axpy", n, axpy_ns, 3 * n * 4, max_abs(got_axpy - (2.5 * xd + yd)))


def main() -> None:
    if not torch.cuda.is_available():
        print("No CUDA device; this baseline is the GPU half of the comparison.")
        return

    device = "cuda"
    print(
        f"PyTorch {torch.__version__}  NumPy {np.__version__}  "
        f"device={torch.cuda.get_device_name(0)}  dtype=float32  "
        f"TF32 matmul={torch.backends.cuda.matmul.allow_tf32}"
    )

    print()
    print("Factorizations (ms is per call, GFLOP/s from the LAPACK count)")
    print("op\tn\tms\tGFLOP/s\tmax |residual|")
    for n in FACTOR_SIZES:
        bench_gemm(n, device)
    for n in FACTOR_SIZES:
        bench_cholesky(n, device)
    for n in FACTOR_SIZES:
        bench_lu(n, device)
    for n in FACTOR_SIZES:
        bench_solve(n, device)
    for n in FACTOR_SIZES:
        bench_qr(n, device)

    print()
    print("BLAS-1 (us is per call, GB/s over the traffic the op must move)")
    print("op\tn\tus\tGB/s\terror")
    for n in BLAS_SIZES:
        bench_blas1(n, device)


if __name__ == "__main__":
    main()
