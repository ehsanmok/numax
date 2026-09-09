"""PyTorch GPU baseline for the linalg sweep ../bench_linalg_gpu.mojo runs.

Same matrices, sizes, dtype, flop counts and residual definitions as the
Mojo GPU harness and as ../scipy/linalg.py -- see the SciPy file for the
list, since all three deliberately spell them identically.

Runs on whichever accelerator torch finds, CUDA first and Metal second, the
same order and the same `pick_gpu`/`sync` shape ../torch/gaussian.py uses.
On CUDA the factorizations are cuSOLVER underneath, which is the comparison
that matters: numax's blocked `Tensor` tier against a vendor library rather
than against another portable implementation. On Metal they are PyTorch's
own MPS kernels, and they are the *only* baseline available there -- MLX
ships no GPU linalg at all, refusing `cholesky`, `lu_factor`, `qr` and
`solve` on a GPU stream.

Three methodological points, all of which change the numbers:

- **The synchronize is inside the timed loop.** `torch.linalg.cholesky`
  returns as soon as the work is queued, so timing without a per-iteration
  synchronize measures dispatch and nothing else. numax's GPU harness
  synchronizes per call for the same reason, so both report
  launch-through-completion latency.
- **TF32 is left at PyTorch's default**, and printed in the header on CUDA.
  On Ampere that means `matmul` may use TF32 tensor cores while the
  factorizations stay `float32`, which is also what MAX's `matmul` does. A
  TF32 GEMM row is not comparable to an FP32 one and the flag is the only
  way to tell which this is. Metal has no such mode, so the line is CUDA's.
- **The float64 references run on the host, on both backends.** MPS has no
  float64 at all, so the reference cannot be computed on that device. It is
  moved to the CPU unconditionally rather than branched, so there is one
  spelling and one claim -- but it is a slightly different claim than this
  file made before the port: the residual column now checks a device
  float32 result against a *host* float64 reference rather than a device
  one, and a host GEMM and a cuBLAS GEMM do not associate identically even
  at float64, so CUDA residuals shift in their last digits. No published
  table records a residual (`bench/README.md` and `docs/performance.md`
  carry ms, GFLOP/s and GB/s only), and every reference is computed outside
  `time_call`, so neither a reproducible number nor a timing moves.

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


def pick_gpu() -> torch.device | None:
    """The accelerator to measure, or `None` if torch sees none.

    CUDA first: on a machine with both, CUDA is the real accelerator and MPS
    would not be present anyway. The same order ../torch/gaussian.py picks
    in, so the two files never disagree about which device a run describes.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return None


def sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def f64(t: torch.Tensor) -> torch.Tensor:
    """`t` widened to float64 on the host.

    The one place a reference recomputation is spelled. MPS refuses
    `.double()` outright -- "the MPS framework doesn't support float64" --
    so the widening cannot happen on that device at all, and it comes back
    to the host on *both* backends rather than branching. One spelling, one
    claim: branching would put a device float64 reference and a host one in
    the same column, which is the failure the sync-placement note above
    warns about. See the module docstring for what that changes.
    """
    return t.detach().cpu().double()


def spd(n: int, device: torch.device) -> torch.Tensor:
    i = torch.arange(n, device=device).reshape(-1, 1)
    j = torch.arange(n, device=device).reshape(1, -1)
    a = 1.0 / (1.0 + (i - j).abs().to(DTYPE))
    a.fill_diagonal_(float(n))
    return a.contiguous()


def general(n: int, device: torch.device) -> torch.Tensor:
    i = torch.arange(n, device=device).reshape(-1, 1)
    j = torch.arange(n, device=device).reshape(1, -1)
    a = (((i * 37 + j * 11) % 17).to(DTYPE)) * 0.0625 - 0.5
    a.fill_diagonal_(float(n))
    return a.contiguous()


def ramp(n: int, salt: int, device: torch.device) -> torch.Tensor:
    i = torch.arange(n, device=device)
    return (((i * 37 + salt * 11) % 17).to(DTYPE) - 8.0).contiguous()


def row(name: str, n: int, ns: float, flops: float, resid: float) -> None:
    print(f"{name}\t{n}\t{ns / 1e6:.4f}\t{flops / (ns / 1e9) / 1e9:.1f}\t{resid}")


def band_row(name: str, n: int, ns: float, nbytes: int, err: float) -> None:
    print(f"{name}\t{n}\t{ns / 1e3:.3f}\t{nbytes / (ns / 1e9) / 1e9:.2f}\t{err}")


def time_call(fn, count: int, device: torch.device) -> float:
    """Average nanoseconds per call, synchronizing inside every iteration so
    each one is a full launch-through-completion round trip."""
    for _ in range(WARMUP_ITERS):
        fn()
    sync(device)
    t0 = time.perf_counter()
    for _ in range(count):
        fn()
        sync(device)
    return (time.perf_counter() - t0) * 1e9 / count


def max_abs(t: torch.Tensor) -> float:
    return float(t.abs().max().item())


def bench_gemm(n: int, device: torch.device) -> None:
    a = general(n, device)
    b = general(n, device) + 1.0
    ns = time_call(lambda: a @ b, iters(2 * n**3), device)
    want = f64(a) @ f64(b)
    row("matmul (ceiling)", n, ns, 2.0 * n**3, max_abs(f64(a @ b) - want))


def bench_cholesky(n: int, device: torch.device) -> None:
    a = spd(n, device)
    ns = time_call(lambda: torch.linalg.cholesky(a), iters(n**3 // 3), device)
    lower = f64(torch.linalg.cholesky(a))
    resid = max_abs(lower @ lower.T - f64(a))
    row("cholesky", n, ns, n**3 / 3.0, resid)


def bench_lu(n: int, device: torch.device) -> None:
    a = general(n, device)
    ns = time_call(lambda: torch.linalg.lu_factor(a), iters(2 * n**3 // 3), device)
    lu, piv = torch.linalg.lu_factor(a)
    b = ramp(n, 3, device).reshape(-1, 1)
    x = torch.linalg.lu_solve(lu, piv, b)
    resid = max_abs(f64(a) @ f64(x) - f64(b))
    row("lu_factor", n, ns, 2.0 * n**3 / 3.0, resid)


def bench_solve(n: int, device: torch.device) -> None:
    a = general(n, device)
    b = ramp(n, 5, device)
    ns = time_call(lambda: torch.linalg.solve(a, b), iters(2 * n**3 // 3), device)
    x = torch.linalg.solve(a, b)
    resid = max_abs(f64(a) @ f64(x) - f64(b))
    row("solve", n, ns, 2.0 * n**3 / 3.0 + 2.0 * n**2, resid)


def bench_qr(n: int, device: torch.device) -> None:
    a = general(n, device)
    ns = time_call(
        lambda: torch.linalg.qr(a, mode="reduced"), iters(4 * n**3 // 3), device
    )
    q, r = torch.linalg.qr(a, mode="reduced")
    resid = max_abs(f64(q) @ f64(r) - f64(a))
    row("qr", n, ns, 2.0 * n**2 * (n - n / 3.0), resid)


def bench_blas1(n: int, device: torch.device) -> None:
    # Four host float64 references live at once below, and at n = 1 << 26
    # each is 512 MB -- about 2 GB of host RAM at the largest size. That is
    # a reduction against the pre-port version, which held the same four in
    # device memory, but do not add a fifth without checking.
    x = ramp(n, 1, device)
    y = ramp(n, 2, device)
    count = iters(n)

    dot_ns = time_call(lambda: torch.dot(x, y), count, device)
    nrm2_ns = time_call(lambda: torch.linalg.vector_norm(x), count, device)
    asum_ns = time_call(lambda: x.abs().sum(), count, device)
    axpy_ns = time_call(lambda: torch.add(y, x, alpha=2.5), count, device)

    xd = f64(x)
    yd = f64(y)
    want_dot = float(torch.dot(xd, yd).item())
    want_dot_abs = float((xd * yd).abs().sum().item())
    want_sq = float(torch.dot(xd, xd).item())
    want_nrm2 = want_sq**0.5
    want_asum = float(xd.abs().sum().item())

    got_dot = float(torch.dot(x, y).item())
    got_nrm2 = float(torch.linalg.vector_norm(x).item())
    got_asum = float(x.abs().sum().item())
    got_axpy = f64(torch.add(y, x, alpha=2.5))

    band_row("dot", n, dot_ns, 2 * n * 4, abs(got_dot - want_dot) / want_dot_abs)
    band_row("nrm2", n, nrm2_ns, n * 4, abs(got_nrm2 - want_nrm2) / want_nrm2)
    band_row("asum", n, asum_ns, n * 4, abs(got_asum - want_asum) / want_asum)
    band_row("axpy", n, axpy_ns, 3 * n * 4, max_abs(got_axpy - (2.5 * xd + yd)))


def main() -> None:
    device = pick_gpu()
    if device is None:
        print("No GPU visible to torch; this baseline is the GPU half of the")
        print("comparison. ../scipy/linalg.py is the CPU half.")
        return

    if device.type == "cuda":
        name = torch.cuda.get_device_name(0)
        precision = f"TF32 matmul={torch.backends.cuda.matmul.allow_tf32}"
    else:
        # Metal's answer to the TF32 question is that there is no question:
        # MPS has no reduced-precision GEMM mode to fall into, so a float32
        # matmul row here is a float32 matmul row. Printed in the same field
        # as the CUDA flag so the two headers stay diffable.
        name = "mps"
        precision = "TF32 matmul=n/a (Metal float32)"
    print(
        f"PyTorch {torch.__version__}  NumPy {np.__version__}  "
        f"device={name}  dtype=float32  {precision}"
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
