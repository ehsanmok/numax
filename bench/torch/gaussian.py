"""PyTorch CPU and MPS baseline for the same gaussian(x) = exp(-x^2) sweep
the other benchmarks in ../ run, at the same sizes -- see ../README.md.
Reports four numbers per size: eager and `torch.compile`'d, each on CPU and
on MPS (the same Metal GPU `numax`'s own GPU path and `../mlx/gaussian.py`
target).

`torch.compile` traces and JIT-compiles the function the first time it sees
a given input shape, so that cost has to land in the warmup calls, not the
timed ones -- this reuses one compiled callable per device across every
size in the sweep, so each new size pays its own one-time compile during
warmup, the same way a real caller moving between array sizes would.

MPS dispatch is asynchronous like CUDA's, so `torch.mps.synchronize()` is
required before stopping the clock -- without it, "MPS" timings would just
measure how fast PyTorch can enqueue work, not how fast the GPU runs it.

*Where* that synchronize goes turns out to matter more than whether it is
there, so this reports both placements rather than picking one:

- **per-call**: synchronize inside every timed iteration, so each iteration
  measures one full launch-through-completion round trip. This is latency as
  a caller experiences a single call.
- **amortized**: enqueue all timed iterations and synchronize once at the
  end, so the host round trip is paid once across all of them and the
  dispatches pipeline. This is steady-state throughput for back-to-back
  work.

The two differ by more than 2x at some sizes on the same hardware and the
same kernel, so a table mixing one library's per-call number with another's
amortized number is measuring the methodology, not the libraries. This file
used to report amortized only, which is what `numax`'s own per-call GPU
benchmark was being compared against in `bench/README.md`.

Run: pixi run bench-torch (from the repo root; MPS results need Apple Silicon)
"""

import time

import torch

DTYPE = torch.float32
SIZES = [1 << 16, 1 << 18, 1 << 20, 1 << 22, 1 << 24, 1 << 26]
WARMUP_ITERS = 3
TIMED_ITERS = 10


def gaussian(x: torch.Tensor) -> torch.Tensor:
    return torch.exp(-(x * x))


def make_input(n: int, device: torch.device) -> torch.Tensor:
    return torch.arange(n, dtype=DTYPE, device=device) * 0.0001 - 50.0


def sync(device: torch.device) -> None:
    if device.type == "mps":
        torch.mps.synchronize()


def bench_fn(
    fn,
    xs: torch.Tensor,
    device: torch.device,
    label: str,
    n: int,
    amortize_sync: bool = True,
):
    """Time `fn(xs)`, synchronizing either once per iteration or once total.

    On CPU both placements are the same measurement (the work is synchronous
    already), so `amortize_sync` only changes anything for MPS.
    """
    for _ in range(WARMUP_ITERS):
        fn(xs)
    sync(device)

    t0 = time.perf_counter()
    if amortize_sync:
        for _ in range(TIMED_ITERS):
            ys = fn(xs)
        sync(device)
    else:
        for _ in range(TIMED_ITERS):
            ys = fn(xs)
            sync(device)
    elapsed_ns = (time.perf_counter() - t0) * 1e9

    avg_ms = elapsed_ns / TIMED_ITERS / 1e6
    elems_per_sec_m = n / (avg_ms / 1e3) / 1e6
    print(f"  {label}={avg_ms:8.4f} ms ({elems_per_sec_m:8.1f} M elem/s)", end="")
    return ys


def check(label: str, xs_ref: torch.Tensor, ys: torch.Tensor) -> None:
    reference = torch.exp(-(xs_ref.double() ** 2))
    max_err = float((ys.detach().cpu().double() - reference).abs().max())
    if max_err > 1e-5:
        print(f"  WARNING: max |{label} - f64 reference| = {max_err}")


def bench_at_size(n: int, compiled_cpu, compiled_mps, has_mps: bool) -> None:
    print(f"n={n:>9} ", end="")

    cpu = torch.device("cpu")
    xs_cpu = make_input(n, cpu)
    ys = bench_fn(gaussian, xs_cpu, cpu, "eager-CPU", n)
    ys = bench_fn(compiled_cpu, xs_cpu, cpu, "compile-CPU", n)

    if has_mps:
        mps = torch.device("mps")
        xs_mps = make_input(n, mps)
        bench_fn(gaussian, xs_mps, mps, "eager-MPS-amort", n)
        ys = bench_fn(compiled_mps, xs_mps, mps, "compile-MPS-amort", n)
        print()
        print(f"{'':>11}", end="")
        bench_fn(
            gaussian, xs_mps, mps, "eager-MPS-percall", n, amortize_sync=False
        )
        bench_fn(
            compiled_mps,
            xs_mps,
            mps,
            "compile-MPS-percall",
            n,
            amortize_sync=False,
        )

    print()
    check("torch", xs_cpu, ys)


def main() -> None:
    has_mps = torch.backends.mps.is_available()
    print(f"PyTorch {torch.__version__}  dtype=float32  MPS available={has_mps}")

    compiled_cpu = torch.compile(gaussian)
    compiled_mps = torch.compile(gaussian) if has_mps else None

    for n in SIZES:
        bench_at_size(n, compiled_cpu, compiled_mps, has_mps)


if __name__ == "__main__":
    main()
