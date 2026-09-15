"""NumPy CPU baseline for the core-surface sweep ../bench_core_surface.mojo
runs.

Same hashed data on `[-3, 3)`, same sizes, same `float32`, same columns,
and the same byte count per row -- `np.exp`, `a + b`, `a * 2`, the
comparison, and `a.sum()` are the routines numax's names come from.

Two comparison rows, because the two libraries are not writing the same
program. `a > 0` broadcasts a Python scalar, so it moves 5 bytes per
element; numax has no tensor-scalar comparison, so its spelling names a
zero tensor and moves 9. `np.greater(a, z)` is here so the 9-byte row has
a baseline that does the same work, and `a > 0` is here because it is what
a NumPy user actually writes.

Run: pixi run -e bench-python bench-numpy-core-surface (from the repo root)
"""

import time

import numpy as np

DTYPE = np.float32
SIZES = [1 << 10, 1 << 14, 1 << 20, 1 << 22, 1 << 24]
WARMUP_ITERS = 2
BUDGET_SECS = 1.0


def entry(i: np.ndarray) -> np.ndarray:
    return 6.0 * (((i * 2654435761 + 12345) % 16777216) / 16777216.0 - 0.5)


def data(n: int, salt: int) -> np.ndarray:
    return entry(np.arange(n, dtype=np.int64) + salt).astype(DTYPE)


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


def row(name: str, n: int, ns: float, per_element: int) -> None:
    print(
        f"{name}\t{n}\t{ns / 1e3:.3f}\t{n / (ns / 1e9) / 1e6:.1f}"
        f"\t{per_element}\t{n * per_element / (ns / 1e9) / 1e9:.2f}"
    )


def bench_size(n: int) -> None:
    a = data(n, 0)
    b = data(n, 1)
    z = np.zeros(n, dtype=DTYPE)
    two = DTYPE(2)
    row("exp(a)", n, time_call(lambda: np.exp(a)), 8)
    row("a + b", n, time_call(lambda: a + b), 12)
    row("a * 2", n, time_call(lambda: a * two), 8)
    row("a > 0", n, time_call(lambda: a > 0), 5)
    row("greater(a, zeros)", n, time_call(lambda: np.greater(a, z)), 9)
    row("sum(a)", n, time_call(lambda: a.sum()), 4)


def main() -> None:
    print(f"NumPy {np.__version__}  dtype={DTYPE.__name__}  target=cpu")
    print()
    print("Core surface (us is per call, B/elem is bytes moved per element)")
    print("op\tn\tus\tM elem/s\tB/elem\tGB/s")
    for n in SIZES:
        bench_size(n)


if __name__ == "__main__":
    main()
