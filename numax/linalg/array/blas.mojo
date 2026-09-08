"""BLAS-1 and the small matrix products over `Array`.
`scipy.linalg.blas`, `numpy.linalg.matmul`.

**Tier 1**, all of it: fixed trip counts and no per-lane branching, so
every one of these launches inside a GPU thread at any conformer and
differentiates when instantiated at `Dual`.

The vectorization axis is the batch, not the vector: at
`Plain[dtype, w]` each `Array` element holds a `w`-wide `SIMD`, so a loop
here is `w` independent problems in lockstep. That is the opposite of what
the `Tensor` tier in `numax.linalg.blas` does, which vectorizes along the
vector and hands the work to MAX. At `w = 1` these are scalar loops and
the `Tensor` overloads are the ones to reach for.
"""

from std.collections import Array

from ...core.numeric import FloatLike

from ..common import _zeros


def dot[T: FloatLike, n: Int](a: Array[T, n], b: Array[T, n]) -> T:
    """The inner product `sum(a[i] * b[i])` -- BLAS-1 `dot`.

    Summed in order rather than pairwise, so the rounding is the obvious
    one and a caller who cares can recover the lost bits by calling this at
    `Compensated` instead of `Plain`. That is the whole reason a `dot` this
    small is worth writing: MAX ships no BLAS-1 at all, and no BLAS
    anywhere is generic over its scalar type.
    """
    var total = T.constant(0.0)
    for i in range(n):
        total = total + a[i] * b[i]
    return total^


def nrm2[T: FloatLike, n: Int](a: Array[T, n]) -> T:
    """The Euclidean norm `sqrt(sum(a[i]**2))` -- BLAS-1 `nrm2`.

    Not rescaled, so a vector whose entries approach the square root of
    `dtype`'s overflow threshold will overflow here where LAPACK's `nrm2`
    would not. Rescaling needs a running maximum and a data-dependent
    branch, which the fixed-iteration invariant rules out; the same
    trade-off `norm` documents.
    """
    return dot[T, n](a, a).sqrt()


def asum[T: FloatLike, n: Int](a: Array[T, n]) -> T:
    """The sum of magnitudes `sum(|a[i]|)` -- BLAS-1 `asum`, and the vector
    1-norm `norm` gives for a matrix.

    Cannot overflow the way `nrm2` can, which is why a convergence check
    that only needs a magnitude usually wants this one.
    """
    var total = T.constant(0.0)
    for i in range(n):
        total = total + a[i].abs()
    return total^


def axpy[
    T: FloatLike, n: Int
](alpha: T, x: Array[T, n], y: Array[T, n]) -> Array[T, n]:
    """`alpha * x + y` -- BLAS-1 `axpy`.

    Returns a new vector rather than updating `y` in place, since a
    register-resident `Array` has no aliasing to avoid and an expression
    reads better than a mutation at these sizes.
    """
    var out = _zeros[T, n]()
    for i in range(n):
        out[i] = alpha * x[i] + y[i]
    return out^


def outer[
    T: FloatLike, n: Int
](a: Array[T, n], b: Array[T, n]) -> Array[T, n * n]:
    """The outer product `out[i, j] = a[i] * b[j]`, row-major.

    The rank-1 update every quasi-Newton method and every Householder
    reflector is built from. MAX's `outer_product_acc` is the nearest
    thing and is denied twice over: it is on the older `LayoutTensor`,
    which numax does not bridge to, and it only accumulates into an
    existing matrix rather than producing one.
    """
    var out = _zeros[T, n * n]()
    for i in range(n):
        for j in range(n):
            out[i * n + j] = a[i] * b[j]
    return out^


def matvec[
    T: FloatLike, n: Int
](a: Array[T, n * n], x: Array[T, n]) -> Array[T, n]:
    """`A @ x` for a row-major `n x n` matrix.

    For a large, plain-`dtype` `A` (past the crossover `matmul`'s own
    docstring documents), MAX's `linalg.matmul` still applies -- a
    matrix-vector product is a matrix-matrix product against an `n x 1`
    `TileTensor`, and MAX has no separate matvec-specific fast path to
    prefer over that.
    """
    var out = _zeros[T, n]()
    for i in range(n):
        var total = T.constant(0.0)
        for j in range(n):
            total = total + a[i * n + j] * x[j]
        out[i] = total^
    return out^


def matmul[
    T: FloatLike, n: Int
](a: Array[T, n * n], b: Array[T, n * n]) -> Array[T, n * n]:
    """`A @ B` for two row-major `n x n` matrices.

    The naive triple loop, which is the right algorithm at these sizes and
    the wrong one past them. Measured against MAX's `linalg.matmul` on an M3
    Pro (`bench/bench_matmul.mojo`), the crossover is at `n = 8` for a
    single matrix and `n = 16` for the 4-wide batched form; by `n = 64` MAX
    is ~130x faster. So call `linalg.matmul` directly for anything
    larger than about 8x8 whose entries are plain `dtype` values.

    What this version has instead: it is generic in `T`, so calling it at
    `Dual` differentiates the product and calling it at `Compensated` runs
    it at extra precision, neither of which a `dtype`-monomorphic kernel
    can do. And since the matrix is an `Array` in registers rather than a
    `TileTensor` in memory, it can be called from inside a single GPU
    thread -- one matrix per SIMD lane, if `T` is itself a vector.
    """
    var out = _zeros[T, n * n]()
    for i in range(n):
        for k in range(n):
            var aik = a[i * n + k].copy()
            for j in range(n):
                out[i * n + j] = out[i * n + j] + aik * b[k * n + j]
    return out^
