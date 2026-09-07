"""Products and BLAS-1: `numpy.linalg`'s `matmul`/`dot`/`outer` and the
level-1 routines `scipy.linalg.blas` wraps.

**Two tiers under one set of names**, resolved by argument type.

Over `Tensor` this module is a delegation layer and nothing else.
`matmul`, `matvec` and `batched_matmul` call `linalg.matmul` and
`linalg.bmm`, so they inherit MAX's whole dispatch tree -- Apple simdgroup,
SM100, SM90, Ampere/CDNA, GEMV, vendor cuBLAS/rocBLAS/hipBLASLt, AMD RDNA
-- and numax names no architecture. `gpu: Bool` picks MAX's `target`; see
this subpackage's `__init__` docstring for why it is a parameter and not a
run-time test.

Over `Array[T, n*n]` the same names are `FloatLike`-generic, tier 1 and
register-resident, which is what lets one call factor a matrix per SIMD
lane inside a kernel. Past roughly 8x8 the `Tensor` overload is the faster
one; `docs/performance.md` has the crossover.

MAX ships no BLAS-1, and no BLAS anywhere is generic over its scalar type,
so `dot`/`nrm2`/`asum`/`axpy`/`outer` are `Array`-only.
"""

from layout import Coord, TileTensor
from layout.tile_layout import row_major
from linalg.bmm import batched_matmul as _max_batched_matmul
from linalg.matmul import matmul as _max_matmul
from std.collections import Array

from ..core.array import Dynamic, Shaped, zeros_dyn
from ..core.numeric import FloatLike

from .common import _zeros


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


def matvec[
    dtype: DType, m: Int, k: Int, gpu: Bool = False
](mut a: Shaped[dtype, m, k], mut x: Shaped[dtype, k]) raises -> Shaped[
    dtype, m
]:
    """The matrix-vector product `a @ x`, on `a`'s own device.

    Routed through `linalg.matmul` rather than `linalg.gemv`, with the
    vectors relaid as `k x 1` and `m x 1` over their own pointers. MAX's
    dispatch already sends a matmul with `n == 1` to its GEMV kernels --
    including the split-K and vector variants it picks between by shape --
    so calling `gemv` directly would be a second numax path to the same
    kernels, and one that would have to reproduce the choice.

    The relayout is free: a rank-1 buffer of `k` elements and a `k x 1`
    row-major layout address the same memory in the same order, so this
    hands MAX a different description of bytes it was going to read
    anyway, not a copy.
    """
    var ctx = a.context()
    var result = Shaped[dtype, m](ctx)
    var xv = x.view()
    var yv = result.view()
    var x_col = TileTensor(xv.ptr_at_offset(Coord(0)), row_major(Coord(k, 1)))
    var y_col = TileTensor(yv.ptr_at_offset(Coord(0)), row_major(Coord(m, 1)))
    _max_matmul[target="gpu" if gpu else "cpu"](y_col, a.view(), x_col, ctx)
    ctx.synchronize()
    return result^


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


def matmul[
    dtype: DType, m: Int, k: Int, n: Int, gpu: Bool = False
](mut a: Shaped[dtype, m, k], mut b: Shaped[dtype, k, n]) raises -> Shaped[
    dtype, m, n
]:
    """The matrix product `a @ b`, on `a`'s own device.

    `linalg.matmul` does the work, and that one call is a whole dispatch
    tree: Apple simdgroup kernels, SM100 `tcgen05`, SM90, Ampere and CDNA
    multistage GEMM with tile shapes chosen by heuristic, GEMV when a
    dimension is 1, vendor cuBLAS/rocBLAS/hipBLASLt, AMD RDNA WMMA, and a
    naive kernel when nothing else fits. numax names no architecture and
    picks no kernel.

    The sibling `matmul` over `Array[T, n*n]` is the one to call inside a
    kernel, or at any conformer other than a raw `dtype`; `to_tensor`
    crosses from there to here and `to_array` back.
    """
    var ctx = a.context()
    var result = Shaped[dtype, m, n](ctx)
    var c = result.view()
    _max_matmul[target="gpu" if gpu else "cpu"](c, a.view(), b.view(), ctx)
    ctx.synchronize()
    return result^


def matmul[
    dtype: DType, gpu: Bool = False
](mut a: Dynamic[dtype, 2], mut b: Dynamic[dtype, 2]) raises -> Dynamic[
    dtype, 2
]:
    """The matrix product `a @ b` at extents known only at run time.

    The run-time-shaped overload of the one above, selected by argument
    type rather than by a `where` clause: `Shaped` and `Dynamic` are
    different layouts, so the two can never be ambiguous. MAX reads the
    extents from the layout either way -- a compile-time shape buys kernel
    specialization, not correctness.

    Raises when `a`'s columns and `b`'s rows disagree, which is the check
    the static overload gets from the type system for free.
    """
    if a.dim[1]() != b.dim[0]():
        raise Error(
            "matmul shape mismatch: a is ",
            a.dim[0](),
            "x",
            a.dim[1](),
            " and b is ",
            b.dim[0](),
            "x",
            b.dim[1](),
        )
    var ctx = a.context()
    var result = zeros_dyn[dtype, 2](a.dim[0](), b.dim[1](), ctx=ctx)
    var c = result.view()
    _max_matmul[target="gpu" if gpu else "cpu"](c, a.view(), b.view(), ctx)
    ctx.synchronize()
    return result^


def batched_matmul[
    dtype: DType, batch: Int, m: Int, k: Int, n: Int, gpu: Bool = False
](
    mut a: Shaped[dtype, batch, m, k], mut b: Shaped[dtype, batch, k, n]
) raises -> Shaped[dtype, batch, m, n]:
    """`batch` independent matrix products, one per leading index.

    `linalg.bmm.batched_matmul` does the work, in one launch rather than
    `batch` of them. There is no `Array[T, n*n]` counterpart: a batch of
    small matrices is what a `map` over a conformer already expresses, one
    matrix per SIMD lane, so the batched form only earns its own kernel at
    sizes past the crossover.
    """
    var ctx = a.context()
    var result = Shaped[dtype, batch, m, n](ctx)
    var c = result.view()
    _max_batched_matmul[target="gpu" if gpu else "cpu"](
        c, a.view(), b.view(), context=ctx
    )
    ctx.synchronize()
    return result^
