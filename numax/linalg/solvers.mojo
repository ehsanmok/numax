"""Matrix equation solvers over `Tensor`: `solve_sylvester`,
`solve_continuous_lyapunov` and `solve_discrete_lyapunov`, SciPy's
`scipy.linalg._solvers`.

**Tier 2.** Every one is the Bartels-Stewart algorithm or a map onto it:
the real Schur forms `a = U R U^T` and `b = V S V^T` (`schur`, device
reduction and `O(n^3)` products), the right-hand side rotated to `F = U^T
Q V`, the quasi-triangular `R Y + Y S = F` solved by `trsyl_column` --
one single-block device kernel per column of `S`, `O(n^2)` each -- and `X
= U Y V^T`. The host drives the column loop and reads nothing: each
launch reads its own block role off `S`'s subdiagonal. So `gpu=True`
keeps the whole solve on the device.

A unique solution needs `a` and `-b` to share no eigenvalue. Where they
come close the quasi-triangular system is near-singular and the answer
grows without warning, as LAPACK's `trsyl` warns but still returns; the
residual `a X + X b - q` is the check.

## The MAX gate

Nothing: MAX has no Schur form and no Sylvester solver. `schur` is
numax's (`numax.linalg.eigen`), and the products are `linalg.matmul`.
**Extend.**
"""

from max.gpu.host import DeviceContext

from ..core.tensorlike import TensorLike, dim
from ..core.tensor import Static, _same_order, eye, transpose
from ..core.ops import add, multiply, subtract
from .basic import inverse
from .blas import _target, matmul
from .common import _mut_view
from .eigen import schur
from .panel import _PANEL_THREADS, trsyl_column


def _trsyl[
    dtype: DType, n: Int, m: Int, gpu: Bool
](
    r: Static[dtype, n, n],
    s: Static[dtype, m, m],
    mut y: Static[dtype, n, m],
) raises where dtype.is_floating_point():
    """Overwrite `y` (holding `F`) with the `Y` of `R Y + Y S = F`, one
    `trsyl_column` launch per column of `S`."""
    var ctx = y.context()
    var work = Static[dtype, 4 * _PANEL_THREADS](ctx)
    var rv = _mut_view(r)
    var sv = _mut_view(s)
    var yv = y.tile()
    var wv = work.tile()
    # The Schur forms and `F` come from asynchronous products.
    ctx.synchronize()
    for k in range(m):
        comptime if gpu:
            ctx.enqueue_function[
                trsyl_column[
                    dtype,
                    RLayout=type_of(rv).LayoutType,
                    SLayout=type_of(sv).LayoutType,
                    YLayout=type_of(yv).LayoutType,
                    WLayout=type_of(wv).LayoutType,
                    gpu=True,
                ]
            ](
                rv,
                sv,
                yv,
                wv,
                Int32(k),
                Int32(n),
                Int32(m),
                grid_dim=1,
                block_dim=_PANEL_THREADS,
            )
        else:
            trsyl_column(rv, sv, yv, wv, Int32(k), Int32(n), Int32(m))
    ctx.synchronize()
    _ = work^


def _square[
    dtype: DType, n: Int, T: TensorLike
](a: T) raises -> Static[dtype, n, n] where (
    T.LayoutType.rank == 2 and T.LayoutType.all_dims_known
):
    """`a` copied to a concrete `Static[dtype, n, n]` on its device, so the
    products between several inputs see one type -- the layout prover does
    not carry a `dtype` equality through a callee's result."""
    return rebind_var[Static[dtype, n, n]](
        _same_order(a, Static[T.dtype, n, n]._static_layout())
    )


def _sylvester[
    dtype: DType, n: Int, m: Int, gpu: Bool
](
    a: Static[dtype, n, n], b: Static[dtype, m, m], q: Static[dtype, n, m]
) raises -> Static[dtype, n, m] where dtype.is_floating_point():
    var sa = schur[gpu=gpu](a)
    var sb = schur[gpu=gpu](b)
    var f = matmul[gpu=gpu](transpose[gpu=gpu](sa.z), matmul[gpu=gpu](q, sb.z))
    _trsyl[gpu=gpu](sa.t, sb.t, f)
    return matmul[gpu=gpu](sa.z, matmul[gpu=gpu](f, transpose[gpu=gpu](sb.z)))


def solve_sylvester[
    A: TensorLike,
    B: TensorLike,
    Q: TensorLike,
    gpu: Bool = False,
](a: A, b: B, q: Q) raises -> Static[A.dtype, dim[A, 0], dim[B, 0]] where (
    A.dtype.is_floating_point()
    and A.LayoutType.rank == 2
    and A.LayoutType.all_dims_known
    and dim[A, 1] == dim[A, 0]
    and B.dtype == A.dtype
    and B.LayoutType.rank == 2
    and B.LayoutType.all_dims_known
    and dim[B, 1] == dim[B, 0]
    and Q.dtype == A.dtype
    and Q.LayoutType.rank == 2
    and Q.LayoutType.all_dims_known
    and dim[Q, 0] == dim[A, 0]
    and dim[Q, 1] == dim[B, 0]
):
    """**Tier 2.** The `X` with `a X + X b = q`. `scipy.linalg.solve_sylvester`.

    Bartels-Stewart, per this module's docstring: the real Schur forms of
    `a` and `b`, the rotated right-hand side, `trsyl_column` over the
    quasi-triangular system, and the rotation back. SciPy takes the
    complex Schur form of `b`'s conjugate transpose instead; for real
    input the two solve the same equation.

    Parameters:
        A: The tensor type of `a`, square `n x n`.
        B: The tensor type of `b`, square `m x m`, the same `dtype`.
        Q: The tensor type of `q`, `n x m`, the same `dtype`.
        gpu: Whether to run on the inputs' device.

    Args:
        a: The left coefficient.
        b: The right coefficient.
        q: The right-hand side.

    Returns:
        The `n x m` solution, on the inputs' device.

    Raises:
        If a Schur decomposition or a device operation fails.
    """
    comptime dtype = A.dtype
    comptime n = dim[A, 0]
    comptime m = dim[B, 0]
    var ac = _square[dtype, n](a)
    var bc = _square[dtype, m](b)
    var qc = rebind_var[Static[dtype, n, m]](
        _same_order(q, Static[Q.dtype, n, m]._static_layout())
    )
    return _sylvester[gpu=gpu](ac, bc, qc)


def solve_continuous_lyapunov[
    A: TensorLike,
    Q: TensorLike,
    gpu: Bool = False,
](a: A, q: Q) raises -> Static[A.dtype, dim[A, 0], dim[A, 0]] where (
    A.dtype.is_floating_point()
    and A.LayoutType.rank == 2
    and A.LayoutType.all_dims_known
    and dim[A, 1] == dim[A, 0]
    and Q.dtype == A.dtype
    and Q.LayoutType.rank == 2
    and Q.LayoutType.all_dims_known
    and dim[Q, 0] == dim[A, 0]
    and dim[Q, 1] == dim[A, 0]
):
    """**Tier 2.** The `X` with `a X + X a^T = q`.
    `scipy.linalg.solve_continuous_lyapunov`.

    `solve_sylvester(a, a^T, q)`, as SciPy's is. For a stable `a` (every
    eigenvalue in the left half-plane) and a symmetric negative
    semidefinite `q`, `X` is the controllability Gramian and symmetric
    positive semidefinite; nothing here forces the symmetry, so it holds
    to rounding.

    Parameters:
        A: The tensor type of `a`, square.
        Q: The tensor type of `q`, the same shape and `dtype`.
        gpu: Whether to run on the inputs' device.

    Args:
        a: The coefficient.
        q: The right-hand side.

    Returns:
        The `n x n` solution.

    Raises:
        If the Schur decomposition or a device operation fails.
    """
    comptime dtype = A.dtype
    comptime n = dim[A, 0]
    var ac = _square[dtype, n](a)
    var qc = _square[dtype, n](q)
    var at = transpose[gpu=gpu](ac)
    return _sylvester[gpu=gpu](ac, at, qc)


def solve_discrete_lyapunov[
    A: TensorLike,
    Q: TensorLike,
    gpu: Bool = False,
](a: A, q: Q) raises -> Static[A.dtype, dim[A, 0], dim[A, 0]] where (
    A.dtype.is_floating_point()
    and A.LayoutType.rank == 2
    and A.LayoutType.all_dims_known
    and dim[A, 1] == dim[A, 0]
    and Q.dtype == A.dtype
    and Q.LayoutType.rank == 2
    and Q.LayoutType.all_dims_known
    and dim[Q, 0] == dim[A, 0]
    and dim[Q, 1] == dim[A, 0]
):
    """**Tier 2.** The `X` with `X - a X a^T = q`, the Stein equation.
    `scipy.linalg.solve_discrete_lyapunov(a, q, method="bilinear")`.

    SciPy's bilinear transformation onto the continuous equation: with
    `B = (a^T - I)(a^T + I)^-1` and `C = 2 (a + I)^-1 q (a^T + I)^-1`, `X`
    solves `B^T X + X B = -C`, which `solve_continuous_lyapunov` takes.
    Two inverses and four products beside it, all on the device at
    `gpu=True`. SciPy's `"direct"` method, the `n^2 x n^2` Kronecker
    system it uses below `n = 10`, is not taken; it and this agree to
    rounding. `a` must have no eigenvalue at `-1`, where the transformation
    is singular.

    Parameters:
        A: The tensor type of `a`, square.
        Q: The tensor type of `q`, the same shape and `dtype`.
        gpu: Whether to run on the inputs' device.

    Args:
        a: The coefficient.
        q: The right-hand side.

    Returns:
        The `n x n` solution.

    Raises:
        If an inverse, the Schur decomposition or a device operation fails.
    """
    comptime dtype = A.dtype
    comptime n = dim[A, 0]
    var ac = _square[dtype, n](a)
    var qc = _square[dtype, n](q)
    var ctx = ac.context()
    var identity = eye[n, dtype](ctx=ctx)
    var ah = transpose[gpu=gpu](ac)
    var ahi_inv = inverse[gpu=gpu](add[gpu=gpu](ah, identity))
    var bmat = matmul[gpu=gpu](subtract[gpu=gpu](ah, identity), ahi_inv)
    var c = multiply[gpu=gpu](
        matmul[gpu=gpu](
            matmul[gpu=gpu](inverse[gpu=gpu](add[gpu=gpu](ac, identity)), qc),
            ahi_inv,
        ),
        Scalar[dtype](2.0),
    )
    var neg_c = multiply[gpu=gpu](c, Scalar[dtype](-1.0))
    var bt = transpose[gpu=gpu](bmat)
    return _sylvester[gpu=gpu](bt, bmat, neg_c)
