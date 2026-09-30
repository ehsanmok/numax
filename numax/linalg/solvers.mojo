"""Matrix equation solvers over `Tensor`: `solve_sylvester`,
`solve_continuous_lyapunov`, `solve_discrete_lyapunov` and
`solve_continuous_are`, SciPy's `scipy.linalg._solvers`.

**Tier 2.** The Sylvester and Lyapunov solvers are the Bartels-Stewart
algorithm or a map onto it:
the real Schur forms `a = U R U^T` and `b = V S V^T` (`schur`, device
reduction and `O(n^3)` products), the right-hand side rotated to `F = U^T
Q V`, the quasi-triangular `R Y + Y S = F` solved by `trsyl_column` --
one single-block device kernel per column of `S`, `O(n^2)` each -- and `X
= U Y V^T`. The host drives the column loop and reads nothing: each
launch reads its own block role off `S`'s subdiagonal. So `gpu=True`
keeps the whole solve on the device. `solve_continuous_are` is the
matrix sign function of the Hamiltonian instead; its docstring has it.

A unique solution needs `a` and `-b` to share no eigenvalue. Where they
come close the quasi-triangular system is near-singular and the answer
grows without warning, as LAPACK's `trsyl` warns but still returns; the
residual `a X + X b - q` is the check.

## The MAX gate

Nothing: MAX has no Schur form and no Sylvester solver. `schur` is
numax's (`numax.linalg.eigen`), and the products are `linalg.matmul`.
**Extend.**
"""

from layout import Coord, coord_to_index_list
from max.algorithm.functional import elementwise
from max.gpu.host import DeviceContext
from std.math import exp as _exp

from ..core.tensorlike import TensorLike, dim
from ..core.tensor import Static, _same_order, eye, transpose
from ..core.ops import add, multiply, subtract
from .basic import inverse
from .blas import _target, matmul
from .common import _mut_view
from .eigen import schur
from .lu import lu_factor
from .misc import norm
from .qr import qr_factor
from .triangular import solve_triangular
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
    var rv = _mut_view(r)
    var sv = _mut_view(s)
    var yv = y.tile()
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
                    gpu=True,
                ]
            ](
                rv,
                sv,
                yv,
                Int32(k),
                Int32(n),
                Int32(m),
                grid_dim=1,
                block_dim=_PANEL_THREADS,
            )
        else:
            trsyl_column(rv, sv, yv, Int32(k), Int32(n), Int32(m))
    ctx.synchronize()


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


comptime _SIGN_MAX_STEPS = 100
"""Newton steps the sign iteration may take. Determinant scaling brings a
well-conditioned Hamiltonian to convergence in 10 to 20; the cap only
bounds a problem with an eigenvalue on the imaginary axis, which has no
stabilizing solution."""


def _hamiltonian[
    dtype: DType, n: Int, gpu: Bool
](
    a: Static[dtype, n, n], g: Static[dtype, n, n], q: Static[dtype, n, n]
) raises -> Static[dtype, 2 * n, 2 * n] where dtype.is_floating_point():
    """`[[a, -g], [-q, -a^T]]`, assembled in one launch."""
    var ctx = a.context()
    var h = Static[dtype, 2 * n, 2 * n](ctx)
    var av = _mut_view(a)
    var gv = _mut_view(g)
    var qv = _mut_view(q)
    var hv = h.tile()

    @always_inline
    def fill[
        w: Int, alignment: Int = 1
    ](coord: Coord) {var av, var gv, var qv, var hv}:
        var at = coord_to_index_list(coord)
        var i = at[0]
        var j = at[1]
        var value: Scalar[dtype]
        if i < n and j < n:
            value = av[Coord(i, j)]
        elif i < n:
            value = -gv[Coord(i, j - n)]
        elif j < n:
            value = -qv[Coord(i - n, j)]
        else:
            value = -av[Coord(j - n, i - n)]
        hv.store[1](coord, value)

    elementwise[simd_width=1, target=_target[gpu]()](
        fill, Coord(2 * n, 2 * n), ctx
    )
    ctx.synchronize()
    return h^


def _stable_subspace_system[
    dtype: DType, n: Int, gpu: Bool
](w: Static[dtype, 2 * n, 2 * n]) raises -> Tuple[
    Static[dtype, 2 * n, n], Static[dtype, 2 * n, n]
] where dtype.is_floating_point():
    """`([W12; W22 + I], -[W11 + I; W21])`: `(W + I) [I; X] = 0` on the
    stable invariant subspace `[I; X]` spans, split into `M X = R`."""
    var ctx = w.context()
    var m = Static[dtype, 2 * n, n](ctx)
    var r = Static[dtype, 2 * n, n](ctx)
    var wv = _mut_view(w)
    var mv = m.tile()
    var rv = r.tile()

    @always_inline
    def split[
        lanes: Int, alignment: Int = 1
    ](coord: Coord) {var wv, var mv, var rv}:
        var at = coord_to_index_list(coord)
        var i = at[0]
        var j = at[1]
        var one = Scalar[dtype](1)
        var zero = Scalar[dtype](0)
        mv.store[1](coord, wv[Coord(i, n + j)] + (one if i == n + j else zero))
        rv.store[1](coord, -(wv[Coord(i, j)] + (one if i == j else zero)))

    elementwise[simd_width=1, target=_target[gpu]()](
        split, Coord(2 * n, n), ctx
    )
    ctx.synchronize()
    return (m^, r^)


def solve_continuous_are[
    A: TensorLike,
    B: TensorLike,
    Q: TensorLike,
    R: TensorLike,
    gpu: Bool = False,
](a: A, b: B, q: Q, r: R) raises -> Static[
    A.dtype, dim[A, 0], dim[A, 0]
] where (
    A.dtype.is_floating_point()
    and A.LayoutType.rank == 2
    and A.LayoutType.all_dims_known
    and dim[A, 1] == dim[A, 0]
    and B.dtype == A.dtype
    and B.LayoutType.rank == 2
    and B.LayoutType.all_dims_known
    and dim[B, 0] == dim[A, 0]
    and Q.dtype == A.dtype
    and Q.LayoutType.rank == 2
    and Q.LayoutType.all_dims_known
    and dim[Q, 0] == dim[A, 0]
    and dim[Q, 1] == dim[A, 0]
    and R.dtype == A.dtype
    and R.LayoutType.rank == 2
    and R.LayoutType.all_dims_known
    and dim[R, 0] == dim[B, 1]
    and dim[R, 1] == dim[B, 1]
    # True for every size; stated so the least-squares `qr_factor` of the
    # `2n x n` system can see its `m >= n`.
    and 2 * dim[A, 0] >= dim[A, 0]
):
    """**Tier 2.** The stabilizing solution of the continuous algebraic
    Riccati equation `a^T X + X a - X b r^-1 b^T X + q = 0`.
    `scipy.linalg.solve_continuous_are(a, b, q, r)`.

    By the matrix sign function of the Hamiltonian `H = [[a, -G], [-q,
    -a^T]]`, `G = b r^-1 b^T` (Roberts; Byers' determinant scaling): Newton's
    `Z <- (Z / c + c Z^-1) / 2` with `c = |det Z|^(1/2n)`, until the step
    is below `sqrt(eps)` of `Z`'s norm, then two more steps, which the
    quadratic convergence turns into full precision. `sign(H) = W` has
    eigenvalue `-1` on the stable invariant subspace, spanned by `[I; X]`,
    so `[W12; W22 + I] X = -[W11 + I; W21]`, solved by least squares
    through `qr_factor`, and `X` is symmetrized. Every step is an LU, a
    product or a QR, so `gpu=True` keeps it on the device; the host reads
    one determinant and two norms per step to scale and to stop.

    SciPy reduces the extended pencil by QZ with balancing; numax has no
    QZ. The sign iteration reaches the same solution to about `1e-13`
    relative on well-conditioned problems and, like any Hamiltonian method
    without balancing, loses accuracy as the closed-loop spectrum
    approaches the imaginary axis.

    Parameters:
        A: The tensor type of `a`, `n x n`.
        B: The tensor type of `b`, `n x m`.
        Q: The tensor type of `q`, `n x n`, symmetric.
        R: The tensor type of `r`, `m x m`, symmetric positive definite.
        gpu: Whether to run on the inputs' device.

    Args:
        a: The state matrix.
        b: The input matrix.
        q: The state weight.
        r: The input weight.

    Returns:
        The symmetric stabilizing `X`.

    Raises:
        If the iteration does not converge in `_SIGN_MAX_STEPS` steps --
        `H` then has eigenvalues on or near the imaginary axis, and no
        stabilizing solution exists -- or a device operation fails.
    """
    comptime dtype = A.dtype
    comptime n = dim[A, 0]
    comptime k = dim[B, 1]
    comptime two_n = 2 * n
    var ac = _square[dtype, n](a)
    var qc = _square[dtype, n](q)
    var rc = _square[dtype, k](r)
    var bc = rebind_var[Static[dtype, n, k]](
        _same_order(b, Static[B.dtype, n, k]._static_layout())
    )
    var g = matmul[gpu=gpu](
        bc, matmul[gpu=gpu](inverse[gpu=gpu](rc), transpose[gpu=gpu](bc))
    )
    var z = _hamiltonian[gpu=gpu](ac, g, qc)
    var ctx = z.context()
    var identity = eye[two_n, dtype](ctx=ctx)
    comptime eps = 2.220446049250313e-16 if dtype == DType.float64 else 1.1920928955078125e-07
    var threshold = Scalar[dtype](eps**0.5)
    var converged = False
    var polish = 0
    for _ in range(_SIGN_MAX_STEPS):
        var factored = lu_factor[gpu=gpu](z)
        var inv = factored.solve(identity)
        var logdet = factored.slogdet()[1]
        var c = _exp(logdet / Scalar[dtype](two_n))
        if converged:
            c = Scalar[dtype](1)
        var next = multiply[gpu=gpu](
            add[gpu=gpu](
                multiply[gpu=gpu](z, Scalar[dtype](1) / c),
                multiply[gpu=gpu](inv, c),
            ),
            Scalar[dtype](0.5),
        )
        var step = norm[ord=1, gpu=gpu](subtract[gpu=gpu](next, z))
        var size = norm[ord=1, gpu=gpu](next)
        z = next^
        if converged:
            polish += 1
            if polish == 2:
                break
        elif step <= threshold * size:
            converged = True
    if not converged or polish < 2:
        raise Error(
            "solve_continuous_are: the sign iteration did not converge in ",
            _SIGN_MAX_STEPS,
            (
                " steps; the Hamiltonian has eigenvalues on or near the"
                " imaginary axis, so no stabilizing solution exists"
            ),
        )
    var system = _stable_subspace_system[gpu=gpu](z)
    var factor = qr_factor[gpu=gpu](system[0])
    var qt_rhs = matmul[gpu=gpu](transpose[gpu=gpu](factor.q()), system[1])
    var x = solve_triangular[upper=True, gpu=gpu](factor.r(), qt_rhs)
    return multiply[gpu=gpu](
        add[gpu=gpu](x, transpose[gpu=gpu](x)), Scalar[dtype](0.5)
    )
