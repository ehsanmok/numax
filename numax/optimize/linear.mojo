"""Constrained linear least squares over `numax.core.array.Tensor`: `nnls`
and `lsq_linear`. `scipy.optimize.nnls`, `scipy.optimize.lsq_linear`.

**This module is tier 2.** Both are one box-constrained quadratic program
`min (1/2) x^T G x - c^T x` over `l <= x <= u`, with `G = A^T A` and `c =
A^T b` formed once on the device through `matmul` and `matvec` -- the
`O(m n^2)` term, and the one that grows with the data -- and then solved on
the host by a projected Newton active-set iteration over `n` unknowns:

1. the gradient `g = G x - c`, and the *free* set of variables not held at
   a bound by a gradient pointing outward;
2. the Newton step on the free set, `G_FF d_F = -g_F`, by elimination on
   the host (falling back to `-g_F` if that block is singular);
3. the step projected onto the box and backtracked on the quadratic until
   it is a decrease.

Convergence is the projected gradient below `tol`, and a run that hits
`max_iter` returns `converged=False` at the best point. On a strictly
convex problem the active set settles in finitely many steps and the
last Newton step is exact, which is what makes this competitive with
Lawson-Hanson for `nnls` and Stark-Parker for `lsq_linear` on the
problems they are used for; both of those are active-set methods with the
same subproblem and a more elaborate bookkeeping of which bound is
released when.

`ponytail:` the subproblem is solved through the normal equations, which
square `A`'s condition number -- fine for the well-conditioned designs
these are usually applied to, and the ceiling for an ill-conditioned one.
The upgrade is a QR-based free-set solve through `numax.linalg.lstsq`,
which the augmented Levenberg-Marquardt step in `least_squares` already
does for the unconstrained case. An unbounded `lsq_linear` *is*
`numax.linalg.lstsq`, and is not duplicated here.
"""

from std.math import sqrt as _sqrt

from max.gpu.host import DeviceContext

from ..core.array import Static, transpose
from ..linalg.blas import matmul, matvec

from .common import _as_tensor
from .minimize import _to_list


struct TensorLinearResult[dtype: DType, n: Int](Movable):
    """What `nnls` and `lsq_linear` return: the solution, `||A x - b||_2`
    at it, the iteration count and whether the projected gradient met the
    tolerance. Movable but not `Copyable`: `x` owns a `DeviceBuffer`."""

    var x: Static[Self.dtype, Self.n]
    var residual_norm: Float64
    var iterations: Int
    var converged: Bool

    def __init__(
        out self,
        var x: Static[Self.dtype, Self.n],
        residual_norm: Float64,
        iterations: Int,
        converged: Bool,
    ):
        self.x = x^
        self.residual_norm = residual_norm
        self.iterations = iterations
        self.converged = converged


def _quadratic(
    g: List[Float64], c: List[Float64], x: List[Float64], n: Int
) -> Float64:
    """`(1/2) x^T G x - c^T x`."""
    var total = 0.0
    for i in range(n):
        var row = 0.0
        for j in range(n):
            row += g[i * n + j] * x[j]
        total += 0.5 * x[i] * row - c[i] * x[i]
    return total


def _solve_free(
    g: List[Float64], rhs: List[Float64], free: List[Int], n: Int
) -> List[Float64]:
    """`d` with `G_FF d = rhs` on the free indices by Gaussian elimination
    with partial pivoting; `-rhs`'s direction (steepest descent) where
    that block is singular. Returned as an `n`-vector, zero off the free
    set."""
    var k = len(free)
    var system = List[Float64](length=k * k, fill=0.0)
    var b = List[Float64](capacity=k)
    for p in range(k):
        for q in range(k):
            system[p * k + q] = g[free[p] * n + free[q]]
        b.append(rhs[p])
    var out = List[Float64](length=n, fill=0.0)
    var scale = 0.0
    for e in range(k * k):
        scale = max(scale, abs(system[e]))
    for col in range(k):
        var pivot = col
        for row in range(col + 1, k):
            if abs(system[row * k + col]) > abs(system[pivot * k + col]):
                pivot = row
        if abs(system[pivot * k + col]) <= 1e-14 * scale:
            for p in range(k):
                out[free[p]] = rhs[p]
            return out^
        if pivot != col:
            for e in range(k):
                var tmp = system[col * k + e]
                system[col * k + e] = system[pivot * k + e]
                system[pivot * k + e] = tmp
            var t = b[col]
            b[col] = b[pivot]
            b[pivot] = t
        for row in range(col + 1, k):
            var factor = system[row * k + col] / system[col * k + col]
            if factor != 0:
                for e in range(col, k):
                    system[row * k + e] -= factor * system[col * k + e]
                b[row] -= factor * b[col]
    var i = k - 1
    while i >= 0:
        var acc = b[i]
        for e in range(i + 1, k):
            acc -= system[i * k + e] * b[e]
        b[i] = acc / system[i * k + i]
        i -= 1
    for p in range(k):
        out[free[p]] = b[p]
    return out^


def _box_qp(
    g: List[Float64],
    c: List[Float64],
    lower: List[Float64],
    upper: List[Float64],
    var x: List[Float64],
    n: Int,
    tol: Float64,
    max_iter: Int,
) -> Tuple[List[Float64], Int, Bool]:
    """Projected Newton on the free set for `min (1/2) x^T G x - c^T x`
    over `lower <= x <= upper`; the module docstring has the steps."""
    var scale = 1.0
    for i in range(n):
        scale = max(scale, abs(c[i]))
    for i in range(n):
        x[i] = min(max(x[i], lower[i]), upper[i])
    var value = _quadratic(g, c, x, n)

    for iteration in range(max_iter):
        var grad = List[Float64](length=n, fill=0.0)
        for i in range(n):
            var row = 0.0
            for j in range(n):
                row += g[i * n + j] * x[j]
            grad[i] = row - c[i]
        var free = List[Int]()
        var projected = 0.0
        for i in range(n):
            var at_lower = x[i] <= lower[i] and grad[i] > 0
            var at_upper = x[i] >= upper[i] and grad[i] < 0
            if not (at_lower or at_upper):
                free.append(i)
                projected = max(projected, abs(grad[i]))
        if projected <= tol * scale:
            return (x^, iteration, True)

        var negated = List[Float64](capacity=len(free))
        for p in range(len(free)):
            negated.append(-grad[free[p]])
        var direction = _solve_free(g, negated, free, n)

        var step = 1.0
        var accepted = False
        var candidate = List[Float64](length=n, fill=0.0)
        var next_value = value
        for _ in range(60):
            var slope = 0.0
            for i in range(n):
                candidate[i] = min(
                    max(x[i] + step * direction[i], lower[i]), upper[i]
                )
                slope += grad[i] * (candidate[i] - x[i])
            next_value = _quadratic(g, c, candidate, n)
            if next_value <= value + 1e-4 * slope and slope < 0:
                accepted = True
                break
            step = step / 2
        if not accepted:
            return (x^, iteration + 1, False)
        x = candidate^
        value = next_value
    return (x^, max_iter, False)


def _normal_equations[
    dtype: DType, m: Int, n: Int, gpu: Bool
](mut a: Static[dtype, m, n], mut b: Static[dtype, m]) raises -> Tuple[
    List[Float64], List[Float64]
]:
    """`(A^T A, A^T b)` formed on the device and read back."""
    var at = transpose[gpu=gpu](a)
    var at_again = transpose[gpu=gpu](a)
    var gram = matmul[dtype, n, m, n, gpu](at, a)
    var rhs = matvec[dtype, n, m, gpu](at_again, b)
    var g_host = gram.to_host()
    var g = List[Float64](capacity=n * n)
    for i in range(n * n):
        g.append(Float64(g_host[i]))
    return (g^, _to_list[dtype, n](rhs))


def _residual_norm[
    dtype: DType, m: Int, n: Int, gpu: Bool
](
    mut a: Static[dtype, m, n],
    mut b: Static[dtype, m],
    x: List[Float64],
    ctx: DeviceContext,
) raises -> Float64:
    var xs = _as_tensor[dtype, n](x, ctx)
    var ax = _to_list[dtype, m](matvec[dtype, m, n, gpu](a, xs))
    var bh = _to_list[dtype, m](b)
    var total = 0.0
    for i in range(m):
        var d = ax[i] - bh[i]
        total += d * d
    return _sqrt(total)


def lsq_linear[
    dtype: DType, m: Int, n: Int, gpu: Bool = False
](
    mut a: Static[dtype, m, n],
    mut b: Static[dtype, m],
    lower: Static[dtype, n],
    upper: Static[dtype, n],
    tol: Optional[Float64] = None,
    max_iter: Optional[Int] = None,
) raises -> TensorLinearResult[dtype, n] where (
    dtype.is_floating_point() and m >= 1 and n >= 1
):
    """`min ||A x - b||` over `lower <= x <= upper`, elementwise.
    `scipy.optimize.lsq_linear(A, b, bounds=(lower, upper))`.

    The module docstring has the algorithm. `tol` (default `1e-10`) is on
    the projected gradient relative to `max|A^T b|`; `max_iter` defaults
    to `50 n`. Infinite bounds are ordinary entries -- pass a large value
    for an unbounded side, or `numax.linalg.lstsq` for no bounds at all.
    """
    var ctx = a.context()
    var normal = _normal_equations[dtype, m, n, gpu](a, b)
    var lo = _to_list[dtype, n](lower)
    var hi = _to_list[dtype, n](upper)
    var start = List[Float64](length=n, fill=0.0)
    var solved = _box_qp(
        normal[0],
        normal[1],
        lo,
        hi,
        start^,
        n,
        tol.value() if tol else 1e-10,
        max_iter.value() if max_iter else 50 * n,
    )
    var norm = _residual_norm[dtype, m, n, gpu](a, b, solved[0], ctx)
    return TensorLinearResult[dtype, n](
        _as_tensor[dtype, n](solved[0], ctx), norm, solved[1], solved[2]
    )


def nnls[
    dtype: DType, m: Int, n: Int, gpu: Bool = False
](
    mut a: Static[dtype, m, n],
    mut b: Static[dtype, m],
    tol: Optional[Float64] = None,
    max_iter: Optional[Int] = None,
) raises -> TensorLinearResult[dtype, n] where (
    dtype.is_floating_point() and m >= 1 and n >= 1
):
    """`min ||A x - b||` over `x >= 0`. `scipy.optimize.nnls(A, b)`, whose
    `(x, rnorm)` pair is `.x` and `.residual_norm` here.

    `lsq_linear` with the box `[0, inf)`; the module docstring has the
    algorithm and the comparison with Lawson-Hanson.
    """
    var ctx = a.context()
    var normal = _normal_equations[dtype, m, n, gpu](a, b)
    var lo = List[Float64](length=n, fill=0.0)
    var hi = List[Float64](length=n, fill=1e300)
    var start = List[Float64](length=n, fill=0.0)
    var solved = _box_qp(
        normal[0],
        normal[1],
        lo,
        hi,
        start^,
        n,
        tol.value() if tol else 1e-10,
        max_iter.value() if max_iter else 50 * n,
    )
    var norm = _residual_norm[dtype, m, n, gpu](a, b, solved[0], ctx)
    return TensorLinearResult[dtype, n](
        _as_tensor[dtype, n](solved[0], ctx), norm, solved[1], solved[2]
    )
