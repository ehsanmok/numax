"""General solves and inverses: `scipy.linalg._basic`.

**The `Tensor` tier**, with `numax.linalg.array.basic` holding the
`FloatLike`-generic one -- unpivoted LU plus two substitutions for
`solve`, and a `pinv` over one-sided Jacobi where this one goes through
`numax.linalg.svd`.

`solve` factors with the blocked pivoted `lu_factor` -- whose trailing
update is MAX's GEMM -- and substitutes. It pivots, so it solves systems
the `Array` overload cannot start on. Tier 2. Reach for `lu_factor`
directly and reuse the `TensorLU` when there is more than one right-hand
side; this spelling throws the factorization away.

`inverse` factors once rather than calling `solve` `n` times, and sends
the `n` right-hand sides through the factorization together, so the
substitutions are `trsm` and the work between diagonal blocks is a GEMM.
Note that inverting explicitly is rarely the right move at any size:
solving against a specific right-hand side is both cheaper and better
conditioned.

`pinv` is `svd` followed by one `inner` product: `V diag(1/s) U^T` with
the small singular values dropped, which is what to reach for instead of
`inverse` when the matrix may be singular, rectangular, or both.

MAX ships no `solve`, no `inv` and no pseudo-inverse at any size, so
nothing here delegates.
"""

from layout import Coord
from layout.tile_layout import TensorLayout, row_major

from ..core.array import (
    Dynamic,
    Static,
    Tensor,
    _LayoutOf,
    _dyn_shape_from,
    eye,
)

from .blas import inner
from .eigen import svd
from .lu import lu_factor


def solve[
    dtype: DType, n: Int, gpu: Bool = False, block: Int = 16 if gpu else 32
](mut a: Static[dtype, n, n], mut b: Static[dtype, n]) raises -> Static[
    dtype, n
] where dtype.is_floating_point():
    """**Tier 2.** `x` with `a @ x == b`. `scipy.linalg.solve`.

    Factors with `lu_factor` -- blocked, partially pivoted, trailing update
    in MAX -- and substitutes. Call `lu_factor` directly and reuse the
    `TensorLU` when there is more than one right-hand side; this spelling
    throws the factorization away.

    Unlike the `Array[T, n*n]` sibling, this pivots, so it solves systems
    that one cannot start on. The trade is the generic `T`: picking a row
    by magnitude is a branch on data, so there is no conformer axis here.
    Both halves do run on the accelerator at `gpu=True` -- the
    factorization and the two substitutions alike, since `TensorLU` carries
    `gpu` in its type and cannot be solved against on the wrong device.
    """
    var factorization = lu_factor[dtype, n, gpu, block](a)
    return factorization.solve[block](b)


def inverse[
    dtype: DType, n: Int, gpu: Bool = False, block: Int = 16 if gpu else 32
](mut a: Static[dtype, n, n]) raises -> Static[
    dtype, n, n
] where dtype.is_floating_point():
    """**Tier 2.** `A^-1`, by factoring once and solving against the whole
    identity at once. `scipy.linalg.inv`.

    One `lu_factor` and one `TensorLU.solve` with `n` right-hand sides, so
    the substitutions are `trsm` rather than `n` separate `trsv`s: the
    update between diagonal blocks is a matrix product and goes to MAX's
    `matmul`. That is the difference from the `Array[T, n*n]` sibling,
    which substitutes column by column because at register residency
    there is nothing else to do.

    Inverting explicitly is still rarely the right move. `solve` against
    the right-hand side actually wanted is cheaper and better conditioned,
    and `cholesky_solve` cheaper again when the matrix is positive
    definite. This exists for the cases that genuinely need the entries of
    `A^-1` -- a covariance matrix's precision, for instance.
    """
    var factorization = lu_factor[dtype, n, gpu, block](a)
    var identity = eye[n, dtype](a.context())
    return factorization.solve[n, block](identity)


def pinv[
    dtype: DType, m: Int, n: Int, gpu: Bool = False
](mut a: Static[dtype, m, n], rcond: Float64 = 1e-15) raises -> Static[
    dtype, n, m
] where (dtype.is_floating_point() and m >= n and n >= 1):
    """**Tier 2.** The Moore-Penrose pseudoinverse, `V diag(1/s) U^T`, with
    singular values below `rcond` times the largest dropped.
    `numpy.linalg.pinv`, and `scipy.linalg.pinv`'s shape.

    **`gpu=True` does not compile**, since this is `svd` plus a reciprocal
    and `svd` refuses it.

    Reach for this instead of `inverse` when the matrix might be singular,
    nearly so, or not square at all: `inverse` solves against the identity
    through a pivoted LU and returns enormous garbage for a near-singular
    input, whereas here a singular direction contributes nothing rather
    than dominating. `n x m` for an `m x n` input, so `pinv(a) @ a` is the
    `n x n` identity when `a` has full column rank.

    `svd` device-resident, then `V` scaled column by column on the host --
    `n x n`, the small factor -- and one `inner` product against `U`,
    which is `linalg.matmul` reading `U` transposed in place. `rcond`
    defaults to NumPy's `1e-15`; the `Array` tier uses `1e-12`, because at
    a fixed Jacobi sweep count its small singular values carry more noise.
    """
    comptime assert not gpu, (
        "pinv: gpu=True is a known-wrong device path and is refused;"
        " run the default gpu=False. See the docstring."
    )
    var factored = svd[gpu=gpu](a)
    var s = factored.s.to_host()
    var threshold = Float64(s[0]) * rcond

    var v_scaled = factored.v.to_host()
    for j in range(n):
        var inv = Scalar[dtype](0)
        if Float64(s[j]) > threshold:
            inv = Scalar[dtype](1) / s[j]
        for i in range(n):
            v_scaled[i * n + j] = v_scaled[i * n + j] * inv
    var scaled = Static[dtype, n, n](a.context(), v_scaled^)
    return inner[gpu=gpu](scaled, factored.u)


def _svd_tolerance[
    dtype: DType
](s: List[Scalar[dtype]], m: Int, n: Int, rcond: Optional[Float64]) -> Float64:
    """The singular-value cutoff `orth` and `null_space` share.

    SciPy's default: the largest singular value times `max(m, n)` times the
    machine epsilon of `dtype`. An explicit `rcond` is used as a multiplier
    of the largest singular value instead, which is `pinv`'s convention in
    this module.
    """
    var largest = Float64(s[0])
    if rcond:
        return largest * rcond.value()
    # The same two constants `matrix_rank` uses, for the same reason: this
    # has to be the cutoff that routine reports a rank against.
    var eps: Float64
    comptime if dtype == DType.float32:
        eps = 1.1920929e-07
    else:
        eps = 2.220446049250313e-16
    return largest * Float64(max(m, n)) * eps


def orth[
    dtype: DType, m: Int, n: Int, gpu: Bool = False
](
    mut a: Static[dtype, m, n], rcond: Optional[Float64] = None
) raises -> Dynamic[dtype, 2] where (
    dtype.is_floating_point() and m >= n and n >= 1
):
    """An orthonormal basis for the range of `a`, as columns.
    `scipy.linalg.orth`.

    The columns of `U` whose singular value clears the tolerance, so the
    result is `m x rank`. **`rank` is a run-time fact**, which is why this
    returns a `Dynamic` where `svd` returns a statically shaped
    `TensorSVD`: the number of columns depends on the matrix's *values*,
    and nothing in the type system can see it. `matrix_rank` is the same
    count on its own.

    `rcond` defaults to SciPy's cutoff -- the largest singular value times
    `max(m, n)` times the machine epsilon -- and an explicit one is read as
    a multiplier of the largest singular value, matching `pinv` above.

    **Tier 2, and `gpu=True` does not compile**, since `svd` refuses it.
    """
    comptime assert not gpu, (
        "orth: gpu=True is a known-wrong device path and is refused;"
        " run the default gpu=False. See the docstring."
    )
    var factored = svd[gpu=gpu](a)
    var s = factored.s.to_host()
    var cutoff = _svd_tolerance(s, m, n, rcond)

    var keep = List[Int](capacity=n)
    for j in range(n):
        if Float64(s[j]) > cutoff:
            keep.append(j)

    var u = factored.u.to_host()
    var rank = len(keep)
    var values = List[Scalar[dtype]](length=m * rank, fill=0)
    for i in range(m):
        for c in range(rank):
            values[i * rank + c] = u[i * n + keep[c]]

    var extents = List[Int](capacity=2)
    extents.append(m)
    extents.append(rank)
    return Dynamic[dtype, 2](
        a.context(), row_major(_dyn_shape_from[2](extents)), values^
    )


def null_space[
    dtype: DType, m: Int, n: Int, gpu: Bool = False
](
    mut a: Static[dtype, m, n], rcond: Optional[Float64] = None
) raises -> Dynamic[dtype, 2] where (
    dtype.is_floating_point() and m >= n and n >= 1
):
    """An orthonormal basis for the null space of `a`, as columns.
    `scipy.linalg.null_space`.

    `orth`'s complement: the columns of `V` whose singular value falls
    *below* the tolerance, so the result is `n x (n - rank)` and is
    `Dynamic` for the same reason. `a @ null_space(a)` is zero to rounding.

    A full-rank `a` gives a basis with zero columns rather than an error,
    which is SciPy's behavior and the useful one -- the emptiness is the
    answer.
    """
    comptime assert not gpu, (
        "null_space: gpu=True is a known-wrong device path and is refused;"
        " run the default gpu=False. See the docstring."
    )
    var factored = svd[gpu=gpu](a)
    var s = factored.s.to_host()
    var cutoff = _svd_tolerance(s, m, n, rcond)

    var drop = List[Int](capacity=n)
    for j in range(n):
        if Float64(s[j]) <= cutoff:
            drop.append(j)

    # `v` holds the right singular vectors as columns, so a null-space
    # basis vector is a column of `v` -- no transpose, unlike SciPy, whose
    # `Vh` makes it a row.
    var v = factored.v.to_host()
    var nullity = len(drop)
    var values = List[Scalar[dtype]](length=n * nullity, fill=0)
    for i in range(n):
        for c in range(nullity):
            values[i * nullity + c] = v[i * n + drop[c]]

    var extents = List[Int](capacity=2)
    extents.append(n)
    extents.append(nullity)
    return Dynamic[dtype, 2](
        a.context(), row_major(_dyn_shape_from[2](extents)), values^
    )


struct Polar[dtype: DType, n: Int](
    Movable where dtype.is_floating_point() and n >= 1
):
    """`polar`'s result: `A = U P` with `U` orthogonal and `P` symmetric
    positive semidefinite.

    A struct rather than SciPy's `(u, p)` tuple, for the reason `TensorQR`
    and `TensorSVD` are structs: Mojo 1.0 cannot destructure a `Tuple` of
    two `Tensor`s.
    """

    var u: Static[Self.dtype, Self.n, Self.n]
    """The orthogonal factor -- the closest orthogonal matrix to `A` in
    the Frobenius norm, which is what makes `polar` an orthogonalizer."""

    var p: Static[Self.dtype, Self.n, Self.n]
    """The symmetric positive semidefinite factor, `sqrtm(A^T A)`."""

    def __init__(
        out self,
        var u: Static[Self.dtype, Self.n, Self.n],
        var p: Static[Self.dtype, Self.n, Self.n],
    ):
        self.u = u^
        self.p = p^


def polar[
    dtype: DType, n: Int, gpu: Bool = False
](mut a: Static[dtype, n, n]) raises -> Polar[dtype, n] where (
    dtype.is_floating_point() and n >= n and n >= 1
):
    """The right polar decomposition `a = U P`: `U` orthogonal, `P`
    symmetric positive semidefinite. `scipy.linalg.polar`, `side="right"`.

    The `n >= n` in the `where` clause is not a typo. `svd` requires
    `m >= n` and the prover does no arithmetic on evidence, so a square
    call has to restate that clause with both sides spelled `n` for it to
    be discharged (`findings.mdc`).

    From the SVD, which is how SciPy computes it too: with `a = W S V^T`,
    `U` is `W V^T` and `P` is `V S V^T`. Both are two products over the
    factors `svd` already returns, so this adds a name and no algorithm.

    Square only. SciPy takes a rectangular `a`, but the left and right
    factors then have different shapes and only one of them is square, so
    the general form wants a `side` parameter changing the return type --
    deferred until a caller needs it rather than guessed at.

    `sqrtm(a^T a)` is the same `P` by definition and is the slower route;
    `P` here never forms `a^T a`, so it does not square the condition
    number.

    **Tier 2, and `gpu=True` does not compile**, since `svd` refuses it.
    """
    comptime assert not gpu, (
        "polar: gpu=True is a known-wrong device path and is refused;"
        " run the default gpu=False. See the docstring."
    )
    var factored = svd[gpu=gpu](a)

    # U = W V^T, with `inner` supplying the transpose of its second
    # operand rather than materializing one.
    var orthogonal = inner[gpu=gpu](factored.u, factored.v)

    # P = V S V^T: scale V's columns by the singular values first, then
    # the same transposed product.
    var s = factored.s.to_host()
    var v_scaled = factored.v.to_host()
    for j in range(n):
        for i in range(n):
            v_scaled[i * n + j] = v_scaled[i * n + j] * s[j]
    var scaled = Static[dtype, n, n](a.context(), v_scaled^)
    var positive = inner[gpu=gpu](scaled, factored.v)

    return Polar[dtype, n](orthogonal^, positive^)


# ---------------------------------------------- tensorsolve and tensorinv


def _isqrt(n: Int) -> Int:
    """The integer square root, exact where `n` is a perfect square;
    evaluable at compile time."""
    if n <= 0:
        return 0
    var x = n
    var y = (x + 1) // 2
    while y < x:
        x = y
        y = (x + n // x) // 2
    return x


def _static_size[LayoutType: TensorLayout]() -> Int:
    """The element count of a compile-time shape, read off the layout
    *type* so it can fix a `where`-clause-sized matrix before any tensor
    exists."""
    return row_major(Coord[*LayoutType._shape_types]()).size()


def tensorsolve[
    dtype: DType,
    ALayout: TensorLayout,
    BLayout: TensorLayout,
    gpu: Bool = False,
](
    mut a: Tensor[dtype, ALayout], mut b: Tensor[dtype, BLayout]
) raises -> Dynamic[dtype, ALayout.rank - BLayout.rank] where (
    dtype.is_floating_point() and ALayout.rank - BLayout.rank >= 1
):
    """Solve `tensordot(a, x, axes=x.ndim) == b` for `x`.
    `numpy.linalg.tensorsolve(a, b)`.

    `a`'s shape is `b.shape + x.shape` with the two halves of equal
    element count `M`, so `a` read as `M x M` is a square system and `x`
    is its solution read at `a`'s trailing extents -- one `solve`, and
    therefore LAPACK-shaped LU on the device, with every reshape a
    retyping of the same buffer. `M` is the integer square root of `a`'s
    element count, fixed at compile time from its layout; `b`'s size and
    `a`'s leading extents are checked against it at run time.
    """
    comptime total = _static_size[ALayout]()
    comptime m = _isqrt(total)
    comptime assert (
        m * m == total
    ), "tensorsolve: a's element count must be a perfect square"
    comptime rank_b = BLayout.rank
    comptime rank_x = ALayout.rank - rank_b
    if b.size() != m:
        raise Error("tensorsolve: b has ", b.size(), " elements; a implies ", m)
    var leading = 1
    for d in range(rank_b):
        leading *= a.dim_at(d)
    if leading != m:
        raise Error(
            "tensorsolve: a's first ",
            rank_b,
            " extents multiply to ",
            leading,
            ", not the ",
            m,
            " its element count implies",
        )
    var square = Static[dtype, m, m](
        a.buffer.copy(),
        rebind[_LayoutOf[m, m]](row_major[m, m]()),
        a.host_addressable,
    )
    var rhs = Static[dtype, m](
        b.buffer.copy(),
        rebind[_LayoutOf[m]](row_major[m]()),
        b.host_addressable,
    )
    var x = solve[dtype, m, gpu](square, rhs)
    var extents = List[Int](capacity=rank_x)
    for d in range(rank_b, rank_b + rank_x):
        extents.append(a.dim_at(d))
    return Dynamic[dtype, rank_x](
        x.buffer.copy(),
        row_major(_dyn_shape_from[rank_x](extents)),
        x.host_addressable,
    )


def tensorinv[
    dtype: DType, ALayout: TensorLayout, ind: Int = 2, gpu: Bool = False
](mut a: Tensor[dtype, ALayout]) raises -> Dynamic[dtype, ALayout.rank] where (
    dtype.is_floating_point() and ind >= 1 and ind < ALayout.rank
):
    """The inverse of `a` with respect to `tensordot` at `ind` axes:
    `tensordot(tensorinv(a), a, ind)` is the identity.
    `numpy.linalg.tensorinv(a, ind)`.

    `a`'s first `ind` extents and its remaining ones must each multiply to
    the same `M` -- `M` fixed at compile time as the square root of the
    element count, the split checked at run time; `a` read as `M x M` is
    inverted with `inverse` and the result read back with the two halves
    of the shape swapped, so it contracts against `a`'s leading axes.
    """
    comptime total = _static_size[ALayout]()
    comptime m = _isqrt(total)
    comptime assert (
        m * m == total
    ), "tensorinv: a's element count must be a perfect square"
    comptime rank = ALayout.rank
    var leading = 1
    for d in range(ind):
        leading *= a.dim_at(d)
    if leading != m:
        raise Error(
            "tensorinv: the first ",
            ind,
            " extents multiply to ",
            leading,
            ", not the ",
            m,
            " the element count implies",
        )
    var square = Static[dtype, m, m](
        a.buffer.copy(),
        rebind[_LayoutOf[m, m]](row_major[m, m]()),
        a.host_addressable,
    )
    var inv = inverse[dtype, m, gpu](square)
    var extents = List[Int](capacity=rank)
    for d in range(ind, rank):
        extents.append(a.dim_at(d))
    for d in range(ind):
        extents.append(a.dim_at(d))
    return Dynamic[dtype, rank](
        inv.buffer.copy(),
        row_major(_dyn_shape_from[rank](extents)),
        inv.host_addressable,
    )
