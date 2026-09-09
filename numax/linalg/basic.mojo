"""General solves and inverses: `scipy.linalg._basic`.

**The `Tensor` tier**, with `numax.linalg.array.basic` holding the
`FloatLike`-generic one -- unpivoted LU plus two substitutions for
`solve`, and `pinv`, which has no `Tensor` form because `svd` does not.

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

MAX ships no `solve`, no `inv` and no pseudo-inverse at any size, so
nothing here delegates.
"""

from ..core.array import Static, eye

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
