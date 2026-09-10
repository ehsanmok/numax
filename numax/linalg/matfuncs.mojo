"""Functions *of* a matrix: `scipy.linalg`'s `_matfuncs`.

Not elementwise. `expm(A)` is the matrix whose power series is
`I + A + A^2/2 + ...`, which is a different object from `exp` applied entry
by entry -- that one is `numax.core.elementwise.exp` and agrees with this
only when `A` is diagonal.

## What is here

`expm` at both tiers, by scaling and squaring with a degree-13 Pade
approximant. This is the algorithm every serious implementation uses, and
it is a good fit for numax: the whole of it is matrix products plus one
solve, so at the `Tensor` tier the cubic term goes straight to
`linalg.matmul` and this is an **extend** in MAX's own idiom.

The two tiers differ in one decision and it is the interesting one. The
`Tensor` overload chooses its squaring count from `norm(A)`, which is a
branch on data and makes it **tier 2**. The `Array` overload takes the
count as a compile-time parameter instead, so its work is fixed and it is
**tier 1** -- it differentiates at `Dual` and runs inside a GPU kernel body.
That is a property SciPy has no way to offer, and it costs the caller a
parameter they have to choose.

## What is deferred, and what it is waiting for

`logm`, `sqrtm` for a general matrix, `funm`, `fractional_matrix_power`,
`cosm`/`sinm`/`tanm`, `expm_frechet` and `expm_cond` are **not here**, and
they are all waiting on the same thing: a **Schur decomposition**. Every one
of them is "reduce to a triangular form, apply the scalar function to the
diagonal, then recur up the superdiagonals" -- and numax has no `schur`. It
has no `hessenberg` either, except as a private step inside
`numax/linalg/array/eigen.mojo`'s `eigvals`.

That is not a small gap and it is the same one `numax/linalg/__init__.mojo`
records for the spectral four: a real Schur form's reduction phase is the
block reflector `qr_factor` already runs, but its iterative phase is a
sequential QR sweep with data-dependent deflation, which no GEMM helps.
`expm` is the one matrix function that needs none of it, which is exactly
why it is the one that ships.

When this resumes the first commit is `schur` on top of that reduction, and
`logm` and `sqrtm` follow immediately -- they are short once the triangular
form exists.

`sqrtm` for a **symmetric positive definite** matrix needs no Schur form at
all, since `eigh` already diagonalizes one, and it lives in
`numax.linalg.array.matfuncs` beside its tier.
"""

from std.math import ceil as _ceil, log2 as _log2

from ..core.array import Static, copy, eye
from ..core.ops import add, multiply, subtract

from .blas import matmul
from .lu import lu_factor
from .misc import norm


comptime _B0 = 64764752532480000.0
comptime _B1 = 32382376266240000.0
comptime _B2 = 7771770303897600.0
comptime _B3 = 1187353796428800.0
comptime _B4 = 129060195264000.0
comptime _B5 = 10559470521600.0
comptime _B6 = 670442572800.0
comptime _B7 = 33522128640.0
comptime _B8 = 1323241920.0
comptime _B9 = 40840800.0
comptime _B10 = 960960.0
comptime _B11 = 16380.0
comptime _B12 = 182.0
comptime _B13 = 1.0
"""The degree-13 Pade coefficients, Higham's.

Named one by one rather than held in a `comptime Array`, because indexing
one of those from runtime code fails with "cannot materialize comptime
value ... it is not `ImplicitlyCopyable`" -- the same limitation
`findings.mdc` records for a runtime loop over a comptime table. Fourteen
names is the cheaper workaround here, since every coefficient is used
exactly once at a known position.

Only their *ratios* matter: the approximant is `(V - U)^-1 (V + U)` and
scaling `U` and `V` by the same constant leaves that unchanged, which is
why coefficients past `2^53` are not a problem despite not being exactly
representable."""


comptime _THETA13 = 5.371920351148152
"""The largest `||A||_1` for which the degree-13 Pade approximant has a
backward error below `float64`'s unit roundoff. Higham (2005). Scaling `A`
until its norm is under this is what "scaling and squaring" means, and it
is why the result is accurate to working precision rather than to the
approximant's own truncation error."""


def expm[
    dtype: DType, n: Int, gpu: Bool = False, block: Int = 16 if gpu else 32
](mut a: Static[dtype, n, n]) raises -> Static[dtype, n, n] where (
    dtype.is_floating_point() and n >= 1
):
    """The matrix exponential of `a`. `scipy.linalg.expm`.

    **Tier 2**: the squaring count comes from `norm(a)`, which is a branch
    on data. `numax.linalg.array.expm` is the tier-1 sibling that takes the
    count as a parameter instead and therefore differentiates.

    Scaling and squaring with a degree-13 Pade approximant, which is
    Higham's algorithm and what SciPy runs:

    1. Pick `s` so that `||a / 2^s||_1 <= 5.372`, the threshold at which the
       approximant's backward error drops below unit roundoff.
    2. Evaluate the approximant on the scaled matrix. Six matrix products
       and one solve, arranged so the even and odd halves share the powers
       `A^2`, `A^4` and `A^6` -- that arrangement is the reason degree 13 is
       affordable at all.
    3. Square the result `s` times, undoing the scaling.

    **The accuracy claim, and its limit.** The backward error is below
    `2^-53` -- that is what `_THETA13` is chosen for -- so the computed
    result is the exact exponential of a matrix within rounding of `a`. That
    is a *backward* statement, and it is the honest one: for a matrix whose
    exponential is ill-conditioned, a tiny backward error still permits a
    large forward one, and no algorithm avoids that. The condition number is
    what `scipy.linalg.expm_cond` reports, which numax does not have.

    Every product goes to `linalg.matmul` and the solve to `lu_factor`, so
    the cubic work is MAX's and the matrix stays on its device throughout.

    `expm(a) @ expm(-a)` is the identity to rounding, which the tests check;
    `expm(a + b) == expm(a) @ expm(b)` is **not** generally true and holds
    only when `a` and `b` commute.
    """
    var ctx = a.context()

    # 1. Scale.
    var magnitude = Float64(norm[dtype, n, 1, gpu](a))
    var squarings = 0
    if magnitude > _THETA13:
        squarings = Int(_ceil(_log2(magnitude / _THETA13)))
    var scale = Scalar[dtype](1.0 / Float64(1 << squarings))
    var scaled = multiply(a, scale)

    # 2. The approximant. `a2`, `a4` and `a6` are shared by both halves,
    # which is what makes degree 13 cost six products rather than thirteen.
    # `matmul` takes both operands mutably -- a writable `TileTensor` view
    # cannot be built from an immutable binding -- and Mojo will not pass
    # one binding through two `mut` arguments, so a squaring needs a second
    # named copy of the same matrix. `numax.core.array.copy` is the explicit
    # spelling; `matrix_power` in `blas.mojo` pays the same cost.
    var scaled_again = copy(scaled)
    var a2 = matmul[dtype, n, n, n, gpu](scaled, scaled_again)
    var a2_again = copy(a2)
    var a4 = matmul[dtype, n, n, n, gpu](a2, a2_again)
    var a2_third = copy(a2)
    var a6 = matmul[dtype, n, n, n, gpu](a4, a2_third)
    var identity = eye[n, dtype](ctx)

    var odd_inner = add(
        add(
            multiply(a6, Scalar[dtype](_B13)),
            multiply(a4, Scalar[dtype](_B11)),
        ),
        multiply(a2, Scalar[dtype](_B9)),
    )
    var odd_outer = add(
        add(
            add(
                multiply(a6, Scalar[dtype](_B7)),
                multiply(a4, Scalar[dtype](_B5)),
            ),
            multiply(a2, Scalar[dtype](_B3)),
        ),
        multiply(identity, Scalar[dtype](_B1)),
    )
    var odd_part = add(matmul[dtype, n, n, n, gpu](a6, odd_inner), odd_outer)
    var u = matmul[dtype, n, n, n, gpu](scaled, odd_part)

    var even_inner = add(
        add(
            multiply(a6, Scalar[dtype](_B12)),
            multiply(a4, Scalar[dtype](_B10)),
        ),
        multiply(a2, Scalar[dtype](_B8)),
    )
    var even_outer = add(
        add(
            add(
                multiply(a6, Scalar[dtype](_B6)),
                multiply(a4, Scalar[dtype](_B4)),
            ),
            multiply(a2, Scalar[dtype](_B2)),
        ),
        multiply(identity, Scalar[dtype](_B0)),
    )
    var v = add(matmul[dtype, n, n, n, gpu](a6, even_inner), even_outer)

    # `(V - U) r = V + U`, solved against the whole right-hand side at once
    # so the update between diagonal blocks is a GEMM rather than `n`
    # separate `gemv`s.
    var denominator = subtract(v, u)
    var numerator = add(v, u)
    var factored = lu_factor[dtype, n, gpu, block](denominator)
    var result = factored.solve[n, block](numerator)

    # 3. Undo the scaling.
    for _ in range(squarings):
        var mirror = copy(result)
        result = matmul[dtype, n, n, n, gpu](result, mirror)
    return result^
