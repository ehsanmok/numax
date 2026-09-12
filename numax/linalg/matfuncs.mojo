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

## What is here beyond `expm`, and how

Everything else is built on the real Schur form `numax.linalg.schur`
returns, `A = Z T Z^T` with `T` quasi-triangular -- upper triangular
except for `2 x 2` diagonal blocks, each holding one complex conjugate
pair -- and every function below is "apply the scalar function to `T`,
then `Z f(T) Z^T`": two `matmul`s on the device around a host recurrence
over `T`, which is `O(n^3)` scalar work at a small constant and the same
kind of ceiling `schur` itself names.

- `funm(a, f)` is the block Parlett recurrence: `f` on each diagonal block
  -- a `1 x 1` block directly, a `2 x 2` block `B = mu I + nu J` with `J^2 =
  -I` as `Re f(mu + i nu) I + Im f(mu + i nu) J`, evaluating `f` over
  `Complex[Plain]` so the same `FloatLike` function serves both -- and then
  `T_ii F_ij - F_ij T_jj = F_ii T_ij - T_ij F_jj + sum_k (F_ik T_kj - T_ik
  F_kj)` for the blocks above the diagonal, each a Sylvester equation of
  order at most four. That recurrence divides by eigenvalue differences,
  so two Schur blocks with the *same* eigenvalue and a nonzero coupling
  between them (a Jordan block) leave it singular, and `funm` raises
  rather than return the `inf`s SciPy's does; a repeated eigenvalue with
  zero coupling (a diagonal matrix, `2 I`) is fine. The upgrade is the
  Schur-Parlett algorithm with eigenvalue clustering and Taylor blocks
  (Davies and Higham), which is what SciPy does not do either.
- `sqrtm` is not `funm(sqrt)`: it is the Bjorck-Hammarling block
  recurrence `U_ii U_ij + U_ij U_jj = T_ij - sum_k U_ik U_kj` with `U_ii =
  sqrt(T_ii)`, whose denominators are sums of principal square roots and
  never vanish, so it handles repeated and defective eigenvalues where the
  Parlett recurrence cannot -- `sqrtm([[4, 1], [0, 4]])` is `[[2, 1/4],
  [0, 2]]`. Higham's algorithm, and SciPy's.
- `logm` is inverse scaling and squaring on `T`: square roots by that
  same recurrence until `||T^{1/2^k} - I||_1 <= 1/4`, the degree-8 Pade
  approximant of `log(I + X)` in its partial-fraction form (eight solves
  against `I + x_j X` at the Gauss-Legendre nodes), and `2^k` times the
  result. Higham's inverse scaling and squaring without the sharper
  degree selection; the ceiling is the `1/4` threshold, which costs one
  or two more square roots than the optimal choice would.
- `cosm` and `sinm` are `funm` at `cos` and `sin`;
  `fractional_matrix_power(a, t)` is `expm(t logm(a))`.

The real forms are what these return, so a matrix with an eigenvalue on
the closed negative real axis -- where `sqrt` and `log` are complex --
comes back with NaN in it from the scalar `sqrt`/`ln` at `Plain`, where
SciPy switches to a complex result. `expm_frechet` and `expm_cond` are
still not here.

`sqrtm` for a **symmetric positive definite** matrix needs no Schur form
at all, since `eigh` already diagonalizes one, and it lives in
`numax.linalg.array.matfuncs` beside its tier.
"""

from std.collections import Array
from std.math import ceil as _ceil, log2 as _log2, sqrt as _sqrt

from max.gpu.host import DeviceContext

from ..core.array import Static, copy, eye, transpose
from ..core.complex import Complex
from ..core.numeric import FloatLike
from ..core.ops import add, multiply, subtract
from ..core.plain import Plain

from .blas import matmul
from .eigen import schur
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


# ------------------------------------------------- Schur-based functions

comptime _GL8_X: Array[Float64, 8] = [
    0.0198550717512318842,
    0.10166676129318663,
    0.237233795041835507,
    0.408282678752175098,
    0.591717321247824902,
    0.762766204958164493,
    0.89833323870681337,
    0.980144928248768116,
]
comptime _GL8_W: Array[Float64, 8] = [
    0.0506142681451881296,
    0.111190517226687235,
    0.156853322938943644,
    0.181341891689180991,
    0.181341891689180991,
    0.156853322938943644,
    0.111190517226687235,
    0.0506142681451881296,
]
"""Eight-point Gauss-Legendre nodes and weights on `[0, 1]`: the
partial-fraction form of the degree-8 Pade approximant of `log(1 + x)` is
`sum_j w_j x / (1 + x_j x)`."""

comptime _LOGM_THRESHOLD = 0.25
"""`||T^{1/2^k} - I||_1` at which the square-rooting stops; the degree-8
approximant is at unit roundoff below about `0.34`."""

comptime _MAX_SQUARE_ROOTS = 60


def _block_starts[dtype: DType](t: List[Scalar[dtype]], n: Int) -> List[Int]:
    """The first row of every diagonal block of a real Schur form: a `2 x
    2` block wherever the subdiagonal entry is nonzero, `1 x 1` elsewhere.
    """
    var starts = List[Int]()
    var i = 0
    while i < n:
        starts.append(i)
        if i + 1 < n and t[(i + 1) * n + i] != 0:
            i += 2
        else:
            i += 1
    return starts^


def _take[
    dtype: DType
](
    t: List[Scalar[dtype]], n: Int, i0: Int, j0: Int, rows: Int, cols: Int
) -> List[Scalar[dtype]]:
    var out = List[Scalar[dtype]](capacity=rows * cols)
    for i in range(rows):
        for j in range(cols):
            out.append(t[(i0 + i) * n + (j0 + j)])
    return out^


def _put[
    dtype: DType
](
    mut t: List[Scalar[dtype]],
    n: Int,
    i0: Int,
    j0: Int,
    rows: Int,
    cols: Int,
    block: List[Scalar[dtype]],
):
    for i in range(rows):
        for j in range(cols):
            t[(i0 + i) * n + (j0 + j)] = block[i * cols + j]


def _mul_small[
    dtype: DType
](
    a: List[Scalar[dtype]],
    ra: Int,
    ca: Int,
    b: List[Scalar[dtype]],
    cb: Int,
) -> List[Scalar[dtype]]:
    var out = List[Scalar[dtype]](length=ra * cb, fill=0)
    for i in range(ra):
        for j in range(cb):
            var acc = Scalar[dtype](0)
            for k in range(ca):
                acc += a[i * ca + k] * b[k * cb + j]
            out[i * cb + j] = acc
    return out^


def _sylvester_small[
    dtype: DType
](
    a: List[Scalar[dtype]],
    r: Int,
    b: List[Scalar[dtype]],
    c: Int,
    rhs: List[Scalar[dtype]],
    sign: Scalar[dtype],
    coupled: Bool,
) raises -> List[Scalar[dtype]]:
    """Solve `A X + sign X B = C` for the `r x c` block `X`, `r, c <= 2`, as
    the `rc x rc` linear system on `vec(X)` by Gaussian elimination with
    partial pivoting. A singular system means the two blocks share an
    eigenvalue: where nothing couples them (`coupled` is false -- `T_ij`
    and every `T_ik`, `T_kj` between are zero) the solution is `X = 0`;
    where something does, this is the Jordan-block case the Parlett
    recurrence cannot serve, since `f'` would be needed, and it raises.
    """
    var m = r * c
    var system = List[Scalar[dtype]](length=m * m, fill=0)
    var vec = List[Scalar[dtype]](capacity=m)
    var scale = Scalar[dtype](0)
    for p in range(r):
        for q in range(c):
            var row = p * c + q
            for k in range(r):
                system[row * m + (k * c + q)] += a[p * r + k]
            for k in range(c):
                system[row * m + (p * c + k)] += sign * b[k * c + q]
            vec.append(rhs[p * c + q])
    for i in range(m * m):
        scale = max(scale, abs(system[i]))
    var tiny = Scalar[dtype](1e-13) * (scale if scale > 0 else Scalar[dtype](1))

    for col in range(m):
        var pivot = col
        for row in range(col + 1, m):
            if abs(system[row * m + col]) > abs(system[pivot * m + col]):
                pivot = row
        if abs(system[pivot * m + col]) <= tiny:
            if not coupled:
                return List[Scalar[dtype]](length=m, fill=0)
            raise Error(
                "funm: two Schur blocks share an eigenvalue and are coupled"
                " (a Jordan block); the Parlett recurrence is singular there"
            )
        if pivot != col:
            for k in range(m):
                var tmp = system[col * m + k]
                system[col * m + k] = system[pivot * m + k]
                system[pivot * m + k] = tmp
            var t = vec[col]
            vec[col] = vec[pivot]
            vec[pivot] = t
        for row in range(col + 1, m):
            var factor = system[row * m + col] / system[col * m + col]
            if factor != 0:
                for k in range(col, m):
                    system[row * m + k] -= factor * system[col * m + k]
                vec[row] -= factor * vec[col]
    var x = List[Scalar[dtype]](length=m, fill=0)
    var i = m - 1
    while i >= 0:
        var acc = vec[i]
        for k in range(i + 1, m):
            acc -= system[i * m + k] * x[k]
        x[i] = acc / system[i * m + i]
        i -= 1
    return x^


def _complex_block[
    dtype: DType, f: def[T: FloatLike](T) thin -> T
](block: List[Scalar[dtype]]) -> List[
    Scalar[dtype]
] where dtype.is_floating_point():
    """`f` of a `2 x 2` Schur block with a complex pair `mu +- i nu`: `B =
    mu I + nu J` with `J^2 = -I`, so `f(B) = Re f(lambda) I + Im f(lambda) J`
    with `f` evaluated once over `Complex[Plain]`."""
    comptime P = Plain[dtype, 1]
    var a = block[0]
    var b = block[1]
    var c = block[2]
    var d = block[3]
    var mu = (a + d) / Scalar[dtype](2)
    var disc = (a - d) * (a - d) + Scalar[dtype](4) * b * c
    var nu = _sqrt(-disc) / Scalar[dtype](2)
    var value = f[Complex[P]](Complex[P](P(mu), P(nu)))
    var re = value.re.v[0]
    var im = value.im.v[0]
    var ratio = im / nu
    var out = List[Scalar[dtype]](capacity=4)
    out.append(re + ratio * (a - mu))
    out.append(ratio * b)
    out.append(ratio * c)
    out.append(re + ratio * (d - mu))
    return out^


def _funm_host[
    dtype: DType, f: def[T: FloatLike](T) thin -> T
](t: List[Scalar[dtype]], n: Int) raises -> List[
    Scalar[dtype]
] where dtype.is_floating_point():
    """The block Parlett recurrence over the real Schur form `t`."""
    comptime P = Plain[dtype, 1]
    var starts = _block_starts(t, n)
    var m = len(starts)
    var sizes = List[Int](capacity=m)
    for k in range(m):
        var next = starts[k + 1] if k + 1 < m else n
        sizes.append(next - starts[k])
    var out = List[Scalar[dtype]](length=n * n, fill=0)
    for k in range(m):
        var i0 = starts[k]
        if sizes[k] == 1:
            out[i0 * n + i0] = f[P](P(t[i0 * n + i0])).v[0]
        else:
            _put(
                out,
                n,
                i0,
                i0,
                2,
                2,
                _complex_block[dtype, f](_take(t, n, i0, i0, 2, 2)),
            )
    for sep in range(1, m):
        for bi in range(m - sep):
            var bj = bi + sep
            var i0 = starts[bi]
            var j0 = starts[bj]
            var r = sizes[bi]
            var c = sizes[bj]
            var t_ij = _take(t, n, i0, j0, r, c)
            var coupled = False
            for e in range(r * c):
                if t_ij[e] != 0:
                    coupled = True
            for bk in range(bi + 1, bj):
                var k0 = starts[bk]
                var kk = sizes[bk]
                var t_ik = _take(t, n, i0, k0, r, kk)
                var t_kj = _take(t, n, k0, j0, kk, c)
                for e in range(r * kk):
                    if t_ik[e] != 0:
                        coupled = True
                for e in range(kk * c):
                    if t_kj[e] != 0:
                        coupled = True
            var f_ii = _take(out, n, i0, i0, r, r)
            var f_jj = _take(out, n, j0, j0, c, c)
            var rhs = _mul_small(f_ii, r, r, t_ij, c)
            var second = _mul_small(t_ij, r, c, f_jj, c)
            for e in range(r * c):
                rhs[e] -= second[e]
            for bk in range(bi + 1, bj):
                var k0 = starts[bk]
                var kk = sizes[bk]
                var left = _mul_small(
                    _take(out, n, i0, k0, r, kk),
                    r,
                    kk,
                    _take(t, n, k0, j0, kk, c),
                    c,
                )
                var right = _mul_small(
                    _take(t, n, i0, k0, r, kk),
                    r,
                    kk,
                    _take(out, n, k0, j0, kk, c),
                    c,
                )
                for e in range(r * c):
                    rhs[e] += left[e] - right[e]
            var x = _sylvester_small(
                _take(t, n, i0, i0, r, r),
                r,
                _take(t, n, j0, j0, c, c),
                c,
                rhs,
                Scalar[dtype](-1),
                coupled,
            )
            _put(out, n, i0, j0, r, c, x)
    return out^


def _sqrt_f[T: FloatLike](x: T) -> T:
    return x.sqrt()


def _sqrtm_host[
    dtype: DType
](t: List[Scalar[dtype]], n: Int) raises -> List[
    Scalar[dtype]
] where dtype.is_floating_point():
    """The Bjorck-Hammarling recurrence for the principal square root of a
    real Schur form: `U_ii = sqrt(T_ii)`, then `U_ii U_ij + U_ij U_jj = T_ij
    - sum_{i<k<j} U_ik U_kj` block by block. The result is again quasi-
    triangular, which is what lets `logm` iterate it."""
    var starts = _block_starts(t, n)
    var m = len(starts)
    var sizes = List[Int](capacity=m)
    for k in range(m):
        var next = starts[k + 1] if k + 1 < m else n
        sizes.append(next - starts[k])
    var out = List[Scalar[dtype]](length=n * n, fill=0)
    for k in range(m):
        var i0 = starts[k]
        if sizes[k] == 1:
            out[i0 * n + i0] = _sqrt(t[i0 * n + i0])
        else:
            _put(
                out,
                n,
                i0,
                i0,
                2,
                2,
                _complex_block[dtype, _sqrt_f](_take(t, n, i0, i0, 2, 2)),
            )
    for sep in range(1, m):
        for bi in range(m - sep):
            var bj = bi + sep
            var i0 = starts[bi]
            var j0 = starts[bj]
            var r = sizes[bi]
            var c = sizes[bj]
            var rhs = _take(t, n, i0, j0, r, c)
            for bk in range(bi + 1, bj):
                var k0 = starts[bk]
                var kk = sizes[bk]
                var prod = _mul_small(
                    _take(out, n, i0, k0, r, kk),
                    r,
                    kk,
                    _take(out, n, k0, j0, kk, c),
                    c,
                )
                for e in range(r * c):
                    rhs[e] -= prod[e]
            var x = _sylvester_small(
                _take(out, n, i0, i0, r, r),
                r,
                _take(out, n, j0, j0, c, c),
                c,
                rhs,
                Scalar[dtype](1),
                True,
            )
            _put(out, n, i0, j0, r, c, x)
    return out^


def _solve_dense_host[
    dtype: DType
](a: List[Scalar[dtype]], b: List[Scalar[dtype]], n: Int) raises -> List[
    Scalar[dtype]
]:
    """`X` with `A X = B`, both `n x n`, by Gaussian elimination with
    partial pivoting on the host."""
    var lu = a.copy()
    var x = b.copy()
    for col in range(n):
        var pivot = col
        for row in range(col + 1, n):
            if abs(lu[row * n + col]) > abs(lu[pivot * n + col]):
                pivot = row
        if lu[pivot * n + col] == 0:
            raise Error("logm: singular system in the Pade approximant")
        if pivot != col:
            for k in range(n):
                var t1 = lu[col * n + k]
                lu[col * n + k] = lu[pivot * n + k]
                lu[pivot * n + k] = t1
                var t2 = x[col * n + k]
                x[col * n + k] = x[pivot * n + k]
                x[pivot * n + k] = t2
        for row in range(col + 1, n):
            var factor = lu[row * n + col] / lu[col * n + col]
            if factor != 0:
                for k in range(col, n):
                    lu[row * n + k] -= factor * lu[col * n + k]
                for k in range(n):
                    x[row * n + k] -= factor * x[col * n + k]
    var i = n - 1
    while i >= 0:
        for k in range(n):
            var acc = x[i * n + k]
            for j in range(i + 1, n):
                acc -= lu[i * n + j] * x[j * n + k]
            x[i * n + k] = acc / lu[i * n + i]
        i -= 1
    return x^


def _logm_host[
    dtype: DType
](t: List[Scalar[dtype]], n: Int) raises -> List[
    Scalar[dtype]
] where dtype.is_floating_point():
    """Inverse scaling and squaring on the real Schur form `t`: square
    roots until `||T - I||_1 <= 1/4`, the degree-8 Pade approximant of
    `log(I + X)` as `sum_j w_j X (I + x_j X)^{-1}`, then times `2^k`."""
    var s = t.copy()
    var k = 0
    while k < _MAX_SQUARE_ROOTS:
        var distance = Scalar[dtype](0)
        for j in range(n):
            var column = Scalar[dtype](0)
            for i in range(n):
                var value = s[i * n + j]
                if i == j:
                    value -= Scalar[dtype](1)
                column += abs(value)
            distance = max(distance, column)
        if distance <= Scalar[dtype](_LOGM_THRESHOLD):
            break
        s = _sqrtm_host(s, n)
        k += 1
    var x = s.copy()
    for i in range(n):
        x[i * n + i] -= Scalar[dtype](1)
    var out = List[Scalar[dtype]](length=n * n, fill=0)
    comptime for j in range(8):
        comptime node = _GL8_X[j]
        comptime weight = _GL8_W[j]
        var system = x.copy()
        for e in range(n * n):
            system[e] = system[e] * Scalar[dtype](node)
        for i in range(n):
            system[i * n + i] += Scalar[dtype](1)
        var term = _solve_dense_host(system, x, n)
        for e in range(n * n):
            out[e] += Scalar[dtype](weight) * term[e]
    var factor = Scalar[dtype](Float64(1 << k))
    for e in range(n * n):
        out[e] *= factor
    return out^


def _similar[
    dtype: DType, n: Int, gpu: Bool
](
    mut z: Static[dtype, n, n], f_host: List[Scalar[dtype]], ctx: DeviceContext
) raises -> Static[dtype, n, n]:
    """`Z F Z^T` on the device, the step every Schur-based function ends
    with."""
    var f = Static[dtype, n, n](ctx, f_host.copy())
    var zt = transpose[gpu=gpu](z)
    var half = matmul[dtype, n, n, n, gpu](z, f)
    return matmul[dtype, n, n, n, gpu](half, zt)


def funm[
    dtype: DType, n: Int, f: def[T: FloatLike](T) thin -> T, gpu: Bool = False
](mut a: Static[dtype, n, n]) raises -> Static[dtype, n, n] where (
    dtype.is_floating_point() and n >= 1
):
    """`f(a)` for a scalar `FloatLike` function `f`, as a function of the
    matrix. `scipy.linalg.funm(a, f)`.

    **Tier 2.** `schur` reduces `a` to real Schur form with the cubic term
    in `linalg.matmul`, the block Parlett recurrence evaluates `f` on the
    quasi-triangular factor on the host -- `f` itself at `Plain` on a `1 x
    1` block and over `Complex[Plain]` on a `2 x 2` one -- and `Z f(T) Z^T`
    returns through two `matmul`s. The module docstring has the recurrence
    and its one singularity: two coupled Schur blocks with the same
    eigenvalue raise.

    `f` is any `def[T: FloatLike](T) -> T`, so a kernel written for
    `numax.special` serves here unchanged:

    ```mojo
    def my_f[T: FloatLike](x: T) -> T:
        return x.exp() * x
    var b = funm[f=my_f](a)
    ```
    """
    var ctx = a.context()
    var decomposed = schur[dtype, n, gpu](a)
    var t = decomposed.t.to_host()
    var f_t = _funm_host[dtype, f](t, n)
    return _similar[dtype, n, gpu](decomposed.z, f_t, ctx)


def sqrtm[
    dtype: DType, n: Int, gpu: Bool = False
](mut a: Static[dtype, n, n]) raises -> Static[dtype, n, n] where (
    dtype.is_floating_point() and n >= 1
):
    """The principal matrix square root, `X @ X == a`. `scipy.linalg.sqrtm`.

    **Tier 2.** `schur`, then the Bjorck-Hammarling recurrence on the real
    Schur form -- Higham's algorithm, whose denominators are sums of
    principal square roots and never vanish, so repeated and defective
    eigenvalues are fine -- then `Z U Z^T`. Real for a matrix with no
    eigenvalue on the closed negative real axis; where there is one the
    scalar `sqrt` at `Plain` is NaN and so is the result, which is at least
    loud. `numax.linalg.array.sqrtm` is the symmetric positive definite
    route through `eigh` for matrices small enough to live in registers.
    """
    var ctx = a.context()
    var decomposed = schur[dtype, n, gpu](a)
    var t = decomposed.t.to_host()
    var u = _sqrtm_host(t, n)
    return _similar[dtype, n, gpu](decomposed.z, u, ctx)


def logm[
    dtype: DType, n: Int, gpu: Bool = False
](mut a: Static[dtype, n, n]) raises -> Static[dtype, n, n] where (
    dtype.is_floating_point() and n >= 1
):
    """The principal matrix logarithm, `expm(logm(a)) == a`.
    `scipy.linalg.logm`.

    **Tier 2.** `schur`, then inverse scaling and squaring on the real
    Schur form -- square roots by the `sqrtm` recurrence until the factor
    is within `1/4` of the identity, the degree-8 Pade approximant of
    `log(I + X)`, and `2^k` times that -- then `Z L Z^T`. The module
    docstring has the ceiling. Real for a matrix with no eigenvalue on the
    closed negative real axis, NaN otherwise.
    """
    var ctx = a.context()
    var decomposed = schur[dtype, n, gpu](a)
    var t = decomposed.t.to_host()
    var l = _logm_host(t, n)
    return _similar[dtype, n, gpu](decomposed.z, l, ctx)


def _cos_f[T: FloatLike](x: T) -> T:
    return x.cos()


def _sin_f[T: FloatLike](x: T) -> T:
    return x.sin()


def cosm[
    dtype: DType, n: Int, gpu: Bool = False
](mut a: Static[dtype, n, n]) raises -> Static[dtype, n, n] where (
    dtype.is_floating_point() and n >= 1
):
    """The matrix cosine, `funm` at `cos`. `scipy.linalg.cosm`. `cosm(a) @
    cosm(a) + sinm(a) @ sinm(a)` is the identity."""
    return funm[dtype, n, _cos_f, gpu](a)


def sinm[
    dtype: DType, n: Int, gpu: Bool = False
](mut a: Static[dtype, n, n]) raises -> Static[dtype, n, n] where (
    dtype.is_floating_point() and n >= 1
):
    """The matrix sine, `funm` at `sin`. `scipy.linalg.sinm`."""
    return funm[dtype, n, _sin_f, gpu](a)


def fractional_matrix_power[
    dtype: DType, n: Int, gpu: Bool = False
](mut a: Static[dtype, n, n], t: Float64) raises -> Static[dtype, n, n] where (
    dtype.is_floating_point() and n >= 1
):
    """`a^t` for real `t`, as `expm(t logm(a))`.
    `scipy.linalg.fractional_matrix_power`. Real for a matrix with no
    eigenvalue on the closed negative real axis, and `t = 1/2` agrees with
    `sqrtm` to rounding; SciPy's Schur-Pade route is sharper for `t` near
    an integer and is the upgrade."""
    var l = logm[dtype, n, gpu](a)
    var scaled = multiply(l, Scalar[dtype](t))
    return expm[dtype, n, gpu](scaled)
