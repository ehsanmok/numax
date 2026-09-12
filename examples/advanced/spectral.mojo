"""Eigenvalues, singular values and matrix functions over `Tensor`.

Everything here used to be `Array`-tier only, a few dozen values held in
registers. It now runs over a device-resident `Tensor`, which is what lets
it be pointed at a matrix too large to sit in a kernel's stack frame.

**The split to understand before reading the numbers.** Each of these is
two phases with different characters:

1. A **reduction** to band form -- `sytrd` to tridiagonal, `gebrd` to
   bidiagonal, `gehrd` to Hessenberg. Blocked, device-resident, every
   heavy step a `linalg.matmul`, exactly like `cholesky` and `qr_factor`.
2. An **iteration** on that band -- implicit QL/QR, Golub-Kahan, Francis
   double-shift. Sequential, data-dependent, `float64` on the host. It is
   tier 2 and says so, and when eigenvectors are wanted it is also where
   the time goes: accumulating Givens rotations into `Z` is `O(n^3)` of
   scalar work with no GEMM to send it to.

`docs/performance.md` measures that split rather than asserting it. Read
it before assuming `eigh` costs what `cholesky` costs.

The identities each block checks are the definitions, so a reader can see
the factorization is real rather than take the printout on faith:
`A V = V diag(w)`, `A = U diag(s) V^T`, `A = Z T Z^T`, `expm(logm(A)) = A`.

Run: `pixi run example-spectral`
"""

from max.gpu.host import DeviceContext

from numax import FloatLike
from numax.core.array import Static, transpose
from numax.linalg import (
    cond,
    eigh,
    eigvals,
    eigvalsh,
    expm,
    funm,
    logm,
    matmul,
    matrix_rank,
    pinv,
    schur,
    sqrtm,
    svd,
    svdvals,
)

comptime dtype = DType.float64
comptime n = 4


def symmetric(ctx: DeviceContext) raises -> Static[dtype, n, n]:
    """A symmetric matrix with a positive spectrum, built fresh per call:
    every entry point below takes its argument `mut` and factors in place.
    """
    return Static[dtype, n, n](
        ctx,
        [
            7.0,
            1.0,
            -2.0,
            2.0,
            1.0,
            5.0,
            0.0,
            1.0,
            -2.0,
            0.0,
            6.0,
            -2.0,
            2.0,
            1.0,
            -2.0,
            2.0,
        ],
    )


def general(ctx: DeviceContext) raises -> Static[dtype, n, n]:
    """A nonsymmetric matrix with two complex conjugate pairs, `1 +- 2.449i`
    and `2 +- 2i` -- the case a real Schur form keeps in `2 x 2` blocks
    rather than splitting."""
    return Static[dtype, n, n](
        ctx,
        [
            1.0,
            -3.0,
            0.5,
            0.0,
            2.0,
            1.0,
            0.0,
            1.5,
            0.0,
            0.0,
            2.0,
            -4.0,
            0.0,
            0.0,
            1.0,
            2.0,
        ],
    )


def show_row(label: String, mut a: Static[dtype, n]) raises:
    var h = a.to_host()
    print(label, h[0], h[1], h[2], h[3])


def worst_difference(
    mut lhs: Static[dtype, n, n], mut rhs: Static[dtype, n, n]
) raises -> Float64:
    var a = lhs.to_host()
    var b = rhs.to_host()
    var worst = Float64(0)
    for i in range(n * n):
        var diff = abs(Float64(a[i]) - Float64(b[i]))
        if diff > worst:
            worst = diff
    return worst


def exp_of[T: FloatLike](x: T) -> T:
    """The scalar function `funm` lifts to a matrix. Any `FloatLike` kernel
    works here, which is the one place the two axes of the library meet:
    the conformer tier supplies `f`, the `Tensor` tier supplies the Schur
    form it is evaluated on."""
    return x.exp()


def main() raises:
    var ctx = DeviceContext(api="cpu")

    print("Symmetric: eigvalsh and eigh")
    print("----------------------------")

    var a1 = symmetric(ctx)
    var values = eigvalsh[dtype, n](a1)
    show_row("  eigvalsh(A)  w =", values)
    print("    ascending, and they sum to trace(A) = 20")

    var a2 = symmetric(ctx)
    var decomposition = eigh[dtype, n](a2)
    show_row("  eigh(A).values  ", decomposition.values)

    # `A V == V diag(w)`, the eigen-equation column by column. `diag(w)` is
    # built here rather than imported because scaling columns is cheaper
    # than a second matmul, and it makes the check explicit.
    var a_for_product = symmetric(ctx)
    var av = matmul(a_for_product, decomposition.vectors)
    var vectors = decomposition.vectors.to_host()
    var w = decomposition.values.to_host()
    var av_host = av.to_host()
    var residual = Float64(0)
    for i in range(n):
        for j in range(n):
            var want = Float64(vectors[i * n + j]) * Float64(w[j])
            var diff = abs(Float64(av_host[i * n + j]) - want)
            if diff > residual:
                residual = diff
    print("    max |A V - V diag(w)| =", residual)

    print()
    print("Rectangular: svdvals and svd")
    print("----------------------------")

    var b1 = general(ctx)
    var singular = svdvals[dtype, n, n](b1)
    show_row("  svdvals(A)   s =", singular)
    print("    descending, unlike the Array tier's, which is unsorted")

    var b2 = general(ctx)
    var factored = svd[dtype, n, n](b2)
    # `A == U diag(s) V^T`. `v` holds the right singular vectors as columns,
    # so SciPy's `Vh` is `transpose(v)` -- stated in `TensorSVD`'s docstring
    # and worth seeing once.
    var u_host = factored.u.to_host()
    var s_host = factored.s.to_host()
    var scaled_entries = List[Scalar[dtype]](capacity=n * n)
    for i in range(n):
        for j in range(n):
            scaled_entries.append(u_host[i * n + j] * s_host[j])
    var scaled = Static[dtype, n, n](ctx, scaled_entries^)
    var v_transposed = transpose[dtype, n, n](factored.v)
    var reconstructed = matmul(scaled, v_transposed)
    var original = general(ctx)
    print(
        "    max |U diag(s) V^T - A| =",
        worst_difference(reconstructed, original),
    )

    var b3 = general(ctx)
    print("  cond(A)      =", cond[dtype, n, n](b3))
    var b4 = general(ctx)
    print("  matrix_rank(A) =", matrix_rank[dtype, n, n](b4))

    # `pinv` is the SVD's first dependent: `V diag(1/s) U^T`, with the
    # singular values below `rcond` dropped. On a nonsingular matrix it is
    # the inverse, so `A^+ A` is the identity.
    var b5 = general(ctx)
    var pseudo = pinv[dtype, n, n](b5)
    var b6 = general(ctx)
    var back = matmul(pseudo, b6)
    var identity_error = Float64(0)
    var product = back.to_host()
    for i in range(n):
        for j in range(n):
            var want = 1.0 if i == j else 0.0
            var diff = abs(Float64(product[i * n + j]) - want)
            if diff > identity_error:
                identity_error = diff
    print("    max |pinv(A) A - I| =", identity_error)

    print()
    print("Nonsymmetric: eigvals and schur")
    print("-------------------------------")

    # A `Tensor` is monomorphic in a `DType` and so cannot hold a complex
    # number. `eigvals` returns the real and imaginary parts as two
    # tensors, the same answer `numax.fft`'s `Spectrum` gives to the same
    # constraint.
    var b7 = general(ctx)
    var spectrum = eigvals[dtype, n](b7)
    var re = spectrum.re.to_host()
    var im = spectrum.im.to_host()
    for i in range(n):
        print("  lambda =", re[i], "+", im[i], "i")

    var b8 = general(ctx)
    var form = schur[dtype, n](b8)
    # `A == Z T Z^T` with `T` quasi-triangular: a complex pair leaves a
    # `2 x 2` block on the diagonal rather than splitting into two real
    # eigenvalues that do not exist.
    var zt = matmul(form.z, form.t)
    var z_transposed = transpose[dtype, n, n](form.z)
    var rebuilt = matmul(zt, z_transposed)
    var source = general(ctx)
    print("  max |Z T Z^T - A| =", worst_difference(rebuilt, source))

    print()
    print("Matrix functions on that Schur form")
    print("-----------------------------------")

    # `logm` is the inverse of `expm`, which is the check worth printing:
    # the two are independent implementations (inverse scaling and squaring
    # against scaling and squaring), so agreement is evidence.
    var b9 = general(ctx)
    var logarithm = logm[dtype, n](b9)
    var exponentiated = expm[dtype, n](logarithm)
    var original_again = general(ctx)
    print(
        "  max |expm(logm(A)) - A| =",
        worst_difference(exponentiated, original_again),
    )

    # `sqrtm` over `Tensor` is the general one -- the Bjorck-Hammarling
    # recurrence on the real Schur form -- not the SPD-only Array version.
    var b10 = general(ctx)
    var root = sqrtm[dtype, n](b10)
    var b11 = general(ctx)
    var root_again = sqrtm[dtype, n](b11)
    var squared = matmul(root, root_again)
    var third = general(ctx)
    print("  max |sqrtm(A)^2 - A| =", worst_difference(squared, third))

    # `funm` takes the scalar function as a compile-time parameter, so
    # `funm[f=exp_of]` and `expm` must agree.
    var b12 = general(ctx)
    var through_funm = funm[dtype, n, f=exp_of](b12)
    var b13 = general(ctx)
    var through_expm = expm[dtype, n](b13)
    print(
        "  max |funm(A, exp) - expm(A)| =",
        worst_difference(through_funm, through_expm),
    )
