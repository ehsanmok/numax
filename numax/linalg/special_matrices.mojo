"""Structured matrix constructors. `scipy.linalg`'s `_special_matrices`.

**The `Tensor` tier, `Plain`-only, tier 2.** Every function here builds a
matrix from a rule on its indices, so none of them branches on a value and
all of them carry `gpu: Bool`; the tier is about the host orchestrating the
launch, not about anything in the body.

This is the fourth gate outcome rather than the usual three -- not
delegate, extend or diverge, but the **`Plain`-only surface**. MAX ships
none of these (searched at the `max ==26.5` pin across `linalg`, `nn`,
`algorithm` and `layout`), *and* instantiating a Hilbert matrix at `Dual` or
`Interval` would add no meaning: the entries are constants of the index, so
the derivative is zero everywhere and the interval is a point. There is
therefore no `Array` sibling of this module and none is missing.

Each one is a single `max.algorithm.elementwise` map over the output, the
same shape `numax.linalg.blas`'s `outer` and `kron` use. Written width-1
throughout: every rule here reads its inputs at an index computed from the
output coordinate -- `i - j`, `i + j`, `i % p` -- and none of those is
contiguous in the output's fastest axis, so a wider store would gather
anyway.

## What each is for

`toeplitz`, `hankel` and `circulant` are the three constant-diagonal
families, and the reason to build them explicitly rather than exploit the
structure is that numax has no structured solver to exploit it with:
`numax.linalg.banded` covers the banded case, and a dense `solve` on the
materialized matrix covers the rest. Materializing an `n x n` from an
`n`-vector is `O(n^2)` of memory for an operator that could be applied in
`O(n log n)`, so these are for the sizes where that does not matter and for
handing a structured operator to code that wants a matrix.

`companion` turns a polynomial into the matrix whose eigenvalues are its
roots, which is how `numpy.roots` is implemented and what
`numax.linalg.array.eigvals` can then be pointed at. `convolution_matrix`
does the same job for `numax.signal.convolve`: it is the matrix `C` with
`C @ v == convolve(a, v)`. `block_diag` and `khatri_rao` are assembly.

The seven index rules that were once deferred -- `pascal`, `invpascal`,
`hadamard`, `helmert`, `fiedler`, `fiedler_companion` and `leslie` -- are
here in the same shape: one `elementwise` map each, the entry a function
of `(i, j)` and at most a vector argument. `fiedler_companion` is the
better-conditioned sibling of `companion` for a root finder; `invpascal`
is the closed-form inverse rather than a call to `inverse`.

`dft` and `invhilbert` are deliberately absent, on the grounds
`hilbert` and the parity record give (`invhilbert`'s entries need integer
arithmetic to mean anything past `n = 12`). `dft`'s entries are complex and there is no
complex `Tensor` -- `numax.fft` carries a spectrum as a real/imaginary
`Spectrum` pair instead -- so the choice is a signature that is not SciPy's
or a wait for a complex tensor surface, which `docs/parity.md` records as
not absorbed on purpose. A half-shape is worse than the absence.
"""

from std.math import sqrt

from layout import Coord, coord_to_index_list
from max.algorithm.functional import elementwise
from max.gpu.host import DeviceContext

from ..core.array import Static

from .blas import _target


def toeplitz[
    dtype: DType, m: Int, n: Int, gpu: Bool = False
](mut c: Static[dtype, m], mut r: Static[dtype, n]) raises -> Static[
    dtype, m, n
]:
    """The Toeplitz matrix with first column `c` and first row `r`.
    `scipy.linalg.toeplitz(c, r)`.

    Constant along every diagonal: `out[i, j]` is `c[i - j]` on and below
    the diagonal and `r[j - i]` above it.

    `r[0]` is **ignored**, as it is in SciPy: the corner entry belongs to
    both the first row and the first column, and `c[0]` is the one that
    wins. Passing an `r` whose first entry disagrees with `c[0]` is not an
    error and not a silent average -- it is `c[0]`.
    """
    var ctx = c.context()
    var out = Static[dtype, m, n]._uninitialized(ctx)
    var cv = c.view()
    var rv = r.view()
    var ov = out.view()

    @always_inline
    def step[w: Int, alignment: Int = 1](coord: Coord) {var cv, var rv, var ov}:
        var at = coord_to_index_list(coord)
        var i = at[0]
        var j = at[1]
        if i >= j:
            ov.store[1](coord, cv[Coord(i - j)])
        else:
            ov.store[1](coord, rv[Coord(j - i)])

    elementwise[simd_width=1, target=_target[gpu]()](step, Coord(m, n), ctx)
    return out^


def toeplitz[
    dtype: DType, n: Int, gpu: Bool = False
](mut c: Static[dtype, n]) raises -> Static[dtype, n, n]:
    """The **symmetric** Toeplitz matrix with first column and first row
    both `c`. `scipy.linalg.toeplitz(c)`.

    The one-argument form, selected by arity. `out[i, j] = c[abs(i - j)]`,
    which is the common case -- an autocorrelation matrix is exactly this
    shape -- and is what `numax.linalg.banded.solve_toeplitz` expects when
    it is handed a single vector.
    """
    var ctx = c.context()
    var out = Static[dtype, n, n]._uninitialized(ctx)
    var cv = c.view()
    var ov = out.view()

    @always_inline
    def step[w: Int, alignment: Int = 1](coord: Coord) {var cv, var ov}:
        var at = coord_to_index_list(coord)
        var i = at[0]
        var j = at[1]
        ov.store[1](coord, cv[Coord(i - j if i >= j else j - i)])

    elementwise[simd_width=1, target=_target[gpu]()](step, Coord(n, n), ctx)
    return out^


def hankel[
    dtype: DType, m: Int, n: Int, gpu: Bool = False
](mut c: Static[dtype, m], mut r: Static[dtype, n]) raises -> Static[
    dtype, m, n
]:
    """The Hankel matrix with first column `c` and last row `r`.
    `scipy.linalg.hankel(c, r)`.

    Constant along every *anti*-diagonal, which is `toeplitz` with the rows
    reversed: `out[i, j]` is `c[i + j]` while that index is in range and
    `r[i + j - m + 1]` after it.

    `r[0]` is ignored for the same reason `toeplitz` ignores it -- the
    bottom-left corner belongs to both `c`'s end and `r`'s start, and `c`
    wins.
    """
    var ctx = c.context()
    var out = Static[dtype, m, n]._uninitialized(ctx)
    var cv = c.view()
    var rv = r.view()
    var ov = out.view()

    @always_inline
    def step[w: Int, alignment: Int = 1](coord: Coord) {var cv, var rv, var ov}:
        var at = coord_to_index_list(coord)
        var sum_index = at[0] + at[1]
        if sum_index < m:
            ov.store[1](coord, cv[Coord(sum_index)])
        else:
            ov.store[1](coord, rv[Coord(sum_index - m + 1)])

    elementwise[simd_width=1, target=_target[gpu]()](step, Coord(m, n), ctx)
    return out^


def circulant[
    dtype: DType, n: Int, gpu: Bool = False
](mut c: Static[dtype, n]) raises -> Static[dtype, n, n]:
    """The circulant matrix whose first column is `c`.
    `scipy.linalg.circulant(c)`.

    `out[i, j] = c[(i - j) mod n]`: each column is the one before it rotated
    down by one. Every circulant is diagonalized by the DFT, which is what
    makes `numax.linalg.banded.solve_circulant` an FFT rather than a solve;
    this materializes the matrix for code that wants one.

    The index is written `(i - j + n) % n` rather than `(i - j) % n` so the
    negative case does not depend on how Mojo rounds a modulus.
    """
    var ctx = c.context()
    var out = Static[dtype, n, n]._uninitialized(ctx)
    var cv = c.view()
    var ov = out.view()

    @always_inline
    def step[w: Int, alignment: Int = 1](coord: Coord) {var cv, var ov}:
        var at = coord_to_index_list(coord)
        ov.store[1](coord, cv[Coord((at[0] - at[1] + n) % n)])

    elementwise[simd_width=1, target=_target[gpu]()](step, Coord(n, n), ctx)
    return out^


def companion[
    dtype: DType, n: Int, gpu: Bool = False
](mut a: Static[dtype, n]) raises -> Static[dtype, n - 1, n - 1] where (
    dtype.is_floating_point() and n >= 2
):
    """The companion matrix of the polynomial with coefficients `a`, highest
    degree first. `scipy.linalg.companion(a)`.

    `n` coefficients describe a degree-`n - 1` polynomial, so the result is
    `(n-1) x (n-1)`: first row `-a[1:] / a[0]`, ones on the first
    subdiagonal, zeros elsewhere.

    Its eigenvalues are the polynomial's roots, which is how `numpy.roots`
    is implemented and what makes this the bridge to
    `numax.linalg.array.eigvals` -- build the companion, take its
    eigenvalues, and those are the roots.

    `a[0]` must be nonzero, since the first row divides by it. A leading
    zero means the polynomial is of lower degree than its coefficient list
    claims, and trimming it is the caller's decision to make rather than
    something to do silently: the two readings give different-sized
    matrices.
    """
    var ctx = a.context()
    var out = Static[dtype, n - 1, n - 1]._uninitialized(ctx)
    var av = a.view()
    var ov = out.view()

    @always_inline
    def step[w: Int, alignment: Int = 1](coord: Coord) {var av, var ov}:
        var at = coord_to_index_list(coord)
        var i = at[0]
        var j = at[1]
        if i == 0:
            ov.store[1](coord, -av[Coord(j + 1)] / av[Coord(0)])
        elif i == j + 1:
            ov.store[1](coord, Scalar[dtype](1))
        else:
            ov.store[1](coord, Scalar[dtype](0))

    elementwise[simd_width=1, target=_target[gpu]()](
        step, Coord(n - 1, n - 1), ctx
    )
    return out^


def hilbert[
    n: Int, dtype: DType = DType.float64, gpu: Bool = False
](ctx: Optional[DeviceContext] = None) raises -> Static[
    dtype, n, n
] where dtype.is_floating_point():
    """The `n x n` Hilbert matrix, `out[i, j] = 1 / (i + j + 1)`.
    `scipy.linalg.hilbert(n)`.

    The standard ill-conditioned test matrix: its condition number grows
    like `e^(3.5 n)`, so a `12 x 12` Hilbert matrix is already past what
    `float64` can solve to any accuracy at all. That is what it is for --
    checking that a solver degrades honestly rather than confidently.

    Takes `n` as a parameter and no matrix argument, so the `DeviceContext`
    is the last argument and optional, matching `eye` and the rest of
    `numax.core.array`'s factories rather than the operations in this
    module.

    There is no `invhilbert`. Its entries are large integers -- the
    `12 x 12` inverse has entries past `1e17` -- so computing them in
    `float64` and calling the result an inverse would be a claim numax
    cannot back; an exact one needs integer arithmetic this tier does not
    have.
    """
    var device = ctx.value() if ctx else DeviceContext(api="cpu")
    var out = Static[dtype, n, n]._uninitialized(device)
    var ov = out.view()

    @always_inline
    def step[w: Int, alignment: Int = 1](coord: Coord) {var ov}:
        var at = coord_to_index_list(coord)
        ov.store[1](coord, Scalar[dtype](1) / Scalar[dtype](at[0] + at[1] + 1))

    elementwise[simd_width=1, target=_target[gpu]()](step, Coord(n, n), device)
    return out^


def block_diag[
    dtype: DType,
    rows_a: Int,
    cols_a: Int,
    rows_b: Int,
    cols_b: Int,
    gpu: Bool = False,
](
    mut a: Static[dtype, rows_a, cols_a], mut b: Static[dtype, rows_b, cols_b]
) raises -> Static[dtype, rows_a + rows_b, cols_a + cols_b]:
    """`a` and `b` on the diagonal of a larger matrix, zeros elsewhere.
    `scipy.linalg.block_diag(a, b)`.

    Two blocks rather than SciPy's varargs, because each operand's shape
    lives in its type and a variadic of differently typed tensors has no
    spelling here. Three blocks is `block_diag(block_diag(a, b), c)`, which
    costs one extra pass over the first result and is the composition SciPy
    is doing internally anyway.
    """
    var ctx = a.context()
    var out = Static[dtype, rows_a + rows_b, cols_a + cols_b]._uninitialized(
        ctx
    )
    var av = a.view()
    var bv = b.view()
    var ov = out.view()

    @always_inline
    def step[w: Int, alignment: Int = 1](coord: Coord) {var av, var bv, var ov}:
        var at = coord_to_index_list(coord)
        var i = at[0]
        var j = at[1]
        if i < rows_a and j < cols_a:
            ov.store[1](coord, av[Coord(i, j)])
        elif i >= rows_a and j >= cols_a:
            ov.store[1](coord, bv[Coord(i - rows_a, j - cols_a)])
        else:
            ov.store[1](coord, Scalar[dtype](0))

    elementwise[simd_width=1, target=_target[gpu]()](
        step, Coord(rows_a + rows_b, cols_a + cols_b), ctx
    )
    return out^


def khatri_rao[
    dtype: DType, m: Int, p: Int, k: Int, gpu: Bool = False
](mut a: Static[dtype, m, k], mut b: Static[dtype, p, k]) raises -> Static[
    dtype, m * p, k
]:
    """The column-wise Kronecker product of `a` and `b`.
    `scipy.linalg.khatri_rao(a, b)`.

    Column `j` of the result is `kron(a[:, j], b[:, j])`, so the two
    operands must agree on their column count and the result has `m * p`
    rows. Not the same as `kron`, which pairs every entry of `a` with every
    entry of `b`; here the pairing is only within a column, which is what
    makes it the building block of a CP tensor decomposition.
    """
    var ctx = a.context()
    var out = Static[dtype, m * p, k]._uninitialized(ctx)
    var av = a.view()
    var bv = b.view()
    var ov = out.view()

    @always_inline
    def step[w: Int, alignment: Int = 1](coord: Coord) {var av, var bv, var ov}:
        var at = coord_to_index_list(coord)
        var row = at[0]
        var col = at[1]
        ov.store[1](coord, av[Coord(row // p, col)] * bv[Coord(row % p, col)])

    elementwise[simd_width=1, target=_target[gpu]()](step, Coord(m * p, k), ctx)
    return out^


def convolution_matrix[
    dtype: DType, m: Int, n: Int, gpu: Bool = False
](mut a: Static[dtype, m]) raises -> Static[dtype, m + n - 1, n]:
    """The matrix `C` with `C @ v == convolve(a, v)` for any `v` of length
    `n`. `scipy.linalg.convolution_matrix(a, n)`, in its `"full"` mode.

    `out[i, j] = a[i - j]` where that index is in range and zero elsewhere
    -- a Toeplitz matrix, banded by `a`'s length, `(m + n - 1) x n`.

    Why materialize a convolution: `numax.signal.convolve` applies one, and
    this is for the problems that need it as an *operator* -- deconvolution
    is `lstsq` against this matrix, and a regularized one is that with rows
    appended. Those are things a matrix can be handed to and a function
    cannot.

    `"full"` only. SciPy's `"same"` and `"valid"` modes are row slices of
    this one, and numax has no owned slice type to return them as -- which
    `docs/parity.md` lists under what is still missing rather than
    something this module should work around.
    """
    var ctx = a.context()
    var out = Static[dtype, m + n - 1, n]._uninitialized(ctx)
    var av = a.view()
    var ov = out.view()

    @always_inline
    def step[w: Int, alignment: Int = 1](coord: Coord) {var av, var ov}:
        var at = coord_to_index_list(coord)
        var offset = at[0] - at[1]
        if offset >= 0 and offset < m:
            ov.store[1](coord, av[Coord(offset)])
        else:
            ov.store[1](coord, Scalar[dtype](0))

    elementwise[simd_width=1, target=_target[gpu]()](
        step, Coord(m + n - 1, n), ctx
    )
    return out^


# ---------------------------------------------------- the index rules


def _binomial[dtype: DType](top: Int, bottom: Int) -> Scalar[dtype]:
    """`C(top, bottom)` by the multiplicative rule in the tensor's own
    `dtype` -- not `Float64`, which a Metal kernel cannot hold -- exact
    while the result is below the dtype's integer range (`2^53` at
    float64, `2^24` at float32); zero outside `0 <= bottom <= top`."""
    if bottom < 0 or bottom > top:
        return Scalar[dtype](0)
    var k = bottom if bottom < top - bottom else top - bottom
    var value = Scalar[dtype](1)
    for step in range(1, k + 1):
        value = value * Scalar[dtype](top - k + step) / Scalar[dtype](step)
    return value


def pascal[
    n: Int,
    dtype: DType = DType.float64,
    kind: StaticString = "symmetric",
    gpu: Bool = False,
](ctx: Optional[DeviceContext] = None) raises -> Static[dtype, n, n] where (
    n >= 1
):
    """The `n x n` Pascal matrix. `scipy.linalg.pascal(n, kind)`.

    `"symmetric"` (the default) has `out[i, j] = C(i + j, i)`, `"lower"`
    has `C(i, j)` on and below the diagonal, `"upper"` is its transpose;
    the symmetric one is `lower @ upper`. Entries are binomial
    coefficients computed by the multiplicative rule in the tensor's
    `dtype`, exact while below `2^53` at float64 -- through `n = 28` for
    the symmetric kind, whose largest entry is `C(2n - 2, n - 1)`, and `n =
    56` for the triangular ones -- and below `2^24` at float32.
    SciPy's `exact=True` integer form is not offered at this tier.
    """
    var device = ctx.value() if ctx else DeviceContext(api="cpu")
    var out = Static[dtype, n, n]._uninitialized(device)
    var ov = out.view()

    @always_inline
    def step[w: Int, alignment: Int = 1](coord: Coord) {var ov}:
        var at = coord_to_index_list(coord)
        var i = at[0]
        var j = at[1]
        var value: Scalar[dtype]
        comptime if kind == "lower":
            value = _binomial[dtype](i, j)
        elif kind == "upper":
            value = _binomial[dtype](j, i)
        else:
            value = _binomial[dtype](i + j, i)
        ov.store[1](coord, value)

    comptime if not (kind == "symmetric" or kind == "lower" or kind == "upper"):
        raise Error("pascal: kind must be 'symmetric', 'lower' or 'upper'")
    elementwise[simd_width=1, target=_target[gpu]()](step, Coord(n, n), device)
    return out^


def invpascal[
    n: Int,
    dtype: DType = DType.float64,
    kind: StaticString = "symmetric",
    gpu: Bool = False,
](ctx: Optional[DeviceContext] = None) raises -> Static[dtype, n, n] where (
    n >= 1
):
    """The inverse of the `n x n` Pascal matrix of the same `kind`, in
    closed form rather than by inverting. `scipy.linalg.invpascal(n, kind)`.

    The lower kind's inverse is `(-1)^(i - j) C(i, j)`, the upper's its
    transpose, and the symmetric's is `(-1)^(i - j) sum_{k} C(i + k, k)
    C(i + k, i + k - j)` over `k < n - i` for `j <= i`, mirrored -- SciPy's
    formula, which is the product of the two triangular inverses written
    out. Exact while the sums stay below `2^53`, the same range as
    `pascal`.
    """
    var device = ctx.value() if ctx else DeviceContext(api="cpu")
    var out = Static[dtype, n, n]._uninitialized(device)
    var ov = out.view()

    @always_inline
    def step[w: Int, alignment: Int = 1](coord: Coord) {var ov}:
        var at = coord_to_index_list(coord)
        var i = at[0]
        var j = at[1]
        var value: Scalar[dtype]
        comptime if kind == "lower":
            value = _binomial[dtype](i, j) * Scalar[dtype](
                1 if (i - j) % 2 == 0 else -1
            )
        elif kind == "upper":
            value = _binomial[dtype](j, i) * Scalar[dtype](
                1 if (j - i) % 2 == 0 else -1
            )
        else:
            var row = i if i >= j else j
            var col = j if i >= j else i
            var total = Scalar[dtype](0)
            for k in range(n - row):
                total += _binomial[dtype](row + k, k) * _binomial[dtype](
                    row + k, row + k - col
                )
            value = total * Scalar[dtype](1 if (row - col) % 2 == 0 else -1)
        ov.store[1](coord, value)

    comptime if not (kind == "symmetric" or kind == "lower" or kind == "upper"):
        raise Error("invpascal: kind must be 'symmetric', 'lower' or 'upper'")
    elementwise[simd_width=1, target=_target[gpu]()](step, Coord(n, n), device)
    return out^


def hadamard[
    n: Int, dtype: DType = DType.float64, gpu: Bool = False
](ctx: Optional[DeviceContext] = None) raises -> Static[dtype, n, n] where (
    n >= 1
):
    """The `n x n` Sylvester Hadamard matrix, `n` a power of two.
    `scipy.linalg.hadamard(n)`.

    `out[i, j] = (-1)^popcount(i & j)`, the closed form of the recursive
    `[[H, H], [H, -H]]` construction, so it is one index rule rather than
    `log2 n` stackings. `H^T H == n I`. A non-power-of-two `n` is a
    compile-time error.
    """
    comptime assert (n & (n - 1)) == 0, "hadamard: n must be a power of two"
    var device = ctx.value() if ctx else DeviceContext(api="cpu")
    var out = Static[dtype, n, n]._uninitialized(device)
    var ov = out.view()

    @always_inline
    def step[w: Int, alignment: Int = 1](coord: Coord) {var ov}:
        var at = coord_to_index_list(coord)
        var bits = at[0] & at[1]
        var parity = 0
        while bits != 0:
            parity ^= bits & 1
            bits >>= 1
        ov.store[1](coord, Scalar[dtype](1 - 2 * parity))

    elementwise[simd_width=1, target=_target[gpu]()](step, Coord(n, n), device)
    return out^


def helmert[
    n: Int, dtype: DType = DType.float64, full: Bool = False, gpu: Bool = False
](ctx: Optional[DeviceContext] = None) raises -> Static[
    dtype, n - 1 + (1 if full else 0), n
] where (dtype.is_floating_point() and n >= 2):
    """The Helmert matrix of order `n`: `n - 1` orthonormal rows, each
    orthogonal to the constant vector, or with `full=True` the `n x n`
    orthogonal matrix whose first row is the constant `1 / sqrt(n)`.
    `scipy.linalg.helmert(n, full)`.

    Row `i` of the full matrix (`i >= 1`) is `1 / sqrt(i (i + 1))` in its
    first `i` entries, `-i / sqrt(i (i + 1))` at entry `i`, zero after --
    the contrasts that turn a vector into its successive deviations from
    running means, which is what the matrix is for in the analysis of
    variance.
    """
    var device = ctx.value() if ctx else DeviceContext(api="cpu")
    comptime rows = n - 1 + (1 if full else 0)
    var out = Static[dtype, rows, n]._uninitialized(device)
    var ov = out.view()

    @always_inline
    def step[w: Int, alignment: Int = 1](coord: Coord) {var ov}:
        var at = coord_to_index_list(coord)
        var i = at[0] + (0 if full else 1)
        var j = at[1]
        var value: Scalar[dtype]
        if i == 0:
            value = Scalar[dtype](1) / sqrt(Scalar[dtype](n))
        else:
            var scale = Scalar[dtype](1) / sqrt(
                Scalar[dtype](i) * Scalar[dtype](i + 1)
            )
            if j < i:
                value = scale
            elif j == i:
                value = -Scalar[dtype](i) * scale
            else:
                value = Scalar[dtype](0)
        ov.store[1](coord, value)

    elementwise[simd_width=1, target=_target[gpu]()](
        step, Coord(rows, n), device
    )
    return out^


def fiedler[
    dtype: DType, n: Int, gpu: Bool = False
](mut a: Static[dtype, n]) raises -> Static[dtype, n, n] where n >= 1:
    """The Fiedler matrix of `a`, `out[i, j] = |a[i] - a[j]|`.
    `scipy.linalg.fiedler(a)`. Symmetric with a zero diagonal; for a
    strictly increasing `a` its inverse is tridiagonal and it has one
    positive and `n - 1` negative eigenvalues, the property it is named
    for."""
    var ctx = a.context()
    var out = Static[dtype, n, n]._uninitialized(ctx)
    var av = a.view()
    var ov = out.view()

    @always_inline
    def step[w: Int, alignment: Int = 1](coord: Coord) {var av, var ov}:
        var at = coord_to_index_list(coord)
        ov.store[1](coord, abs(av[Coord(at[0])] - av[Coord(at[1])]))

    elementwise[simd_width=1, target=_target[gpu]()](step, Coord(n, n), ctx)
    return out^


def fiedler_companion[
    dtype: DType, n: Int, gpu: Bool = False
](mut a: Static[dtype, n]) raises -> Static[dtype, n - 1, n - 1] where (
    dtype.is_floating_point() and n >= 3
):
    """Fiedler's pentadiagonal companion matrix of the polynomial with
    coefficients `a`, highest degree first. `scipy.linalg.fiedler_companion(a)`.

    The same eigenvalues as `companion(a)` -- the polynomial's roots -- in
    a matrix with only `2 n - 3` nonzeros arranged within two of the
    diagonal, which is what makes it the better-conditioned choice for a
    root finder. SciPy's index rule, with `c = a / a[0]`: `out[0, 0] =
    -c[1]`, `out[1, 0] = 1`; on even rows `i`, `out[i, i + 1] = -c[i + 2]`
    and `out[i, i + 2] = 1`; on even rows `i >= 2`, `out[i, i - 1] = -c[i +
    1]`; on odd rows `i >= 3`, `out[i, i - 2] = 1`. `a[0]` must be
    nonzero, as for `companion`.
    """
    var ctx = a.context()
    var out = Static[dtype, n - 1, n - 1]._uninitialized(ctx)
    var av = a.view()
    var ov = out.view()

    @always_inline
    def step[w: Int, alignment: Int = 1](coord: Coord) {var av, var ov}:
        var at = coord_to_index_list(coord)
        var i = at[0]
        var j = at[1]
        var lead = av[Coord(0)]
        var value = Scalar[dtype](0)
        if i == 0 and j == 0:
            value = -av[Coord(1)] / lead
        elif i == 1 and j == 0:
            value = Scalar[dtype](1)
        elif i % 2 == 0:
            if j == i + 1:
                value = -av[Coord(i + 2)] / lead
            elif j == i + 2:
                value = Scalar[dtype](1)
            elif i >= 2 and j == i - 1:
                value = -av[Coord(i + 1)] / lead
        elif i >= 3 and j == i - 2:
            value = Scalar[dtype](1)
        ov.store[1](coord, value)

    elementwise[simd_width=1, target=_target[gpu]()](
        step, Coord(n - 1, n - 1), ctx
    )
    return out^


def leslie[
    dtype: DType, n: Int, gpu: Bool = False
](mut f: Static[dtype, n], mut s: Static[dtype, n - 1]) raises -> Static[
    dtype, n, n
] where (n >= 2):
    """The Leslie matrix of fecundities `f` and survivals `s`: `f` along
    the first row, `s` along the first subdiagonal, zero elsewhere.
    `scipy.linalg.leslie(f, s)`. Its dominant eigenvalue is the
    population's asymptotic growth rate."""
    var ctx = f.context()
    var out = Static[dtype, n, n]._uninitialized(ctx)
    var fv = f.view()
    var sv = s.view()
    var ov = out.view()

    @always_inline
    def step[w: Int, alignment: Int = 1](coord: Coord) {var fv, var sv, var ov}:
        var at = coord_to_index_list(coord)
        var i = at[0]
        var j = at[1]
        if i == 0:
            ov.store[1](coord, fv[Coord(j)])
        elif j == i - 1:
            ov.store[1](coord, sv[Coord(j)])
        else:
            ov.store[1](coord, Scalar[dtype](0))

    elementwise[simd_width=1, target=_target[gpu]()](step, Coord(n, n), ctx)
    return out^
