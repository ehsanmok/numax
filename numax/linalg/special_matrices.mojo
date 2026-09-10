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

`dft` is deliberately absent. Its entries are complex and there is no
complex `Tensor` -- `numax.fft` carries a spectrum as a real/imaginary
`Spectrum` pair instead -- so the choice is a signature that is not SciPy's
or a wait for a complex tensor surface, which `docs/parity.md` records as
not absorbed on purpose. A half-shape is worse than the absence.
"""

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
