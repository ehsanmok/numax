"""Spectral factorizations over `Tensor`: `scipy.linalg`'s `_decomp`.

**This module is tier 2.** Every routine here is two phases, and only the
first is device-resident: a reduction to a condensed form, which is
Householder work whose cubic term goes to `linalg.matmul`, then a sweep on
the condensed band, which loops to a tolerance and deflates on a test of
the data. The second phase has no GEMM to hand anything to -- that is a
property of the algorithm, not of this implementation -- so it runs on the
host over `O(n)` numbers and each function says so where it happens.

MAX ships nothing to delegate to here. There is no eigensolver, no SVD, no
`sytrd`, no Jacobi or Givens helper anywhere in `linalg`, `nn`,
`algorithm` or `layout`; the one decomposition in the tree,
`linalg.qr_factorization`, is `LayoutTensor`-only and a CPU scalar loop,
denied twice over. So everything here is an **extend**, written in MAX's
idiom.

`numax.linalg.array.eigen` is the `FloatLike`-generic sibling, one import
away: cyclic Jacobi at a fixed sweep count, square matrices small enough to
live in registers, differentiable and GPU-launchable in a way this tier
cannot be. This tier is for the sizes that one cannot reach.
"""

from layout import Coord, TileTensor, coord_to_index_list
from layout.tile_layout import TensorLayout, row_major
from layout.tile_tensor import PointerStorage
from linalg.matmul import matmul as _max_matmul
from max.algorithm.functional import elementwise
from max.gpu.host import DeviceContext
from std.builtin.sort import sort as _std_sort
from std.math import hypot as _hypot
from std.sys.info import align_of
from std.utils import IndexList

from ..core.array import Dynamic, Static, zeros, zeros_dyn
from .blas import _target, dot, matmul, matvec
from .common import _Dense
from .panel import (
    _PANEL_THREADS,
    gebd2_col,
    gebd2_row,
    pack_block,
    sytd2_column,
    sytd2_rank_two,
)


struct TensorTridiagonal[dtype: DType, n: Int, gpu: Bool = False](
    Movable where dtype.is_floating_point() and n >= 1
):
    """The symmetric tridiagonal reduction `A = Q T Q^T`, device-resident.

    `T` is the tridiagonal matrix `diag(d)` with `e` on both off-diagonals;
    `Q` is orthogonal and is the product of the Householder reflections the
    reduction applied, held packed rather than formed. `.q()` materializes
    it when a caller wants the vectors; `eigvalsh` never asks.

    `e` is `n` long where only `e[0 .. n-2]` carries a subdiagonal entry,
    rather than the `n - 1` LAPACK uses: an `n == 1` matrix would otherwise
    need a zero-length field, and the last entry is documented as unused
    rather than made unrepresentable.
    """

    var d: Static[Self.dtype, Self.n]
    """`T`'s diagonal."""

    var e: Static[Self.dtype, Self.n]
    """`T`'s subdiagonal in `e[0 .. n-2]`; `e[n-1]` is unused and zero."""

    var reflectors: Static[Self.dtype, Self.n, Self.n]
    """The reduced matrix, holding `v` for column `k` below row `k + 1`
    with its leading `1` implicit -- LAPACK's packed form."""

    var taus: Static[Self.dtype, Self.n]
    """One Householder scale per column; `taus[k] == 0` where the column
    was already reduced and the reflection is the identity."""

    def __init__(
        out self,
        var d: Static[Self.dtype, Self.n],
        var e: Static[Self.dtype, Self.n],
        var reflectors: Static[Self.dtype, Self.n, Self.n],
        var taus: Static[Self.dtype, Self.n],
    ):
        self.d = d^
        self.e = e^
        self.reflectors = reflectors^
        self.taus = taus^

    def q(mut self) raises -> Static[Self.dtype, Self.n, Self.n]:
        """Materialize `Q`, the orthogonal matrix of the reduction.
        LAPACK's `orgtr`.

        `Q = H_0 H_1 ... H_{n-3}`, accumulated in reverse so each
        reflection meets a matrix that is already the product of the ones
        after it. Each step is `Q -= tau v (v^T Q)`, which is a
        matrix-vector product and a rank-one update, both `linalg.matmul`.

        `O(n^3)` and one allocation per step -- the same shape as `sytrd`
        itself, and the same `ponytail:` ceiling applies. A caller who only
        wants eigenvalues should not call this; `eigvalsh` does not.
        """
        var ctx = self.reflectors.context()
        var result = zeros[Self.dtype, Self.n, Self.n](ctx)
        var host = result.to_host()
        for i in range(Self.n):
            host[i * Self.n + i] = Scalar[Self.dtype](1)
        result.copy_from_host(host)

        if Self.n < 3:
            return result^

        var packed = self.reflectors.view()
        var taus = self.taus.view()
        var vpad = zeros[Self.dtype, Self.n](ctx)
        var product = zeros[Self.dtype, Self.n, Self.n](ctx)

        var k = Self.n - 3
        while k >= 0:
            var this_tau = self.taus.to_host()[k]
            if this_tau != 0:
                _write_vpad[gpu=Self.gpu](packed, vpad.view(), k, Self.n, ctx)
                # `y = Q^T v`, then `Q -= tau v y^T`.
                var y = _row_combination[gpu=Self.gpu](result, vpad)
                _rank_one_subtract[gpu=Self.gpu](
                    result, vpad, y, this_tau, product, ctx
                )
            k -= 1
        return result^


def _write_vpad[
    dtype: DType,
    ALayout: TensorLayout,
    VLayout: TensorLayout,
    gpu: Bool = False,
](
    packed: TileTensor[
        dtype, ALayout, MutAnyOrigin, Storage=PointerStorage[element_width=1]
    ],
    vpad: TileTensor[
        dtype, VLayout, MutAnyOrigin, Storage=PointerStorage[element_width=1]
    ],
    k: Int,
    n: Int,
    ctx: DeviceContext,
) raises:
    """Unpack column `k`'s reflector into a full-length vector: zero at and
    above `k`, `1` at `k + 1`, `v` below."""

    @always_inline
    def fill[
        w: Int, alignment: Int = 1
    ](coord: Coord) {var packed, var vpad, var k}:
        var i = coord_to_index_list(coord)[0]
        var value = Scalar[dtype](0)
        if i == k + 1:
            value = Scalar[dtype](1)
        elif i > k + 1:
            value = packed[Coord(i, k)]
        vpad.store[1](coord, value)

    elementwise[simd_width=1, target=_target[gpu]()](fill, Coord(n), ctx)


def _row_combination[
    dtype: DType, n: Int, gpu: Bool = False
](mut q: Static[dtype, n, n], mut v: Static[dtype, n]) raises -> Static[
    dtype, n
]:
    """`Q^T v`, as the matrix-vector product `linalg.matmul` reaches
    through its own `transpose_b`-free path: `v^T Q` read as a row."""
    var ctx = q.context()
    var result = zeros[dtype, n](ctx)
    var row: _Dense[dtype] = TileTensor(
        v.view().ptr_at_offset(Coord(0)), row_major(Coord(1, n))
    )
    var out: _Dense[dtype] = TileTensor(
        result.view().ptr_at_offset(Coord(0)), row_major(Coord(1, n))
    )
    _max_matmul[target=_target[gpu]()](out, row, q.view(), ctx)
    ctx.synchronize()
    return result^


def _rank_one_subtract[
    dtype: DType, n: Int, gpu: Bool = False
](
    mut q: Static[dtype, n, n],
    mut v: Static[dtype, n],
    mut y: Static[dtype, n],
    scale: Scalar[dtype],
    mut product: Static[dtype, n, n],
    ctx: DeviceContext,
) raises:
    """`Q -= scale * v y^T`, through `matmul`'s epilogue so the scattered
    store is the only pass over `Q`."""
    var target = q.view()

    @parameter
    @always_inline
    @__copy_capture(target, scale)
    def subtract[
        _dtype: DType,
        lanes: SIMDLength,
        *,
        alignment: Int = align_of[SIMD[_dtype, lanes]](),
    ](idx: IndexList[2], value: SIMD[_dtype, lanes]) capturing -> None:
        var at = Coord(idx[0], idx[1])
        target.store[lanes](
            at,
            target.load[lanes](at) - scale * rebind[SIMD[dtype, lanes]](value),
        )

    var col: _Dense[dtype] = TileTensor(
        v.view().ptr_at_offset(Coord(0)), row_major(Coord(n, 1))
    )
    var row: _Dense[dtype] = TileTensor(
        y.view().ptr_at_offset(Coord(0)), row_major(Coord(1, n))
    )
    _max_matmul[elementwise_lambda_fn=subtract, target=_target[gpu]()](
        product.view(), col, row, ctx
    )
    ctx.synchronize()


def sytrd[
    dtype: DType, n: Int, gpu: Bool = False
](mut a: Static[dtype, n, n]) raises -> TensorTridiagonal[
    dtype, n, gpu
] where dtype.is_floating_point():
    """**Tier 2.** Reduce a symmetric `a` to tridiagonal form by
    Householder reflections, device-resident. LAPACK's `sytrd`.

    `a` is read as symmetric and is **not checked** -- checking costs a
    full pass and the reduction is meaningless on a matrix that is not,
    in a way the caller is better placed to notice. Only the lower triangle
    and the diagonal are read.

    This is the half of an eigendecomposition that has a GEMM in it, and
    it is over half the arithmetic. Each column is four launches and no
    host round trip for the matrix: a single-block kernel forms the
    reflector, `matvec` multiplies the *whole* matrix by the padded
    reflector, `dot` reduces one scalar, and one `matmul` applies the
    symmetric rank-two update `a -= v w^T + w v^T` as a single product
    `[v | w] @ [w | v]^T` with `transpose_b=True`. That last identity is
    what lets numax skip the `syr2k` MAX does not ship.

    `ponytail:` this is the unblocked reduction, and it has two ceilings,
    both closed by the same upgrade. It multiplies the **whole** matrix by
    each reflector rather than the shrinking trailing block, because a
    strided sub-block cannot be handed to `matmul` without staging it dense
    and staging it per column would be `O(n^3)` of memcpy -- so it does
    about `3n^3` flops against LAPACK's `4n^3/3`. And its products are
    matrix-vector shaped, which is bandwidth-bound rather than
    GEMM-bound. The upgrade is LAPACK's blocked `latrd`: accumulate `V` and
    `W` over a panel of `block` columns and apply
    `A -= [V | W] @ [W | V]^T` once per panel with `K = 2 * block`, which is
    the same single-GEMM identity this already uses at `block == 1`. The
    `dot` per column is also a device-to-host synchronization; a blocked
    panel amortizes those too.

    `numax.linalg.array.eigh` is the small-matrix route and needs none of
    this: cyclic Jacobi at a fixed sweep count, differentiable, and
    launchable inside a GPU thread.
    """
    var ctx = a.context()
    var work = zeros[dtype, n, n](ctx)
    var taus = zeros[dtype, n](ctx)
    var vpad = zeros[dtype, n](ctx)
    var scratch = zeros[dtype, _PANEL_THREADS + 1](ctx)
    var left = zeros[dtype, n, 2](ctx)
    var right = zeros[dtype, n, 2](ctx)
    var product = zeros[dtype, n, n](ctx)

    var wv = work.view()
    var tv = taus.view()
    var vv = vpad.view()
    var sv = scratch.view()

    pack_block[target=_target[gpu]()](a.view(), wv, 0, 0, n, n, ctx)

    for k in range(n - 2):
        comptime if gpu:
            ctx.enqueue_function[
                sytd2_column[
                    dtype,
                    ALayout=type_of(wv).LayoutType,
                    VLayout=type_of(vv).LayoutType,
                    TauLayout=type_of(tv).LayoutType,
                    SLayout=type_of(sv).LayoutType,
                    gpu=True,
                ]
            ](
                wv,
                vv,
                tv,
                sv,
                Int32(k),
                Int32(n),
                grid_dim=1,
                block_dim=_PANEL_THREADS,
            )
            ctx.synchronize()
        else:
            sytd2_column(wv, vv, tv, sv, Int32(k), Int32(n))

        var this_tau = taus.to_host()[k]
        if this_tau == 0:
            continue

        # `p = A v` over the whole matrix: `vpad` is zero at and above `k`,
        # so the leading block contributes nothing and needs no staging.
        var p = matvec[gpu=gpu](work, vpad)
        var kappa = dot[gpu=gpu](p, vpad)

        var lv = left.view()
        var rv = right.view()
        comptime if gpu:
            ctx.enqueue_function[
                sytd2_rank_two[
                    dtype,
                    VLayout=type_of(vv).LayoutType,
                    PLayout=type_of(p.view()).LayoutType,
                    LLayout=type_of(lv).LayoutType,
                    gpu=True,
                ]
            ](
                vv,
                p.view(),
                lv,
                rv,
                this_tau,
                kappa,
                Int32(k),
                Int32(n),
                grid_dim=1,
                block_dim=_PANEL_THREADS,
            )
            ctx.synchronize()
        else:
            sytd2_rank_two(
                vv, p.view(), lv, rv, this_tau, kappa, Int32(k), Int32(n)
            )

        _subtract_rank_two[gpu=gpu](work, left, right, product, ctx)

    # `sv` is read by every launch above and `scratch` is named nowhere
    # else, so without this Mojo would destroy it after `.view()`; see
    # `findings.mdc` on the origin-erased view and the queued free.
    _ = scratch^
    var reduced = work.to_host()
    var d = zeros[dtype, n](ctx)
    var e = zeros[dtype, n](ctx)
    var d_host = d.to_host()
    var e_host = e.to_host()
    for i in range(n):
        d_host[i] = reduced[i * n + i]
        if i + 1 < n:
            e_host[i] = reduced[(i + 1) * n + i]
    d.copy_from_host(d_host)
    e.copy_from_host(e_host)

    return TensorTridiagonal[dtype, n, gpu](d^, e^, work^, taus^)


def _subtract_rank_two[
    dtype: DType, n: Int, gpu: Bool = False
](
    mut target: Static[dtype, n, n],
    mut left: Static[dtype, n, 2],
    mut right: Static[dtype, n, 2],
    mut product: Static[dtype, n, n],
    ctx: DeviceContext,
) raises:
    """`target -= left @ right^T`, in one GEMM with the subtraction fused
    into the epilogue.

    With `left = [v | w]` and `right = [w | v]` this is the symmetric
    rank-two update `v w^T + w v^T`, which is the whole trailing update of
    a tridiagonal reduction step.
    """
    var into = target.view()

    @parameter
    @always_inline
    @__copy_capture(into)
    def subtract[
        _dtype: DType,
        lanes: SIMDLength,
        *,
        alignment: Int = align_of[SIMD[_dtype, lanes]](),
    ](idx: IndexList[2], value: SIMD[_dtype, lanes]) capturing -> None:
        var at = Coord(idx[0], idx[1])
        into.store[lanes](
            at, into.load[lanes](at) - rebind[SIMD[dtype, lanes]](value)
        )

    _max_matmul[
        transpose_b=True,
        elementwise_lambda_fn=subtract,
        target=_target[gpu](),
    ](product.view(), left.view(), right.view(), ctx)
    ctx.synchronize()


@always_inline
def _eps[dtype: DType]() -> Scalar[dtype]:
    """Machine epsilon for `dtype`, the deflation threshold the sweeps use."""
    comptime if dtype == DType.float32:
        return Scalar[dtype](1.1920929e-07)
    else:
        return Scalar[dtype](2.220446049250313e-16)


comptime _MAX_QL_SWEEPS = 60
"""Sweeps allowed per eigenvalue before `eigvalsh` gives up.

LAPACK's `sterf` uses `30 * n` in total; per eigenvalue that is a bound
nothing well-formed approaches, since a Wilkinson-shifted QL step converges
cubically once the off-diagonal is small. Reaching it means `e` carries a
NaN or an infinity, and the raise says so rather than looping forever.
"""


def _tql[
    dtype: DType, vectors: Bool
](
    mut d: List[Scalar[dtype]],
    mut e: List[Scalar[dtype]],
    mut z: List[Scalar[dtype]],
) raises where dtype.is_floating_point():
    """Implicit QL with Wilkinson shifts on the symmetric tridiagonal
    `(d, e)`, in place into `d`. LAPACK's `sterf` at `vectors=False` and
    `steqr` at `vectors=True`, where the rotations are also accumulated
    into the row-major `n x n` matrix `z`, which the caller passes in as
    the identity.

    **Tier 2, host-side.** Each sweep is a chain of Givens rotations down
    the band that touches two entries at a time and stops at the first
    negligible subdiagonal -- there is no GEMM to hand any of it to, and
    the deflation test branches on the data. This is the half of an
    eigendecomposition with no MAX in it.

    The two settings cost very differently, and that is the reason for the
    parameter rather than two functions. Values alone are `O(n^2)`,
    negligible beside the reduction. Accumulating `z` applies every
    rotation to a full column pair, `O(n)` per rotation and `O(n^3)`
    overall, at a small constant and in scalar host code -- the one
    genuinely host-bound term in `eigh`, and the `ponytail:` ceiling it
    names. The upgrade is `stedc`, divide and conquer, whose merge phase
    multiplies the sub-problems' eigenvector blocks together and so *is*
    GEMM-shaped.

    `e` is read as the `n` entries `TensorTridiagonal` carries, with
    `e[n-1]` unused and treated as zero. Numerical Recipes' `tqli`, which
    is also Golub and Van Loan's Algorithm 8.3.3 read left to right.
    """
    var n = len(d)
    if n <= 1:
        return
    e[n - 1] = Scalar[dtype](0)
    var eps = _eps[dtype]()

    for l in range(n):
        var sweeps = 0
        while True:
            # Find the first negligible subdiagonal at or past `l`; the
            # block `l..m` is what this sweep works on.
            var m = l
            while m < n - 1:
                var scale = abs(d[m]) + abs(d[m + 1])
                if abs(e[m]) <= eps * scale:
                    break
                m += 1
            if m == l:
                break

            sweeps += 1
            if sweeps > _MAX_QL_SWEEPS:
                raise Error(
                    "eigvalsh: eigenvalue ",
                    l,
                    " did not converge in ",
                    _MAX_QL_SWEEPS,
                    " sweeps -- the matrix holds a NaN or an infinity",
                )

            # Wilkinson shift from the leading 2x2 of the block.
            var g = (d[l + 1] - d[l]) / (Scalar[dtype](2) * e[l])
            var r = _hypot(g, Scalar[dtype](1))
            var signed_r = r if g >= 0 else -r
            g = d[m] - d[l] + e[l] / (g + signed_r)

            var s_ = Scalar[dtype](1)
            var c = Scalar[dtype](1)
            var p = Scalar[dtype](0)
            var i = m - 1
            var underflowed = False
            while i >= l:
                var f = s_ * e[i]
                var b = c * e[i]
                r = _hypot(f, g)
                e[i + 1] = r
                if r == 0:
                    # The chain broke early: a zero rotation splits the
                    # block here and the next pass restarts from `l`.
                    d[i + 1] -= p
                    e[m] = Scalar[dtype](0)
                    underflowed = True
                    break
                s_ = f / r
                c = g / r
                g = d[i + 1] - p
                r = (d[i] - g) * s_ + Scalar[dtype](2) * c * b
                p = s_ * r
                d[i + 1] = g + p
                g = c * r - b
                comptime if vectors:
                    # Rotate columns `i` and `i + 1` of `z`.
                    for row in range(n):
                        var zi = z[row * n + i]
                        var zn = z[row * n + i + 1]
                        z[row * n + i + 1] = s_ * zi + c * zn
                        z[row * n + i] = c * zi - s_ * zn
                i -= 1
            if underflowed:
                continue
            d[l] -= p
            e[l] = g
            e[m] = Scalar[dtype](0)


def eigvalsh[
    dtype: DType, n: Int, gpu: Bool = False
](mut a: Static[dtype, n, n]) raises -> Static[
    dtype, n
] where dtype.is_floating_point():
    """**Tier 2.** The eigenvalues of a symmetric `a`, ascending, without
    the eigenvectors. `numpy.linalg.eigvalsh` / `scipy.linalg.eigvalsh`.

    Two phases, and the split is the whole story. `sytrd` reduces `a` to
    tridiagonal form device-resident -- `O(4n^3/3)` with the cubic term in
    `linalg.matmul` -- and then `_tql` runs implicit QL sweeps on the
    two diagonals on the host. That sweep is `O(n^2)`, sequential, and
    branches on the data, which is tier 2 by numax's definition; beside the
    reduction it is genuinely negligible, and the eigenvalues never asked
    for a vector to be accumulated, which is where the `O(n^3)` of a
    host-side `eigh` would hide.

    Ascending, as SciPy returns them. The `Array` tier's `eigvalsh` is
    cyclic Jacobi at a fixed sweep count and returns its values unsorted --
    the two agree as multisets, and a test pins that.

    `a` is read as symmetric and not checked; see `sytrd`.
    """
    var reduced = sytrd[gpu=gpu](a)
    var d = reduced.d.to_host()
    var e = reduced.e.to_host()
    var unused = List[Scalar[dtype]]()
    _tql[vectors=False](d, e, unused)
    _std_sort(d)
    return Static[dtype, n](a.context(), d^)


struct TensorEigh[dtype: DType, n: Int](
    Movable where dtype.is_floating_point() and n >= 1
):
    """`eigh`'s result: eigenvalues ascending, eigenvectors as columns.

    A struct rather than the `(w, v)` tuple SciPy returns, for the reason
    `qr_factor` returns a `TensorQR`: a `Tuple` of two `Tensor`s cannot be
    destructured in Mojo 1.0, so a tuple-shaped return would hand back a
    pair no caller could take apart. `numax.linalg.array.eigh` returns the
    tuple because `Array` copies.
    """

    var values: Static[Self.dtype, Self.n]
    """The eigenvalues, ascending."""

    var vectors: Static[Self.dtype, Self.n, Self.n]
    """The eigenvectors, one per column, in the order of `values`, so
    `a @ vectors[:, j] == values[j] * vectors[:, j]`. Orthonormal."""

    def __init__(
        out self,
        var values: Static[Self.dtype, Self.n],
        var vectors: Static[Self.dtype, Self.n, Self.n],
    ):
        self.values = values^
        self.vectors = vectors^


def eigh[
    dtype: DType, n: Int, gpu: Bool = False
](mut a: Static[dtype, n, n]) raises -> TensorEigh[
    dtype, n
] where dtype.is_floating_point():
    """**Tier 2.** The eigendecomposition of a symmetric `a`: eigenvalues
    ascending and orthonormal eigenvectors as columns.
    `numpy.linalg.eigh` / `scipy.linalg.eigh`.

    Three steps, two of them GEMM-shaped. `sytrd` reduces `a` to
    tridiagonal form device-resident, `O(4n^3/3)` through `linalg.matmul`.
    Implicit QL then diagonalizes the tridiagonal on the host,
    accumulating its rotations into `Z`, the tridiagonal's own eigenvector
    matrix. And the eigenvectors of `a` are `Q Z`, where `Q` is the
    reduction's orthogonal factor -- LAPACK's `ormtr` -- which is one
    `matmul`.

    `ponytail:` the middle step is the ceiling, and it is stated rather
    than hidden. Accumulating `Z` is `O(n^3)` of scalar Givens rotations
    on the host -- a small constant, but the one term here with no MAX in
    it, and the reason `eigvalsh` exists as a separate name: values alone
    make that step `O(n^2)`. The upgrade is `stedc`, divide and conquer,
    whose merge phase multiplies the sub-problems' eigenvector blocks and
    so is GEMM-shaped; nothing above or below it changes when it lands.

    The `Array` tier's `eigh` is cyclic Jacobi at a fixed sweep count,
    unsorted, differentiable and launchable inside a GPU thread. For a
    matrix small enough to live in registers it is the better tool; this
    is for the sizes it cannot reach.

    `a` is read as symmetric and not checked; see `sytrd`.
    """
    var ctx = a.context()
    var reduced = sytrd[gpu=gpu](a)
    var d = reduced.d.to_host()
    var e = reduced.e.to_host()

    var z = List[Scalar[dtype]](length=n * n, fill=0)
    for i in range(n):
        z[i * n + i] = Scalar[dtype](1)
    _tql[vectors=True](d, e, z)

    # Sort ascending, carrying each eigenvalue's column with it.
    var order = List[Int](capacity=n)
    for i in range(n):
        order.append(i)
    for i in range(1, n):
        var j = i
        while j > 0 and d[order[j]] < d[order[j - 1]]:
            var tmp = order[j]
            order[j] = order[j - 1]
            order[j - 1] = tmp
            j -= 1

    var sorted_values = List[Scalar[dtype]](capacity=n)
    var sorted_z = List[Scalar[dtype]](length=n * n, fill=0)
    for j in range(n):
        var src = order[j]
        sorted_values.append(d[src])
        for row in range(n):
            sorted_z[row * n + j] = z[row * n + src]

    var values = Static[dtype, n](ctx, sorted_values^)
    var z_dev = Static[dtype, n, n](ctx, sorted_z^)
    var q = reduced.q()
    var vectors = matmul[gpu=gpu](q, z_dev)
    return TensorEigh[dtype, n](values^, vectors^)


# ----------------------------------------------------------------- SVD


def _zero_prefix[
    dtype: DType, n: Int, gpu: Bool = False
](mut v: Static[dtype, n], count: Int, ctx: DeviceContext) raises:
    """Zero `v[0 .. count)` on `v`'s device.

    What confines a reflector's update to the trailing block: the product
    `v^T A` or `A u` is taken over the whole matrix, and the entries that
    would write the row or column just finished are cleared before the
    rank-one update reads them -- the same load-bearing detail `sytrd`
    records for its `w`.
    """
    var view = v.view()

    @always_inline
    def fill[w: Int, alignment: Int = 1](coord: Coord) {var view, var count}:
        var i = coord_to_index_list(coord)[0]
        if i < count:
            view.store[1](coord, Scalar[dtype](0))

    elementwise[simd_width=1, target=_target[gpu]()](fill, Coord(n), ctx)


def _left_products[
    dtype: DType, m: Int, n: Int, gpu: Bool = False
](mut a: Static[dtype, m, n], mut v: Static[dtype, m]) raises -> Static[
    dtype, n
]:
    """`v^T A` as an `n`-vector: `v` read as a `1 x m` row against `A`."""
    var ctx = a.context()
    var result = zeros[dtype, n](ctx)
    var row: _Dense[dtype] = TileTensor(
        v.view().ptr_at_offset(Coord(0)), row_major(Coord(1, m))
    )
    var out: _Dense[dtype] = TileTensor(
        result.view().ptr_at_offset(Coord(0)), row_major(Coord(1, n))
    )
    _max_matmul[target=_target[gpu]()](out, row, a.view(), ctx)
    ctx.synchronize()
    return result^


def _rank_one_subtract_rect[
    dtype: DType, m: Int, n: Int, gpu: Bool = False
](
    mut a: Static[dtype, m, n],
    mut col: Static[dtype, m],
    mut row: Static[dtype, n],
    scale: Scalar[dtype],
    mut product: Static[dtype, m, n],
    ctx: DeviceContext,
) raises:
    """`A -= scale * col row^T` for a rectangular `A`, through `matmul`'s
    epilogue -- `_rank_one_subtract` at `m x n`."""
    var target = a.view()

    @parameter
    @always_inline
    @__copy_capture(target, scale)
    def subtract[
        _dtype: DType,
        lanes: SIMDLength,
        *,
        alignment: Int = align_of[SIMD[_dtype, lanes]](),
    ](idx: IndexList[2], value: SIMD[_dtype, lanes]) capturing -> None:
        var at = Coord(idx[0], idx[1])
        target.store[lanes](
            at,
            target.load[lanes](at) - scale * rebind[SIMD[dtype, lanes]](value),
        )

    var c: _Dense[dtype] = TileTensor(
        col.view().ptr_at_offset(Coord(0)), row_major(Coord(m, 1))
    )
    var r: _Dense[dtype] = TileTensor(
        row.view().ptr_at_offset(Coord(0)), row_major(Coord(1, n))
    )
    _max_matmul[elementwise_lambda_fn=subtract, target=_target[gpu]()](
        product.view(), c, r, ctx
    )
    ctx.synchronize()


def _column_of[
    dtype: DType, m: Int, n: Int, gpu: Bool = False
](mut dense: Static[dtype, m, n], k: Int) raises -> Static[dtype, m]:
    """Column `k` of a dense `m x n` as its own vector, gathered on the
    device."""
    var ctx = dense.context()
    var out = zeros[dtype, m](ctx)
    var src = dense.view()
    var dst = out.view()

    @always_inline
    def gather[
        w: Int, alignment: Int = 1
    ](coord: Coord) {var src, var dst, var k}:
        var i = coord_to_index_list(coord)[0]
        dst.store[1](coord, src[Coord(i, k)])

    elementwise[simd_width=1, target=_target[gpu]()](gather, Coord(m), ctx)
    return out^


def _row_of[
    dtype: DType, m: Int, n: Int, gpu: Bool = False
](mut dense: Static[dtype, m, n], k: Int) raises -> Static[dtype, n]:
    """Row `k` of a dense `m x n` as its own vector, on the device -- a
    contiguous copy, so a `pack_block` of one row."""
    var ctx = dense.context()
    var out = zeros[dtype, n](ctx)
    var dst: _Dense[dtype] = TileTensor(
        out.view().ptr_at_offset(Coord(0)), row_major(Coord(1, n))
    )
    pack_block[target=_target[gpu]()](dense.view(), dst, k, 0, 1, n, ctx)
    ctx.synchronize()
    return out^


struct TensorBidiagonal[dtype: DType, m: Int, n: Int, gpu: Bool = False](
    Movable where dtype.is_floating_point() and m >= n and n >= 1
):
    """The bidiagonal reduction `A = Q B P^T` of an `m x n` matrix with
    `m >= n`, device-resident. LAPACK's `gebrd` output.

    `B` is upper bidiagonal: `d` on the diagonal, `e[0 .. n-2]` on the
    superdiagonal (`e[n-1]` is unused and zero, for the reason
    `TensorTridiagonal` gives). `Q` (`m x n`, the thin form) and `P`
    (`n x n`) are orthogonal and held as their Householder vectors -- one
    full-length vector per column of `left` and per row of `right` -- so
    `.q()` and `.p()` materialize them and `svdvals` never asks.
    """

    var d: Static[Self.dtype, Self.n]
    """`B`'s diagonal."""

    var e: Static[Self.dtype, Self.n]
    """`B`'s superdiagonal in `e[0 .. n-2]`; `e[n-1]` unused and zero."""

    var left: Static[Self.dtype, Self.m, Self.n]
    """Column `k` is the left reflector `v_k`, full length, `1` at `k`."""

    var right: Static[Self.dtype, Self.n, Self.n]
    """Row `k` is the right reflector `u_k`, full length, `1` at `k + 1`;
    rows `n-2` and `n-1` are unused."""

    var taus_left: Static[Self.dtype, Self.n]
    var taus_right: Static[Self.dtype, Self.n]

    def __init__(
        out self,
        var d: Static[Self.dtype, Self.n],
        var e: Static[Self.dtype, Self.n],
        var left: Static[Self.dtype, Self.m, Self.n],
        var right: Static[Self.dtype, Self.n, Self.n],
        var taus_left: Static[Self.dtype, Self.n],
        var taus_right: Static[Self.dtype, Self.n],
    ):
        self.d = d^
        self.e = e^
        self.left = left^
        self.right = right^
        self.taus_left = taus_left^
        self.taus_right = taus_right^

    def q(mut self) raises -> Static[Self.dtype, Self.m, Self.n]:
        """The thin `Q`, `m x n`: `H_0 H_1 ... H_{n-1}` applied to the
        first `n` columns of the identity, in reverse so each reflection
        meets a matrix already carrying the ones after it. Each step is a
        `v^T E` product and a rank-one update, both `linalg.matmul`."""
        var ctx = self.left.context()
        var result = zeros[Self.dtype, Self.m, Self.n](ctx)
        var host = result.to_host()
        for i in range(Self.n):
            host[i * Self.n + i] = Scalar[Self.dtype](1)
        result.copy_from_host(host)
        var product = zeros[Self.dtype, Self.m, Self.n](ctx)
        var taus = self.taus_left.to_host()
        var k = Self.n - 1
        while k >= 0:
            if taus[k] != 0:
                var v = _column_of[gpu=Self.gpu](self.left, k)
                var w = _left_products[gpu=Self.gpu](result, v)
                _rank_one_subtract_rect[gpu=Self.gpu](
                    result, v, w, taus[k], product, ctx
                )
            k -= 1
        return result^

    def p(mut self) raises -> Static[Self.dtype, Self.n, Self.n]:
        """`P`, `n x n`: `G_0 G_1 ... G_{n-2}` applied to the identity in
        reverse, the same way `q()` forms `Q`."""
        var ctx = self.right.context()
        var result = zeros[Self.dtype, Self.n, Self.n](ctx)
        var host = result.to_host()
        for i in range(Self.n):
            host[i * Self.n + i] = Scalar[Self.dtype](1)
        result.copy_from_host(host)
        if Self.n < 2:
            return result^
        var product = zeros[Self.dtype, Self.n, Self.n](ctx)
        var taus = self.taus_right.to_host()
        var k = Self.n - 2
        while k >= 0:
            if taus[k] != 0:
                var u = _row_of[gpu=Self.gpu](self.right, k)
                var w = _row_combination[gpu=Self.gpu](result, u)
                _rank_one_subtract[gpu=Self.gpu](
                    result, u, w, taus[k], product, ctx
                )
            k -= 1
        return result^


def gebrd[
    dtype: DType, m: Int, n: Int, gpu: Bool = False
](mut a: Static[dtype, m, n]) raises -> TensorBidiagonal[
    dtype, m, n, gpu
] where (dtype.is_floating_point() and m >= n and n >= 1):
    """**Tier 2.** Reduce an `m x n` matrix, `m >= n`, to upper bidiagonal
    form by alternating left and right Householder reflections,
    device-resident. LAPACK's `gebrd`, unblocked.

    Column `k` takes a left reflector from `a[k.., k]` and row `k` a right
    one from `a[k, k+1..]`; each is applied to the rest of the matrix as a
    matrix-vector product and a rank-one update through `linalg.matmul`,
    with the product's entry for the row or column just finished cleared
    first so the update never touches it. That is the whole of the
    arithmetic, `O(4 m n^2)` at matrix-vector shapes.

    `ponytail:` unblocked, like the first `sytrd`, and with the same
    ceiling: matrix-vector shapes are bandwidth-bound where LAPACK's
    blocked `labrd` accumulates a panel and applies it as GEMMs. The
    upgrade is that panel; nothing above changes when it lands.
    """
    var ctx = a.context()
    var work = zeros[dtype, m, n](ctx)
    var left = zeros[dtype, m, n](ctx)
    var right = zeros[dtype, n, n](ctx)
    var taus_left = zeros[dtype, n](ctx)
    var taus_right = zeros[dtype, n](ctx)
    var scratch = zeros[dtype, _PANEL_THREADS + 1](ctx)
    var product = zeros[dtype, m, n](ctx)

    var wv = work.view()
    var lv = left.view()
    var rv = right.view()
    var tlv = taus_left.view()
    var trv = taus_right.view()
    var sv = scratch.view()
    pack_block[target=_target[gpu]()](a.view(), wv, 0, 0, m, n, ctx)

    for k in range(n):
        # Left reflector on column k.
        comptime if gpu:
            ctx.enqueue_function[
                gebd2_col[
                    dtype,
                    ALayout=type_of(wv).LayoutType,
                    VLayout=type_of(lv).LayoutType,
                    TauLayout=type_of(tlv).LayoutType,
                    SLayout=type_of(sv).LayoutType,
                    gpu=True,
                ]
            ](
                wv,
                lv,
                tlv,
                sv,
                Int32(k),
                Int32(m),
                grid_dim=1,
                block_dim=_PANEL_THREADS,
            )
            ctx.synchronize()
        else:
            gebd2_col(wv, lv, tlv, sv, Int32(k), Int32(m))

        var tau_l = taus_left.to_host()[k]
        if tau_l != 0:
            var v = _column_of[gpu=gpu](left, k)
            var w = _left_products[gpu=gpu](work, v)
            _zero_prefix[gpu=gpu](w, k + 1, ctx)
            _rank_one_subtract_rect[gpu=gpu](work, v, w, tau_l, product, ctx)

        # Right reflector on row k, when there is a superdiagonal to reduce.
        if k + 1 < n:
            comptime if gpu:
                ctx.enqueue_function[
                    gebd2_row[
                        dtype,
                        ALayout=type_of(wv).LayoutType,
                        ULayout=type_of(rv).LayoutType,
                        TauLayout=type_of(trv).LayoutType,
                        SLayout=type_of(sv).LayoutType,
                        gpu=True,
                    ]
                ](
                    wv,
                    rv,
                    trv,
                    sv,
                    Int32(k),
                    Int32(n),
                    grid_dim=1,
                    block_dim=_PANEL_THREADS,
                )
                ctx.synchronize()
            else:
                gebd2_row(wv, rv, trv, sv, Int32(k), Int32(n))

            var tau_r = taus_right.to_host()[k]
            if tau_r != 0:
                var u = _row_of[gpu=gpu](right, k)
                var pcol = matvec[gpu=gpu](work, u)
                _zero_prefix[gpu=gpu](pcol, k + 1, ctx)
                _rank_one_subtract_rect[gpu=gpu](
                    work, pcol, u, tau_r, product, ctx
                )

    _ = scratch^  # as in `sytrd`: pin the workspace past its last launch
    var reduced = work.to_host()
    var d = zeros[dtype, n](ctx)
    var e = zeros[dtype, n](ctx)
    var d_host = d.to_host()
    var e_host = e.to_host()
    for i in range(n):
        d_host[i] = reduced[i * n + i]
        if i + 1 < n:
            e_host[i] = reduced[i * n + i + 1]
    d.copy_from_host(d_host)
    e.copy_from_host(e_host)
    return TensorBidiagonal[dtype, m, n, gpu](
        d^, e^, left^, right^, taus_left^, taus_right^
    )


def _golub_kahan[
    dtype: DType, n: Int, vectors: Bool
](d: List[Scalar[dtype]], e: List[Scalar[dtype]]) raises -> Tuple[
    List[Scalar[dtype]], List[Scalar[dtype]]
] where dtype.is_floating_point():
    """The singular values of the upper bidiagonal `(d, e)`, descending, and
    -- at `vectors=True` -- the `2n x 2n` eigenvector matrix of the
    Golub-Kahan tridiagonal they came from, row-major.

    The Golub-Kahan matrix is the `2n x 2n` symmetric tridiagonal with zero
    diagonal and off-diagonal `d_1, e_1, d_2, e_2, ..., d_n`. Its
    eigenvalues are `+-sigma_i`, and the eigenvector for `+sigma_i`
    interleaves the singular vectors: even entries are `v_i / sqrt(2)`, odd
    entries `u_i / sqrt(2)`. So the sweep `eigh` already runs gives the SVD
    of `B` with no new numerics -- LAPACK's `dbdsvdx` takes the same route.

    `ponytail:` at `vectors=True` this is `O((2n)^3)` of scalar host
    rotations, eight times `eigh`'s on the same `n`; a Golub-Kahan implicit
    QR sweep on the bidiagonal itself (`bdsqr`) is the upgrade. Singular
    vectors for an *exactly* zero singular value are not guaranteed
    orthonormal, since the two zero eigenvalues' vectors may mix; the
    values are exact, and `pinv`/`matrix_rank` never use those vectors.
    """
    comptime size = 2 * n
    var gd = List[Scalar[dtype]](length=size, fill=0)
    var ge = List[Scalar[dtype]](length=size, fill=0)
    for i in range(n):
        ge[2 * i] = d[i]
        if i + 1 < n:
            ge[2 * i + 1] = e[i]
    var z = List[Scalar[dtype]]()
    comptime if vectors:
        z = List[Scalar[dtype]](length=size * size, fill=0)
        for i in range(size):
            z[i * size + i] = Scalar[dtype](1)
    _tql[vectors=vectors](gd, ge, z)
    return (gd^, z^)


def _top_n_descending[
    dtype: DType, n: Int
](values: List[Scalar[dtype]]) -> List[Int]:
    """Indices of the `n` largest of `2n` eigenvalues, descending -- the
    positive half of the Golub-Kahan spectrum."""
    var order = List[Int](capacity=len(values))
    for i in range(len(values)):
        order.append(i)
    for i in range(1, len(values)):
        var j = i
        while j > 0 and values[order[j]] > values[order[j - 1]]:
            var tmp = order[j]
            order[j] = order[j - 1]
            order[j - 1] = tmp
            j -= 1
    var top = List[Int](capacity=n)
    for i in range(n):
        top.append(order[i])
    return top^


def svdvals[
    dtype: DType, m: Int, n: Int, gpu: Bool = False
](mut a: Static[dtype, m, n]) raises -> Static[dtype, n] where (
    dtype.is_floating_point() and m >= n and n >= 1
):
    """**Tier 2.** The singular values of an `m x n` matrix, `m >= n`,
    descending. `scipy.linalg.svdvals`.

    `gebrd` device-resident, then the Golub-Kahan tridiagonal's
    eigenvalues by the `O(n^2)` implicit-QL sweep, the positive half taken.
    Descending, as SciPy returns them; the `Array` tier's `svdvals` is
    one-sided Jacobi and comes back unsorted, and a test pins the two as
    multisets.
    """
    var reduced = gebrd[gpu=gpu](a)
    var gk = _golub_kahan[dtype, n, vectors=False](
        reduced.d.to_host(), reduced.e.to_host()
    )
    var top = _top_n_descending[dtype, n](gk[0])
    var out = List[Scalar[dtype]](capacity=n)
    for i in range(n):
        # `|.|`: an exactly singular matrix's zero pair can land with its
        # nominally positive member a rounding error below zero.
        out.append(abs(gk[0][top[i]]))
    return Static[dtype, n](a.context(), out^)


struct TensorSVD[dtype: DType, m: Int, n: Int](
    Movable where dtype.is_floating_point() and m >= n and n >= 1
):
    """`svd`'s result: `A = U diag(s) V^T`, singular values descending.

    A struct rather than SciPy's `(U, s, Vh)` tuple, for the reason
    `qr_factor` returns a `TensorQR`. The fields follow the `Array` tier's
    `svd` -- `v` holds the right singular vectors as *columns*, so SciPy's
    `Vh` is `transpose(v)`, and `A == U @ diag(s) @ V^T` reads the way the
    reconstruction is written.
    """

    var u: Static[Self.dtype, Self.m, Self.n]
    """The left singular vectors as columns; `m x n`, the thin form."""

    var s: Static[Self.dtype, Self.n]
    """The singular values, descending."""

    var v: Static[Self.dtype, Self.n, Self.n]
    """The right singular vectors as columns."""

    def __init__(
        out self,
        var u: Static[Self.dtype, Self.m, Self.n],
        var s: Static[Self.dtype, Self.n],
        var v: Static[Self.dtype, Self.n, Self.n],
    ):
        self.u = u^
        self.s = s^
        self.v = v^


def svd[
    dtype: DType, m: Int, n: Int, gpu: Bool = False
](mut a: Static[dtype, m, n]) raises -> TensorSVD[dtype, m, n] where (
    dtype.is_floating_point() and m >= n and n >= 1
):
    """**Tier 2.** The thin singular value decomposition of an `m x n`
    matrix, `m >= n`: `A = U diag(s) V^T` with `s` descending.
    `scipy.linalg.svd(a, full_matrices=False)`.

    Three steps, two of them GEMM-shaped, the shape `eigh` has. `gebrd`
    reduces `A` to bidiagonal `B = Q^T A P` device-resident. The
    Golub-Kahan tridiagonal of `B` is diagonalized on the host by the same
    implicit-QL sweep `eigh` uses, its eigenvectors interleaving `B`'s
    singular vectors. And `U = Q U_B`, `V = P V_B` are two `matmul`s.

    `ponytail:` the middle step is the ceiling, and it is larger than
    `eigh`'s by the factor the doubling costs -- `O((2n)^3)` scalar host
    rotations. The upgrade is a bidiagonal QR sweep (`bdsqr`) or divide and
    conquer; nothing above or below changes when it lands. `svdvals` skips
    the vectors and is `O(n^2)` past the reduction.

    Rectangular, unlike the `Array` tier's square-only one-sided Jacobi,
    and descending where that one is unsorted; both docstrings say so.
    """
    var ctx = a.context()
    var reduced = gebrd[gpu=gpu](a)
    var gk = _golub_kahan[dtype, n, vectors=True](
        reduced.d.to_host(), reduced.e.to_host()
    )
    var values = gk[0].copy()
    var z = gk[1].copy()
    comptime size = 2 * n
    var top = _top_n_descending[dtype, n](values)

    var s_host = List[Scalar[dtype]](capacity=n)
    var ub = List[Scalar[dtype]](length=n * n, fill=0)
    var vb = List[Scalar[dtype]](length=n * n, fill=0)
    var root_two = Scalar[dtype](1.4142135623730951)
    for j in range(n):
        var col = top[j]
        s_host.append(abs(values[col]))
        for i in range(n):
            vb[i * n + j] = z[(2 * i) * size + col] * root_two
            ub[i * n + j] = z[(2 * i + 1) * size + col] * root_two

    var u_b = Static[dtype, n, n](ctx, ub^)
    var v_b = Static[dtype, n, n](ctx, vb^)
    var q = reduced.q()
    var p = reduced.p()
    var u = matmul[gpu=gpu](q, u_b)
    var v = matmul[gpu=gpu](p, v_b)
    return TensorSVD[dtype, m, n](u^, Static[dtype, n](ctx, s_host^), v^)


def matrix_rank[
    dtype: DType, m: Int, n: Int, gpu: Bool = False
](
    mut a: Static[dtype, m, n], tol: Optional[Float64] = None
) raises -> Int where (dtype.is_floating_point() and m >= n and n >= 1):
    """**Tier 2.** How many singular values exceed `tol`.
    `numpy.linalg.matrix_rank`.

    `tol` defaults to NumPy's: the largest singular value times
    `max(m, n)` times the machine epsilon of `dtype`, which is the noise
    floor an SVD of that size can be expected to carry. Pass an explicit
    `tol` to ask a different question -- "how many directions carry more
    than one part in a thousand" is `tol = 1e-3 * s_max`.

    An `Int`, where the `Array` tier returns `T`: one `Tensor` is one
    matrix, so there is one rank, and nothing here needs to stay branchless.
    """
    var s = svdvals[gpu=gpu](a).to_host()
    var eps: Float64
    comptime if dtype == DType.float32:
        eps = 1.1920929e-07
    else:
        eps = 2.220446049250313e-16
    var threshold = (
        tol.value() if tol else Float64(s[0]) * Float64(max(m, n)) * eps
    )
    var rank = 0
    for i in range(n):
        if Float64(s[i]) > threshold:
            rank += 1
    return rank
