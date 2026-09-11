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
from .blas import _target, dot, matvec
from .common import _Dense
from .panel import (
    _PANEL_THREADS,
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


def _tql_values[
    dtype: DType
](
    mut d: List[Scalar[dtype]], mut e: List[Scalar[dtype]]
) raises where dtype.is_floating_point():
    """Implicit QL with Wilkinson shifts on the symmetric tridiagonal
    `(d, e)`, eigenvalues only, in place into `d`. LAPACK's `sterf`.

    **Tier 2, host-side, and `O(n^2)`.** Each sweep is a chain of Givens
    rotations down the band that touches two entries at a time and stops
    at the first negligible subdiagonal -- there is no GEMM to hand any of
    it to, and the deflation test branches on the data. This is the half of
    an eigendecomposition with no MAX in it, and at `O(n^2)` beside the
    reduction's `O(n^3)` it is the half that costs nothing to run on the
    host.

    `e` is read as the `n` entries `TensorTridiagonal` carries, with
    `e[n-1]` unused and treated as zero. Numerical Recipes' `tqli` without
    the eigenvector accumulation, which is also Golub and Van Loan's
    Algorithm 8.3.3 read left to right.
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
    `linalg.matmul` -- and then `_tql_values` runs implicit QL sweeps on the
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
    _tql_values(d, e)
    _std_sort(d)
    return Static[dtype, n](a.context(), d^)
