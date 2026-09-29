"""The symmetric indefinite factorization `a = L D L^T`: `ldl`, SciPy's
`scipy.linalg.ldl`, and the `LDL` it returns.

**Tier 2.** Bunch-Kaufman pivoting chooses `1 x 1` or `2 x 2` blocks by
comparing magnitudes, a branch on data. The factorization is
`sytf2_lower` (`numax.linalg.panel`), LAPACK's unblocked `sytf2`, and
SciPy's post-processing into `(lu, d, perm)` is `ldl_unpack_lower`: two
single-block launches, so `gpu=True` runs the whole of it on the device
and the host reads nothing. SciPy calls the blocked `sytrf`, whose pivot
choices are `sytf2`'s -- LAPACK runs `sytf2` itself below its block size
-- so the factors agree with SciPy's to rounding.

**Ceiling.** One thread block does all `O(n^3 / 3)` of the update and the
`O(n^2)` of serial pivot scans, the single-SM ceiling `getrf_panel`
states; a blocked `sytrf` over `lasyf` panels, its trailing update in
`linalg.matmul`, is the upgrade.

## The MAX gate

Nothing: MAX ships no factorization of a symmetric indefinite matrix.
**Extend.**
"""

from ..core.tensorlike import TensorLike, dim
from ..core.tensor import Static, _same_order
from .common import _mut_view
from .panel import _PANEL_THREADS, ldl_unpack_lower, sytf2_lower


struct LDL[dtype: DType, n: Int](Movable where dtype.is_floating_point()):
    """What `ldl` returns: SciPy's `(lu, d, perm)`, with `a = lu @ d @
    lu^T` and `lu[perm]` lower triangular."""

    var lu: Static[Self.dtype, Self.n, Self.n]
    """The (row-permuted) unit lower-triangular factor."""
    var d: Static[Self.dtype, Self.n, Self.n]
    """Block diagonal, `1 x 1` and symmetric `2 x 2` blocks."""
    var perm: Static[DType.int64, Self.n]
    """The row order that makes `lu[perm]` lower triangular."""

    def __init__(
        out self,
        var lu: Static[Self.dtype, Self.n, Self.n],
        var d: Static[Self.dtype, Self.n, Self.n],
        var perm: Static[DType.int64, Self.n],
    ):
        """Build from the three parts.

        Args:
            lu: The permuted unit lower-triangular factor.
            d: The block-diagonal factor.
            perm: The triangularizing row order.
        """
        self.lu = lu^
        self.d = d^
        self.perm = perm^


def ldl[
    T: TensorLike, gpu: Bool = False
](a: T) raises -> LDL[T.dtype, dim[T, 0]] where (
    T.dtype.is_floating_point()
    and T.LayoutType.rank == 2
    and T.LayoutType.all_dims_known
    and dim[T, 1] == dim[T, 0]
):
    """**Tier 2.** The Bunch-Kaufman factorization of a symmetric
    (possibly indefinite) `a`: `a = lu @ d @ lu^T`.
    `scipy.linalg.ldl(a)`, `lower=True`.

    Reads the lower triangle. `d` is block diagonal with `1 x 1` and `2 x
    2` blocks, and `lu` is unit lower triangular after the row
    permutation `perm`, exactly SciPy's layout -- see the module docstring
    for how and where it runs.

    Parameters:
        T: The tensor type of `a`, square, extents known at compile time.
        gpu: Whether to factor on `a`'s device.

    Args:
        a: The symmetric matrix; only its lower triangle is read.

    Returns:
        An `LDL` with `lu`, `d` and `perm`.

    Raises:
        If a device allocation, copy or launch fails. A singular `a` does
        not raise: its zero pivot yields non-finite entries, as LAPACK's
        `info > 0` does.
    """
    comptime dtype = T.dtype
    comptime n = dim[T, 0]
    var work = _same_order(a, Static[dtype, n, n]._static_layout())
    var ctx = work.context()
    var ipiv = Static[DType.int32, n](ctx)
    var control = Static[DType.int32, 2](ctx)
    var lu = Static[dtype, n, n](ctx)
    var d = Static[dtype, n, n](ctx)
    var perm = Static[DType.int64, n](ctx)
    var scratch = Static[DType.int32, 3 * n](ctx)
    var wv = work.tile()
    var pv = ipiv.tile()
    var cv = control.tile()
    var lv = lu.tile()
    var dv = d.tile()
    var ov = perm.tile()
    var sv = scratch.tile()
    comptime if gpu:
        ctx.enqueue_function[
            sytf2_lower[
                dtype,
                ALayout=type_of(wv).LayoutType,
                PLayout=type_of(pv).LayoutType,
                CLayout=type_of(cv).LayoutType,
                gpu=True,
            ]
        ](wv, pv, cv, Int32(n), grid_dim=1, block_dim=_PANEL_THREADS)
        ctx.enqueue_function[
            ldl_unpack_lower[
                dtype,
                ALayout=type_of(wv).LayoutType,
                PLayout=type_of(pv).LayoutType,
                LLayout=type_of(lv).LayoutType,
                DLayout=type_of(dv).LayoutType,
                OLayout=type_of(ov).LayoutType,
                SLayout=type_of(sv).LayoutType,
                gpu=True,
            ]
        ](
            wv,
            pv,
            lv,
            dv,
            ov,
            sv,
            Int32(n),
            grid_dim=1,
            block_dim=_PANEL_THREADS,
        )
    else:
        sytf2_lower(wv, pv, cv, Int32(n))
        ldl_unpack_lower(wv, pv, lv, dv, ov, sv, Int32(n))
    ctx.synchronize()
    _ = work^
    _ = ipiv^
    _ = control^
    _ = scratch^
    return LDL[dtype, n](lu^, d^, perm^)
