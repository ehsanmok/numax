"""`logsumexp`: `log(sum(exp(x)))` without the overflow, over a `Tensor`
through MAX's own monoid and over an `Array` as a fixed-iteration
`FloatLike` kernel. `scipy.special.logsumexp`.

## The MAX gate, and the one delegation in `numax.special`

MAX has no `logsumexp` entry point -- `nn.softmax`'s `logsoftmax=True` is
the closest, and it returns the whole normalized tensor rather than the
scalar. But it ships the *fold*: `algorithm.reduce_op.OnlineLogSumExp` is
the flash-style monoid (Milakov and Gimelshein 2018) that carries `m =
max` and `l = sum(exp(x - m))` and combines two such states without ever
forming `exp(x)`. So the `Tensor` overload here is `numax.linalg.dot`'s
shape exactly: MAX's `rowwise` scaffolder driving MAX's monoid, threaded
on CPU and warp-tiered on GPU, with only `m + log(l)` coming back to the
host. **Delegate** for the monoid, **extend** for the entry point, and the
first time anything in `numax.special` routes to MAX -- recorded in
`docs/parity.md`.

**That overload is tier 2** (host-driven, device-resident). The `Array`
overload is **tier 1**: the running maximum is `max_of`, the sum is a
fixed loop over the comptime length, so it runs per lane inside a kernel
and differentiates at `Dual` -- the softmax normalizer's gradient with no
adjoint rule written.

Both are the stable form: `m + log(sum(exp(x - m)))` with `m = max(x)`,
so `logsumexp([-1000, -1000.5])` is `-999.526` rather than `log(0)`.
"""

from std.collections import Array
from std.math import log as _log

from algorithm import rowwise
from algorithm.reduce_op import OnlineLogSumExp
from layout import Coord, TileTensor
from layout.tile_tensor import PointerStorage
from max.gpu.host import DeviceContext
from std.utils import IndexList

from ..core.array import Static
from ..core.numeric import FloatLike, max_of


def logsumexp[T: FloatLike, n: Int](xs: Array[T, n]) -> T where n > 0:
    """`log(sum(exp(xs)))` over a register-resident `Array`, stably.

    Tier 1: `n - 1` `max_of`s for the shift, `n` exponentials, one log.
    At `Dual` the derivative is the softmax of `xs`, which is the quantity
    a cross-entropy gradient needs.
    """
    var m = xs[0].copy()
    for i in range(1, n):
        m = max_of(m, xs[i])
    var total = T.constant(0.0)
    for i in range(n):
        total = total + (xs[i] - m).exp()
    return m + total.ln()


def logsumexp[
    dtype: DType, n: Int, gpu: Bool = False
](mut xs: Static[dtype, n]) raises -> Scalar[dtype] where (
    dtype.is_floating_point() and n > 0
):
    """`log(sum(exp(xs)))` over every element of a `Tensor`, stably, on
    its own device. `scipy.special.logsumexp(a)`.

    MAX's `OnlineLogSumExp` monoid through its `rowwise` scaffolder --
    `numax.linalg.dot`'s shape with a different fold -- so the tensor is
    read once, no `exp(x)` is ever materialized, and only the scalar
    returns. Tier 2; the `Array` overload is the tier-1 sibling.
    """
    comptime target = "gpu" if gpu else "cpu"
    comptime simd_width = rowwise.pick_simd_width[
        OnlineLogSumExp[dtype, 1], target, 64, dtype
    ]()
    var ctx = xs.context()
    var out = Static[dtype, 1](ctx)
    var src = xs.view()
    var dst = out.view()

    @always_inline
    def identity[
        w: Int
    ](tile: SIMD[dtype, w], idx: IndexList[1]) -> SIMD[dtype, w]:
        return tile

    @always_inline
    def body[
        params: rowwise.ContextParams
    ](row_coords: Coord, mut c: rowwise.Context[params]) {var src, var dst}:
        @always_inline
        def load[
            width: Int, alignment: Int, coord_rank: Int
        ](idx: IndexList[coord_rank]) {var src} -> SIMD[dtype, width]:
            return src.load[width](Coord(idx))

        var row = rowwise.Row[params, dtype, dtype, 0, 1, is_cached=False](
            row_coords, n, c, load
        )
        var state = row.reduce[OnlineLogSumExp[dtype, params.simd_width]](
            identity, load
        )
        var value = state.m[0] + _log(state.l[0])

        @always_inline
        def write(oc: IndexList[1]) {var value, var dst}:
            dst.store[1](Coord(0), value)

        row.emit(write)

    rowwise.launch[
        axis=0,
        simd_width=simd_width,
        target=target,
        num_phases=1,
        associative=True,
    ](body, Coord(IndexList[1](n)), Optional(ctx))
    return out.to_host()[0]
