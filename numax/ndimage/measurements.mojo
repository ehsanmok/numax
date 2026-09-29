"""Connected-component labeling over `Tensor`: `label`, SciPy's
`scipy.ndimage.label`.

**Tier 2.** Minimum-label propagation with pointer jumping, on the input's
device at `gpu=True`: every foreground element starts labeled with its own
flat index, and each round takes the smallest label among itself and its
neighbors, then replaces its label by that pixel's label (the jump that
makes long components converge in a logarithmic number of rounds rather
than in their diameter). A round is one launch; the host reads one
changed-flag maximum per round to stop. At the fixed point every
component carries the flat index of its first element in raster order,
which is exactly the order SciPy numbers components in, so a scan over
the components' roots renumbers them `1, 2, ...` to match.

Connectivity is SciPy's `generate_binary_structure(rank, connectivity)`:
neighbors whose offsets are each in `{-1, 0, 1}` with at most
`connectivity` of them nonzero -- faces at `1`, all `3^rank - 1` at
`rank`.

## The MAX gate

Nothing: MAX has no connected-component labeling. **Extend.**
"""

from algorithm.rowwise_types import RowCoord
from layout import Coord, coord_to_index_list
from layout.tile_layout import row_major
from max.algorithm.functional import elementwise

from ..core.rowwise import reduce_all
from ..core.tensorlike import TensorLike
from ..core.tensor import (
    Dynamic,
    _dyn_shape,
    _dyn_shape_from,
    _same_order,
    _scan_device,
)


def _target[gpu: Bool]() -> StaticString:
    comptime if gpu:
        return "gpu"
    else:
        return "cpu"


struct Labeled[rank: Int](Movable):
    """What `label` returns: SciPy's `(labeled_array, num_features)`."""

    var labels: Dynamic[DType.int32, Self.rank]
    """`0` for background, `1 .. num_features` for the components."""
    var num_features: Int
    """How many components."""

    def __init__(
        out self, var labels: Dynamic[DType.int32, Self.rank], num_features: Int
    ):
        """Build from the parts.

        Args:
            labels: The labeled array.
            num_features: The component count.
        """
        self.labels = labels^
        self.num_features = num_features


def label[
    T: TensorLike, gpu: Bool = False
](input: T, connectivity: Int = 1) raises -> Labeled[T.LayoutType.rank] where (
    T.LayoutType.rank >= 1 and T.LayoutType.rank <= 8
):
    """Label the connected components of the nonzero elements of `input`.
    `scipy.ndimage.label(input, structure=generate_binary_structure(rank,
    connectivity))`.

    Parameters:
        T: The tensor type of `input`, rank 1 to 8.
        gpu: Whether to label on the input's device.

    Args:
        input: The array; nonzero elements are foreground.
        connectivity: `1` joins elements sharing a face, `rank` elements
            sharing any corner.

    Returns:
        A `Labeled` with the `int32` labels, SciPy's numbering, and the
        component count.

    Raises:
        On a connectivity outside `[1, rank]`, or a device failure.
    """
    comptime rank = T.LayoutType.rank
    if connectivity < 1 or connectivity > rank:
        raise Error(
            "label: connectivity must be in [1, ", rank, "], got ", connectivity
        )
    var dims = SIMD[DType.int64, 8](1)
    comptime for a in range(rank):
        dims[a] = Int64(input.dim_at(a))
    var strides = SIMD[DType.int64, 8](1)
    for a in range(rank - 2, -1, -1):
        strides[a] = strides[a + 1] * dims[a + 1]
    var count = input.size()
    var src = _same_order(input, row_major(_dyn_shape[1](count)))
    var ctx = src.context()
    var lab = Dynamic[DType.int32, 1](row_major(_dyn_shape[1](count)), ctx)
    var nxt = Dynamic[DType.int32, 1](row_major(_dyn_shape[1](count)), ctx)
    var flags = Dynamic[DType.int32, 1](row_major(_dyn_shape[1](count)), ctx)
    var sp = src.tile().ptr.unsafe_origin_cast[MutAnyOrigin]()
    var lp = lab.tile()

    @always_inline
    def seed[w: Int, alignment: Int = 1](coord: Coord) {var sp, var lp}:
        var i = coord_to_index_list(coord)[0]
        lp.ptr[unsafe_offset=i] = Int32(i + 1) if sp[
            unsafe_offset=i
        ] != 0 else Int32(0)

    elementwise[simd_width=1, target=_target[gpu]()](seed, Coord(count), ctx)
    ctx.synchronize()

    var neighbors = 1
    for _ in range(rank):
        neighbors *= 3
    while True:
        var ap = lab.tile()
        var bp = nxt.tile()
        var fp = flags.tile()

        @always_inline
        def step[
            w: Int, alignment: Int = 1
        ](coord: Coord) {
            var ap,
            var bp,
            var fp,
            var dims,
            var strides,
            var neighbors,
            var connectivity,
        }:
            var i = coord_to_index_list(coord)[0]
            var own = ap.ptr[unsafe_offset=i]
            var best = own
            if own != 0:
                var at = SIMD[DType.int64, 8](0)
                var rest = i
                comptime for a in range(rank):
                    at[a] = Int64(rest // Int(strides[a]))
                    rest = rest % Int(strides[a])
                for code in range(neighbors):
                    var c = code
                    var nonzero = 0
                    var offset = 0
                    var inside = True
                    comptime for a in range(rank):
                        var d = c % 3 - 1
                        c //= 3
                        if d != 0:
                            nonzero += 1
                        var j = Int(at[a]) + d
                        if j < 0 or j >= Int(dims[a]):
                            inside = False
                        offset += j * Int(strides[a])
                    if nonzero == 0 or nonzero > connectivity or not inside:
                        continue
                    var other = ap.ptr[unsafe_offset=offset]
                    if other != 0 and other < best:
                        best = other
                # The jump: the label of the pixel the label names.
                var jumped = ap.ptr[unsafe_offset=Int(best) - 1]
                if jumped != 0 and jumped < best:
                    best = jumped
            bp.ptr[unsafe_offset=i] = best
            fp.ptr[unsafe_offset=i] = Int32(1) if best != own else Int32(0)

        elementwise[simd_width=1, target=_target[gpu]()](
            step, Coord(count), ctx
        )
        ctx.synchronize()
        var changed: Int
        comptime if gpu:
            var peak = Dynamic[DType.int32, 1](row_major(_dyn_shape[1](1)), ctx)

            @always_inline
            def identity[
                ww: Int
            ](tile: SIMD[DType.int32, ww], idx: RowCoord[1]) {} -> SIMD[
                DType.int32, ww
            ]:
                return tile

            reduce_all[monoid="max", gpu=True](
                flags.tile(), peak.tile(), identity, count, Optional(ctx)
            )
            changed = Int(peak.to_host()[0])
        else:
            var host = flags.to_host()
            changed = 0
            for j in range(count):
                if host[j] != 0:
                    changed = 1
                    break
        var held = lab^
        lab = nxt^
        nxt = held^
        if changed == 0:
            break

    # Roots are the elements whose label is their own index; their
    # running count renumbers every component in raster order.
    var roots = Dynamic[DType.int32, 1](row_major(_dyn_shape[1](count)), ctx)
    var rp = roots.tile()
    var lp2 = lab.tile()

    @always_inline
    def mark[w: Int, alignment: Int = 1](coord: Coord) {var rp, var lp2}:
        var i = coord_to_index_list(coord)[0]
        rp.ptr[unsafe_offset=i] = Int32(1) if lp2.ptr[unsafe_offset=i] == Int32(
            i + 1
        ) else Int32(0)

    elementwise[simd_width=1, target=_target[gpu]()](mark, Coord(count), ctx)
    ctx.synchronize()
    var ranks: Dynamic[DType.int32, 1]
    var total: Int
    comptime if gpu:
        ranks = _scan_device["sum"](
            roots, row_major(_dyn_shape[1](count)), count, 1
        )
        total = Int(ranks.to_host()[count - 1])
    else:
        var host = roots.to_host()
        var running = Int32(0)
        for j in range(count):
            running += host[j]
            host[j] = running
        total = Int(running)
        ranks = Dynamic[DType.int32, 1](row_major(_dyn_shape[1](count)), ctx)
        ranks.copy_from_host(host^)
    var out = Dynamic[DType.int32, 1](row_major(_dyn_shape[1](count)), ctx)
    var kp = ranks.tile()
    var op = out.tile()
    var lp3 = lab.tile()

    @always_inline
    def renumber[
        w: Int, alignment: Int = 1
    ](coord: Coord) {var kp, var op, var lp3}:
        var i = coord_to_index_list(coord)[0]
        var l = Int(lp3.ptr[unsafe_offset=i])
        op.ptr[unsafe_offset=i] = kp.ptr[
            unsafe_offset=l - 1
        ] if l != 0 else Int32(0)

    elementwise[simd_width=1, target=_target[gpu]()](
        renumber, Coord(count), ctx
    )
    ctx.synchronize()
    _ = src^
    var extents = List[Int](capacity=rank)
    for a in range(rank):
        extents.append(Int(dims[a]))
    var shaped = Dynamic[DType.int32, rank](
        out._buffer,
        row_major(_dyn_shape_from[rank](extents)),
        out.host_addressable,
    )
    return Labeled[rank](shaped^, total)
