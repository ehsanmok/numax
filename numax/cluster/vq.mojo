"""Vector quantization and k-means over `Tensor`: `whiten`, `vq`, `kmeans`
and `kmeans2`, SciPy's `scipy.cluster.vq`.

**Tier 2**, on the observations' device at `gpu=True`.

- `vq` assigns each observation its nearest code, one lane per
  observation scanning the codebook: `cdist` and a row argmin fused into
  one pass, returning SciPy's Euclidean distortions.
- The centroid update is one lane per (code, feature) pair, summing the
  observations the code owns; a code that owns none is flagged.
- `kmeans(obs, guess)` is SciPy's `_kmeans`: Lloyd steps until the mean
  distortion changes by at most `thresh`, dropping codes that lose all
  their observations, and the final distortion measured against the final
  codebook. `kmeans(obs, k, rng)` restarts it `iter` times from `k`
  distinct observations and keeps the best.
- `kmeans2(data, guess, iter)` is SciPy's fixed-count loop with
  `minit="matrix"`: an empty cluster keeps its previous centroid, and the
  labels returned are the last assignment, made before the last update.
  `kmeans2(data, k, rng, iter)` starts from `k` distinct observations,
  SciPy's `minit="points"`.

The random starts draw their indices by Floyd's algorithm from the
`Generator`'s Philox stream on the host -- `k` integers, control data --
so they are reproducible with a seed but not SciPy's draws.

## The MAX gate

MAX has `nn.argmaxmin` but no distance-then-argmin and no clustering;
the fused lane is one pass where the two would be two, and the Lloyd loop
is numax's. **Extend.**
"""

from algorithm.rowwise_types import RowCoord
from layout import Coord, coord_to_index_list
from layout.tile_layout import row_major
from max.algorithm.functional import elementwise
from std.math import sqrt as _sqrt
from std.random.philox import Random

from ..core.rowwise import reduce_all
from ..core.tensorlike import TensorLike
from ..core.tensor import Dynamic, _dyn_shape, _same_order, asarray, copy
from ..stats.random import Generator


def _target[gpu: Bool]() -> StaticString:
    comptime if gpu:
        return "gpu"
    else:
        return "cpu"


def _rows[
    T: TensorLike
](a: T) raises -> Dynamic[T.dtype, 2] where T.LayoutType.rank == 2:
    return _same_order(a, row_major(_dyn_shape[2](a.dim_at(0), a.dim_at(1))))


struct VQResult[dtype: DType](Movable):
    """What `vq` returns: SciPy's `(code, dist)`."""

    var code: Dynamic[DType.int64, 1]
    """Each observation's nearest code."""
    var dist: Dynamic[Self.dtype, 1]
    """The Euclidean distance to it."""

    def __init__(
        out self,
        var code: Dynamic[DType.int64, 1],
        var dist: Dynamic[Self.dtype, 1],
    ):
        """Build from the parts.

        Args:
            code: The assignments.
            dist: The distortions.
        """
        self.code = code^
        self.dist = dist^


struct KMeansResult[dtype: DType](Movable):
    """What `kmeans` returns: SciPy's `(codebook, distortion)`."""

    var codebook: Dynamic[Self.dtype, 2]
    """The centroids, one per row; codes that lost every observation are
    dropped, as SciPy drops them."""
    var distortion: Float64
    """The mean distance from an observation to its nearest centroid."""

    def __init__(
        out self, var codebook: Dynamic[Self.dtype, 2], distortion: Float64
    ):
        """Build from the parts.

        Args:
            codebook: The centroids.
            distortion: The mean distortion.
        """
        self.codebook = codebook^
        self.distortion = distortion


struct KMeans2Result[dtype: DType](Movable):
    """What `kmeans2` returns: SciPy's `(centroid, label)`."""

    var centroid: Dynamic[Self.dtype, 2]
    """The centroids, one per row."""
    var label: Dynamic[DType.int64, 1]
    """Each observation's cluster, from the last assignment."""

    def __init__(
        out self,
        var centroid: Dynamic[Self.dtype, 2],
        var label: Dynamic[DType.int64, 1],
    ):
        """Build from the parts.

        Args:
            centroid: The centroids.
            label: The assignments.
        """
        self.centroid = centroid^
        self.label = label^


def _vq[
    dtype: DType, gpu: Bool
](obs: Dynamic[dtype, 2], book: Dynamic[dtype, 2]) raises -> VQResult[
    dtype
] where dtype.is_floating_point():
    var ctx = obs.context()
    var n = obs.dim[0]()
    var k = obs.dim[1]()
    var c = book.dim[0]()
    var code = Dynamic[DType.int64, 1](row_major(_dyn_shape[1](n)), ctx)
    var dist = Dynamic[dtype, 1](row_major(_dyn_shape[1](n)), ctx)
    var op = obs.tile().ptr.unsafe_origin_cast[ImmutAnyOrigin]()
    var bp = book.tile().ptr.unsafe_origin_cast[ImmutAnyOrigin]()
    var cp = code.tile()
    var dp = dist.tile()

    @always_inline
    def body[
        w: Int, alignment: Int = 1
    ](coord: Coord) {var op, var bp, var cp, var dp, var k, var c}:
        var i = coord_to_index_list(coord)[0]
        var best = 0
        var best_d = Scalar[dtype].MAX
        for j in range(c):
            var acc = Scalar[dtype](0)
            for f in range(k):
                var diff = (
                    op[unsafe_offset=i * k + f] - bp[unsafe_offset=j * k + f]
                )
                acc += diff * diff
            if acc < best_d:
                best_d = acc
                best = j
        cp.ptr[unsafe_offset=i] = Int64(best)
        dp.ptr[unsafe_offset=i] = _sqrt(best_d)

    elementwise[simd_width=1, target=_target[gpu]()](body, Coord(n), ctx)
    ctx.synchronize()
    return VQResult[dtype](code^, dist^)


def _mean_of[
    dtype: DType, gpu: Bool
](mut v: Dynamic[dtype, 1]) raises -> Float64 where dtype.is_floating_point():
    var count = v.size()
    comptime if gpu:
        var ctx = v.context()
        var out = Dynamic[dtype, 1](row_major(_dyn_shape[1](1)), ctx)

        @always_inline
        def identity[
            w: Int
        ](tile: SIMD[dtype, w], idx: RowCoord[1]) {} -> SIMD[dtype, w]:
            return tile

        reduce_all[monoid="sum", gpu=True](
            v.tile(), out.tile(), identity, count, Optional(ctx)
        )
        return Float64(out.to_host()[0]) / Float64(count)
    else:
        var host = v.to_host()
        var total = 0.0
        for i in range(count):
            total += Float64(host[i])
        return total / Float64(count)


def _update[
    dtype: DType, gpu: Bool
](
    obs: Dynamic[dtype, 2],
    code: Dynamic[DType.int64, 1],
    old: Dynamic[dtype, 2],
) raises -> Tuple[
    Dynamic[dtype, 2], Dynamic[DType.int64, 1]
] where dtype.is_floating_point():
    """SciPy's `update_cluster_means`: each code's centroid over the
    observations it owns, and whether it owns any; an empty code keeps
    `old`'s row, which `kmeans2` wants and `kmeans` discards."""
    var ctx = obs.context()
    var n = obs.dim[0]()
    var k = obs.dim[1]()
    var c = old.dim[0]()
    var means = Dynamic[dtype, 2](row_major(_dyn_shape[2](c, k)), ctx)
    var members = Dynamic[DType.int64, 1](row_major(_dyn_shape[1](c)), ctx)
    var op = obs.tile().ptr.unsafe_origin_cast[ImmutAnyOrigin]()
    var cp = code.tile().ptr.unsafe_origin_cast[ImmutAnyOrigin]()
    var gp = old.tile().ptr.unsafe_origin_cast[ImmutAnyOrigin]()
    var mp = means.tile()
    var hp = members.tile()

    @always_inline
    def body[
        w: Int, alignment: Int = 1
    ](coord: Coord) {var op, var cp, var gp, var mp, var hp, var n, var k}:
        var e = coord_to_index_list(coord)[0]
        var j = e // k
        var f = e % k
        var total = Scalar[dtype](0)
        var count = 0
        for i in range(n):
            if Int(cp[unsafe_offset=i]) == j:
                total += op[unsafe_offset=i * k + f]
                count += 1
        mp.ptr[unsafe_offset=e] = (
            total / Scalar[dtype](count) if count > 0 else gp[unsafe_offset=e]
        )
        if f == 0:
            hp.ptr[unsafe_offset=j] = Int64(count)

    elementwise[simd_width=1, target=_target[gpu]()](body, Coord(c * k), ctx)
    ctx.synchronize()
    return (means^, members^)


def _keep_rows[
    dtype: DType, gpu: Bool
](book: Dynamic[dtype, 2], keep: List[Int]) raises -> Dynamic[
    dtype, 2
] where dtype.is_floating_point():
    """The rows `keep` of `book`, in order, gathered on its device."""
    var ctx = book.context()
    var k = book.dim[1]()
    var rows = len(keep)
    var idx = List[Scalar[DType.int64]](capacity=max(rows, 1))
    for r in keep:
        idx.append(Int64(r))
    if rows == 0:
        idx.append(0)
    var ip = asarray(idx^, ctx)
    var out = Dynamic[dtype, 2](row_major(_dyn_shape[2](rows, k)), ctx)
    var bp = book.tile().ptr.unsafe_origin_cast[ImmutAnyOrigin]()
    var ipp = ip.tile()
    var op = out.tile()

    @always_inline
    def body[
        w: Int, alignment: Int = 1
    ](coord: Coord) {var bp, var ipp, var op, var k}:
        var e = coord_to_index_list(coord)[0]
        var r = e // k
        var f = e % k
        op.ptr[unsafe_offset=e] = bp[
            unsafe_offset=Int(ipp.ptr[unsafe_offset=r]) * k + f
        ]

    if rows > 0:
        elementwise[simd_width=1, target=_target[gpu]()](
            body, Coord(rows * k), ctx
        )
        ctx.synchronize()
    _ = ip^
    return out^


def _distinct_indices(seed: UInt64, n: Int, k: Int) -> List[Int]:
    """`k` distinct indices from `0 .. n-1` by Floyd's algorithm, each draw
    a Philox word multiply-shifted into range."""
    var chosen = List[Int]()
    var word = 0
    for j in range(n - k, n):
        var r = Random(seed=seed, offset=UInt64(word // 4))
        var bits = UInt64(r.step()[word % 4])
        word += 1
        var t = Int((UInt64(j + 1) * bits) >> 32)
        var taken = False
        for c in chosen:
            if c == t:
                taken = True
                break
        chosen.append(j if taken else t)
    return chosen^


def whiten[
    T: TensorLike, gpu: Bool = False
](obs: T) raises -> Dynamic[T.dtype, 2] where (
    T.dtype.is_floating_point() and T.LayoutType.rank == 2
):
    """Each feature divided by its standard deviation over the
    observations. `scipy.cluster.vq.whiten(obs)`; a zero deviation divides
    by one, as SciPy's does.

    Parameters:
        T: The tensor type of `obs`, `n x k`.
        gpu: Whether to run on the observations' device.

    Args:
        obs: The observations, one per row.

    Returns:
        The rescaled observations.

    Raises:
        If a device operation fails.
    """
    comptime dtype = T.dtype
    var x = _rows(obs)
    var ctx = x.context()
    var n = x.dim[0]()
    var k = x.dim[1]()
    var scale = Dynamic[dtype, 1](row_major(_dyn_shape[1](k)), ctx)
    var out = Dynamic[dtype, 2](row_major(_dyn_shape[2](n, k)), ctx)
    var xp = x.tile().ptr.unsafe_origin_cast[MutAnyOrigin]()
    var sp = scale.tile()
    var op = out.tile()

    @always_inline
    def deviation[
        w: Int, alignment: Int = 1
    ](coord: Coord) {var xp, var sp, var n, var k}:
        var f = coord_to_index_list(coord)[0]
        var mean = Scalar[dtype](0)
        for i in range(n):
            mean += xp[unsafe_offset=i * k + f]
        mean /= Scalar[dtype](n)
        var ss = Scalar[dtype](0)
        for i in range(n):
            var d = xp[unsafe_offset=i * k + f] - mean
            ss += d * d
        var s = _sqrt(ss / Scalar[dtype](n))
        sp.ptr[unsafe_offset=f] = (
            Scalar[dtype](1) if s == Scalar[dtype](0) else s
        )

    elementwise[simd_width=1, target=_target[gpu]()](deviation, Coord(k), ctx)
    ctx.synchronize()

    @always_inline
    def divide[
        w: Int, alignment: Int = 1
    ](coord: Coord) {var xp, var sp, var op, var k}:
        var e = coord_to_index_list(coord)[0]
        op.ptr[unsafe_offset=e] = (
            xp[unsafe_offset=e] / sp.ptr[unsafe_offset=e % k]
        )

    elementwise[simd_width=1, target=_target[gpu]()](divide, Coord(n * k), ctx)
    ctx.synchronize()
    _ = x^
    return out^


def vq[
    O: TensorLike, B: TensorLike, gpu: Bool = False
](obs: O, code_book: B) raises -> VQResult[O.dtype] where (
    O.dtype.is_floating_point()
    and B.dtype == O.dtype
    and O.LayoutType.rank == 2
    and B.LayoutType.rank == 2
):
    """Assign each observation to its nearest code.
    `scipy.cluster.vq.vq(obs, code_book)`.

    Parameters:
        O: The tensor type of `obs`, `n x k`.
        B: The tensor type of `code_book`, `c x k`.
        gpu: Whether to run on the observations' device.

    Args:
        obs: The observations, one per row.
        code_book: The codes, one per row.

    Returns:
        A `VQResult` with each observation's code and distance.

    Raises:
        If the rows differ in length, or a device operation fails.
    """
    if obs.dim_at(1) != code_book.dim_at(1):
        raise Error(
            "vq: observations have ",
            obs.dim_at(1),
            " features, codes ",
            code_book.dim_at(1),
        )
    var book = rebind_var[Dynamic[O.dtype, 2]](_rows(code_book))
    return _vq[O.dtype, gpu](_rows(obs), book)


def _kmeans_run[
    dtype: DType, gpu: Bool
](
    obs: Dynamic[dtype, 2], var book: Dynamic[dtype, 2], thresh: Float64
) raises -> KMeansResult[dtype] where dtype.is_floating_point():
    var previous = Float64.MAX
    var diff = Float64.MAX
    var first = True
    while diff > thresh:
        var assigned = _vq[dtype, gpu](obs, book)
        var mean = _mean_of[dtype, gpu](assigned.dist)
        var updated = _update[dtype, gpu](obs, assigned.code, book)
        var counts = updated[1].to_host()
        var keep = List[Int]()
        for j in range(len(counts)):
            if counts[j] > 0:
                keep.append(j)
        book = _keep_rows[dtype, gpu](updated[0], keep)
        diff = Float64.MAX if first else abs(previous - mean)
        first = False
        previous = mean
    var final = _vq[dtype, gpu](obs, book)
    var distortion = _mean_of[dtype, gpu](final.dist)
    return KMeansResult[dtype](book^, distortion)


def kmeans[
    O: TensorLike, G: TensorLike, gpu: Bool = False
](obs: O, guess: G, thresh: Float64 = 1e-5) raises -> KMeansResult[
    O.dtype
] where (
    O.dtype.is_floating_point()
    and G.dtype == O.dtype
    and O.LayoutType.rank == 2
    and G.LayoutType.rank == 2
):
    """k-means from an initial codebook. `scipy.cluster.vq.kmeans(obs,
    guess, thresh=thresh)`, which runs once from a given codebook.

    Parameters:
        O: The tensor type of `obs`, `n x k`.
        G: The tensor type of `guess`, `c x k`.
        gpu: Whether to run on the observations' device.

    Args:
        obs: The observations, one per row, usually `whiten`ed.
        guess: The starting codes, one per row.
        thresh: The change in mean distortion that stops the iteration.

    Returns:
        A `KMeansResult` with the codebook and its mean distortion.

    Raises:
        If the rows differ in length, `guess` is empty, or a device
        operation fails.
    """
    if obs.dim_at(1) != guess.dim_at(1):
        raise Error(
            "kmeans: observations have ",
            obs.dim_at(1),
            " features, the guess ",
            guess.dim_at(1),
        )
    if guess.dim_at(0) < 1:
        raise Error("kmeans: asked for 0 clusters")
    var book = rebind_var[Dynamic[O.dtype, 2]](_rows(guess))
    return _kmeans_run[O.dtype, gpu](_rows(obs), book^, thresh)


def kmeans[
    O: TensorLike, gpu: Bool = False
](
    obs: O, k: Int, mut rng: Generator, iter: Int = 20, thresh: Float64 = 1e-5
) raises -> KMeansResult[O.dtype] where (
    O.dtype.is_floating_point() and O.LayoutType.rank == 2
):
    """k-means with `iter` random restarts, each from `k` distinct
    observations; the lowest distortion wins. `scipy.cluster.vq.kmeans(obs,
    k, iter, thresh, rng=...)`, with numax's generator.

    Parameters:
        O: The tensor type of `obs`, `n x k`.
        gpu: Whether to run on the observations' device.

    Args:
        obs: The observations, one per row, usually `whiten`ed.
        k: The number of codes, `1 <= k <= n`.
        rng: The generator the starts are drawn from; advanced once per
            restart.
        iter: The number of restarts, at least 1.
        thresh: The change in mean distortion that stops each run.

    Returns:
        The best `KMeansResult`.

    Raises:
        On `k` outside `[1, n]`, `iter < 1`, or a device failure.
    """
    comptime dtype = O.dtype
    var n = obs.dim_at(0)
    if k < 1 or k > n:
        raise Error(
            "kmeans: asked for ", k, " clusters of ", n, " observations"
        )
    if iter < 1:
        raise Error("kmeans: iter must be at least 1")
    var x = _rows(obs)
    var best = KMeansResult[dtype](
        Dynamic[dtype, 2](row_major(_dyn_shape[2](0, x.dim[1]())), x.context()),
        Float64.MAX,
    )
    for _ in range(iter):
        var start = _distinct_indices(rng._advance(), n, k)
        var run = _kmeans_run[dtype, gpu](
            x, _keep_rows[dtype, gpu](x, start), thresh
        )
        if run.distortion < best.distortion:
            best = run^
    return best^


def _kmeans2_run[
    dtype: DType, gpu: Bool
](
    data: Dynamic[dtype, 2], var book: Dynamic[dtype, 2], iter: Int
) raises -> KMeans2Result[dtype] where dtype.is_floating_point():
    var label = Dynamic[DType.int64, 1](
        row_major(_dyn_shape[1](data.dim[0]())), data.context()
    )
    for _ in range(iter):
        var assigned = _vq[dtype, gpu](data, book)
        var updated = _update[dtype, gpu](data, assigned.code, book)
        book = copy(updated[0])
        label = copy(assigned.code)
    return KMeans2Result[dtype](book^, label^)


def kmeans2[
    D: TensorLike, G: TensorLike, gpu: Bool = False
](data: D, guess: G, iter: Int = 10) raises -> KMeans2Result[D.dtype] where (
    D.dtype.is_floating_point()
    and G.dtype == D.dtype
    and D.LayoutType.rank == 2
    and G.LayoutType.rank == 2
):
    """k-means for a fixed number of Lloyd steps from a given codebook.
    `scipy.cluster.vq.kmeans2(data, guess, iter, minit="matrix")`; an empty
    cluster keeps its previous centroid.

    Parameters:
        D: The tensor type of `data`, `n x k`.
        G: The tensor type of `guess`, `c x k`.
        gpu: Whether to run on the data's device.

    Args:
        data: The observations, one per row.
        guess: The starting centroids, one per row.
        iter: The number of Lloyd steps, at least 1.

    Returns:
        A `KMeans2Result` with the centroids and the last labels.

    Raises:
        If the rows differ in length, `iter < 1`, or a device failure.
    """
    if data.dim_at(1) != guess.dim_at(1):
        raise Error(
            "kmeans2: data have ",
            data.dim_at(1),
            " features, the guess ",
            guess.dim_at(1),
        )
    if iter < 1:
        raise Error("kmeans2: iter must be at least 1")
    var book = rebind_var[Dynamic[D.dtype, 2]](_rows(guess))
    return _kmeans2_run[D.dtype, gpu](_rows(data), book^, iter)


def kmeans2[
    D: TensorLike, gpu: Bool = False
](data: D, k: Int, mut rng: Generator, iter: Int = 10) raises -> KMeans2Result[
    D.dtype
] where (D.dtype.is_floating_point() and D.LayoutType.rank == 2):
    """k-means for a fixed number of Lloyd steps from `k` distinct
    observations. `scipy.cluster.vq.kmeans2(data, k, iter,
    minit="points", rng=...)`, with numax's generator.

    Parameters:
        D: The tensor type of `data`, `n x k`.
        gpu: Whether to run on the data's device.

    Args:
        data: The observations, one per row.
        k: The number of clusters, `1 <= k <= n`.
        rng: The generator the start is drawn from; advanced once.
        iter: The number of Lloyd steps, at least 1.

    Returns:
        A `KMeans2Result`.

    Raises:
        On `k` outside `[1, n]`, `iter < 1`, or a device failure.
    """
    var n = data.dim_at(0)
    if k < 1 or k > n:
        raise Error(
            "kmeans2: asked for ", k, " clusters of ", n, " observations"
        )
    if iter < 1:
        raise Error("kmeans2: iter must be at least 1")
    var x = _rows(data)
    var start = _keep_rows[D.dtype, gpu](
        x, _distinct_indices(rng._advance(), n, k)
    )
    return _kmeans2_run[D.dtype, gpu](x, start^, iter)
