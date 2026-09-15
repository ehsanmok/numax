"""Correlation over `numax.core.array.Tensor`: `cov`, `corrcoef`,
`pearsonr`, `spearmanr`, `kendalltau`, `linregress`, `rankdata` and
`zscore`, with NumPy's and SciPy's conventions and SciPy's p-values.

**`cov` and `corrcoef` run where the tensor lives; the rest is tier 2,
host-side, in `Float64`.**

The split is about what the answer costs, not about taste. `pearsonr`,
`spearmanr`, `kendalltau`, `linregress` and `zscore` are a handful of sums
over one or two vectors -- or, for the rank correlations, a sort and a
pair count -- followed by a tail probability of Student's `t` or the
normal, and the p-value is the part that decides the placement: it comes
from `numax.stats.t.sf` and `numax.stats.norm.sf`, scalar `FloatLike`
kernels evaluated once. The data comes down once and the answer is a few
scalars, so a device pass would move more than it computed. `zscore` is
the one in spirit that could move, and is host-side only because the
whole-tensor `mean`/`stddev` it standardizes by are read back as scalars
anyway.

`cov` and `corrcoef` are the ones that cannot stay: the covariance of
`rows` variables over `n` observations is `O(rows^2 n)`, which this module
used to spend in a `Float64` host loop over a `List[List[Float64]]`. It is
three steps that MAX already has:

1. **Means.** `numax.stats.mean[axis=1]`, which is MAX's `Welford` monoid
   under its `rowwise` scaffolder, on either target.
2. **Centering.** One `elementwise` writing `m[i, j] - mean[i]` into a
   scratch, the means read through a freshly built rank-1 `Coord` -- the
   lower-rank read `.cursor/rules/findings.mdc` records as safe.
3. **The Gram matrix.** One `linalg.matmul` with `transpose_b=True`, so
   `C C^T` is a single GEMM with no transposed copy materialized. This is
   the MAX-first gate's answer: the expensive step is a matrix product,
   MAX ships the matrix product, and numax writes the bookkeeping.

`cov` then scales by `1 / (n - ddof)` and `corrcoef` divides by the outer
product of the diagonal's square roots, each one small launch over
`rows x rows`. Both take `gpu: Bool = False` last, and a call whose target
and whose tensor's residency disagree falls back to the old host loop with
one line on `stderr` -- `numax.core._drive`'s policy, the same one the
elementwise surface and the reductions use.

The GEMM **reassociates**: it sums the `n` products in tiles rather than
left to right, so a `float32` covariance differs from the host loop's in
the last bits, and the matrix is no longer forced symmetric by mirroring
one triangle -- `c[i, j]` and `c[j, i]` are computed independently from
the same data, as `numpy.cov`'s own `dot` computes them. `corrcoef`'s
diagonal is still written as exactly `1`, as it was.

## Conventions

`cov` and `corrcoef` are NumPy's: rows are variables, columns are
observations, `ddof = 1` unless `bias` says otherwise. `pearsonr`'s
p-value is the two-sided `t` test with `n - 2` degrees of freedom;
`spearmanr` is `pearsonr` on average ranks, with the same test;
`kendalltau` is tau-b with tie corrections and SciPy's asymptotic normal
p-value -- always the asymptotic one, where SciPy switches to an exact
enumeration for small untied samples, so a tiny tie-free sample's p-value
differs from SciPy's by the approximation's error and nothing else.
`linregress` is SciPy's, standard errors included. `rankdata`'s five
`method`s are SciPy's, `"average"` by default.

## The MAX gate

MAX has no covariance, correlation or ranking entry point, and no
partition or ranking kernel to build one on: **extend**. But the
`O(rows^2 n)` step inside `cov` is a Gram matrix, and that is `linalg`'s
`matmul` with `transpose_b=True`, so the extension is bookkeeping around a
delegation rather than a kernel. Searched at the 26.5 pin across `linalg`,
`nn`, `algorithm` and `layout`; `nn` has no covariance operator and
`algorithm`'s reductions fold a row to a value, which is the mean, not the
outer product.
"""

from std.builtin.sort import sort as _sort
from std.math import sqrt as _sqrt
from std.sys.info import simd_width_of

from layout import Coord, TileTensor, coord_to_index_list
from layout.tile_layout import row_major, TensorLayout
from layout.tile_tensor import PointerStorage
from linalg.matmul import matmul as _max_matmul
from max.algorithm.functional import elementwise

from ..core.array import Static, Tensor
from ..core._drive import _check_device, _dense, _flat, _notice, _target
from ..core.plain import Plain
from .distributions import norm, t
from .statistics import mean as _mean_axis

comptime _P = Plain[DType.float64]


def _as_float64[dtype: DType](values: List[Scalar[dtype]]) -> List[Float64]:
    var out = List[Float64](capacity=len(values))
    for i in range(len(values)):
        out.append(Float64(values[i]))
    return out^


def _mean(values: List[Float64]) -> Float64:
    var total = 0.0
    for i in range(len(values)):
        total += values[i]
    return total / Float64(len(values))


def _covariance(
    xs: List[Float64], ys: List[Float64], ddof: Int
) raises -> Float64:
    """`sum((x - mx)(y - my)) / (n - ddof)`."""
    var n = len(xs)
    if n - ddof <= 0:
        raise Error("cov: not enough observations for ddof ", ddof)
    var mx = _mean(xs)
    var my = _mean(ys)
    var total = 0.0
    for i in range(n):
        total += (xs[i] - mx) * (ys[i] - my)
    return total / Float64(n - ddof)


def _t_two_sided(statistic: Float64, df: Float64) -> Float64:
    """`2 * P(T > |statistic|)` on `df` degrees of freedom."""
    var tail = t.sf[_P](_P(abs(statistic)), _P(df))
    return 2.0 * Float64(tail.v)


def _normal_two_sided(z: Float64) -> Float64:
    var tail = norm.sf[_P](_P(abs(z)), _P(0.0), _P(1.0))
    return 2.0 * Float64(tail.v)


comptime _Block[dtype: DType] = TileTensor[
    dtype,
    type_of(row_major(Coord(0, 0))),
    MutAnyOrigin,
    Storage=PointerStorage[element_width=1],
]
"""A run-time-shaped contiguous rank-2 view over a buffer this module owns
-- the operand type `linalg.matmul` accepts, the shape
`numax.linalg.common` uses for the same reason."""

comptime _Row[dtype: DType] = TileTensor[
    dtype,
    type_of(row_major(Coord(0))),
    MutAnyOrigin,
    Storage=PointerStorage[element_width=1],
]
"""One row of a `_Block` retyped as a rank-1 destination, so a copy into it
has the *same* extents as its rank-1 source. A cross-shape `elementwise`
is the pattern `.cursor/rules/findings.mdc` records as unreliable at small
extents; retyping the buffer costs nothing and cannot miscompute."""


@always_inline
def _lanes[dtype: DType, gpu: Bool]() -> Int:
    """Native SIMD width on the host, one element per thread on the
    device."""
    comptime if gpu:
        return 1
    else:
        return simd_width_of[dtype]()


def _stack_rows[
    dtype: DType, n: Int, gpu: Bool
](mut x: Static[dtype, n], mut y: Static[dtype, n]) raises -> Static[
    dtype, 2, n
]:
    """`x` and `y` as the two rows of one `2 x n` matrix, where they live.

    Two same-shape copies rather than one `elementwise` over `Coord(2, n)`
    picking a source per row: each row of the destination is retyped as a
    rank-1 view, so source and destination extents match.
    """
    var ctx = x.context()
    var out = Static[dtype, 2, n]._uninitialized(ctx)
    var ov = out.view()
    var xs = _flat(x)
    var ys = _flat(y)
    var top: _Row[dtype] = TileTensor(
        ov.ptr_at_offset(Coord(0, 0)), row_major(Coord(n))
    )
    var bottom: _Row[dtype] = TileTensor(
        ov.ptr_at_offset(Coord(1, 0)), row_major(Coord(n))
    )

    @always_inline
    def first[w: Int, alignment: Int = 1](coord: Coord) {var xs, var top}:
        top.store[w](coord, xs.load[w](coord))

    @always_inline
    def second[w: Int, alignment: Int = 1](coord: Coord) {var ys, var bottom}:
        bottom.store[w](coord, ys.load[w](coord))

    comptime lanes = _lanes[dtype, gpu]()
    elementwise[simd_width=lanes, target=_target[gpu]()](first, Coord(n), ctx)
    elementwise[simd_width=lanes, target=_target[gpu]()](second, Coord(n), ctx)
    return out^


def _centered[
    dtype: DType, rows: Int, n: Int, gpu: Bool
](mut m: Static[dtype, rows, n]) raises -> Static[
    dtype, rows, n
] where dtype.is_floating_point():
    """`m` with each row's own mean subtracted, where `m` lives.

    The means come from `numax.stats.mean[axis=1]`, which is MAX's
    `Welford` monoid; the subtraction is one `elementwise` whose body reads
    the rank-1 means through a freshly built `Coord`, the lower-rank read
    recorded as safe. Source and destination have the same extents.
    """
    var ctx = m.context()
    var means = _mean_axis[axis=1, gpu=gpu](m)
    var out = Static[dtype, rows, n]._uninitialized(ctx)
    var src = _dense(m)
    var avg = _flat(means)
    var dst = out.view()

    @always_inline
    def center[
        w: Int, alignment: Int = 1
    ](coord: Coord) {var src, var avg, var dst}:
        var at = coord_to_index_list(coord)
        dst.store[w](
            coord,
            src.load[w](coord) - SIMD[dtype, w](avg[Coord(at[0])]),
        )

    elementwise[simd_width=_lanes[dtype, gpu](), target=_target[gpu]()](
        center, Coord(rows, n), ctx
    )
    # `avg` is a view over `means`, whose last mention is the view itself.
    _ = means^
    return out^


def _gram[
    dtype: DType, rows: Int, n: Int, gpu: Bool
](mut centered: Static[dtype, rows, n]) raises -> Static[dtype, rows, rows]:
    """`C C^T` for the centered `C`, in one GEMM.

    `linalg.matmul` with `transpose_b=True` reads the right operand
    transposed in place, so the Gram matrix costs one product and no
    transposed copy. Two views of the same buffer because `matmul` takes
    both operands mutably and rejects two live views sharing an origin --
    the spelling `numax.linalg.cholesky`'s trailing update uses.
    """
    var ctx = centered.context()
    var out = Static[dtype, rows, rows]._uninitialized(ctx)
    var cv = centered.view()
    var ov = out.view()
    var left: _Block[dtype] = TileTensor(
        cv.ptr_at_offset(Coord(0, 0)), row_major(Coord(rows, n))
    )
    var right: _Block[dtype] = TileTensor(
        cv.ptr_at_offset(Coord(0, 0)), row_major(Coord(rows, n))
    )
    var product: _Block[dtype] = TileTensor(
        ov.ptr_at_offset(Coord(0, 0)), row_major(Coord(rows, rows))
    )
    _max_matmul[transpose_b=True, target=_target[gpu]()](
        product, left, right, ctx
    )
    ctx.synchronize()
    return out^


def _dof_of(bias: Bool, ddof: Optional[Int], n: Int) raises -> Int:
    """NumPy's `ddof` rule, spelled once: `ddof` if given, else `0` under
    `bias` and `1` otherwise."""
    var dof = ddof.value() if ddof else (0 if bias else 1)
    if n - dof <= 0:
        raise Error("cov: not enough observations for ddof ", dof)
    return dof


def _cov_device[
    dtype: DType, rows: Int, n: Int, gpu: Bool
](mut m: Static[dtype, rows, n], dof: Int) raises -> Static[
    dtype, rows, rows
] where dtype.is_floating_point():
    """Centre, one GEMM, one scaling launch."""
    var ctx = m.context()
    var centered = _centered[gpu=gpu](m)
    var gram = _gram[gpu=gpu](centered)
    var gv = gram.view()
    var scale = Scalar[dtype](1) / Scalar[dtype](n - dof)

    @always_inline
    def rescale[w: Int, alignment: Int = 1](coord: Coord) {var gv, var scale}:
        gv.store[1](coord, gv[coord] * scale)

    elementwise[simd_width=1, target=_target[gpu]()](
        rescale, Coord(rows, rows), ctx
    )
    _ = centered^
    return gram^


def _cov_host[
    dtype: DType, rows: Int, n: Int
](m: Static[dtype, rows, n], dof: Int) raises -> Static[dtype, rows, rows]:
    """The pre-0.2 `O(rows^2 n)` host loop, kept as the fallback for a call
    whose target and whose tensor's residency disagree."""
    var host = _as_float64(m.to_host())
    var series = List[List[Float64]]()
    for r in range(rows):
        var row = List[Float64](capacity=n)
        for c in range(n):
            row.append(host[r * n + c])
        series.append(row^)
    var values = List[Scalar[dtype]](length=rows * rows, fill=0)
    for a in range(rows):
        for b in range(a, rows):
            var c = Scalar[dtype](_covariance(series[a], series[b], dof))
            values[a * rows + b] = c
            values[b * rows + a] = c
    return Static[dtype, rows, rows](m.context(), values^)


def _corrcoef_host[
    dtype: DType, rows: Int, n: Int
](m: Static[dtype, rows, n]) raises -> Static[dtype, rows, rows]:
    """`_cov_host`'s sibling: the pre-0.2 walk, for the mismatch path."""
    var host = _as_float64(m.to_host())
    var series = List[List[Float64]]()
    for r in range(rows):
        var row = List[Float64](capacity=n)
        for c in range(n):
            row.append(host[r * n + c])
        series.append(row^)
    var scale = List[Float64](capacity=rows)
    for r in range(rows):
        scale.append(_sqrt(_covariance(series[r], series[r], 0)))
    var values = List[Scalar[dtype]](length=rows * rows, fill=0)
    for a in range(rows):
        for b in range(a, rows):
            var r = 1.0 if a == b else _covariance(series[a], series[b], 0) / (
                scale[a] * scale[b]
            )
            values[a * rows + b] = Scalar[dtype](r)
            values[b * rows + a] = Scalar[dtype](r)
    return Static[dtype, rows, rows](m.context(), values^)


def cov[
    dtype: DType, rows: Int, n: Int, gpu: Bool = False
](
    mut m: Static[dtype, rows, n],
    bias: Bool = False,
    ddof: Optional[Int] = None,
) raises -> Static[dtype, rows, rows] where (
    dtype.is_floating_point() and rows > 0 and n > 1
):
    """The covariance matrix of `rows` variables observed `n` times each,
    one variable per row. `numpy.cov(m)` with its default `rowvar=True`.

    `ddof = 1` by default, `0` under `bias`, or as given.

    Centering through MAX's `Welford` monoid, then one
    `linalg.matmul(transpose_b=True)` for `C C^T`, then one scaling launch
    -- so this runs where `m` lives and is `O(rows^2 n)` at GEMM speed
    rather than in a host loop. `gpu=True` on a tensor that is not on a GPU
    context (or the reverse) falls back to that loop and says so on
    `stderr`.

    The GEMM reassociates, so a `float32` result differs from the host
    loop's in the last bits, and `c[i, j]` and `c[j, i]` are computed
    independently rather than mirrored -- `numpy.cov` does the same.
    """
    var dof = _dof_of(bias, ddof, n)
    if not _check_device[gpu=gpu](m):
        _notice[gpu]("cov")
        return _cov_host(m, dof)
    return _cov_device[gpu=gpu](m, dof)


def cov[
    dtype: DType, n: Int, gpu: Bool = False
](
    mut x: Static[dtype, n],
    mut y: Static[dtype, n],
    bias: Bool = False,
    ddof: Optional[Int] = None,
) raises -> Static[dtype, 2, 2] where (dtype.is_floating_point() and n > 1):
    """The covariance matrix of two variables, `[[var x, cov], [cov, var
    y]]`. `numpy.cov(x, y, bias, ddof)`: `ddof = 1` by default, `0` with
    `bias`, or as given.

    The two vectors are stacked into one `2 x n` matrix and handed to the
    matrix overload, so there is one covariance algorithm here, not two.
    """
    var stacked = _stack_rows[gpu=gpu](x, y)
    return cov[gpu=gpu](stacked, bias, ddof)


def corrcoef[
    dtype: DType, rows: Int, n: Int, gpu: Bool = False
](mut m: Static[dtype, rows, n]) raises -> Static[dtype, rows, rows] where (
    dtype.is_floating_point() and rows > 0 and n > 1
):
    """The Pearson correlation matrix of `rows` variables, one per row.
    `numpy.corrcoef(m)`.

    `cov`'s Gram matrix divided by the outer product of its diagonal's
    square roots. The `1 / (n - ddof)` cancels, so the unscaled Gram matrix
    is what this reads and `corrcoef` takes no `ddof`. The diagonal is
    written as exactly `1` rather than computed, which is what the host
    walk did and what `numpy.corrcoef`'s clip amounts to.
    """
    if not _check_device[gpu=gpu](m):
        _notice[gpu]("corrcoef")
        return _corrcoef_host(m)
    var ctx = m.context()
    var centered = _centered[gpu=gpu](m)
    var gram = _gram[gpu=gpu](centered)
    var out = Static[dtype, rows, rows]._uninitialized(ctx)
    var gv = _dense(gram)
    var dst = out.view()

    @always_inline
    def normalize[w: Int, alignment: Int = 1](coord: Coord) {var gv, var dst}:
        var at = coord_to_index_list(coord)
        var i = at[0]
        var j = at[1]
        if i == j:
            dst.store[1](coord, Scalar[dtype](1))
        else:
            dst.store[1](
                coord,
                gv[Coord(i, j)]
                / (_sqrt(gv[Coord(i, i)]) * _sqrt(gv[Coord(j, j)])),
            )

    elementwise[simd_width=1, target=_target[gpu]()](
        normalize, Coord(rows, rows), ctx
    )
    _ = centered^
    _ = gram^
    return out^


def corrcoef[
    dtype: DType, n: Int, gpu: Bool = False
](mut x: Static[dtype, n], mut y: Static[dtype, n]) raises -> Static[
    dtype, 2, 2
] where (dtype.is_floating_point() and n > 1):
    """The Pearson correlation matrix of two variables, ones on the
    diagonal. `numpy.corrcoef(x, y)`. The pair is stacked into a `2 x n`
    and handed to the matrix overload."""
    var stacked = _stack_rows[gpu=gpu](x, y)
    return corrcoef[gpu=gpu](stacked)


@fieldwise_init
struct CorrelationResult(Copyable, Movable):
    """What `pearsonr`, `spearmanr` and `kendalltau` return: the
    correlation `statistic` and the two-sided `pvalue` of the test that it
    is zero, SciPy's result shape."""

    var statistic: Float64
    var pvalue: Float64


def _pearson(xs: List[Float64], ys: List[Float64]) raises -> CorrelationResult:
    var n = len(xs)
    if n < 3:
        raise Error("pearsonr: at least three observations are needed")
    var r = _covariance(xs, ys, 0) / _sqrt(
        _covariance(xs, xs, 0) * _covariance(ys, ys, 0)
    )
    r = min(1.0, max(-1.0, r))
    var df = Float64(n - 2)
    if abs(r) == 1.0:
        return CorrelationResult(r, 0.0)
    var statistic = r * _sqrt(df / (1.0 - r * r))
    return CorrelationResult(r, _t_two_sided(statistic, df))


def pearsonr[
    dtype: DType, n: Int
](
    mut x: Static[dtype, n], mut y: Static[dtype, n]
) raises -> CorrelationResult where (dtype.is_floating_point() and n > 2):
    """Pearson's `r` and the two-sided p-value of `r = 0`.
    `scipy.stats.pearsonr(x, y)`.

    The p-value is the `t` test on `n - 2` degrees of freedom, `t = r
    sqrt((n - 2) / (1 - r^2))`, which is the same number SciPy's beta form
    gives.
    """
    return _pearson(_as_float64(x.to_host()), _as_float64(y.to_host()))


def _ranks(values: List[Float64], method: StaticString) raises -> List[Float64]:
    """`scipy.stats.rankdata` on the host: one-based ranks under the five
    tie methods."""
    var n = len(values)
    var order = List[Int](capacity=n)
    for i in range(n):
        order.append(i)

    @parameter
    def by_value(a: Int, b: Int) -> Bool:
        return values[a] < values[b] or (values[a] == values[b] and a < b)

    _sort[by_value](order)
    var ranks = List[Float64](length=n, fill=0.0)
    var dense = 0
    var i = 0
    while i < n:
        var j = i
        while j + 1 < n and values[order[j + 1]] == values[order[i]]:
            j += 1
        dense += 1
        for k in range(i, j + 1):
            var rank: Float64
            if method == "average":
                rank = Float64(i + j + 2) / 2.0
            elif method == "min":
                rank = Float64(i + 1)
            elif method == "max":
                rank = Float64(j + 1)
            elif method == "dense":
                rank = Float64(dense)
            elif method == "ordinal":
                rank = Float64(k + 1)
            else:
                raise Error(
                    "rankdata: unknown method '",
                    method,
                    "'; expected average, min, max, dense or ordinal",
                )
            ranks[order[k]] = rank
        i = j + 1
    return ranks^


def rankdata[
    dtype: DType, LayoutType: TensorLayout
](
    xs: Tensor[dtype, LayoutType], method: StaticString = "average"
) raises -> Tensor[dtype, LayoutType] where dtype.is_floating_point():
    """The one-based rank of every element, ties resolved by `method`:
    `"average"` (the default), `"min"`, `"max"`, `"dense"` or
    `"ordinal"`. `scipy.stats.rankdata(a, method)`, the same shape back."""
    var ranks = _ranks(_as_float64(xs.to_host()), method)
    var values = List[Scalar[dtype]](capacity=len(ranks))
    for i in range(len(ranks)):
        values.append(Scalar[dtype](ranks[i]))
    return Tensor[dtype, LayoutType](xs.context(), xs.layout, values^)


def spearmanr[
    dtype: DType, n: Int
](
    mut x: Static[dtype, n], mut y: Static[dtype, n]
) raises -> CorrelationResult where (dtype.is_floating_point() and n > 2):
    """Spearman's rank correlation and its two-sided p-value:
    `pearsonr` on the average ranks, with the same `t` test on `n - 2`
    degrees of freedom SciPy uses. `scipy.stats.spearmanr(x, y)`."""
    return _pearson(
        _ranks(_as_float64(x.to_host()), "average"),
        _ranks(_as_float64(y.to_host()), "average"),
    )


def kendalltau[
    dtype: DType, n: Int
](
    mut x: Static[dtype, n], mut y: Static[dtype, n]
) raises -> CorrelationResult where (dtype.is_floating_point() and n > 1):
    """Kendall's tau-b and its two-sided p-value.
    `scipy.stats.kendalltau(x, y, method="asymptotic")`.

    Tau-b counts concordant less discordant pairs over the geometric mean
    of the pairs untied in each variable; the p-value is the normal
    approximation with SciPy's tie-corrected variance. SciPy switches to an
    exact enumeration for a small sample with no ties, which this does not
    do, so there and only there its p-value differs by the approximation's
    error. `O(n^2)` in the pair count, host-side.
    """
    var xs = _as_float64(x.to_host())
    var ys = _as_float64(y.to_host())
    var concordant = 0
    var discordant = 0
    var xtie = 0
    var ytie = 0
    var both = 0
    for i in range(n):
        for j in range(i + 1, n):
            var dx = xs[i] - xs[j]
            var dy = ys[i] - ys[j]
            if dx == 0 and dy == 0:
                both += 1
            elif dx == 0:
                xtie += 1
            elif dy == 0:
                ytie += 1
            elif (dx > 0) == (dy > 0):
                concordant += 1
            else:
                discordant += 1
    var total = n * (n - 1) // 2
    var x_tied = xtie + both
    var y_tied = ytie + both
    if x_tied == total or y_tied == total:
        raise Error("kendalltau: one variable is constant")
    var tau = Float64(concordant - discordant) / (
        _sqrt(Float64(total - x_tied)) * _sqrt(Float64(total - y_tied))
    )
    tau = min(1.0, max(-1.0, tau))

    # SciPy's variance of `con - dis` with tie corrections: the group sizes
    # of equal values in each variable enter through three sums.
    var stats_x = _tie_sums(xs)
    var stats_y = _tie_sums(ys)
    var m = Float64(n) * Float64(n - 1)
    var variance = (
        (m * Float64(2 * n + 5) - stats_x[1] - stats_y[1]) / 18.0
        + 2.0 * stats_x[0] * stats_y[0] / m
        + stats_x[2] * stats_y[2] / (9.0 * m * Float64(n - 2))
    )
    var z = Float64(concordant - discordant) / _sqrt(variance)
    return CorrelationResult(tau, _normal_two_sided(z))


def _tie_sums(values: List[Float64]) -> Tuple[Float64, Float64, Float64]:
    """Over the groups of equal values with sizes `c`: `sum c(c-1)/2`,
    `sum c(c-1)(2c+5)` and `sum c(c-1)(c-2)` -- SciPy's `count_rank_tie`."""
    var sorted = values.copy()
    _sort(sorted)
    var pairs = 0.0
    var weighted = 0.0
    var cubic = 0.0
    var i = 0
    while i < len(sorted):
        var j = i
        while j + 1 < len(sorted) and sorted[j + 1] == sorted[i]:
            j += 1
        var c = Float64(j - i + 1)
        if c > 1:
            pairs += c * (c - 1.0) / 2.0
            weighted += c * (c - 1.0) * (2.0 * c + 5.0)
            cubic += c * (c - 1.0) * (c - 2.0)
        i = j + 1
    return (pairs, weighted, cubic)


@fieldwise_init
struct LinregressResult(Copyable, Movable):
    """What `linregress` returns: `scipy.stats.linregress`'s result fields,
    the fitted line, its correlation, the two-sided p-value of a zero
    slope, and the standard errors of slope and intercept."""

    var slope: Float64
    var intercept: Float64
    var rvalue: Float64
    var pvalue: Float64
    var stderr: Float64
    var intercept_stderr: Float64


def linregress[
    dtype: DType, n: Int
](
    mut x: Static[dtype, n], mut y: Static[dtype, n]
) raises -> LinregressResult where (dtype.is_floating_point() and n > 2):
    """The least-squares line `y = slope x + intercept` through the points,
    with `r`, the two-sided p-value of `slope = 0` and both standard
    errors. `scipy.stats.linregress(x, y)`, formula for formula.
    """
    var xs = _as_float64(x.to_host())
    var ys = _as_float64(y.to_host())
    var ssxm = _covariance(xs, xs, 0)
    var ssym = _covariance(ys, ys, 0)
    var ssxym = _covariance(xs, ys, 0)
    if ssxm == 0:
        raise Error("linregress: x is constant")
    var slope = ssxym / ssxm
    var intercept = _mean(ys) - slope * _mean(xs)
    var r = 0.0
    if ssym > 0:
        r = min(1.0, max(-1.0, ssxym / _sqrt(ssxm * ssym)))
    var df = Float64(n - 2)
    comptime tiny = 1e-20
    var statistic = r * _sqrt(df / ((1.0 - r + tiny) * (1.0 + r + tiny)))
    var pvalue = _t_two_sided(statistic, df)
    var slope_stderr = _sqrt((1.0 - r * r) * ssym / ssxm / df)
    var xm = _mean(xs)
    var intercept_stderr = slope_stderr * _sqrt(ssxm + xm * xm)
    return LinregressResult(
        slope, intercept, r, pvalue, slope_stderr, intercept_stderr
    )


def zscore[
    dtype: DType, LayoutType: TensorLayout
](xs: Tensor[dtype, LayoutType], ddof: Int = 0) raises -> Tensor[
    dtype, LayoutType
] where dtype.is_floating_point():
    """Every element standardized by the tensor's mean and standard
    deviation, `(x - mean) / std` with `ddof` degrees of freedom in the
    standard deviation. `scipy.stats.zscore(a, ddof)`, the same shape
    back. A constant tensor raises rather than dividing by zero."""
    var values = _as_float64(xs.to_host())
    var n = len(values)
    if n - ddof <= 0:
        raise Error("zscore: not enough elements for ddof ", ddof)
    var centre = _mean(values)
    var scale = _sqrt(_covariance(values, values, ddof))
    if scale == 0:
        raise Error("zscore: the tensor is constant")
    var out = List[Scalar[dtype]](capacity=n)
    for i in range(n):
        out.append(Scalar[dtype]((values[i] - centre) / scale))
    return Tensor[dtype, LayoutType](xs.context(), xs.layout, out^)
