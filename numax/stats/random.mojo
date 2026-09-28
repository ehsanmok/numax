"""Random sampling into `numax.core.tensor.Tensor`, `Plain`-only, on the host
or the device, from one counter-based stream.

**This module is tier 2** in the sense that matters for the rest of the
library -- it returns `Plain` values into a `Tensor` and no `FloatLike`
conformer is involved -- but every draw is a fixed amount of branchless
work per element, which is what lets the same fill run as a GPU kernel.

`docs/parity.md` picks random sampling as a genuine `numax` gap:
without it, `examples/advanced/ode.mojo`'s GPU ensemble (which needs
initial conditions) and any Gaussian-process-shaped example reach for a
raw RNG directly and lose the `numax` entry point. **No `Random[FloatLike]`
conformer exists here on purpose** -- RNG is not mathematically
differentiable (seeding a `Dual`'s derivative from a random draw has no
well-defined meaning), so the trait contract does not fit the mathematics
and this stays outside `FloatLike` entirely.

## One stream, host or device

Every function below fills its tensor from `std.random.philox.Random`, the
counter-based Philox generator, with element `i` of the output reading
word `i` of the stream `Random(seed=s, offset=i // 4).step()[i % 4]`:
a 32-bit word per element, four per Philox step, and no element depends
on any other. A `float64` output takes two consecutive words for a 53-bit
uniform; everything else takes one word's top 24 bits, so a `float32`
uniform is exact in `[0, 1)` and never rounds up to `1`. `normal` is
Box-Muller on the two uniforms at words `2i` and `2i + 1`; `exponential`
is `-scale ln(1 - U)` on word `i`; `randint` is `low + ((high - low) * word) >> 32`
in integer arithmetic; `randbool` compares `U < p`.

`Generator.gamma`, `beta`, `lognormal`, `poisson` and `binomial` take
several words per element: element `i` owns a fixed block of the stream
and a rejection sampler gets a fixed number of attempts inside it
(Marsaglia-Tsang for the gamma family, NumPy's PTRS and BTRS transformed
rejection for the counts, inversion below their splits), so they fill on
the device like the rest.

`permutation`, `shuffle` and `choice` are compositions over those fills:
`argsort` of uniform keys, `randint` indices, or inversion through the
running sum of the weights and `searchsorted`, then a gather, each on
the tensor's device. `multivariate_normal` is `z L^T + mean` through
`numax.linalg`'s Cholesky and `matmul`. `spawn` seeds children from the
parent's stream at a distant offset.

Because the value at every position is a pure function of `(seed, i)`,
the fill is the same body on both sides of the launch boundary, driven by
`max.algorithm.elementwise` the way the distributions' `Tensor` overloads
are: with `gpu=False` (the default) it walks the tensor threaded at native
SIMD width on the host; with `gpu=True` and a device `ctx` it runs one
thread per element on the device, and the two produce the same tensor bit
for bit for `uniform`, `randint` and `randbool` (the transcendental ones
agree to the device's `ln`/`cos` rounding). There is no host round trip: a
`normal[float32, 1_000_000](ctx=gpu, gpu=True)` allocates on the device
and fills there, which is what `examples/intermediate/random_ensemble.mojo`
does for its initial conditions.

The module-level functions take their seed from `std.random`'s global
generator (one `random_ui64` per call), so `seed(v)` reproduces a
sequence of draws exactly as before. `Generator` owns its seed and
advances it by one per draw, so two `Generator(seed=7)`s agree with each
other and with nothing else, and the global state is never touched --
which also makes it safe to use from several threads.

## The MAX gate

`nn.rand_uniform`/`nn.rand_normal` exist and fill a `TileTensor` on the
device from the same Philox, which looked like the obvious route --
confirmed otherwise by direct experiment. Both take their fill logic as
an `OutputFn` parameter bound to `RegisterPassable & ImplicitlyCopyable`,
which a capturing closure over a caller's own buffer does not satisfy
(confirmed: passing one fails to compile against that bound, even with an
`imm`-only capture), and they read the seed from a device pointer rather
than a scalar. That shape is graph-op fusion machinery, meant to be
threaded through a MAX `Graph` compilation and not called eagerly from
ordinary Mojo. What numax takes from MAX here is the generator itself,
`std.random.philox.Random`, and the per-element `offset` idiom
`nn/rand_uniform.mojo` uses; the fill around it is numax's, in
MAX's idiom (a `TileTensor` out, one thread per element). **Extend.**
"""

from std.math import cos, exp, floor, log, sqrt
from std.random import Random, random_ui64, seed as _std_seed
from std.sys.info import simd_width_of
from std.utils.coord import coord_to_index_list

from layout import Coord
from layout.tile_layout import row_major
from max.algorithm.functional import elementwise
from max.gpu.host import DeviceContext

from ..core.tensor import (
    Dynamic,
    Static,
    _LayoutOf,
    arange_n,
    _context,
    _dyn_shape,
    _product,
    _same_order,
    transpose,
)
from ..core.tensorlike import TensorLike, dim
from ..core.sorting import argsort, searchsorted, take
from ..core.elementwise import clip
from ..linalg.blas import matmul
from ..linalg.cholesky import cholesky
from .statistics import _target, cumsum

comptime _TWO_PI = 6.283185307179586


def _uniform01[dtype: DType](seed: UInt64, index: Int) -> Scalar[dtype]:
    """Word `index` of the Philox stream as a uniform in `[0, 1)`: two
    words for a 53-bit `float64`, the top 24 bits of one word otherwise,
    so the conversion is exact and never reaches `1`."""
    comptime if dtype == DType.float64:
        var r = Random(seed=seed, offset=UInt64(index // 2))
        var words = r.step()
        var pair = 2 * (index % 2)
        var hi = UInt64(words[pair])
        var lo = UInt64(words[pair + 1])
        var bits = (hi << 32) | lo
        return Scalar[dtype](Float64(bits >> 11) * 1.1102230246251565e-16)
    else:
        var r = Random(seed=seed, offset=UInt64(index // 4))
        var word = r.step()[index % 4]
        return Scalar[dtype](Float32(word >> 8) * 5.960464477539063e-08)


def _uniform_step[
    dtype: DType
](seed: UInt64, index: Int, low: Scalar[dtype], high: Scalar[dtype]) -> Scalar[
    dtype
]:
    return low + (high - low) * _uniform01[dtype](seed, index)


def _normal_step[
    dtype: DType
](
    seed: UInt64, index: Int, mean: Scalar[dtype], stddev: Scalar[dtype]
) -> Scalar[dtype] where dtype.is_floating_point():
    """Box-Muller on the uniforms at words `2i` and `2i + 1`."""
    var u1 = _uniform01[dtype](seed, 2 * index)
    var u2 = _uniform01[dtype](seed, 2 * index + 1)
    var radius = sqrt(Scalar[dtype](-2.0) * log(Scalar[dtype](1.0) - u1))
    return mean + stddev * radius * cos(Scalar[dtype](_TWO_PI) * u2)


def _exponential_step[
    dtype: DType
](
    seed: UInt64, index: Int, scale: Scalar[dtype], unused: Scalar[dtype]
) -> Scalar[dtype] where dtype.is_floating_point():
    return -scale * log(Scalar[dtype](1.0) - _uniform01[dtype](seed, index))


def _randint_step[
    dtype: DType
](seed: UInt64, index: Int, low: Int64, high: Int64) -> Scalar[dtype]:
    """`low + floor((high - low) U)` in 64-bit integer arithmetic on the raw
    32-bit word -- no floating point at all, so the kernel has no `double`
    for Metal to reject and the result is exact for any range below 2^32
    (the modulo bias is `(high - low) / 2^32`)."""
    var r = Random(seed=seed, offset=UInt64(index // 4))
    var word = UInt64(r.step()[index % 4])
    var span = UInt64(high - low)
    return Scalar[dtype](low + Int64((span * word) >> 32))


def _randbool_step[
    dtype: DType
](seed: UInt64, index: Int, p: Float32, unused: Float32) -> Scalar[dtype]:
    return _uniform01[DType.float32](seed, index).lt(p).cast[dtype]()


# ---------------------------------------------------------------------------
# Draws that take several words per element
# ---------------------------------------------------------------------------
#
# Element `i` of these reads only the words in its own block `[i W, (i+1) W)`
# of the stream, `W` fixed per distribution, so elements stay independent
# and the fill stays one body on either side of the launch. A rejection
# sampler gets a fixed number of attempts inside its block rather than an
# open-ended loop; the attempt counts below make running out rarer than
# one in `1e20` draws, and the fallback then is the distribution's own
# center, stated at each.

comptime _GAMMA_TRIES = 16
"""Marsaglia-Tsang attempts per gamma draw; each accepts with probability
above 0.95, so all sixteen failing is below `1e-20`."""
comptime _GAMMA_WORDS = 3 * _GAMMA_TRIES + 1
"""Words per gamma draw: a normal (two) and a uniform per attempt, and one
more for the `shape < 1` boost."""
comptime _REJECT_TRIES = 24
"""Transformed-rejection attempts per Poisson or binomial draw; each accepts
with probability above 0.85 (Hormann's bound), so all failing is below
`1e-20`."""
comptime _REJECT_WORDS = 2 * _REJECT_TRIES
"""Words per Poisson or binomial draw by transformed rejection."""
comptime _INVERSION_WORDS = 1
"""Words per draw by inversion: one uniform."""


@always_inline
def _word_uniform[
    dtype: DType
](seed: UInt64, word: Int) -> Scalar[dtype] where dtype.is_floating_point():
    """Word `word` of the stream as a uniform in `[0, 1)` at `dtype`'s
    precision, one word even at `float64`: these samplers need the
    word count per element fixed across dtypes."""
    var r = Random(seed=seed, offset=UInt64(word // 4))
    var bits = r.step()[word % 4]
    return Scalar[dtype](Float32(bits >> 8) * 5.960464477539063e-08)


@always_inline
def _word_normal[
    dtype: DType
](seed: UInt64, word: Int) -> Scalar[dtype] where dtype.is_floating_point():
    """A standard normal from words `word` and `word + 1`, Box-Muller."""
    var u1 = _word_uniform[dtype](seed, word)
    var u2 = _word_uniform[dtype](seed, word + 1)
    var radius = sqrt(Scalar[dtype](-2.0) * log(Scalar[dtype](1.0) - u1))
    return radius * cos(Scalar[dtype](_TWO_PI) * u2)


def _loggam[
    dtype: DType
](x: Scalar[dtype]) -> Scalar[dtype] where dtype.is_floating_point():
    """`ln Gamma(x)` for `x >= 1`: NumPy's `random_loggam`, Stirling's series
    after shifting `x` above 7, plain arithmetic and `log` so it runs in a
    device kernel."""
    if x == 1 or x == 2:
        return Scalar[dtype](0)
    var x0 = x
    var shift = 0
    if x <= 7:
        shift = Int(7 - x)
        x0 = x + Scalar[dtype](shift)
    var x2 = 1 / (x0 * x0)
    var gl0 = Scalar[dtype](-1.39243221690590e00)
    gl0 = gl0 * x2 + Scalar[dtype](1.796443723688307e-01)
    gl0 = gl0 * x2 + Scalar[dtype](-2.955065359477124e-02)
    gl0 = gl0 * x2 + Scalar[dtype](6.410256410256410e-03)
    gl0 = gl0 * x2 + Scalar[dtype](-1.917526917526918e-03)
    gl0 = gl0 * x2 + Scalar[dtype](8.417508417508418e-04)
    gl0 = gl0 * x2 + Scalar[dtype](-5.952380952380952e-04)
    gl0 = gl0 * x2 + Scalar[dtype](7.936507936507937e-04)
    gl0 = gl0 * x2 + Scalar[dtype](-2.777777777777778e-03)
    gl0 = gl0 * x2 + Scalar[dtype](8.333333333333333e-02)
    var gl = (
        gl0 / x0 + Scalar[dtype](0.9189385332046727) + (x0 - 0.5) * log(x0) - x0
    )
    for _ in range(shift):
        gl -= log(x0 - 1)
        x0 -= 1
    return gl


def _standard_gamma[
    dtype: DType
](seed: UInt64, base: Int, shape: Scalar[dtype]) -> Scalar[
    dtype
] where dtype.is_floating_point():
    """A standard gamma draw from the words at `base`: Marsaglia and Tsang's
    squeeze for `shape >= 1`, and for `shape < 1` a draw at `shape + 1`
    times `U^(1/shape)`, their boost. After `_GAMMA_TRIES` rejections the
    draw is the proposal's center `d`."""
    var boost = shape < 1
    var a = shape + 1 if boost else shape
    var d = a - Scalar[dtype](1.0 / 3.0)
    var c = 1 / sqrt(9 * d)
    var result = d
    for t in range(_GAMMA_TRIES):
        var x = _word_normal[dtype](seed, base + 3 * t)
        var v = 1 + c * x
        if v <= 0:
            continue
        v = v * v * v
        var u = _word_uniform[dtype](seed, base + 3 * t + 2)
        if log(u) < Scalar[dtype](0.5) * x * x + d - d * v + d * log(v):
            result = d * v
            break
    if boost:
        var u = _word_uniform[dtype](seed, base + 3 * _GAMMA_TRIES)
        result = result * (u ** (1 / shape))
    return result


def _gamma_step[
    dtype: DType
](
    seed: UInt64, index: Int, shape: Scalar[dtype], scale: Scalar[dtype]
) -> Scalar[dtype] where dtype.is_floating_point():
    return scale * _standard_gamma[dtype](seed, index * _GAMMA_WORDS, shape)


def _beta_step[
    dtype: DType
](seed: UInt64, index: Int, a: Scalar[dtype], b: Scalar[dtype]) -> Scalar[
    dtype
] where dtype.is_floating_point():
    """`X / (X + Y)` with `X ~ Gamma(a)` and `Y ~ Gamma(b)` from two blocks."""
    var base = index * 2 * _GAMMA_WORDS
    var x = _standard_gamma[dtype](seed, base, a)
    var y = _standard_gamma[dtype](seed, base + _GAMMA_WORDS, b)
    return x / (x + y)


def _lognormal_step[
    dtype: DType
](
    seed: UInt64, index: Int, mean: Scalar[dtype], sigma: Scalar[dtype]
) -> Scalar[dtype] where dtype.is_floating_point():
    return exp(mean + sigma * _normal_step[dtype](seed, index, 0, 1))


def _poisson_count[
    W: DType
](seed: UInt64, base: Int, lam: Scalar[W]) -> Scalar[
    W
] where W.is_floating_point():
    """A Poisson draw from the words at `base`: inversion by a sequential
    search for `lam < 10`, bounded at `lam + 12 sqrt(lam) + 30` steps, and
    Hormann's PTRS transformed rejection above, NumPy's own split and its
    constants. The rejection's fallback is `floor(lam)`."""
    if lam <= 0:
        return Scalar[W](0)
    if lam < 10:
        var u = _word_uniform[W](seed, base)
        var k = Scalar[W](0)
        var term = exp(-lam)
        var total = term
        var bound = Int(lam + 12 * sqrt(lam) + 30)
        for _ in range(bound):
            if u < total:
                break
            k += 1
            term = term * lam / k
            total += term
        return k
    var slam = sqrt(lam)
    var loglam = log(lam)
    var b = Scalar[W](0.931) + Scalar[W](2.53) * slam
    var a = Scalar[W](-0.059) + Scalar[W](0.02483) * b
    var invalpha = Scalar[W](1.1239) + Scalar[W](1.1328) / (b - Scalar[W](3.4))
    var vr = Scalar[W](0.9277) - Scalar[W](3.6224) / (b - 2)
    for t in range(_REJECT_TRIES):
        var u = _word_uniform[W](seed, base + 2 * t) - Scalar[W](0.5)
        var v = _word_uniform[W](seed, base + 2 * t + 1)
        var us = Scalar[W](0.5) - abs(u)
        var k = floor((2 * a / us + b) * u + lam + Scalar[W](0.43))
        if us >= Scalar[W](0.07) and v <= vr:
            return k
        if k < 0 or (us < Scalar[W](0.013) and v > us):
            continue
        if log(v) + log(invalpha) - log(a / (us * us) + b) <= (
            -lam + k * loglam - _loggam[W](k + 1)
        ):
            return k
    return floor(lam)


def _poisson_step[
    dtype: DType, W: DType
](seed: UInt64, index: Int, lam: Scalar[W], unused: Scalar[W]) -> Scalar[
    dtype
] where W.is_floating_point():
    return _poisson_count[W](seed, index * _REJECT_WORDS, lam).cast[dtype]()


def _binomial_count[
    W: DType
](seed: UInt64, base: Int, n: Scalar[W], p_in: Scalar[W]) -> Scalar[
    W
] where W.is_floating_point():
    """A binomial draw from the words at `base`: inversion for
    `n min(p, 1-p) < 10`, and Hormann's BTRS transformed rejection above,
    with `p > 1/2` reflected to `n - Binomial(n, 1 - p)`. The rejection's
    fallback is the mode."""
    if n <= 0 or p_in <= 0:
        return Scalar[W](0)
    if p_in >= 1:
        return n
    var flip = p_in > Scalar[W](0.5)
    var p = 1 - p_in if flip else p_in
    var q = 1 - p
    var k = Scalar[W](0)
    if n * p < 10:
        var u = _word_uniform[W](seed, base)
        var term = q**n
        var total = term
        var ratio = p / q
        var mean = n * p
        var steps = min(Int(n), Int(mean + 12 * sqrt(mean) + 30))
        for _ in range(steps):
            if u < total:
                break
            k += 1
            term = term * ratio * (n - k + 1) / k
            total += term
    else:
        var spq = sqrt(n * p * q)
        var b = Scalar[W](1.15) + Scalar[W](2.53) * spq
        var a = Scalar[W](-0.0873) + Scalar[W](0.0248) * b + Scalar[W](0.01) * p
        var c = n * p + Scalar[W](0.5)
        var vr = Scalar[W](0.92) - Scalar[W](4.2) / b
        var alpha = (Scalar[W](2.83) + Scalar[W](5.1) / b) * spq
        var lpq = log(p / q)
        var m = floor((n + 1) * p)
        var h = _loggam[W](m + 1) + _loggam[W](n - m + 1)
        k = m
        for t in range(_REJECT_TRIES):
            var u = _word_uniform[W](seed, base + 2 * t) - Scalar[W](0.5)
            var v = _word_uniform[W](seed, base + 2 * t + 1)
            var us = Scalar[W](0.5) - abs(u)
            var cand = floor((2 * a / us + b) * u + c)
            if cand < 0 or cand > n:
                continue
            if us >= Scalar[W](0.07) and v <= vr:
                k = cand
                break
            var lv = log(v * alpha / (a / (us * us) + b))
            if (
                lv
                <= h
                - _loggam[W](cand + 1)
                - _loggam[W](n - cand + 1)
                + (cand - m) * lpq
            ):
                k = cand
                break
    return n - k if flip else k


def _binomial_step[
    dtype: DType, W: DType
](seed: UInt64, index: Int, n: Scalar[W], p: Scalar[W]) -> Scalar[
    dtype
] where W.is_floating_point():
    return _binomial_count[W](seed, index * _REJECT_WORDS, n, p).cast[dtype]()


def _nbinom_step[
    dtype: DType, W: DType
](seed: UInt64, index: Int, n: Scalar[W], p: Scalar[W]) -> Scalar[
    dtype
] where W.is_floating_point():
    """A negative binomial draw as NumPy makes it, a gamma-Poisson mixture:
    `lambda ~ Gamma(n, (1-p)/p)` from the element's gamma block, then a
    Poisson at `lambda` from the rejection block after it."""
    var base = index * (_GAMMA_WORDS + _REJECT_WORDS)
    var lam = _standard_gamma[W](seed, base, n) * (1 - p) / p
    return _poisson_count[W](seed, base + _GAMMA_WORDS, lam).cast[dtype]()


comptime _HYPERGEOM_MAX_DRAWS = 4096
"""The most draws `hypergeometric` takes per sample: it simulates the urn,
one uniform per draw, so its words per element are fixed at this."""


def _hypergeom_step[
    dtype: DType, W: DType
](seed: UInt64, index: Int, good: Scalar[W], packed: Scalar[W]) -> Scalar[
    dtype
] where W.is_floating_point():
    """`nsample` draws without replacement from an urn of `ngood` good and
    `nbad` bad balls, one uniform per draw, counting the good ones. The
    two counts beyond `ngood` arrive packed as `nbad * 8192 + nsample`,
    both below 8192."""
    var nbad = floor(packed / 8192)
    var nsample = Int(packed - nbad * 8192)
    var g = good
    var total = good + nbad
    var count = Scalar[W](0)
    var base = index * _HYPERGEOM_MAX_DRAWS
    for d in range(nsample):
        var u = _word_uniform[W](seed, base + d)
        if u * total < g:
            count += 1
            g -= 1
        total -= 1
    return count.cast[dtype]()


@always_inline
def _width[dtype: DType, gpu: Bool]() -> Int:
    comptime if gpu:
        return 1
    else:
        return simd_width_of[dtype]()


def _fill[
    dtype: DType,
    P: DType,
    draw: def(UInt64, Int, Scalar[P], Scalar[P]) thin -> Scalar[dtype],
    gpu: Bool,
    *dims: Int,
](
    seed: UInt64, a: Scalar[P], b: Scalar[P], ctx: Optional[DeviceContext]
) raises -> Static[dtype, *dims]:
    """Allocate on `ctx`'s device and fill it there, element `i` from
    `draw(seed, i, a, b)`: `elementwise` at native SIMD width on the host,
    one thread per element with `gpu=True`.

    The fill runs over a rank-1 tensor of the same element count and the
    result is that tensor's buffer retyped to `*dims` -- the `as_dynamic` /
    `as_static` move, no copy -- because a compile-time shape pack is not
    something the layout prover can show `coalesce`'s `all_dims_known` for,
    while a single extent is a view it can store through directly.
    """
    var device = _context(ctx)
    comptime n = _product[*dims]()
    # `_uninitialized`, not the zero-filling constructor: every element is
    # about to be written, and at `DType.bool` the zero fill is the
    # compiler crash `findings.mdc` records.
    var flat = Static[dtype, n]._uninitialized(device)
    var ys = flat.tile()

    @always_inline
    def body[
        w: Int, alignment: Int = 1
    ](coord: Coord) {var ys, var seed, var a, var b}:
        var base = coord_to_index_list(coord)[0]
        var values = SIMD[dtype, w]()
        comptime for lane in range(w):
            values[lane] = draw(seed, base + lane, a, b)
        ys.store[w](coord, values)

    elementwise[simd_width=_width[dtype, gpu](), target=_target[gpu]()](
        body, Coord(n), device
    )
    device.synchronize()
    return Static[dtype, *dims](
        flat._buffer,
        rebind[_LayoutOf[*dims]](row_major[*dims]()),
        flat.host_addressable,
    )


def _next_seed() -> UInt64:
    """One 64-bit word from `std.random`'s global generator, so `seed(v)`
    reproduces every module-level draw that follows it."""
    return random_ui64(0, UInt64.MAX)


def uniform[
    dtype: DType, *dims: Int, gpu: Bool = False
](
    low: Scalar[dtype] = 0,
    high: Scalar[dtype] = 1,
    ctx: Optional[DeviceContext] = None,
) raises -> Static[dtype, *dims]:
    """A new tensor of the given compile-time shape on `ctx`'s device,
    filled there with values drawn uniformly from `[low, high)`.
    `numpy.random.uniform`. Pass `gpu=True` with a device `ctx` to fill one
    thread per element; the module docstring has the stream layout.

    Parameters:
        dtype: The element type of the result.
        dims: The result's compile-time shape.
        gpu: Whether to fill one thread per element on `ctx`'s device, which
            must then be an accelerator; `False` fills on the host, threaded at
            native SIMD width.

    Args:
        low: The inclusive lower bound.
        high: The exclusive upper bound.
        ctx: The device to allocate and fill on; `None` means the host.

    Returns:
        A new `Static` tensor of shape `dims` on `ctx`'s device, filled with
        uniform draws from `[low, high)`.

    Raises:
        When the host context cannot be created, or the allocation or the fill
        launch fails.
    """
    return _fill[dtype, dtype, _uniform_step[dtype], gpu, *dims](
        _next_seed(), low, high, ctx
    )


def normal[
    dtype: DType, *dims: Int, gpu: Bool = False
](
    mean: Scalar[dtype] = 0,
    stddev: Scalar[dtype] = 1,
    ctx: Optional[DeviceContext] = None,
) raises -> Static[dtype, *dims] where dtype.is_floating_point():
    """A new tensor of the given compile-time shape on `ctx`'s device,
    filled there with values drawn from a normal distribution with the
    given `mean` and `stddev`, by Box-Muller on two uniforms per element.
    `numpy.random.normal`. `gpu=True` fills on the device.

    Parameters:
        dtype: The floating-point element type of the result.
        dims: The result's compile-time shape.
        gpu: Whether to fill one thread per element on `ctx`'s device, which
            must then be an accelerator; `False` fills on the host, threaded at
            native SIMD width.

    Args:
        mean: The mean of the distribution.
        stddev: The standard deviation of the distribution.
        ctx: The device to allocate and fill on; `None` means the host.

    Returns:
        A new `Static` tensor of shape `dims` on `ctx`'s device, filled with
        normal draws.

    Raises:
        When the host context cannot be created, or the allocation or the fill
        launch fails.
    """
    return _fill[dtype, dtype, _normal_step[dtype], gpu, *dims](
        _next_seed(), mean, stddev, ctx
    )


def exponential[
    dtype: DType, *dims: Int, gpu: Bool = False
](
    scale: Scalar[dtype] = 1, ctx: Optional[DeviceContext] = None
) raises -> Static[dtype, *dims] where dtype.is_floating_point():
    """A new tensor of the given compile-time shape on `ctx`'s device,
    filled there with values drawn from an exponential distribution with
    the given `scale` (`1/rate`), by inverse CDF: `-scale ln(1 - U)`, with
    `U` in `[0, 1)` so `1 - U` never reaches `0`. `numpy.random.exponential`.
    `gpu=True` fills on the device.

    Parameters:
        dtype: The floating-point element type of the result.
        dims: The result's compile-time shape.
        gpu: Whether to fill one thread per element on `ctx`'s device, which
            must then be an accelerator; `False` fills on the host, threaded at
            native SIMD width.

    Args:
        scale: The mean of the distribution, `1 / rate`.
        ctx: The device to allocate and fill on; `None` means the host.

    Returns:
        A new `Static` tensor of shape `dims` on `ctx`'s device, filled with
        exponential draws.

    Raises:
        When the host context cannot be created, or the allocation or the fill
        launch fails.
    """
    return _fill[dtype, dtype, _exponential_step[dtype], gpu, *dims](
        _next_seed(), scale, scale, ctx
    )


def randint[
    dtype: DType, *dims: Int, gpu: Bool = False
](low: Int, high: Int, ctx: Optional[DeviceContext] = None) raises -> Static[
    dtype, *dims
]:
    """A new tensor filled with integers drawn uniformly from `[low, high)`.
    `numpy.random.randint`.

    `low + ((high - low) * word) >> 32` on the raw 32-bit Philox word, in
    integer arithmetic, so it is exact and runs on a device without
    `double`. `dtype` is the tensor's own -- an integer dtype gives exact
    integers, a floating one gives integral values in floating storage.
    `gpu=True` fills on the device.

    Parameters:
        dtype: The element type of the result; an integer dtype gives exact
            integers, a floating one integral values in floating storage.
        dims: The result's compile-time shape.
        gpu: Whether to fill one thread per element on `ctx`'s device, which
            must then be an accelerator; `False` fills on the host, threaded at
            native SIMD width.

    Args:
        low: The inclusive lower bound.
        high: The exclusive upper bound; `high - low` must be below `2^32`.
        ctx: The device to allocate and fill on; `None` means the host.

    Returns:
        A new `Static` tensor of shape `dims` on `ctx`'s device, filled with
        integers drawn uniformly from `[low, high)`.

    Raises:
        When the host context cannot be created, or the allocation or the fill
        launch fails.
    """
    return _fill[dtype, DType.int64, _randint_step[dtype], gpu, *dims](
        _next_seed(), Int64(low), Int64(high), ctx
    )


def randbool[
    dtype: DType, *dims: Int, gpu: Bool = False
](p: Float64 = 0.5, ctx: Optional[DeviceContext] = None) raises -> Static[
    DType.bool, *dims
] where (dtype == DType.bool):
    """A new boolean tensor, true with probability `p`.

    `numpy.random.binomial(1, p)` reshaped as a mask, which is what a
    caller wants it for.

    Takes a leading `dtype` parameter like every other draw in this module
    even though the only admissible one is `DType.bool`: a caller writing
    `randbool[DType.bool, 4](ctx=ctx)` beside `uniform[DType.float32, 4](ctx=ctx)`
    does not have to remember that this one is shaped differently.
    `gpu=True` fills on the device.

    Parameters:
        dtype: Must be `DType.bool`; there so the spelling matches the other
            draws.
        dims: The result's compile-time shape.
        gpu: Whether to fill one thread per element on `ctx`'s device, which
            must then be an accelerator; `False` fills on the host, threaded at
            native SIMD width.

    Args:
        p: The probability that an element is true.
        ctx: The device to allocate and fill on; `None` means the host.

    Returns:
        A new `Static` tensor of shape `dims` on `ctx`'s device, a `DType.bool`
        mask true with probability `p`.

    Raises:
        When the host context cannot be created, or the allocation or the fill
        launch fails.
    """
    return _fill[
        DType.bool, DType.float32, _randbool_step[DType.bool], gpu, *dims
    ](_next_seed(), Float32(p), Float32(p), ctx)


def seed(value: Int):
    """Seed the global generator that `uniform`/`normal`/`exponential`/
    `randint`/`randbool` take their per-call Philox seed from.

    Forwards to `std.random.seed`; two draws separated only by the same
    `seed(value)` call reproduce identically, on the host or the device
    (checked in `tests/stats/test_random.mojo`). Has no effect on a
    `Generator`, which owns its seed.

    Args:
        value: The seed handed to `std.random.seed`.
    """
    _std_seed(value)


def _as_vector[T: TensorLike](x: T) raises -> Dynamic[T.dtype, 1]:
    """`x`'s elements as a rank-1 run-time-shaped tensor on its device, the
    shape `take` can prove its axis against from inside a generic method."""
    return _same_order(x, row_major(_dyn_shape[1](x.size())))


struct Generator(Copyable):
    """A named, reproducible source of draws. `numpy.random.Generator`.

    ```mojo
    var rng = Generator(seed=0)
    var xs = rng.uniform[DType.float64, 8]()
    var ys = rng.normal[DType.float64, 8]()
    ```

    Two generators built from the same seed produce the same sequence, and
    a generator's own sequence does not depend on what other code did to
    the global RNG in between -- which is the property the module-level
    `seed`/`uniform`/`normal` pair cannot offer, since they all share one
    process-wide state. Each draw's Philox seed is this generator's
    current seed, advanced by one afterwards, so no global state is read or
    written and several `Generator`s can be used from several threads.

    The module-level functions stay, and are what a program that never
    needs a second stream should keep using.
    """

    var _seed: UInt64
    """The seed the next draw will use; advanced by one after each."""

    def __init__(out self, seed: Int = 0):
        """A generator whose first draw uses `seed`.

        Args:
            seed: The Philox seed of the first draw; later draws use `seed + 1`,
                `seed + 2`, and so on.
        """
        self._seed = UInt64(seed)

    def _advance(mut self) -> UInt64:
        var current = self._seed
        self._seed += 1
        return current

    def uniform[
        dtype: DType, *dims: Int, gpu: Bool = False
    ](
        mut self,
        low: Scalar[dtype] = 0,
        high: Scalar[dtype] = 1,
        ctx: Optional[DeviceContext] = None,
    ) raises -> Static[dtype, *dims]:
        """`uniform`, from this generator's stream.

        Parameters:
            dtype: The element type of the result.
            dims: The result's compile-time shape.
            gpu: Whether to fill one thread per element on `ctx`'s device, which
                must then be an accelerator; `False` fills on the host, threaded
                at native SIMD width.

        Args:
            low: The inclusive lower bound.
            high: The exclusive upper bound.
            ctx: The device to allocate and fill on; `None` means the host.

        Returns:
            A new `Static` tensor of shape `dims` on `ctx`'s device, filled with
            uniform draws from `[low, high)`; this generator's seed advances by
            one.

        Raises:
            When the host context cannot be created, or the allocation or the
            fill launch fails.
        """
        return _fill[dtype, dtype, _uniform_step[dtype], gpu, *dims](
            self._advance(), low, high, ctx
        )

    def normal[
        dtype: DType, *dims: Int, gpu: Bool = False
    ](
        mut self,
        mean: Scalar[dtype] = 0,
        stddev: Scalar[dtype] = 1,
        ctx: Optional[DeviceContext] = None,
    ) raises -> Static[dtype, *dims] where dtype.is_floating_point():
        """`normal`, from this generator's stream.

        Parameters:
            dtype: The floating-point element type of the result.
            dims: The result's compile-time shape.
            gpu: Whether to fill one thread per element on `ctx`'s device, which
                must then be an accelerator; `False` fills on the host, threaded
                at native SIMD width.

        Args:
            mean: The mean of the distribution.
            stddev: The standard deviation of the distribution.
            ctx: The device to allocate and fill on; `None` means the host.

        Returns:
            A new `Static` tensor of shape `dims` on `ctx`'s device, filled with
            normal draws; this generator's seed advances by one.

        Raises:
            When the host context cannot be created, or the allocation or the
            fill launch fails.
        """
        return _fill[dtype, dtype, _normal_step[dtype], gpu, *dims](
            self._advance(), mean, stddev, ctx
        )

    def exponential[
        dtype: DType, *dims: Int, gpu: Bool = False
    ](
        mut self,
        scale: Scalar[dtype] = 1,
        ctx: Optional[DeviceContext] = None,
    ) raises -> Static[dtype, *dims] where dtype.is_floating_point():
        """`exponential`, from this generator's stream.

        Parameters:
            dtype: The floating-point element type of the result.
            dims: The result's compile-time shape.
            gpu: Whether to fill one thread per element on `ctx`'s device, which
                must then be an accelerator; `False` fills on the host, threaded
                at native SIMD width.

        Args:
            scale: The mean of the distribution, `1 / rate`.
            ctx: The device to allocate and fill on; `None` means the host.

        Returns:
            A new `Static` tensor of shape `dims` on `ctx`'s device, filled with
            exponential draws; this generator's seed advances by one.

        Raises:
            When the host context cannot be created, or the allocation or the
            fill launch fails.
        """
        return _fill[dtype, dtype, _exponential_step[dtype], gpu, *dims](
            self._advance(), scale, scale, ctx
        )

    def randint[
        dtype: DType, *dims: Int, gpu: Bool = False
    ](
        mut self, low: Int, high: Int, ctx: Optional[DeviceContext] = None
    ) raises -> Static[dtype, *dims]:
        """`randint`, from this generator's stream.

        Parameters:
            dtype: The element type of the result; an integer dtype gives exact
                integers, a floating one integral values in floating storage.
            dims: The result's compile-time shape.
            gpu: Whether to fill one thread per element on `ctx`'s device, which
                must then be an accelerator; `False` fills on the host, threaded
                at native SIMD width.

        Args:
            low: The inclusive lower bound.
            high: The exclusive upper bound; `high - low` must be below `2^32`.
            ctx: The device to allocate and fill on; `None` means the host.

        Returns:
            A new `Static` tensor of shape `dims` on `ctx`'s device, filled with
            integers drawn uniformly from `[low, high)`; this generator's seed
            advances by one.

        Raises:
            When the host context cannot be created, or the allocation or the
            fill launch fails.
        """
        return _fill[dtype, DType.int64, _randint_step[dtype], gpu, *dims](
            self._advance(), Int64(low), Int64(high), ctx
        )

    def randbool[
        dtype: DType, *dims: Int, gpu: Bool = False
    ](
        mut self, p: Float64 = 0.5, ctx: Optional[DeviceContext] = None
    ) raises -> Static[DType.bool, *dims] where (dtype == DType.bool):
        """`randbool`, from this generator's stream.

        Parameters:
            dtype: Must be `DType.bool`; there so the spelling matches the other
                draws.
            dims: The result's compile-time shape.
            gpu: Whether to fill one thread per element on `ctx`'s device, which
                must then be an accelerator; `False` fills on the host, threaded
                at native SIMD width.

        Args:
            p: The probability that an element is true.
            ctx: The device to allocate and fill on; `None` means the host.

        Returns:
            A new `Static` tensor of shape `dims` on `ctx`'s device, a
            `DType.bool` mask true with probability `p`; this generator's seed
            advances by one.

        Raises:
            When the host context cannot be created, or the allocation or the
            fill launch fails.
        """
        return _fill[
            DType.bool, DType.float32, _randbool_step[DType.bool], gpu, *dims
        ](self._advance(), Float32(p), Float32(p), ctx)

    def gamma[
        dtype: DType, *dims: Int, gpu: Bool = False
    ](
        mut self,
        shape: Scalar[dtype],
        scale: Scalar[dtype] = 1,
        ctx: Optional[DeviceContext] = None,
    ) raises -> Static[dtype, *dims] where dtype.is_floating_point():
        """Gamma draws with the given `shape` and `scale`.
        `numpy.random.Generator.gamma`.

        Marsaglia and Tsang's squeeze method, with their `U^(1/shape)` boost
        below `shape = 1`; each element takes its own fixed block of the
        stream, so the fill runs on the device like every other draw.

        Parameters:
            dtype: The floating-point element type of the result.
            dims: The result's compile-time shape.
            gpu: Whether to fill one thread per element on `ctx`'s device.

        Args:
            shape: The shape `k`, positive.
            scale: The scale `theta`; the mean is `k theta`.
            ctx: The device to allocate and fill on; `None` means the host.

        Returns:
            A new `Static` tensor of gamma draws; this generator's seed
            advances by one.

        Raises:
            When the host context cannot be created, or the allocation or the
            fill launch fails.
        """
        return _fill[dtype, dtype, _gamma_step[dtype], gpu, *dims](
            self._advance(), shape, scale, ctx
        )

    def beta[
        dtype: DType, *dims: Int, gpu: Bool = False
    ](
        mut self,
        a: Scalar[dtype],
        b: Scalar[dtype],
        ctx: Optional[DeviceContext] = None,
    ) raises -> Static[dtype, *dims] where dtype.is_floating_point():
        """Beta draws with shapes `a` and `b`. `numpy.random.Generator.beta`.

        `X / (X + Y)` with `X ~ Gamma(a)` and `Y ~ Gamma(b)`.

        Parameters:
            dtype: The floating-point element type of the result.
            dims: The result's compile-time shape.
            gpu: Whether to fill one thread per element on `ctx`'s device.

        Args:
            a: The first shape, positive.
            b: The second shape, positive.
            ctx: The device to allocate and fill on; `None` means the host.

        Returns:
            A new `Static` tensor of draws in `(0, 1)`; this generator's seed
            advances by one.

        Raises:
            When the host context cannot be created, or the allocation or the
            fill launch fails.
        """
        return _fill[dtype, dtype, _beta_step[dtype], gpu, *dims](
            self._advance(), a, b, ctx
        )

    def lognormal[
        dtype: DType, *dims: Int, gpu: Bool = False
    ](
        mut self,
        mean: Scalar[dtype] = 0,
        sigma: Scalar[dtype] = 1,
        ctx: Optional[DeviceContext] = None,
    ) raises -> Static[dtype, *dims] where dtype.is_floating_point():
        """Log-normal draws: `exp(N(mean, sigma))`.
        `numpy.random.Generator.lognormal`.

        Parameters:
            dtype: The floating-point element type of the result.
            dims: The result's compile-time shape.
            gpu: Whether to fill one thread per element on `ctx`'s device.

        Args:
            mean: The mean of the underlying normal.
            sigma: The standard deviation of the underlying normal.
            ctx: The device to allocate and fill on; `None` means the host.

        Returns:
            A new `Static` tensor of positive draws; this generator's seed
            advances by one.

        Raises:
            When the host context cannot be created, or the allocation or the
            fill launch fails.
        """
        return _fill[dtype, dtype, _lognormal_step[dtype], gpu, *dims](
            self._advance(), mean, sigma, ctx
        )

    def poisson[
        dtype: DType, *dims: Int, gpu: Bool = False
    ](
        mut self, lam: Float64 = 1.0, ctx: Optional[DeviceContext] = None
    ) raises -> Static[dtype, *dims]:
        """Poisson draws with rate `lam`. `numpy.random.Generator.poisson`.

        NumPy's split and constants: inversion by sequential search below
        `lam = 10`, Hormann's PTRS transformed rejection above. The
        arithmetic is `float64` on the host and `float32` on the device
        (Metal has no `double`), which bounds a device draw's fidelity for a
        rate past about `1e6`.

        Parameters:
            dtype: The element type of the result; an integer dtype gives
                exact counts.
            dims: The result's compile-time shape.
            gpu: Whether to fill one thread per element on `ctx`'s device.

        Args:
            lam: The rate, at least 0.
            ctx: The device to allocate and fill on; `None` means the host.

        Returns:
            A new `Static` tensor of counts; this generator's seed advances by
            one.

        Raises:
            When the host context cannot be created, or the allocation or the
            fill launch fails.
        """
        comptime if gpu:
            comptime W = DType.float32
            return _fill[dtype, W, _poisson_step[dtype, W], gpu, *dims](
                self._advance(), Scalar[W](lam), Scalar[W](lam), ctx
            )
        else:
            comptime W = DType.float64
            return _fill[dtype, W, _poisson_step[dtype, W], gpu, *dims](
                self._advance(), Scalar[W](lam), Scalar[W](lam), ctx
            )

    def binomial[
        dtype: DType, *dims: Int, gpu: Bool = False
    ](
        mut self, n: Int, p: Float64, ctx: Optional[DeviceContext] = None
    ) raises -> Static[dtype, *dims]:
        """Binomial draws: successes in `n` trials of probability `p`.
        `numpy.random.Generator.binomial`.

        Inversion for `n min(p, 1-p) < 10` and Hormann's BTRS transformed
        rejection above, `p > 1/2` reflected; `float64` on the host and
        `float32` on the device, as `poisson` is.

        Parameters:
            dtype: The element type of the result; an integer dtype gives
                exact counts.
            dims: The result's compile-time shape.
            gpu: Whether to fill one thread per element on `ctx`'s device.

        Args:
            n: The number of trials, at least 0.
            p: The success probability, in `[0, 1]`.
            ctx: The device to allocate and fill on; `None` means the host.

        Returns:
            A new `Static` tensor of counts in `[0, n]`; this generator's seed
            advances by one.

        Raises:
            When the host context cannot be created, or the allocation or the
            fill launch fails.
        """
        comptime if gpu:
            comptime W = DType.float32
            return _fill[dtype, W, _binomial_step[dtype, W], gpu, *dims](
                self._advance(), Scalar[W](n), Scalar[W](p), ctx
            )
        else:
            comptime W = DType.float64
            return _fill[dtype, W, _binomial_step[dtype, W], gpu, *dims](
                self._advance(), Scalar[W](n), Scalar[W](p), ctx
            )

    def spawn(mut self, count: Int) -> List[Generator]:
        """`count` independent child generators.
        `numpy.random.Generator.spawn`.

        Each child's seed is two fresh words of this generator's Philox
        stream at a distant offset, so children do not overlap each other
        or the parent's draws; the parent's seed then advances by one.

        Args:
            count: How many children.

        Returns:
            The children, each its own reproducible stream.
        """
        var base = self._advance()
        var children = List[Generator](capacity=count)
        for k in range(count):
            var r = Random(seed=base, offset=UInt64(1) << 40 | UInt64(k))
            var words = r.step()
            var child_seed = (UInt64(words[0]) << 32) | UInt64(words[1])
            children.append(Generator(seed=Int(child_seed)))
        return children^

    def permutation[
        n: Int, gpu: Bool = False
    ](mut self, ctx: Optional[DeviceContext] = None) raises -> Dynamic[
        DType.int64, 1
    ]:
        """A random permutation of `0 .. n-1`.
        `numpy.random.Generator.permutation(n)`.

        `argsort` of `n` uniform keys, on the device at `gpu=True`. Keys that
        tie keep their index order, a bias far below any sample size that
        could detect it (`2^-24` per pair at `float32`, `2^-53` at
        `float64`).

        Parameters:
            n: The length of the permutation.
            gpu: Whether to draw and sort on `ctx`'s device.

        Args:
            ctx: The device to allocate on; `None` means the host.

        Returns:
            A `Dynamic` rank-1 `int64` tensor holding each index once.

        Raises:
            When allocation, the fill or the sort fails.
        """
        comptime key_dtype = DType.float32 if gpu else DType.float64
        var keys = self.uniform[key_dtype, n, gpu=gpu](ctx=ctx)
        return argsort[gpu=gpu](keys)

    def permutation[
        T: TensorLike, gpu: Bool = False
    ](mut self, x: T) raises -> Dynamic[T.dtype, 1] where (
        T.LayoutType.rank == 1 and T.LayoutType.all_dims_known
    ):
        """`x` in a random order, a copy. `numpy.random.Generator.permutation(x)`
        for a vector.

        Parameters:
            T: The tensor type of `x`, rank 1, static length.
            gpu: Whether to draw, sort and gather on `x`'s device.

        Args:
            x: The values to permute.

        Returns:
            A new `Dynamic` tensor with `x`'s elements in a random order.

        Raises:
            When allocation, the fill, the sort or the gather fails.
        """
        comptime n = dim[T, 0]
        var order = self.permutation[n, gpu=gpu](x.context())
        return take[axis=0, gpu=gpu](_as_vector(x), order)

    def shuffle[
        dtype: DType, n: Int, gpu: Bool = False
    ](mut self, mut x: Static[dtype, n]) raises:
        """Shuffle `x` in place. `numpy.random.Generator.shuffle` for a
        vector: `permutation(x)` copied back into `x`'s own buffer, on its
        device.

        Parameters:
            dtype: The element type of `x`.
            n: The length of `x`.
            gpu: Whether to draw, sort and gather on `x`'s device.

        Args:
            x: The vector to shuffle.

        Raises:
            When allocation, the fill, the sort, the gather or the copy fails.
        """
        var shuffled = self.permutation[gpu=gpu](x)
        x.context().enqueue_copy(x._buffer, shuffled._buffer)
        x.context().synchronize()

    def choice[
        T: TensorLike, size: Int, replace: Bool = True, gpu: Bool = False
    ](mut self, a: T) raises -> Dynamic[T.dtype, 1] where (
        T.LayoutType.rank == 1 and T.LayoutType.all_dims_known and size >= 0
    ):
        """`size` elements of `a` drawn uniformly, with or without
        replacement. `numpy.random.Generator.choice(a, size, replace)`.

        With replacement it is `size` uniform integers gathered from `a`;
        without, the first `size` of a random permutation, which needs
        `size <= len(a)`.

        Parameters:
            T: The tensor type of `a`, rank 1, static length.
            size: How many to draw.
            replace: Whether an element may be drawn more than once.
            gpu: Whether to draw and gather on `a`'s device.

        Args:
            a: The population.

        Returns:
            A new `Dynamic` tensor of the draws.

        Raises:
            When `replace` is `False` and `size > len(a)`, or a device
            operation fails.
        """
        comptime n = dim[T, 0]
        var ctx = a.context()
        comptime if replace:
            var idx = self.randint[DType.int64, size, gpu=gpu](0, n, ctx=ctx)
            return take[axis=0, gpu=gpu](_as_vector(a), idx)
        else:
            if size > n:
                raise Error(
                    "choice: cannot take ",
                    size,
                    " of ",
                    n,
                    " without replacement",
                )
            var order = self.permutation[n, gpu=gpu](ctx)
            var first = arange_n[size, DType.int64](ctx=ctx)
            var head = take[axis=0, gpu=gpu](order, first)
            return take[axis=0, gpu=gpu](_as_vector(a), head)

    def choice[
        T: TensorLike, P: TensorLike, size: Int, gpu: Bool = False
    ](mut self, a: T, p: P) raises -> Dynamic[T.dtype, 1] where (
        T.LayoutType.rank == 1
        and T.LayoutType.all_dims_known
        and P.LayoutType.rank == 1
        and P.LayoutType.all_dims_known
        and P.dtype.is_floating_point()
        and size >= 0
    ):
        """`size` elements of `a` drawn with replacement at the weights `p`.
        `numpy.random.Generator.choice(a, size, p=p)`.

        Inversion: the running sum of `p`, one uniform per draw scaled by the
        total, and `searchsorted` into the sum -- all on `a`'s device at
        `gpu=True`. `p` need not sum to exactly 1; it is normalized by its
        total, where NumPy raises unless it sums to 1.

        Parameters:
            T: The tensor type of `a`, rank 1, static length.
            P: The tensor type of `p`, rank 1, floating-point, as long as `a`.
            size: How many to draw.
            gpu: Whether to draw and gather on `a`'s device.

        Args:
            a: The population.
            p: The nonnegative weights, one per element of `a`.

        Returns:
            A new `Dynamic` tensor of the draws.

        Raises:
            When `p` is not as long as `a`, or a device operation fails.
        """
        comptime n = dim[T, 0]
        comptime pd = P.dtype
        if p.size() != n:
            raise Error(
                "choice: p has ", p.size(), " weights for ", n, " elements"
            )
        var ctx = a.context()
        var cdf = cumsum[gpu=gpu](p)
        var total = cdf[n - 1]
        var u = self.uniform[pd, size, gpu=gpu](
            Scalar[pd](0), Scalar[pd](total), ctx=ctx
        )
        var idx = searchsorted[right=True, gpu=gpu](cdf, u)
        var clipped = clip[gpu=gpu](idx, Int64(0), Int64(n - 1))
        return take[axis=0, gpu=gpu](_as_vector(a), clipped)

    def multivariate_normal[
        M: TensorLike, C: TensorLike, count: Int, gpu: Bool = False
    ](mut self, mean: M, cov: C) raises -> Static[
        M.dtype, count, dim[M, 0]
    ] where (
        M.dtype.is_floating_point()
        and C.dtype.is_floating_point()
        and C.dtype == M.dtype
        and M.LayoutType.rank == 1
        and M.LayoutType.all_dims_known
        and C.LayoutType.rank == 2
        and C.LayoutType.all_dims_known
        and dim[C, 0] == dim[M, 0]
        and dim[C, 1] == dim[C, 0]
    ):
        """`count` draws from the multivariate normal with the given `mean`
        and covariance `cov`, one per row.
        `numpy.random.Generator.multivariate_normal(mean, cov, count,
        method="cholesky")`.

        `z L^T + mean` with `z` standard normal and `L` the Cholesky factor
        of `cov`, which must be positive definite: the draws, the
        factorization and the product all on `mean`'s device at `gpu=True`.

        Parameters:
            M: The tensor type of `mean`, rank 1, static length `d`.
            C: The tensor type of `cov`, `d x d`, same dtype.
            count: How many draws.
            gpu: Whether to draw and compute on `mean`'s device.

        Args:
            mean: The mean vector.
            cov: The covariance matrix, symmetric positive definite.

        Returns:
            A `count x d` tensor, one draw per row.

        Raises:
            When `cov` is not positive definite, or a device operation fails.
        """
        comptime dtype = M.dtype
        comptime d = dim[M, 0]
        var ctx = mean.context()
        var z = self.normal[dtype, count, d, gpu=gpu](ctx=ctx)
        var lower = cholesky[gpu=gpu](cov)
        var upper = transpose[gpu=gpu](lower)
        var out = rebind_var[Static[dtype, count, d]](matmul[gpu=gpu](z, upper))
        # The shift below reads the product from its own launch.
        ctx.synchronize()
        var mp = mean.tile().ptr.unsafe_origin_cast[MutAnyOrigin]()
        var op = out.tile().ptr.unsafe_origin_cast[MutAnyOrigin]()

        @always_inline
        def shift[w: Int, alignment: Int = 1](coord: Coord) {var mp, var op}:
            var f = coord_to_index_list(coord)[0]
            op[unsafe_offset=f] = op[unsafe_offset=f] + rebind[Scalar[dtype]](
                mp[unsafe_offset=f % d]
            )

        elementwise[simd_width=1, target=_target[gpu]()](
            shift, Coord(count * d), ctx
        )
        ctx.synchronize()
        _ = z^
        _ = upper^
        _ = lower^
        return out^

    def negative_binomial[
        dtype: DType, *dims: Int, gpu: Bool = False
    ](
        mut self, n: Float64, p: Float64, ctx: Optional[DeviceContext] = None
    ) raises -> Static[dtype, *dims]:
        """Negative binomial draws: failures before the `n`-th success at
        probability `p`. `numpy.random.Generator.negative_binomial`.

        NumPy's gamma-Poisson mixture, `Poisson(Gamma(n, (1-p)/p))`, each
        element from its own gamma and rejection blocks.

        Parameters:
            dtype: The element type of the result; an integer dtype gives
                exact counts.
            dims: The result's compile-time shape.
            gpu: Whether to fill one thread per element on `ctx`'s device.

        Args:
            n: The number of successes, positive (need not be an integer).
            p: The success probability, in `(0, 1]`.
            ctx: The device to allocate and fill on; `None` means the host.

        Returns:
            A new `Static` tensor of counts; this generator's seed advances by
            one.

        Raises:
            When the host context cannot be created, or the allocation or the
            fill launch fails.
        """
        comptime if gpu:
            comptime W = DType.float32
            return _fill[dtype, W, _nbinom_step[dtype, W], gpu, *dims](
                self._advance(), Scalar[W](n), Scalar[W](p), ctx
            )
        else:
            comptime W = DType.float64
            return _fill[dtype, W, _nbinom_step[dtype, W], gpu, *dims](
                self._advance(), Scalar[W](n), Scalar[W](p), ctx
            )

    def hypergeometric[
        dtype: DType, *dims: Int, gpu: Bool = False
    ](
        mut self,
        ngood: Int,
        nbad: Int,
        nsample: Int,
        ctx: Optional[DeviceContext] = None,
    ) raises -> Static[dtype, *dims]:
        """Hypergeometric draws: good balls among `nsample` drawn without
        replacement from `ngood` good and `nbad` bad.
        `numpy.random.Generator.hypergeometric`.

        The urn simulated, one uniform per ball drawn, so a draw costs
        `nsample` words; `nsample` is capped at 4096 and `ngood`, `nbad` at
        8191, where NumPy switches to HRUA ratio-of-uniforms rejection.

        Parameters:
            dtype: The element type of the result.
            dims: The result's compile-time shape.
            gpu: Whether to fill one thread per element on `ctx`'s device.

        Args:
            ngood: Good balls in the urn, at most 8191.
            nbad: Bad balls in the urn, at most 8191.
            nsample: Balls drawn, at most `min(ngood + nbad, 4096)`.
            ctx: The device to allocate and fill on; `None` means the host.

        Returns:
            A new `Static` tensor of counts; this generator's seed advances by
            one.

        Raises:
            If a count is negative or past its cap, or the fill fails.
        """
        if ngood < 0 or nbad < 0 or nsample < 0 or nsample > ngood + nbad:
            raise Error("hypergeometric: need 0 <= nsample <= ngood + nbad")
        if ngood > 8191 or nbad > 8191 or nsample > _HYPERGEOM_MAX_DRAWS:
            raise Error(
                "hypergeometric: ngood and nbad at most 8191 and nsample at"
                " most 4096 (the urn is simulated draw by draw)"
            )
        var packed = Float64(nbad * 8192 + nsample)
        comptime if gpu:
            comptime W = DType.float32
            return _fill[dtype, W, _hypergeom_step[dtype, W], gpu, *dims](
                self._advance(), Scalar[W](ngood), Scalar[W](packed), ctx
            )
        else:
            comptime W = DType.float64
            return _fill[dtype, W, _hypergeom_step[dtype, W], gpu, *dims](
                self._advance(), Scalar[W](ngood), Scalar[W](packed), ctx
            )
