"""Random sampling into `numax.core.array.Tensor`, `Plain`-only, on the host
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

from std.math import cos, log, sqrt
from std.random import Random, random_ui64, seed as _std_seed
from std.sys.info import simd_width_of
from std.utils.coord import coord_to_index_list

from layout import Coord
from layout.tile_layout import row_major
from max.algorithm.functional import elementwise
from max.gpu.host import DeviceContext

from ..core.array import Static, _LayoutOf, _context, _product
from .statistics import _target

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
    result is that tensor's buffer retyped to `*dims` -- the `dynamic` /
    `static_view` move, no copy -- because a compile-time shape pack is not
    something the layout prover can show `coalesce`'s `all_dims_known` for,
    while a single extent is a view it can store through directly.
    """
    var device = _context(ctx)
    comptime n = _product[*dims]()
    # `_uninitialized`, not the zero-filling constructor: every element is
    # about to be written, and at `DType.bool` the zero fill is the
    # compiler crash `findings.mdc` records.
    var flat = Static[dtype, n]._uninitialized(device)
    var ys = flat.view()

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
        flat.buffer,
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
    """
    _std_seed(value)


struct Generator(Copyable, Movable):
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
        """A generator whose first draw uses `seed`."""
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
        """`uniform`, from this generator's stream."""
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
        """`normal`, from this generator's stream."""
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
        """`exponential`, from this generator's stream."""
        return _fill[dtype, dtype, _exponential_step[dtype], gpu, *dims](
            self._advance(), scale, scale, ctx
        )

    def randint[
        dtype: DType, *dims: Int, gpu: Bool = False
    ](
        mut self, low: Int, high: Int, ctx: Optional[DeviceContext] = None
    ) raises -> Static[dtype, *dims]:
        """`randint`, from this generator's stream."""
        return _fill[dtype, DType.int64, _randint_step[dtype], gpu, *dims](
            self._advance(), Int64(low), Int64(high), ctx
        )

    def randbool[
        dtype: DType, *dims: Int, gpu: Bool = False
    ](
        mut self, p: Float64 = 0.5, ctx: Optional[DeviceContext] = None
    ) raises -> Static[DType.bool, *dims] where (dtype == DType.bool):
        """`randbool`, from this generator's stream."""
        return _fill[
            DType.bool, DType.float32, _randbool_step[DType.bool], gpu, *dims
        ](self._advance(), Float32(p), Float32(p), ctx)
