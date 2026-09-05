"""Random sampling into `numax.core.array.Tensor`, `Plain`-only, over the host RNG.

`docs/parity.md` picks random sampling as a genuine `numax` gap:
without it, `examples/advanced/ode.mojo`'s GPU ensemble (which needs
initial conditions) and any Gaussian-process-shaped example reach for a
raw RNG directly and lose the `numax` entry point. **No `Random[FloatLike]`
conformer exists here on purpose** -- RNG is not mathematically
differentiable (seeding a `Dual`'s derivative from a random draw has no
well-defined meaning), so the trait contract does not fit the mathematics
and this stays outside `FloatLike` entirely. Every function below returns
`Plain` values into a `Tensor`.

**MAX-first was checked, and rejected for a structural reason, not a speed
one.** `nn.rand_uniform`/`nn.rand_normal` exist and fill a `TileTensor`
directly, which looked like the obvious route -- confirmed otherwise by
direct experiment. Both take their fill logic as an `OutputFn` parameter
bound to `RegisterPassable & ImplicitlyCopyable`, which a capturing
closure over a caller's own buffer does not satisfy (confirmed: passing
one fails to compile against that bound, even with an `imm`-only
capture). That shape is graph-op fusion machinery (the `OutputFn` is meant
to be threaded through a MAX `Graph` compilation, not called eagerly from
ordinary Mojo), not an eager host-side API a library like `numax` can
call directly. `std.random.rand`/`std.random.randn` -- host-only, global
state, not thread-safe, exactly the layer `std.random`'s own docs warn is
unsuitable for concurrent use -- fill a raw pointer directly with no such
constraint, and that is what every function in this module is built on.
This module's default (no-`max`-dependency) CPU host case is therefore
`std.random`, not MAX; MAX's random surface remains the right choice for
graph-level work, which this module does not do.

**GPU-consumed random values are `std.random.philox.{Random,
NormalRandom}` directly, not wrapped further.** Those are per-thread,
counter-based generators -- `Random(seed=s, offset=thread_id)` gives one
thread an independent stream with no shared mutable state, which is
exactly what a `map[gpu=True]` kernel body needs and exactly what
`std.random`'s global generator cannot provide inside a kernel (global
state has no meaning across GPU threads, and mutating it from device code
is not a real operation). Wrapping `Random`/`NormalRandom` in a `numax`
name would only rename them: the fixed-iteration invariant already applies
to them as MAX ships them (each `step_uniform`/`step_normal_4` call is a
fixed amount of branchless work), so there's nothing for `numax` to add.
`examples/intermediate/random_ensemble.mojo` calls them directly inside a
`map[gpu=True]` `step`, seeded from each element's own index so every
thread's stream is independent and the result is reproducible from one
fixed seed.

**`exponential` is the one genuine gap.** Neither `std.random` nor MAX
ships an exponential sampler; this module composes it from `uniform` via
inverse-CDF sampling: `-scale * ln(1 - U)` for `U` drawn from `[0, 1)`.
`std.random.rand` draws its uniform values from `[0, 1)` (half-open, `0`
included, `1` excluded), so `1 - U` is drawn from `(0, 1]` and `ln` never
sees a zero argument -- no epsilon clamp needed, unlike the domain guards
`numax.stats` needs for a general PDF.
"""

from std.math import log
from std.random import rand, randn, seed as _std_seed

from max.gpu.host import DeviceContext

from ..core.array import Shaped, Tensor, _context, _product


def uniform[
    dtype: DType, *dims: Int
](
    low: Scalar[dtype] = 0,
    high: Scalar[dtype] = 1,
    ctx: Optional[DeviceContext] = None,
) raises -> Shaped[dtype, *dims]:
    """A new tensor of the given compile-time shape on `ctx`'s device,
    filled with values drawn uniformly from `[low, high)`.

    The draw itself is `std.random`'s host RNG, so the values are generated
    on the host and copied to the tensor's device -- see this module's own
    docstring for why there is no device-side RNG conformer here.
    """
    comptime n = _product[*dims]()
    var storage = List[Scalar[dtype]](length=n, fill=0)
    rand(storage.unsafe_ptr(), n)
    var span = high - low
    for i in range(n):
        storage[i] = low + span * storage[i]
    return Shaped[dtype, *dims](_context(ctx), storage^)


def normal[
    dtype: DType, *dims: Int
](
    mean: Scalar[dtype] = 0,
    stddev: Scalar[dtype] = 1,
    ctx: Optional[DeviceContext] = None,
) raises -> Shaped[dtype, *dims]:
    """A new tensor of the given compile-time shape on `ctx`'s device,
    filled with values drawn from a normal distribution with the given
    `mean` and `stddev`.
    """
    comptime n = _product[*dims]()
    var storage = List[Scalar[dtype]](length=n, fill=0)
    randn(storage.unsafe_ptr(), n, Float64(mean), Float64(stddev))
    return Shaped[dtype, *dims](_context(ctx), storage^)


def exponential[
    dtype: DType, *dims: Int
](
    scale: Scalar[dtype] = 1, ctx: Optional[DeviceContext] = None
) raises -> Shaped[dtype, *dims]:
    """A new tensor of the given compile-time shape, filled with values drawn
    from an exponential distribution with the given `scale` (`1/rate`).

    See this module's own docstring for the inverse-CDF composition from
    `uniform` and why it needs no domain guard.
    """
    comptime n = _product[*dims]()
    var storage = List[Scalar[dtype]](length=n, fill=0)
    rand(storage.unsafe_ptr(), n)
    for i in range(n):
        storage[i] = -scale * Scalar[dtype](
            log(Float64(1) - Float64(storage[i]))
        )
    return Shaped[dtype, *dims](_context(ctx), storage^)


def randint[
    dtype: DType, *dims: Int
](low: Int, high: Int, ctx: Optional[DeviceContext] = None) raises -> Shaped[
    dtype, *dims
]:
    """A new tensor filled with integers drawn uniformly from `[low, high)`.
    `numpy.random.randint`.

    Drawn as uniform reals and floored, which is the standard construction
    and keeps this on the same host RNG as everything else here. `dtype`
    is the tensor's own -- an integer dtype gives exact integers, a
    floating one gives integral values in floating storage.
    """
    comptime n = _product[*dims]()
    var draws = List[Scalar[DType.float64]](length=n, fill=0)
    rand(draws.unsafe_ptr(), n)
    var span = Float64(high - low)
    var storage = List[Scalar[dtype]](length=n, fill=0)
    for i in range(n):
        var value = Float64(low) + span * Float64(draws[i])
        storage[i] = Scalar[dtype](Int(value))
    return Shaped[dtype, *dims](_context(ctx), storage^)


def randbool[
    dtype: DType, *dims: Int
](p: Float64 = 0.5, ctx: Optional[DeviceContext] = None) raises -> Shaped[
    DType.bool, *dims
] where (dtype == DType.bool):
    """A new boolean tensor, true with probability `p`.

    `numpy.random.binomial(1, p)` reshaped as a mask, which is what a
    caller wants it for.

    Takes a leading `dtype` parameter like every other draw in this module
    even though the only admissible one is `DType.bool`: a caller writing
    `randbool[DType.bool, 4](ctx=ctx)` beside `uniform[DType.float32, 4](ctx=ctx)`
    does not have to remember that this one is shaped differently.
    """
    comptime n = _product[*dims]()
    var draws = List[Scalar[DType.float64]](length=n, fill=0)
    rand(draws.unsafe_ptr(), n)
    var storage = List[Scalar[DType.bool]](length=n, fill=False)
    for i in range(n):
        storage[i] = Float64(draws[i]) < p
    return Shaped[DType.bool, *dims](_context(ctx), storage^)


def seed(value: Int):
    """Seed the host RNG that `uniform`/`normal`/`exponential` draw from.

    Forwards to `std.random.seed`; two draws separated only by the same
    `seed(value)` call reproduce identically (checked in
    `tests/stats/test_random.mojo`). Has no effect on `std.random.philox`'s
    per-thread GPU generators, which take their seed as an explicit
    constructor argument instead of consulting global state.
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
    process-wide state.

    The module-level functions stay, and are what a program that never
    needs a second stream should keep using; they draw from the global
    state the way they always did.

    ponytail: this reseeds the global `std.random` generator before each
    draw and advances its own counter afterwards, because `std.random`
    exposes no instantiable generator to own. That makes it reproducible
    and independent between `Generator`s, but not thread-safe and not
    independent of concurrent use of the global. A counter-based
    generator -- `std.random.philox.Random`, which the GPU path in this
    module's docstring already uses per thread -- is the upgrade if either
    matters.
    """

    var _seed: Int
    """The seed the next draw will use; advanced by one after each."""

    def __init__(out self, seed: Int = 0):
        """A generator whose first draw uses `seed`."""
        self._seed = seed

    def _advance(mut self):
        _std_seed(self._seed)
        self._seed += 1

    def uniform[
        dtype: DType, *dims: Int
    ](
        mut self,
        low: Scalar[dtype] = 0,
        high: Scalar[dtype] = 1,
        ctx: Optional[DeviceContext] = None,
    ) raises -> Shaped[dtype, *dims]:
        """`uniform`, from this generator's stream."""
        self._advance()
        return uniform[dtype, *dims](low, high, ctx=ctx)

    def normal[
        dtype: DType, *dims: Int
    ](
        mut self,
        mean: Scalar[dtype] = 0,
        stddev: Scalar[dtype] = 1,
        ctx: Optional[DeviceContext] = None,
    ) raises -> Shaped[dtype, *dims]:
        """`normal`, from this generator's stream."""
        self._advance()
        return normal[dtype, *dims](mean, stddev, ctx=ctx)

    def exponential[
        dtype: DType, *dims: Int
    ](
        mut self,
        scale: Scalar[dtype] = 1,
        ctx: Optional[DeviceContext] = None,
    ) raises -> Shaped[dtype, *dims]:
        """`exponential`, from this generator's stream."""
        self._advance()
        return exponential[dtype, *dims](scale, ctx=ctx)

    def randint[
        dtype: DType, *dims: Int
    ](
        mut self, low: Int, high: Int, ctx: Optional[DeviceContext] = None
    ) raises -> Shaped[dtype, *dims]:
        """`randint`, from this generator's stream."""
        self._advance()
        return randint[dtype, *dims](low, high, ctx=ctx)

    def randbool[
        dtype: DType, *dims: Int
    ](
        mut self, p: Float64 = 0.5, ctx: Optional[DeviceContext] = None
    ) raises -> Shaped[DType.bool, *dims] where (dtype == DType.bool):
        """`randbool`, from this generator's stream."""
        self._advance()
        return randbool[dtype, *dims](p, ctx=ctx)
