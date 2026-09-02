"""Random sampling into `numax.array.Tensor`, `Plain`-only, over the host RNG.

`docs/parity.md` picks random sampling as a genuine `numax` gap:
without it, `examples/advanced/ode.mojo`'s GPU ensemble (which needs
initial conditions) and any Gaussian-process-shaped example reach for a
raw RNG directly and lose the `numax` entry point. **No `Random[FloatLike]`
conformer exists here on purpose** -- RNG is not mathematically
differentiable (seeding a `Dual`'s derivative from a random draw has no
well-defined meaning), the same scoping shape as Kelvin's units in
`strategy.mdc` Track B: the trait contract doesn't fit the mathematics, so
this stays outside `FloatLike` entirely. Every function below returns
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
`numax.distributions` needs for a general PDF.
"""

from std.math import log
from std.random import rand, randn, seed as _std_seed

from .array import Tensor, _product


def uniform[
    dtype: DType, *dims: Int
](low: Scalar[dtype] = 0, high: Scalar[dtype] = 1) -> Tensor[dtype, *dims]:
    """A new tensor of the given compile-time shape, filled with values drawn
    uniformly from `[low, high)`.
    """
    comptime n = _product[*dims]()
    var storage = List[Scalar[dtype]](length=n, fill=0)
    rand(storage.unsafe_ptr(), n)
    var span = high - low
    for i in range(n):
        storage[i] = low + span * storage[i]
    return Tensor[dtype, *dims](storage^)


def normal[
    dtype: DType, *dims: Int
](mean: Scalar[dtype] = 0, stddev: Scalar[dtype] = 1) -> Tensor[dtype, *dims]:
    """A new tensor of the given compile-time shape, filled with values drawn
    from a normal distribution with the given `mean` and `stddev`.
    """
    comptime n = _product[*dims]()
    var storage = List[Scalar[dtype]](length=n, fill=0)
    randn(storage.unsafe_ptr(), n, Float64(mean), Float64(stddev))
    return Tensor[dtype, *dims](storage^)


def exponential[
    dtype: DType, *dims: Int
](scale: Scalar[dtype] = 1) -> Tensor[dtype, *dims]:
    """A new tensor of the given compile-time shape, filled with values drawn
    from an exponential distribution with the given `scale` (`1/rate`).

    See this module's own docstring for the inverse-CDF composition from
    `uniform` and why it needs no domain guard.
    """
    var out = uniform[dtype, *dims]()
    comptime n = _product[*dims]()
    for i in range(n):
        out[i] = -scale * Scalar[dtype](log(Float64(1) - Float64(out[i])))
    return out^


def seed(value: Int):
    """Seed the host RNG that `uniform`/`normal`/`exponential` draw from.

    Forwards to `std.random.seed`; two draws separated only by the same
    `seed(value)` call reproduce identically (checked in
    `tests/test_random.mojo`). Has no effect on `std.random.philox`'s
    per-thread GPU generators, which take their seed as an explicit
    constructor argument instead of consulting global state.
    """
    _std_seed(value)
