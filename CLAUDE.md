# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

νMAX (`numax`) is a numerical computing library for Mojo built on MAX: special
functions, dense linear algebra, quadrature, ODE solvers, FFTs, distributions,
statistics, interpolation, signal processing, and a NumPy-named array surface.
Pinned to `mojo ==1.0.0` and `max ==26.5` via pixi.

## Commands

Everything runs through pixi tasks (see `pixi.toml`). **Every `mojo`
invocation needs this directory on the import path** — the tasks all spell out
`-I .`; a bare `pixi run mojo <file>` cannot find `numax`.

```bash
pixi run tests                    # all suites (aggregate task)
pixi run test-dual                # one suite; there is a test-<name> task per file
pixi run run tests/core/test_dual.mojo   # arbitrary file with -I . supplied
pixi run examples-cpu             # every example that does not need a GPU
pixi run examples                 # + the GPU ones (needs real Metal/CUDA)
pixi run accuracy                 # max error per function vs. checked-in mpmath refs
pixi run bench                    # map vs. a hand-rolled raw-SIMD loop
pixi run -e dev format            # mojo format over numax examples tests bench
pixi run -e dev format-check      # format + git diff --exit-code
pixi run -e dev docs-build        # mojodoc → target/doc/
```

Cross-language baselines live in separate environments so the default install
stays lean: `pixi run -e bench-python bench-numpy|bench-torch|bench-mlx`,
`pixi run -e bench-rust bench-thermite`, and
`pixi run -e bench-python accuracy-gen` to regenerate
`bench/accuracy/reference_data.mojo`.

`mojodoc` is not a pixi dependency (build-backend pin conflict); the `docs`
tasks shell out to a sibling `../mojodoc` checkout, overridable with
`MOJODOC_CLI`.

### Adding a test suite

Two steps, both required: create `tests/<subpkg>/test_<name>.mojo`, then add a
`test-<name>` task **and** wire it into the `tests` aggregate's `depends-on`
list in `pixi.toml`. A suite not in that list never runs in CI.

Test files use `std.testing`: `def test_*() raises` functions plus
`TestSuite.discover_tests[__functions_in_module()]().run()` in `main`.

### Type-check gate for generics

Mojo instantiates generics lazily, so tests do **not** type-check
uninstantiated generic code — and this library is almost entirely generic.
`mojo doc -I . numax` walks the whole public generic surface and catches
breakage tests cannot. Run it after touching generic signatures. (CI does not
yet run it.)

## Architecture

`docs/architecture.md` is the long form; `docs/parity.md` records what numax
absorbs from NumPy/SciPy, what it routes to MAX, and what it leaves out.
`docs/features.md` is the full public inventory.

Two co-equal axes:

1. **Composable type layer.** One kernel written against the `FloatLike` trait
   (`numax/core/numeric.mojo`) means several things depending on the conformer
   it is instantiated with: `Plain` (value), `Dual` (forward-mode derivative),
   `Gradient` (full gradient), `Compensated` (double-double precision),
   `Decimal` (exact base-10 fixed point), `Complex`, `Interval`. They nest —
   `Gradient[Dual[Plain]]` is a Hessian, `Complex[Dual[Plain]]` differentiates
   holomorphically — so no kernel needs a per-type copy.
2. **NumPy/SciPy parity, MAX-first.** MAX's `TileTensor`, the top-level
   `linalg`/`nn` roots and `max.algorithm` are the substrate (there is no
   `max.linalg` and no `max.random`; RNG is `std.random`). Prefer calling MAX over
   writing a replacement; `docs/parity.md` has the survey the dispositions rest
   on.

Adding a `FloatLike` method is expensive — every present and future conformer
must implement it. The trait grew only when call sites multiplied *and* the
workaround was numerically wrong, not merely verbose.

### The fixed-iteration invariant and the two tiers

**Tier 1** (`FloatLike`-generic): fixed iteration count, no per-lane
branching. A `Self` may hold a SIMD vector whose lanes disagree about which
branch they want, and `FloatLike` has no `select`, so a fixed amount of uniform
work is done and per-lane selection is an arithmetic `0`/`1` blend built from
`copysign` (`max_of`/`min_of`/`ge_indicator`/`blend` in
`numax/core/numeric.mojo`). Both sides of every blend are always evaluated, so
both must be safe across the whole domain — clamp before `ln`. This is what
makes every tier-1 kernel GPU-launchable unmodified.

**Tier 2** (`Plain`-only, host): may loop to a tolerance or branch on data. No
GPU guarantee.

Three rules on the boundary: the tier is **declared** in the module docstring
and the function's own docstring, so a reader at a call site never audits a
body; **tier 1 never calls tier 2**; and where both make sense, both ship,
cross-referenced (a fixed-iteration `newton` and a converge-to-tolerance one
are siblings, not replacements). Adaptive-tolerance tier-1 variants are out of
scope, not missing. Every approximation documents its error bound, checked by
`pixi run accuracy`.

### Tensor layer

`numax/core/tensor.mojo` drives a kernel across MAX's `TileTensor` — the same
type for CPU- and GPU-resident data — with one `gpu: Bool` compile-time
parameter rather than two functions. Primitives: `map` (unary/binary),
`reduce`, `reduce_block_gpu`, `reduce_rows`, `broadcast_op_rows`,
`reduce_axis`, `broadcast_op_axis`, and `map_threaded` (CPU cores via
`max.algorithm.elementwise`). `step`/`combine` are non-capturing functions
passed as compile-time parameters — required by
`DeviceContext.enqueue_function`, and kept the same shape on CPU so `map` has
one signature.

`map` and `reduce` each have two overloads under the same name, selected by
`where` clauses that are exact negations: a **static** one (comptime
`row_major[n]()`, flattens via `coalesce()`, GPU-capable) and a **runtime** one
(`row_major(Coord(n))`, flattens by constructing a rank-1 layout over the same
pointer, no GPU). There is deliberately **no second tensor type** — the
comptime/runtime distinction lives in the layout, where MAX put it.

`numax/core/array.mojo`'s `Tensor` owns a MAX `DeviceBuffer`, so the
`DeviceContext` passed to a factory (last argument, optional) decides host or
device memory; `.view()` yields the `TileTensor`. Bulk element access goes
through `to_host`/`copy_from_host` — on CUDA `unsafe_ptr()` returns a device
pointer and a host read segfaults. `to_array`/`to_tensor` is the seam to the
`Array[T, n]` half of the library.

### Package layout and dependency direction

Subpackages mirror NumPy/SciPy names: `core`, `linalg`, `optimize`,
`integrate`, `interpolate`, `special`, `stats`, `fft`, `signal`, `io`.
`core` depends on nothing else in numax; every other subpackage depends on
`core`. The few cross-subpackage edges are deliberate: `stats` → `special`
(incomplete gamma/beta), `integrate` → `special` (Legendre roots) and
`optimize` (Newton), `interpolate` → `linalg` (tridiagonal solve).

Each subpackage re-exports its public surface and `numax/__init__.mojo`
re-exports all of them, so `from numax import ...` is flat and
`from numax.linalg import ...` is one subsystem. A new public name goes in
three places: its module, its subpackage `__init__.mojo`, and (if it belongs to
the common surface) `numax/prelude.mojo`.

**The root docstring and the README say the same thing.** `numax/__init__.mojo`
opens with the library's pitch — what numax is, the two axes, the subpackage
table — and the README says it again for a reader who never opens the source.
They are one claim in two places, so a change to either is incomplete until the
other matches: same framing, same emphasis, same list of subsystems. Reword the
README's positioning and the package docstring moves in the same commit. The
prose need not be identical (the README carries badges and install steps the
docstring has no use for), but a reader must not be able to find a capability,
a limit, or a headline claim in one and not the other. `recipe.yaml`'s `summary`
and `description` are the third copy, shipped as conda metadata — check them
whenever the pitch shifts.

**Name-collision policy.** `numax.prelude` deliberately omits every name that
shadows a Mojo builtin — `sum`, `prod`, `min`, `max`, `abs`, `all`, `any`,
`round`, `copysign` — because a module-level definition *replaces* the builtin
for the rest of the importing file rather than overloading it, so a star import
would silently break `min(1, 2)` in the caller's own code. Those stay one
qualified import away. The nine `scipy.stats` distribution namespaces are out
of the prelude too (`gamma`/`beta` would collide with the special functions).
Keep this property when adding names.

## Mojo 1.0 constraints

- `fn` is a hard parse error. Everything is `def`.
- Kernel scalar parameters must be fixed-width (`Int32`, not `Int`) to satisfy
  `DevicePassable`.
- Generic SIMD-width parameters want `W: SIMDLength`, not `W: Int`.
- `DeviceBuffer`/`HostBuffer` parameters need an explicit `mut`.
- Conformers are built entirely from `SIMD` fields with no pointers or
  allocations, which is what lets all of them run inside a GPU kernel body.
  Keep it that way (`Compensated`'s `exp()` coefficients had to move from a
  runtime `float64` table to comptime `dtype`-native constants for this).
- `comptime` aliases are the idiom for pinning long generic spellings in tests
  and examples.

## Git flow

**Atomic commit.** A commit is atomic: the code change, its tests, and
the docstring/doc updates that describe it, and nothing else. Unrelated
cleanups found along the way are their own commit.

**Trailer.** Exactly one, and only the Claude agent as co-author — the user is
the author, not a co-author, and there is no "Generated with" line:

```
Co-Authored-By: Claude <noreply@anthropic.com>
```

**Message shape** (follow `git log`): subject is an imperative sentence in
plain prose — no Conventional Commits prefix, no scope, no trailing period.
The body states the problem first (what the old shape forced a caller to do),
then the change, then what deliberately stays; a shortcut with a known ceiling
names the ceiling and the upgrade path, as the commit body does, not only the
code comment.

**Per-feature gate, before the commit exists:**

1. Adequate tests for the new behavior — they assert the specific claim the
   commit message makes, not just that the code runs. Where a convenience
   wraps an existing primitive, a test asserts the two spellings agree.
2. No regression: full `pixi run tests` green (new suite wired into the
   aggregate), `pixi run examples-cpu`, `pixi run accuracy`, `mojo doc -I .`
   and `pixi run -e dev format-check` clean. A feature is not done until the
   whole gate passes, not just its own suite.

## Performance and accuracy claims

CPU and GPU numbers are never mixed into one comparison; every row in
`docs/performance.md` and the README compares implementations on the same
processor. If you change a number, re-measure — don't extrapolate.
