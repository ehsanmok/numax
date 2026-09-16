# Mojo language feedback

Defects, gaps and dead ends in the **Mojo language and standard library**,
found while building [numax](https://github.com/ehsanmok/numax), a
numerical computing library. Companion document:
[`max-feedback.md`](max-feedback.md) for the MAX kernel library.

Each entry stands alone: a repro that does not import numax, the
diagnostic as the compiler printed it, what the behavior should be, and
what the gap cost. Every repro is a file in this directory, runnable with
the `pixi run repro` command shown beside it. See
[`README.md`](README.md) for setup, the severity legend and the full
index.

**Pin: Mojo 1.0.0 (ed45d567), MAX 26.5.** Every diagnostic below was
produced by that toolchain on an Apple M3 Pro. Re-run before citing any
of it against a newer release -- one earlier claim had already gone stale
by the time it was checked.

---

## 1.1 Extensions cannot carry a `where` clause, so a stdlib type cannot conditionally conform to a user trait

**Severity: blocked.** This is the single highest-value item in this file.

Mojo 1.0.0 already supports retroactive conformance under an orphan rule,
which is a genuinely useful feature and better than the manual admits (see
1.2). What it cannot do is make that conformance *conditional*, and for
numeric code that is the whole ballgame: a trait with `exp`, `ln` and
`sqrt` on it is meaningful for floating-point `SIMD` and meaningless for
integer `SIMD`, and there is currently no way to say so.

The result is that numax carries a wrapper struct whose entire purpose is
to be a type it owns, so that it can attach a conditional conformance the
extension mechanism will not accept.

### What already works

`pixi run repro 1.1a-extension-works.mojo`

```mojo
trait Tiny(Copyable, Movable):
    def twice(self) -> Self:
        ...


__extension SIMD(Tiny):
    def twice(self) -> Self:
        return self + self


def use[T: Tiny](x: T) -> T:
    return x.twice()


def main():
    print(use(SIMD[DType.float32, 4](1.5)))
```

Observed, and correct:

```
[3.0, 3.0, 3.0, 3.0]
```

A stdlib type conforming to a trait declared in the caller's own module,
at the pinned release. Good.

### Attempt 1 -- put the constraint on the conformance

`pixi run repro 1.1b-conformance-where-rejected.mojo`

```mojo
trait Tiny(Copyable, Movable):
    def twice(self) -> Self:
        ...


__extension SIMD(Tiny where dtype.is_floating_point()):
    def twice(self) -> Self:
        return self + self
```

Observed:

```
error: 'where' clauses in conformance lists are only supported on structs
__extension SIMD(Tiny where dtype.is_floating_point()):
                      ^
```

### Attempt 2 -- drop the constraint, let the body need it

`pixi run repro 1.1c-no-where-lacks-evidence.mojo`

```mojo
from std.math import exp as std_exp


trait HasExp(Copyable, Movable):
    def exp(self) -> Self:
        ...


__extension SIMD(HasExp):
    def exp(self) -> Self:
        return std_exp(self)
```

Observed:

```
error: invalid call to 'std_exp': lacking evidence to prove correctness
        return std_exp(self)
               ^~~~~~~
note: cannot prove constraint for candidate
def exp[dtype: DType, width: SIMDLength, //](x: SIMD[dtype, width]) -> SIMD[dtype, width] where dtype.is_floating_point()
note: constraint declared here needs evidence for 'dtype.is_floating_point()'
```

So the error does **not** defer to instantiation. An unconditional
extension is rejected at the conformance, because `std.math.exp` itself is
correctly constrained to floating point and the extension has no way to
supply the evidence.

### Attempt 3 -- put the constraint on the method, the way a struct can

`pixi run repro 1.1d-method-where-unprovable.mojo`

```mojo
__extension SIMD(HasExp):
    def exp(self) -> Self where dtype.is_floating_point():
        return std_exp(self)
```

Observed -- and this diagnostic names the missing feature outright:

```
error: 'SIMD[dtype, length]' does not implement all requirements for 'HasExp'
note: method 'exp' has constraints that cannot be proven or disproven from conformance constraint
    def exp(self) -> Self where dtype.is_floating_point():
        ^
note: required by trait method here
    def exp(self) -> Self:
        ^
```

"cannot be proven or disproven **from conformance constraint**" is the
compiler asking for the clause that attempt 1 rejected.

### Attempt 4 -- restrict the extension by naming parameters

`pixi run repro 1.1e-extension-params-rejected.mojo`

```mojo
__extension SIMD[dtype: DType, length: SIMDLength](HasExp):
    def exp(self) -> Self:
        return self
```

Observed:

```
error: cannot specify parameter declarations on extensions
__extension SIMD[dtype: DType, length: SIMDLength](HasExp):
                 ^
note: extension already assumes these parameter declarations
```

**Expected.** One of these should work. The natural spelling is attempt 1:
allow a `where` clause in an extension's conformance list, exactly as
structs have had since v0.26.2. Attempt 3 working instead would be equally
sufficient.

To be clear that the method *bodies* are not the problem, a
`FloatLike`-shaped static method with a `Float64` argument -- the trait's
`constant` -- conforms and runs through an extension without complaint:
`pixi run repro 1.1f-constant-body-works.mojo`. Only the members needing
floating-point evidence are unreachable, and only because the evidence
cannot be stated.

**Impact.** numax defines its central trait, `FloatLike`, and every
numeric kernel in the library is written against it once rather than per
dtype. Because `SIMD` cannot conditionally conform, the library ships a
wrapper:

```mojo
struct Plain[dtype: DType, width: Int](
    Copyable, FloatLike where dtype.is_floating_point(), Movable, Writable
):
    var v: SIMD[dtype, width]
```

That struct is the conditional conformance the extension cannot express,
and nothing else. Its cost is paid by users, not by the library: every
value entering a generic kernel is wrapped, and every value leaving one is
unwrapped through `.v`. A user writing a Gaussian gets
`g(Plain[f32](0.5)).v` where they wanted `g(SIMD[f32, 1](0.5))`. The
wrapper is named in the README, the package docstring and 23 of the
library's 26 example programs, because it cannot be hidden.

**Workaround.** The wrapper struct, indefinitely. There is no partial
version of this: the whole trait is unusable on `SIMD` until one of the
four attempts above compiles.

## 1.2 The manual says retroactive conformance is impossible; it shipped

**Severity: friction (documentation).**

`Mojo/docs/site/manual/traits.mdx`, in "Things to know":

> **You can't add traits to existing types.** Conformance is declared
> where a type is defined. You can't retroactively make `Float64`, `Int`,
> or any other type you don't own conform to a new trait.

`Mojo/docs/site/roadmap.mdx` lists struct extensions as not started:

> - (unchecked) **Struct extensions**: Post-hoc type extension and better
>   modular refactoring.

And `Mojo/proposals/struct-extensions.md` is marked `Status: Draft`.

All three are stale with respect to a feature that works at 1.0.0, as 1.1
above demonstrates, subject to an orphan rule (the extension must live in
the struct's file or the trait's file) that is implemented but documented
nowhere a user would look.

**Impact.** numax's `Plain` wrapper exists partly because its author read
the manual and believed it. The workaround was adopted, documented as a
language limitation in the library's own architecture notes, and built
upon for a full release cycle before a source dive found `__extension`.
Every downstream library that needs a trait over a stdlib type is
currently paying that same cost for a feature it has.

**Expected.** Document `__extension` and the orphan rule, or say in the
manual that an undocumented experimental form exists. If the double
underscore means "do not rely on this yet", that is fine and worth saying
outright -- the current text says something stronger and different, which
is that the capability does not exist.

## 1.3 `enqueue_memset` on a `DType.bool` buffer fails with no diagnostic

**Severity: crash (compiler).**

`pixi run repro 1.3-bool-memset-pass-manager.mojo`

```mojo
from max.gpu.host import DeviceContext


def main() raises:
    var ctx = DeviceContext(api="cpu")
    var buf = ctx.enqueue_create_buffer[DType.bool](16)
    ctx.enqueue_memset(buf, Scalar[DType.bool](0))
    ctx.synchronize()
    print("filled 16 bools")
```

Observed -- the entire output, with no source location, no note, and no
indication of which construct is at fault:

```
mojo: error: failed to run the pass manager
```

Substituting `DType.float32` for `DType.bool`, changing nothing else,
compiles and runs.

**Expected.** Either a boolean memset, or a diagnostic pointing at the
line with the reason.

**Impact.** Boolean tensors are the output of every comparison, so this
sits under a whole family of NumPy-shaped operations. It also resists the
obvious guard: a `comptime if` around the memset does not help, because
both branches are still type-checked and `Scalar[DType.bool](0)` is
rejected on its own. numax's tensor type therefore cannot offer its
cheap zero-filling constructor at `DType.bool` at all, and a boolean mask
has to be built through a host `List` or left uninitialized and written
by a subsequent elementwise pass. A separate place in the library passes
an `int64` flag tensor to `linalg.matrix_band_part` where `bool` would be
the natural dtype, purely to avoid this.

**Workaround.** Never memset a bool buffer. Allocate uninitialized and
write every element from a kernel.

## 1.4 The `where` prover cannot evaluate `%`

**Severity: blocked.**

`pixi run repro 1.4-where-prover-modulo.mojo`

```mojo
def halve[n: Int]() -> Int where n % 2 == 0:
    return n // 2


def main():
    print(halve[8]())
```

Observed:

```
error: invalid call to 'halve': lacking evidence to prove correctness
    print(halve[8]())
          ^~~~~
note: constraint declared here needs evidence for '((Int(8) % Int(2)) == Int(0))'
note: cannot evaluate call to non-builtin function declared here
def __mod__(self, rhs: Self) -> Self
^
```

`8 % 2 == 0` is not merely unproven, it is unevaluable: `Int.__mod__` is
an ordinary function and the prover will not run it, even on two literals.

**Expected.** Constant-fold `%` on integer literals, as `//`, `*` and `+`
already are.

**Impact.** "This algorithm needs an even length" and "this radix needs a
power of two" are ordinary preconditions in numerical code and cannot be
stated. numax restructures parameters to avoid needing them, and where
that is impossible falls back to `comptime assert`, which reports at
instantiation rather than at the call and so names the wrong line for
the user.

## 1.5 A move-only type cannot be returned as a tuple and destructured, and the suggested fix does not work

**Severity: blocked.**

`pixi run repro 1.5a-tuple-destructure.mojo`

```mojo
@fieldwise_init
struct Owned(Movable):
    var tag: Int


def pair() -> Tuple[Owned, Owned]:
    return (Owned(1), Owned(2))


def main():
    var q, r = pair()
    print(q.tag, r.tag)
```

Observed:

```
error: value of type 'Owned' cannot be implicitly copied, it does not conform to 'ImplicitlyCopyable'
    var q, r = pair()
         ^
note: consider transferring the value with '^'
```

Following the compiler's own suggestion
(`pixi run repro 1.5b-tuple-transfer-suggestion.mojo`):

```mojo
    var got = pair()
    var q = got[0]^
    var r = got[1]^
```

Observed:

```
error: expression does not designate a value with an origin
    var q = got[0]^
                  ^
```

**Expected.** Either destructuring that moves each element out, or a
suggestion that can actually be followed.

**Impact.** This decides API shape, not just ergonomics. A QR
factorization returns two matrices; the spelling every NumPy and SciPy
user reaches for is `Q, R = qr(a)`. Matrices own device memory, so they
are `Movable` and not `ImplicitlyCopyable`, and the above is why numax
returns a named result object with `.q()` and `.r()` methods instead.
The same reasoning changed the return type of the LU factorization, the
FFT (a complex spectrum is two real tensors and is never unpacked), and
the ODE solvers. Each of those is a permanent divergence from the API
users expect, caused by a language limitation rather than a design view.

## 1.6 `a > b` on a SIMD vector is a compile error

**Severity: friction.**

`pixi run repro 1.6-simd-strict-inequality.mojo`

```mojo
def main():
    var a = SIMD[DType.float32, 4](1.0, 2.0, 3.0, 4.0)
    var b = SIMD[DType.float32, 4](2.0, 2.0, 2.0, 2.0)
    print(a > b)
```

Observed:

```
note: constraint failed: Strict inequality is only defined for `Scalar`s; did you mean to use `SIMD.gt(...)`?
mojo: error: failed to run the pass manager
```

The message is clear and the suggestion is right, so this is the mildest
entry here. It is included because the failure mode is a `constraint
failed` surfacing through `failed to run the pass manager` rather than an
ordinary type error at the operator, and because the asymmetry with
`__eq__` (which returns a single `Bool` by splatting) is a trap for
anyone porting vectorized code.

**Impact.** Every comparison in numax is spelled `a.gt(b)`. Low cost,
but it is the first thing a NumPy user writes and it does not work.

## 1.7 `std.math`'s `exp`, `log` and `erf` are not correctly rounded at float64

**Severity: wrong.**

`pixi run repro 1.7-stdmath-float64-accuracy.mojo`

```mojo
from std.math import exp, log, erf, erfc, sin, sqrt


def main():
    print("exp(1.0)   =", exp(Float64(1.0)), " true 2.718281828459045")
    print("log(0.02)  =", log(Float64(0.02)), " true -3.912023005428146")
    print("erf(0.5)   =", erf(Float64(0.5)), " true 0.5204998778130465")
    print("sin(1.0)   =", sin(Float64(1.0)), " true 0.8414709848078965")
    print("sqrt(2.0)  =", sqrt(Float64(2.0)), " true 1.4142135623730951")
    print("erfc(0.5)  =", erfc(Float64(0.5)), " true 0.4795001221869535")
```

Observed:

```
exp(1.0)   = 2.718281828459813  true 2.718281828459045
log(0.02)  = -3.9120230055139458  true -3.912023005428146
erf(0.5)   = 0.5204998764899175  true 0.5204998778130465
sin(1.0)   = 0.8414709848078965  true 0.8414709848078965
sqrt(2.0)  = 1.4142135623730951  true 1.4142135623730951
erfc(0.5)  = 0.4795001221869535  true 0.4795001221869535
```

`exp` diverges at the 13th significant digit, `log` at the 11th, `erf` at
the 9th. `sin`, `sqrt` and `erfc` are exact on these inputs, so this is
specific to three functions rather than a general float64 weakness.

Measured over grids against mpmath at 50 digits, the worst cases are
`exp` at about 105,000 ulp, `log` at about 9,300,000 ulp (an absolute
floor near 2e-10) and `erf` at about 196,000,000 ulp, while `erfc`,
`sin`, `cos` and `sqrt` stay within 3 ulp.

**Root cause, from the stdlib source at the pinned release.** `exp`
performs its range reduction with `ln 2` rounded to float32
(`0.69314718055966295651160180568695068359375`), so the error grows with
the reduction multiple; `erf` uses nine-significant-digit polynomial
coefficients, which cannot support a float64 result.

**Expected.** Sub-ulp accuracy at float64, or a documented accuracy bound
so a caller knows to bring their own.

**Impact.** This was the accuracy floor of an entire library. Every
special function numax builds on `exp`, `log` or `erf` inherited it:
`lgamma` at 3.4e-9, the incomplete gamma and beta at 1e-8, `exp1` and the
cosine integral at 3e-9, `erfinv` at 6e-7, the real-order Bessel family
at 1.5e-10. The float32 versions are adequate for float32; this is a
float64 problem only.

**Workaround.** numax transcribed fdlibm's `e_exp.c`, `e_log.c` and
`s_erf.c` into SIMD form with masks, validated against mpmath at 30,000
random points each, and routes float64 through those instead of the
stdlib. That fixed every row at once: `erf` went to 2.1e-16, `lgamma` to
2.8e-14, the incomplete functions to 1e-15, `erfinv` to 1.4e-15. It is
193 lines that should not need to exist in a downstream library.

## 1.8 An unqualified call inside an extension method resolves to the method, then hangs the compiler

**Severity: crash (compiler hang).**

`pixi run repro 1.8-extension-self-recursion-hang.mojo` -- this one never
returns. Interrupt it; the hang is the result.

```mojo
from std.math import exp


trait HasExp(Copyable, Movable):
    def exp(self) -> Self:
        ...


__extension SIMD(HasExp):
    def exp(self) -> Self:
        return exp(self)


def main():
    print(SIMD[DType.float32, 4](1.0).exp())
```

Observed: one warning, then no further output and no termination. Killed
after seven minutes.

```
warning: self recursive call will cause an infinite loop
        return exp(self)
                  ^
```

Inside the extension method body, the unqualified name `exp` resolves to
the method currently being defined rather than to the `std.math.exp`
imported at module scope, so the body calls itself. The compiler
diagnoses this correctly as an infinite loop -- and then, having done so,
proceeds rather than stopping.

**Expected.** A warning that names an unconditional infinite loop should
be an error. At minimum the compilation should terminate.

**Impact.** Cost about eight minutes of wall time and looked like a
compiler bug in the extension feature itself rather than a name
resolution result, which sent the investigation in the wrong direction.
Note that this is a genuine hazard specific to extensions: the natural
way to implement a trait method on a foreign type is to forward to the
free function of the same name, and that is exactly the shape that
silently recurses.

**Workaround.** Alias the import (`from std.math import exp as std_exp`).

---

# What these gaps prevent

Two capabilities a numerical library is expected to have are currently
inexpressible in Mojo. Both are listed here rather than as numbered
entries above, because neither is a defect with a repro -- each is a
missing feature whose absence propagates into a public API.

(Sparse matrices and Krylov solvers are the same kind of item on the
kernel side; they are `max-feedback.md` 2.8 and 2.9.)

## Run-time-shape decompositions

Worth stating precisely, because the intuitive culprit is innocent.
**MAX's `matmul` handles run-time extents perfectly well**: it reads
`c.dim[0]()` into a `GemmShape` at the call, its naive GPU kernel takes
`m_dev`/`n_dev`/`k_dev` as `Int32` arguments over `UNKNOWN_VALUE` layouts,
and the vendor path builds `row_major(Coord(...))` from live dimensions.
Most of `nn` is the same, and `elementwise` takes a run-time `Coord`.
Only the tuned grouped-matmul fast paths gate on
`static_shape != UNKNOWN_VALUE`.

The blocker is on the language side, and it is two things:

- **A kernel launched through `enqueue_function` needs its shape in the
  type.** A shape the compiler cannot see cannot become a kernel
  signature, so the panel kernels inside a blocked factorization are
  reachable only at a compile-time layout.
- **There is no run-time-sized inline array.** `Array[T, length: Int]`
  requires a constant expression, so the register-resident seam a
  factorization crosses is comptime-sized by construction.

Together those are why numax's factorization objects carry their dimension
in the type. This is not asked for as a feature request -- dependent types
are a large ask -- but it is the reason, and it is not MAX's.

## Reverse-mode autodiff

A tape needs three things Mojo does not have, and the first is the one
that matters:

- **No existentials or dynamic dispatch.** The roadmap lists
  "Existentials / dynamic traits" as not started, and type-valued
  variables were explicitly rolled back in v0.24.2 to be redesigned.
  `T: Trait` and `Some[Trait]` are compile-time monomorphization, not
  runtime vtables, so a heterogeneous list of backward operations has no
  representation. `Variant` is the nearest thing and its type list is
  closed at compile time.
- **No escaping or heap-allocated closures.** "A closure can't outlive
  the scope where it's declared", and there is no built-in mechanism for
  heap-allocated or existential closures -- so the chain of partial
  applications a tape records cannot be stored.
- **A tape could never be GPU-resident**, even solved on the host:
  `List` is not `DevicePassable`, and `enqueue_function` needs
  non-capturing compile-time kernel identity. That matters here because
  forward-mode works on the device precisely by being pointer-free, and
  numax's whole claim is that its conformers run inside a kernel.

So forward mode is not a preference. `Gradient` costs one pass per
variable and was measured against a tape at a crossover near sixteen
variables, and the better answer past that point is currently
inexpressible rather than unwritten.

---

# What would help most

Ranked by what it unlocks for a downstream numerical library, not by
implementation cost. MAX-side asks are ranked separately in
[`max-feedback.md`](max-feedback.md).

1. **Conditional conformance on extensions** (1.1). Deletes a wrapper
   struct from numax's public API and from every user's first program. It
   is the difference between a trait-generic numeric library being
   pleasant and being apologetic, and nothing else in either document
   changes a user-facing API this much.
2. **Correctly rounded float64 `exp`, `log` and `erf`** (1.7). Three
   functions that set the accuracy floor for everything built on them. A
   downstream library should not be shipping fdlibm.
3. **Diagnose instead of dying.** The bool memset (1.3) produces
   `failed to run the pass manager` with no source location; the
   extension recursion (1.8) warns that it will loop forever and then
   hangs rather than stopping. Both are cheap to turn into errors.
4. **Fold `%` in the `where` prover** (1.4). "This length must be even"
   and "this radix must be a power of two" are ordinary preconditions in
   numerical code and currently cannot be stated.
5. **Destructure move-only tuples** (1.5), or stop suggesting a fix that
   does not work. This one decides API shape: it is why a QR
   factorization returns an object with `.q()` and `.r()` rather than
   `Q, R = qr(a)`.
6. **Correct the traits manual** (1.2). Pure documentation, and it cost
   this project a release cycle of believing a shipped feature did not
   exist.

## Filing notes

Everything here reproduces on any machine with the pinned toolchain --
none of these entries needs a GPU or a particular architecture, unlike
two of the MAX ones.

One claim was **withdrawn** during verification rather than filed, and is
recorded so nobody re-files it: `SIMD.ne` was believed to be ordered,
making `nan != nan` false where NumPy gives true. At Mojo 1.0.0 it
returns `[True, True]`, matching NumPy
(`pixi run repro disproved-simd-ne-nan.mojo`). The note was either wrong
or has since been fixed.
