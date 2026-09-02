"""Gap-only NumPy-named creation and manipulation surface over `TileTensor`.

`docs/parity.md` picks array creation/manipulation as a genuine
`numax` gap: MAX's `layout` package ships `TileTensor` itself (slicing,
tiling, reshaping, coalescing) but no NumPy-named *factory* functions --
`TileTensor.zeros`/`.ones`/`.full`/`.arange` all fail to resolve (verified
directly against `~/workspace/modular/max/kernels/src/layout/tile_tensor.mojo`),
and `max.algorithm.functional` ships only `elementwise`. This module adds
**only** those factory/manipulation names, comptime-shape and CPU-only, as
a thin layer over `TileTensor` -- not a competing array type (the Track E
rule: "any array/buffer-level work in numax builds as a thin layer over
TileTensor, not a bespoke array type").

**Why a `Tensor` wrapper, not a bare `TileTensor`, is what these functions
return.** `TileTensor` is a *view*: a pointer plus a layout, not the memory
itself (confirmed directly -- a function that builds a local
`List[Scalar[dtype]]`, wraps it in a `TileTensor`, and returns the
`TileTensor` alone produces a dangling pointer the instant the function
returns, since the `List`'s backing heap buffer is freed with it; a
stress test that allocates 2000 more lists between the call and first read
corrupted 100% of the "returned" tensor's elements). `Tensor[dtype, *dims]`
owns the backing `List` alongside a compile-time row-major layout, so the
value `zeros[dtype, 4, 4]()` returns can safely outlive the call that built
it. Call `.view()` on a live `Tensor` to get the `TileTensor` that
`numax.tensor.map`/`reduce` and friends expect; the view is only valid as
long as the owning `Tensor` is.

**Comptime shape only.** `row_major[*dims: Int]()` (compile-time variadic)
is what satisfies `numax.tensor`'s `where all_dims_known and is_row_major`
clause; the runtime-shape sibling `row_major(Coord)` produces
`all_dims_known=False` and fails that same clause. So every function here
takes a compile-time `*dims: Int` shape, matching `numax.tensor`'s existing
contract exactly. Dynamic-shape creation is out of scope.

**What MAX's own `nn` versions are, and why these are written here.**
`nn` does ship `arange`, `reshape`, `concat`, `split`, `slice`, `tile`,
`broadcast`, `cumsum`, `argsort` and `argmax`/`argmin` over `TileTensor`,
but they are graph-operator kernels, not array functions: `nn.arange`
returns one SIMD vector for a given index rather than filling a tensor,
`nn.concat` wants a pre-sized output tensor plus a `DeviceContext`, and
`nn.reshape` returns a *dynamically*-laid-out `TileTensor` that fails
`numax.tensor`'s `all_dims_known` clause. So the four below are written
against this module's own comptime-shaped `Tensor` instead. The ones still
not wrapped -- `slice`, `tile`, `broadcast`, `cumsum`, `argsort` -- are
either genuinely better reached through `nn`/`numax.statistics` or wait on
the runtime-shape array; `numax.statistics.argmax`/`argmin` already route
into `nn.argmaxmin`.

**Manipulation scope, stated plainly.** `TileTensor.transpose()` already
gives a zero-copy *view* with every axis reversed (not merely the last two);
`transpose` here is a genuinely different thing -- an owned-copy 2D matrix
transpose, useful when the result needs to outlive the source or be handed
to something that wants its own storage. `squeeze` covers the two concrete
directions a matrix collapses to a vector (`(1, n) -> (n,)` and
`(n, 1) -> (n,)`); a fully general N-dimensional squeeze would need to
build a new variadic shape parameter pack from an arbitrary subset of an
existing one, which Mojo's parameter-pack machinery doesn't expose a public
way to do. `stack` covers exactly two same-shaped tensors along a new
leading axis (`axis=0`); `axis=1` stacking is not provided -- both are
real, documented scope limits, not oversights.
"""

from layout import TileTensor
from layout.tile_layout import row_major, TensorLayout


def _product[*dims: Int]() -> Int:
    """The element count of a compile-time shape -- the product of `dims`."""
    var p = 1
    comptime for i in range(dims.__len__()):
        p *= dims[i]
    return p


struct Tensor[dtype: DType, *dims: Int](Movable):
    """An owned buffer paired with a compile-time row-major layout.

    `numax.array`'s creation routines return this, not a bare `TileTensor`
    -- see this module's own docstring for why a `TileTensor` alone would
    dangle. `dims` is the shape, exactly as passed to the factory function
    that built this (`Tensor[dtype, 4, 4]` for a 4x4 matrix); `rank` and
    `num_elements` are derived from it at compile time.
    """

    comptime layout = row_major[*Self.dims]()
    comptime LayoutType = type_of(Self.layout)
    comptime rank = Self.dims.__len__()
    comptime num_elements = _product[*Self.dims]()

    var storage: List[Scalar[Self.dtype]]

    def __init__(out self, var storage: List[Scalar[Self.dtype]]):
        """Wrap an existing, already-correctly-sized buffer.

        `storage` must have exactly `Self.num_elements` entries in
        row-major order; this is the escape hatch every factory function
        below funnels through.
        """
        self.storage = storage^

    @staticmethod
    def dim[i: Int]() -> Int:
        """The compile-time extent of axis `i`."""
        return Self.dims[i]

    def view(mut self) -> TileTensor[Self.dtype, Self.LayoutType, MutAnyOrigin]:
        """A `TileTensor` view over this tensor's storage.

        Valid only as long as `self` is alive -- pass `self` (or a `mut`
        reference to it) around, not just the value returned here, if the
        view needs to outlive this call.
        """
        var v: TileTensor[
            Self.dtype, Self.LayoutType, MutAnyOrigin
        ] = TileTensor(self.storage, Self.layout)
        return v

    def __getitem__(self, i: Int) -> Scalar[Self.dtype]:
        """Flat (row-major) element access."""
        return self.storage[i]

    def __setitem__(mut self, i: Int, value: Scalar[Self.dtype]):
        """Flat (row-major) element assignment."""
        self.storage[i] = value


def zeros[dtype: DType, *dims: Int]() -> Tensor[dtype, *dims]:
    """A new tensor of the given compile-time shape, filled with `0`."""
    var storage = List[Scalar[dtype]](length=_product[*dims](), fill=0)
    return Tensor[dtype, *dims](storage^)


def ones[dtype: DType, *dims: Int]() -> Tensor[dtype, *dims]:
    """A new tensor of the given compile-time shape, filled with `1`."""
    var storage = List[Scalar[dtype]](length=_product[*dims](), fill=1)
    return Tensor[dtype, *dims](storage^)


def full[
    dtype: DType, *dims: Int
](fill_value: Scalar[dtype]) -> Tensor[dtype, *dims]:
    """A new tensor of the given compile-time shape, filled with `fill_value`.
    """
    var storage = List[Scalar[dtype]](length=_product[*dims](), fill=fill_value)
    return Tensor[dtype, *dims](storage^)


def empty[dtype: DType, *dims: Int]() -> Tensor[dtype, *dims]:
    """A new tensor of the given compile-time shape, its contents unspecified.

    Unlike NumPy's `empty`, this zero-initializes rather than truly leaving
    the memory uninitialized -- `numax` favors a memory-safe default over
    the small allocation-time saving, matching every other factory function
    here. Callers that write every element before reading (the usual reason
    to reach for `empty` at all) pay nothing extra in practice.
    """
    var storage = List[Scalar[dtype]](length=_product[*dims](), fill=0)
    return Tensor[dtype, *dims](storage^)


def eye[dtype: DType, n: Int]() -> Tensor[dtype, n, n]:
    """The `n`x`n` identity matrix."""
    var storage = List[Scalar[dtype]](length=n * n, fill=0)
    for i in range(n):
        storage[i * n + i] = 1
    return Tensor[dtype, n, n](storage^)


def linspace[
    dtype: DType, num: Int
](start: Scalar[dtype], stop: Scalar[dtype]) -> Tensor[dtype, num]:
    """`num` evenly spaced values from `start` to `stop`, inclusive of both.

    Matches `numpy.linspace`'s default `endpoint=True`. `num == 1` returns
    `[start]`, avoiding a division by zero in the step computation.
    """
    var storage = List[Scalar[dtype]](capacity=num)
    comptime if num == 1:
        storage.append(start)
    else:
        var step = (stop - start) / Scalar[dtype](num - 1)
        for i in range(num):
            storage.append(start + Scalar[dtype](i) * step)
    return Tensor[dtype, num](storage^)


def logspace[
    dtype: DType, num: Int
](
    start: Scalar[dtype], stop: Scalar[dtype], base: Scalar[dtype] = 10
) -> Tensor[dtype, num]:
    """`num` values evenly spaced on a log scale: `base**x` for `x` in
    `linspace(start, stop, num)`. Matches `numpy.logspace`'s defaults."""
    var storage = List[Scalar[dtype]](capacity=num)
    comptime if num == 1:
        storage.append(base**start)
    else:
        var step = (stop - start) / Scalar[dtype](num - 1)
        for i in range(num):
            storage.append(base ** (start + Scalar[dtype](i) * step))
    return Tensor[dtype, num](storage^)


def zeros_like[
    dtype: DType, *dims: Int
](a: Tensor[dtype, *dims]) -> Tensor[dtype, *dims]:
    """A new zero-filled tensor with `a`'s dtype and shape."""
    return zeros[dtype, *dims]()


def ones_like[
    dtype: DType, *dims: Int
](a: Tensor[dtype, *dims]) -> Tensor[dtype, *dims]:
    """A new one-filled tensor with `a`'s dtype and shape."""
    return ones[dtype, *dims]()


def full_like[
    dtype: DType, *dims: Int
](a: Tensor[dtype, *dims], fill_value: Scalar[dtype]) -> Tensor[dtype, *dims]:
    """A new `fill_value`-filled tensor with `a`'s dtype and shape."""
    return full[dtype, *dims](fill_value)


def empty_like[
    dtype: DType, *dims: Int
](a: Tensor[dtype, *dims]) -> Tensor[dtype, *dims]:
    """A new tensor with `a`'s dtype and shape; see `empty`'s own docstring
    for why this zero-initializes rather than leaving memory uninitialized.
    """
    return empty[dtype, *dims]()


def transpose[
    dtype: DType, rows: Int, cols: Int
](mut a: Tensor[dtype, rows, cols]) -> Tensor[dtype, cols, rows]:
    """An owned-copy transpose of a 2D tensor.

    Distinct from `TileTensor.transpose()`, which returns a zero-copy view
    with every axis reversed over the *same* memory -- this allocates a new
    buffer, for when the result needs its own storage (e.g. to outlive the
    source, or to feed something that wants a plain `Tensor`).
    """
    var view = a.view()
    var storage = List[Scalar[dtype]](length=rows * cols, fill=0)
    var result = Tensor[dtype, cols, rows](storage^)
    var result_view = result.view()
    for r in range(rows):
        for c in range(cols):
            result_view[c, r] = view[r, c]
    return result^


def squeeze[
    dtype: DType, n: Int
](mut a: Tensor[dtype, 1, n]) -> Tensor[dtype, n]:
    """Drop a size-1 leading axis: `(1, n) -> (n,)`."""
    var view = a.view()
    var storage = List[Scalar[dtype]](capacity=n)
    for i in range(n):
        storage.append(view[0, i])
    return Tensor[dtype, n](storage^)


def squeeze[
    dtype: DType, n: Int
](mut a: Tensor[dtype, n, 1]) -> Tensor[dtype, n]:
    """Drop a size-1 trailing axis: `(n, 1) -> (n,)`."""
    var view = a.view()
    var storage = List[Scalar[dtype]](capacity=n)
    for i in range(n):
        storage.append(view[i, 0])
    return Tensor[dtype, n](storage^)


def stack[
    dtype: DType, n: Int
](mut a: Tensor[dtype, n], mut b: Tensor[dtype, n]) -> Tensor[dtype, 2, n]:
    """Stack two same-shaped rank-1 tensors along a new leading axis
    (`axis=0`): `ys[0, :] = a`, `ys[1, :] = b`.

    `axis=1` stacking is not provided -- see this module's own docstring.
    """
    var a_view = a.view()
    var b_view = b.view()
    var storage = List[Scalar[dtype]](length=2 * n, fill=0)
    var result = Tensor[dtype, 2, n](storage^)
    var result_view = result.view()
    for i in range(n):
        result_view[0, i] = a_view[i]
        result_view[1, i] = b_view[i]
    return result^


def arange[
    dtype: DType, num: Int
](start: Scalar[dtype] = 0, step: Scalar[dtype] = 1) -> Tensor[dtype, num]:
    """`num` values starting at `start`, spaced by `step`.

    `numpy.arange` takes a `stop` and derives the count from it, which makes
    the output extent depend on runtime values; this module's shapes are
    comptime, so the count is the parameter and `stop` is implied
    (`start + num*step`). `numax.array.linspace` is the one to reach for
    when the endpoints are what matter.
    """
    var storage = List[Scalar[dtype]](capacity=num)
    for i in range(num):
        storage.append(start + Scalar[dtype](i) * step)
    return Tensor[dtype, num](storage^)


def reshape[
    dtype: DType, n: Int, rows: Int, cols: Int
](mut a: Tensor[dtype, n]) -> Tensor[dtype, rows, cols] where rows * cols == n:
    """A rank-2 copy of a rank-1 tensor, in row-major order.

    The `where` clause is the element-count check: a shape that does not
    multiply out to `n` fails to compile rather than producing a tensor
    whose storage is the wrong length.

    Concrete ranks rather than a general `*new_dims` pack, for two reasons.
    A general form would need to pair two variadic packs (`*dims` in, and
    `*new_dims` out) in one signature, which Mojo's pack machinery does not
    expose; and the element-count constraint has to be *provable* to
    `where`, which admits builtin arithmetic like `rows * cols` but not a
    call to an ordinary `def` such as `_product`. `squeeze` above takes the
    same shape for the same kind of reason. `ravel` is the inverse, and the
    two compose into any reshape this module can express.
    """
    var view = a.view()
    var storage = List[Scalar[dtype]](capacity=n)
    for i in range(n):
        storage.append(view[i])
    return Tensor[dtype, rows, cols](storage^)


def reshape[
    dtype: DType, n: Int, d0: Int, d1: Int, d2: Int
](mut a: Tensor[dtype, n]) -> Tensor[dtype, d0, d1, d2] where d0 * d1 * d2 == n:
    """A rank-3 copy of a rank-1 tensor, in row-major order. See the rank-2
    overload above for why the ranks are spelled out."""
    var view = a.view()
    var storage = List[Scalar[dtype]](capacity=n)
    for i in range(n):
        storage.append(view[i])
    return Tensor[dtype, d0, d1, d2](storage^)


def ravel[
    dtype: DType, *dims: Int
](mut a: Tensor[dtype, *dims]) -> Tensor[dtype, _product[*dims]()]:
    """A rank-1 copy in row-major order -- the inverse of `reshape`.

    `Tensor` owns its storage and Mojo will not let a field be moved out of
    a value that still has to be destroyed, so this copies rather than
    retypes in place. Every manipulation in this module copies for the same
    reason; `TileTensor.reshape()` is the zero-copy view when the result
    does not need to outlive its source.
    """
    comptime n = _product[*dims]()
    var storage = List[Scalar[dtype]](capacity=n)
    for i in range(n):
        storage.append(a[i])
    return Tensor[dtype, n](storage^)


def concatenate[
    dtype: DType, n: Int, m: Int
](mut a: Tensor[dtype, n], mut b: Tensor[dtype, m]) -> Tensor[dtype, n + m]:
    """Join two rank-1 tensors end to end: `numpy.concatenate` at `axis=0`.

    Rank-1 only, for the same reason `stack` takes exactly two rank-1
    inputs: an axis-`k` concatenation of arbitrary-rank tensors needs a new
    parameter pack built from an existing one with a single extent changed.
    A real scope limit, stated rather than hidden.
    """
    var a_view = a.view()
    var b_view = b.view()
    var storage = List[Scalar[dtype]](capacity=n + m)
    for i in range(n):
        storage.append(a_view[i])
    for i in range(m):
        storage.append(b_view[i])
    return Tensor[dtype, n + m](storage^)


def split[
    dtype: DType, n: Int, at: Int
](mut a: Tensor[dtype, n]) -> Tuple[
    Tensor[dtype, at], Tensor[dtype, n - at]
] where (at >= 0 and at <= n):
    """Cut a rank-1 tensor in two at comptime index `at`: elements
    `[0, at)` and `[at, n)`. The inverse of `concatenate`.

    `numpy.split` takes a section count or a list of indices and returns a
    variable-length list; both make the *number* of outputs a runtime value,
    which a comptime-shaped tensor cannot express. One index, two outputs,
    is the part that survives that constraint.
    """
    var view = a.view()
    var head = List[Scalar[dtype]](capacity=at)
    for i in range(at):
        head.append(view[i])
    var tail = List[Scalar[dtype]](capacity=n - at)
    for i in range(at, n):
        tail.append(view[i])
    return (Tensor[dtype, at](head^), Tensor[dtype, n - at](tail^))
