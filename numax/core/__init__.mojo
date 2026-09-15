"""numax.core: the numeric types and the tensor engine.

`FloatLike` and its conformers, the `Tensor` that owns its storage, the
walks that drive a kernel over one, and the NumPy-named surface over them.

```mojo
from numax.core import Plain, Dual, Tensor, linspace, sqrt, allclose
from numax.core.tensor import map, reduce, reduce_axis   # the engine
```

| Module | Contents |
|---|---|
| `numeric` | The `FloatLike` trait, plus the branchless helpers (`max_of`, `blend`, `ge_indicator`) every conformer-generic kernel is built from |
| `plain`, `dual`, `gradient`, `compensated`, `decimal`, `interval`, `complex` | The conformers: ordinary SIMD, forward-mode autodiff, multi-variable gradients, error-compensated arithmetic, exact base-10 fixed point, interval enclosures, complex over any of them |
| `tensor` | `map`/`reduce`/`reduce_axis`/`reduce_rows`/`broadcast_op_rows` -- one `gpu: Bool` parameter picks CPU or GPU, comptime and runtime shapes under one name, plus `map_strided`/`reduce_strided` for a transposed or sliced view and `map_blocks` for one whole small problem per lane |
| `rowwise` | `reduce_all`/`argmax_all`/`argmin_all` over a whole tensor and `sum_axis`/`prod_axis`/`max_axis`/`min_axis` along one -- reductions delegated to MAX's `algorithm.rowwise` scaffolder and `reduce_op` monoids, one body for both targets, threaded on CPU and tiered on GPU. The monoid is a `StaticString` parameter dispatched by `comptime if`; `tensor`'s `reduce`/`reduce_axis` remain for a fold outside MAX's monoid set |
| `array` | `Tensor`, the creation surface (`zeros`/`ones`/`full`/`eye`/`linspace`/..., each taking its `DeviceContext` last and optional), manipulation (`reshape`/`transpose`/`stack`/`split`/...), and `to_array`/`to_tensor`, the seam to the `Array[T, n]` half of the library |
| `ops`, `elementwise`, `logic`, `sorting` | Arithmetic and operators on `Tensor`, the elementwise math surface, comparisons returning `Static[DType.bool]`, and sort/search/mask |
| `_drive` | Private: the one launch policy behind `ops`, `elementwise` and `logic` -- flatten, pick a target from `gpu: Bool`, and either launch `max.algorithm.elementwise` or walk serially below `1 << 16` elements |
| `constants` | `pi` and `e` at any conformer |

The conformers and `tensor` are tier 1: fixed iteration counts, no
per-lane branching, launchable inside a GPU thread. `ops`, `elementwise`,
`logic` and `sorting` are tier 2: `Plain`-only. `ops`, `elementwise` and
`logic` are tier 2 in shape only -- like `rowwise`, one body serves both
targets through `_drive`, and every routine in them takes `gpu: Bool = False`
as its last compile-time parameter. The operators on `Tensor` forward at that
default, so `a + b` on a device tensor runs the host walk and says so;
`add[gpu=True]` is the device spelling. `logic`'s `all`/`any` return a `Bool`
and stay host reads, and so does the fold half of `allclose`/`array_equal`.
`numax.stats`' `sum`/`prod`/`min`/`max`/`argmax`/`argmin` take the same
`gpu: Bool` and route through `rowwise` rather than `_drive`, since a
reduction is a monoid MAX already ships rather than a body numax supplies.
`sorting` is still host-side.
"""

from .array import (
    Dynamic,
    Static,
    Tensor,
    arange,
    broadcast_shapes,
    broadcast_to,
    asarray,
    atleast_1d,
    atleast_2d,
    concatenate,
    concatenate_dyn,
    copy,
    diag,
    diagflat,
    diagonal,
    empty,
    empty_dyn,
    empty_like,
    expand_dims,
    eye,
    flip,
    full,
    full_dyn,
    full_like,
    geomspace,
    hstack,
    identity,
    linspace,
    logspace,
    meshgrid,
    moveaxis,
    ones,
    ones_dyn,
    ones_like,
    ravel,
    repeat,
    reshape,
    roll,
    reshape_dyn,
    slice,
    split,
    split_dyn,
    stack_dyn,
    squeeze,
    stack,
    swapaxes,
    tile,
    transpose,
    to_array,
    to_tensor,
    tri,
    pad,
    pad_constant,
    pad_edge,
    pad_reflect,
    tril,
    triu,
    vander,
    vstack,
    zeros,
    zeros_dyn,
    zeros_like,
)
from .compensated import Compensated
from .complex import Complex
from .constants import e, e_at, pi, pi_at
from .decimal import Decimal
from .dual import Dual
from .elementwise import (
    abs,
    arccos,
    arccosh,
    arcsin,
    arcsinh,
    arctan,
    arctan2,
    arctanh,
    cbrt,
    ceil,
    clip,
    copysign,
    cos,
    cosh,
    diff,
    exp,
    exp2,
    expm1,
    floor,
    gradient,
    hypot,
    log,
    log10,
    log1p,
    log2,
    maximum,
    minimum,
    remainder,
    round,
    rsqrt,
    sin,
    sinh,
    sqrt,
    tan,
    tanh,
    trunc,
)
from .gradient import Gradient
from .interval import Interval
from .logic import (
    all,
    allclose,
    any,
    array_equal,
    equal,
    greater,
    greater_equal,
    isclose,
    isfinite,
    isinf,
    isnan,
    isneginf,
    isposinf,
    less,
    less_equal,
    logical_and,
    logical_not,
    logical_or,
    logical_xor,
    not_equal,
)
from .numeric import (
    FloatLike,
    blend,
    default_erf_approx,
    ge_indicator,
    guard_nonzero,
    max_of,
    min_of,
)
from .ops import (
    add,
    invert,
    astype,
    divide,
    floor_divide,
    mod,
    multiply,
    negative,
    power,
    subtract,
)
from .plain import Plain
from .dtypes import (
    bf16,
    bool,
    f8e3m4,
    f8e4m3fn,
    f8e4m3fnuz,
    f8e5m2,
    f8e5m2fnuz,
    f16,
    f32,
    f64,
    i8,
    i16,
    i32,
    i64,
    u8,
    u16,
    u32,
    u64,
)
from .rowwise import (
    argmax_all,
    argmin_all,
    max_axis,
    min_axis,
    prod_axis,
    reduce_all,
    sum_axis,
)
from .tensor import (
    add_combine,
    broadcast_op_axis,
    broadcast_op_rows,
    map,
    map_blocks,
    map_strided,
    map_threaded,
    max_combine,
    reduce,
    reduce_axis,
    reduce_rows,
    reduce_strided,
)
from .sorting import (
    all_nonzero,
    any_nonzero,
    argsort,
    argwhere,
    count_nonzero,
    extract,
    nonzero,
    put,
    searchsorted,
    sort,
    take,
    take_along_axis,
    top_k,
    unique,
    select,
)
