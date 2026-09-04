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
| `tensor` | `map`/`reduce`/`reduce_axis`/`reduce_rows`/`broadcast_op_rows` -- one `gpu: Bool` parameter picks CPU or GPU, comptime and runtime shapes under one name |
| `array` | `Tensor`, the creation surface (`zeros`/`ones`/`full`/`eye`/`linspace`/...), manipulation (`reshape`/`transpose`/`stack`/`split`/...), and `to_array`/`to_tensor`, the seam to the `Array[T, n]` half of the library |
| `ops`, `elementwise`, `logic`, `sorting` | Arithmetic and operators on `Tensor`, the elementwise math surface, comparisons returning `Tensor[DType.bool]`, and sort/search/mask |
| `constants` | `pi` and `e` at any conformer |

The conformers and `tensor` are tier 1: fixed iteration counts, no
per-lane branching, launchable inside a GPU thread. `ops`, `elementwise`,
`logic` and `sorting` are tier 2: `Plain`-only and host-side.
"""

from .array import (
    Tensor,
    arange,
    concatenate,
    copy,
    diag,
    diagflat,
    diagonal,
    empty,
    empty_like,
    eye,
    flip,
    full,
    full_like,
    geomspace,
    hstack,
    identity,
    linspace,
    logspace,
    meshgrid,
    ones,
    ones_like,
    ravel,
    reshape,
    split,
    squeeze,
    stack,
    transpose,
    to_array,
    to_tensor,
    tri,
    tril,
    triu,
    vander,
    vstack,
    zeros,
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
from .tensor import (
    add_combine,
    broadcast_op_axis,
    broadcast_op_rows,
    map,
    map_threaded,
    max_combine,
    reduce,
    reduce_axis,
    reduce_rows,
)
from .sorting import (
    all_nonzero,
    any_nonzero,
    argsort,
    count_nonzero,
    extract,
    nonzero,
    searchsorted,
    sort,
    unique,
    select,
)
