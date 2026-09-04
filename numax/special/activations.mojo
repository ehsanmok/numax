"""Activations: a small vectorized kernel library, generic over `FloatLike`.

Every function below is written once, against the trait, and gets three
meanings for free from whatever type it's called with: plain SIMD, a value
paired with its derivative (`Dual`), or a value carried to roughly double
precision (`Compensated`). None of them name a width, a dtype, or an
instruction set.

`erf`/`erfc` live in `numax.special.erf`;
`gamma`/`lgamma`/`gammainc`/`gammaincc` in `numax.special.gamma`;
`j0` in `numax.special.bessel`; `lambertw` in
`numax.special.lambertw`
-- this module keeps the activations (`gaussian`, `sigmoid`, `swish`,
`tanh`, `relu`, `leaky_relu`, `gelu`) plus `softmax`, matching `numax`'s
one-concept-per-file convention (`plain.mojo`, `dual.mojo`,
`compensated.mojo`).

The `tanh` here is the `FloatLike` scalar, and it is what the root package
exports: it is the one a kernel calls, and `gelu` below is built from it.
`numax.core.tanh` is the elementwise form over a whole `Tensor`.

`softmax`, at the bottom, is the exception: it isn't purely elementwise (it
needs a whole row of a tensor to compute one output element), so it isn't
`FloatLike`-generic like everything above it -- it's a small orchestration
function over `Plain` `SIMD` values, built from `numax.core.tensor`'s `reduce_rows`
and `broadcast_op_rows`.
"""

from layout import TileTensor
from layout.tile_layout import TensorLayout
from layout.tile_tensor import PointerStorage

from ..core.numeric import FloatLike
from ..core.plain import Plain
from ..core.tensor import (
    add_combine,
    broadcast_op_rows,
    max_combine,
    reduce_rows,
)


def gaussian[T: FloatLike](x: T) -> T:
    """The unnormalized Gaussian bump, `exp(-x^2)`."""
    return (-(x * x)).exp()


def sigmoid[T: FloatLike](x: T) -> T:
    """The logistic function, `1 / (1 + exp(-x))`."""
    return T.one() / (T.one() + (-x).exp())


def swish[T: FloatLike](x: T) -> T:
    """`x * sigmoid(x)`, the SiLU activation."""
    return x * sigmoid(x)


def tanh[T: FloatLike](x: T) -> T:
    """Hyperbolic tangent, via `(exp(2x) - 1) / (exp(2x) + 1)`."""
    var e2x = (x + x).exp()
    return (e2x - T.one()) / (e2x + T.one())


def relu[T: FloatLike](x: T) -> T:
    """`max(x, 0)`, via `(x + |x|) / 2` -- no branch on `Self`'s sign."""
    return (x + x.abs()) / T.constant(2.0)


def leaky_relu[T: FloatLike](x: T, alpha: Float64) -> T:
    """`x` for `x >= 0`, `alpha * x` otherwise (`alpha` is usually small).

    `((1 + alpha) * x + (1 - alpha) * |x|) / 2` is the same function with no
    branch: at `x >= 0` the two terms are `(1 + alpha)x + (1 - alpha)x = 2x`;
    at `x < 0`, `|x| = -x` and they cancel down to `2*alpha*x`.
    """
    var ax = x.abs()
    return (
        T.constant(1 + alpha) * x + T.constant(1 - alpha) * ax
    ) / T.constant(2.0)


def gelu[T: FloatLike](x: T) -> T:
    """The GELU activation, tanh approximation.

    `0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715*x^3)))` -- reuses `tanh`
    above rather than `erf` directly, since `erf` isn't defined in closed
    form and GELU's usual definition already has this near-equivalent
    approximation in wide use.
    """
    var x3 = x * x * x
    var inner = T.constant(0.7978845608028654) * (x + T.constant(0.044715) * x3)
    return (x * (T.one() + tanh(inner))) / T.constant(2.0)


def _sub_exp_combine[
    dtype: DType
](a: SIMD[dtype, 1], b: SIMD[dtype, 1]) -> SIMD[
    dtype, 1
] where dtype.is_floating_point():
    # exp(a - b), fused into one pass so `softmax` doesn't need a separate,
    # in-place elementwise `exp` step (which `map` can't express anyway --
    # its `xs`/`ys` parameters may not alias the same buffer).
    return (Plain[dtype](a) - Plain[dtype](b)).exp().v


def _div_combine[
    dtype: DType
](a: SIMD[dtype, 1], b: SIMD[dtype, 1]) -> SIMD[dtype, 1]:
    return (Plain[dtype](a) / Plain[dtype](b)).v


def softmax[
    dtype: DType,
    RowsLayout: TensorLayout,
    RowValuesLayout: TensorLayout,
](
    xs: TileTensor[
        dtype,
        RowsLayout,
        MutAnyOrigin,
        Storage=PointerStorage[element_width=1],
    ],
    tmp: TileTensor[
        dtype,
        RowsLayout,
        MutAnyOrigin,
        Storage=PointerStorage[element_width=1],
    ],
    ys: TileTensor[
        dtype,
        RowsLayout,
        MutAnyOrigin,
        Storage=PointerStorage[element_width=1],
    ],
    row_max: TileTensor[
        dtype,
        RowValuesLayout,
        MutAnyOrigin,
        Storage=PointerStorage[element_width=1],
    ],
    row_sum: TileTensor[
        dtype,
        RowValuesLayout,
        MutAnyOrigin,
        Storage=PointerStorage[element_width=1],
    ],
) where dtype.is_floating_point():
    """Row-wise softmax over a 2D tensor: CPU-side.

    `ys[r, :] = exp(xs[r, :] - max(xs[r, :])) / sum(exp(xs[r, :] -
    max(xs[r, :])))` -- the usual numerically-stable formulation (subtracting
    each row's max before exponentiating keeps every input to `exp` `<= 0`,
    so it can't overflow).

    `tmp` (same shape as `xs`) and `row_max`/`row_sum` (one element per row)
    are caller-provided scratch space -- `numax.core.tensor`'s primitives never
    allocate on their own, and `map`/`broadcast_op_rows` can't write their
    output back into the same buffer they read from, so the exp-and-shift
    step needs a separate destination from both `xs` and the final `ys`.

    Built entirely from `numax.core.tensor.reduce_rows` and
    `numax.core.tensor.broadcast_op_rows`: a per-row max, a fused
    subtract-and-`exp` broadcast into `tmp`, a per-row sum of `tmp`, then a
    divide broadcast into `ys`.
    """
    reduce_rows[combine=max_combine[dtype]](xs, row_max, SIMD[dtype, 1](-1e30))
    broadcast_op_rows[combine=_sub_exp_combine[dtype]](xs, row_max, tmp)
    reduce_rows[combine=add_combine[dtype]](tmp, row_sum, SIMD[dtype, 1](0))
    broadcast_op_rows[combine=_div_combine[dtype]](tmp, row_sum, ys)
