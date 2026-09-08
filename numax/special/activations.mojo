"""Activations: a small vectorized kernel library, generic over `FloatLike`.

**This module is tier 1**, except `softmax`, which says so in its own
docstring: one output element needs its whole row, which a per-element
kernel cannot express, so it is `Plain`-only and host-side.

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
`FloatLike`-generic like everything above it. It is also the one function
here numax does not implement -- it hands the tensors to MAX's `nn.softmax`,
for the reason its own docstring gives.
"""

from layout import TileTensor
from layout.tile_layout import TensorLayout
from layout.tile_tensor import PointerStorage
from nn.softmax import softmax as nn_softmax
from std.sys.info import simd_width_of

from ..core.numeric import FloatLike


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


def softmax[
    dtype: DType,
    RowsLayout: TensorLayout,
](
    xs: TileTensor[
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
    axis: Int = Int(RowsLayout.rank) - 1,
) raises where dtype.is_floating_point():
    """**Tier 2.** Softmax along `axis`, delegated to MAX's `nn.softmax`.

    `ys[r, :] = exp(xs[r, :] - max(xs[r, :])) / sum(exp(xs[r, :] -
    max(xs[r, :])))` -- the numerically stable formulation, subtracting each
    row's max so every input to `exp` is `<= 0` and cannot overflow. `axis`
    defaults to the last, which is the row-wise case.

    numax writes none of that. MAX ships softmax as a `rowwise` kernel with
    `ReduceMax` and `ReduceSum` monoids and a fused normalizing write, and
    re-implementing it here would be the defect the MAX-first gate exists to
    catch -- as it was: this used to be four passes built from
    `numax.core.tensor.reduce_rows` and `broadcast_op_rows`, with three
    caller-provided scratch buffers, none of which are needed now.

    Tier 2 because MAX's target-parameterized entry point is out of reach at
    the pin, so this is the host one. The overload taking a `target` also
    takes its input as a fused closure in a *compile-time* parameter, whose
    implicit `__origins__`
    the compiler cannot infer across the module boundary, and there is no
    keyword to bind it by hand. The overload numax can call takes tensors and
    has no `target`. So a device-resident softmax is still hand-launched from
    `numax.core.tensor`'s primitives -- `examples/intermediate/softmax.mojo`
    shows both, and checks them against each other.
    """
    nn_softmax[dtype, simd_width_of[dtype](), Int(RowsLayout.rank)](
        xs, ys, axis
    )
