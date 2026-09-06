"""One kernel, three meanings: plain SIMD, autodiff, and extra precision.

`numax.gaussian` is a single, ordinary-looking function:

    def gaussian[T: FloatLike](x: T) -> T:
        return (-(x * x)).exp()

This example calls it three times with three different `FloatLike` types and
gets three different things out, with no change to `gaussian` itself:
plain `float32` SIMD, the value plus its derivative (`Dual`), and the value
carried to roughly double precision (`Compensated`).

The 4096 points live in a `Tensor`, which owns its storage. The walk is
`numax.core.tensor.map`, and what it takes is `.view()`: the `TileTensor`
MAX kernels speak, here driven at the CPU's native SIMD width with a scalar
tail for whatever doesn't divide evenly. `examples/gaussian_gpu.mojo` hands
the same view to the same `map` on a GPU.
"""

from std.math import exp
from std.sys.info import simd_width_of

from numax import Compensated, Dual, Plain, gaussian, zeros
from numax.core.tensor import map

comptime dtype = DType.float32
comptime n = 4096
comptime width = simd_width_of[dtype]()


def gaussian_step[w: Int](x: SIMD[dtype, w]) -> SIMD[dtype, w]:
    return gaussian(Plain[dtype, w](x)).v


def gaussian_deriv_step[w: Int](x: SIMD[dtype, w]) -> SIMD[dtype, w]:
    return gaussian(
        Dual[Plain[dtype, w]](Plain[dtype, w](x), Plain[dtype, w](1))
    ).deriv.v


def gaussian_compensated_value_step[
    w: Int
](x: SIMD[dtype, w]) -> SIMD[dtype, w]:
    return gaussian(Compensated[dtype, w](x, 0)).value


def gaussian_compensated_error_step[
    w: Int
](x: SIMD[dtype, w]) -> SIMD[dtype, w]:
    return gaussian(Compensated[dtype, w](x, 0)).error


def main() raises:
    var xs = zeros[dtype, n]()
    for i in range(n):
        xs[i] = Scalar[dtype](i) * 0.001 - 2.0

    var ys = zeros[dtype, n]()
    var dydx = zeros[dtype, n]()
    var precise_values = zeros[dtype, n]()
    var precise_errors = zeros[dtype, n]()

    # 1. Plain SIMD, at the CPU's native width.
    map[width=width, step=gaussian_step](xs.view(), ys.view())

    # 2. The same function, differentiated.
    map[width=width, step=gaussian_deriv_step](xs.view(), dydx.view())

    # 3. The same function again, at extra precision.
    map[width=width, step=gaussian_compensated_value_step](
        xs.view(), precise_values.view()
    )
    map[width=width, step=gaussian_compensated_error_step](
        xs.view(), precise_errors.view()
    )

    for idx in [0, n // 4, n // 2, n - 1]:
        var x0 = Float64(xs[idx])
        var expected_y = exp(-(x0 * x0))
        var expected_d = -2.0 * x0 * expected_y
        print(
            "x=",
            x0,
            " y=",
            ys[idx],
            " (expected ",
            expected_y,
            ") dy/dx=",
            dydx[idx],
            " (expected ",
            expected_d,
            ") compensated=",
            precise_values[idx] + precise_errors[idx],
        )

    var max_err_plain = Float64(0)
    var max_err_compensated = Float64(0)
    for idx in range(n):
        var x0 = Float64(xs[idx])
        var reference = exp(-(x0 * x0))
        max_err_plain = max(max_err_plain, abs(Float64(ys[idx]) - reference))
        max_err_compensated = max(
            max_err_compensated,
            abs(
                (Float64(precise_values[idx]) + Float64(precise_errors[idx]))
                - reference
            ),
        )

    print("SIMD width used:", width)
    print("max |plain f32 - f64 reference|   =", max_err_plain)
    print("max |compensated - f64 reference| =", max_err_compensated)
