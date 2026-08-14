"""One kernel, three meanings: plain SIMD, autodiff, and extra precision.

`ember.gaussian` is a single, ordinary-looking function:

    def gaussian[T: FloatLike](x: T) -> T:
        return (-(x * x)).exp()

This example calls it three times with three different `FloatLike` types and
gets three different things out, with no change to `gaussian` itself:
plain `float32` SIMD, the value plus its derivative (`Dual`), and the value
carried to roughly double precision (`Compensated`).

The walk over the 4096 points is `ember.tensor.map_simd`, driving `gaussian`
over a `TileTensor` (from MAX's `layout` package) at the CPU's native SIMD
width, with a scalar tail for whatever doesn't divide evenly -- the same
`TileTensor`-based approach `examples/gaussian_gpu.mojo` uses on the GPU.
"""

from layout import TileTensor
from layout.tile_layout import row_major
from std.math import exp
from std.sys.info import simd_width_of

from ember import Compensated, Dual, Plain, gaussian
from ember.tensor import map_simd

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
    comptime layout = row_major[n]()

    var xs_storage = List[Scalar[dtype]](capacity=n)
    for i in range(n):
        xs_storage.append(Scalar[dtype](i) * 0.001 - 2.0)
    var xs = TileTensor(xs_storage, layout)

    var ys_storage = List[Scalar[dtype]](length=n, fill=0)
    var dydx_storage = List[Scalar[dtype]](length=n, fill=0)
    var precise_values_storage = List[Scalar[dtype]](length=n, fill=0)
    var precise_errors_storage = List[Scalar[dtype]](length=n, fill=0)
    var ys = TileTensor(ys_storage, layout)
    var dydx = TileTensor(dydx_storage, layout)
    var precise_values = TileTensor(precise_values_storage, layout)
    var precise_errors = TileTensor(precise_errors_storage, layout)

    # 1. Plain SIMD, at the CPU's native width.
    map_simd[width=width, step=gaussian_step](xs, ys)

    # 2. The same function, differentiated.
    map_simd[width=width, step=gaussian_deriv_step](xs, dydx)

    # 3. The same function again, at extra precision.
    map_simd[width=width, step=gaussian_compensated_value_step](
        xs, precise_values
    )
    map_simd[width=width, step=gaussian_compensated_error_step](
        xs, precise_errors
    )

    for idx in [0, n // 4, n // 2, n - 1]:
        var x0 = Float64(xs_storage[idx])
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
        var x0 = Float64(xs_storage[idx])
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
