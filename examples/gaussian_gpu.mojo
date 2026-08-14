"""The same `gaussian` kernel from `examples/gaussian.mojo`, run on the GPU.

`DevicePassable` -- the trait a value must conform to in order to cross the
host/device boundary as a kernel argument -- only constrains what's passed
*into* a kernel function (buffers, pointers, plain scalars, and `TileTensor`
itself). `Plain`, `Dual`, and `Compensated` are all built entirely from
`SIMD` lanes already on the device side, with no pointers or allocations of
their own, so none of them need any changes to be used inside a kernel body:
`gaussian` itself, and every `FloatLike` type it's called with here, are
exactly the same code imported from `ember.special`, `ember.plain`,
`ember.dual`, and `ember.compensated` as the CPU example uses.

The walk itself is `ember.tensor.map_gpu`, launched once per element via
`DeviceContext.enqueue_function` -- one GPU thread per element, since GPU
parallelism comes from thread count rather than per-thread SIMD registers
(unlike the CPU example's `map_simd`, which vectorizes within a thread).
Same `kernel`/`wrap`/`unwrap` shape `map`/`map_simd` use on CPU, over a
`DeviceBuffer`-backed `TileTensor` instead of a `List`-backed one.

`Compensated.exp()` used to be excluded here because its Taylor-series
coefficients were built as an `Array[Float64, ...]` at runtime, and Apple's
Metal backend (the target on this machine) has no float64 support at all.
That's fixed now: the coefficients are split into `dtype`-native hi/lo pairs
at compile time (see `_split_f64` in `ember/compensated.mojo`), so no
float64 arithmetic or storage reaches device code, on Metal or anywhere
else.
"""

from layout import TileTensor
from layout.tile_layout import row_major
from max.gpu.host import DeviceContext
from std.math import exp

from ember import Compensated, Dual, Plain, gaussian
from ember.tensor import map_gpu

comptime dtype = DType.float32
comptime n = 4096


def wrap_plain(x: SIMD[dtype, 1]) -> Plain[dtype, 1]:
    return Plain[dtype, 1](x)


def unwrap_plain(p: Plain[dtype, 1]) -> SIMD[dtype, 1]:
    return p.v


def wrap_dual(x: SIMD[dtype, 1]) -> Dual[Plain[dtype, 1]]:
    return Dual[Plain[dtype, 1]](Plain[dtype, 1](x), Plain[dtype, 1](1))


def unwrap_dual_deriv(d: Dual[Plain[dtype, 1]]) -> SIMD[dtype, 1]:
    return d.deriv.v


def wrap_compensated(x: SIMD[dtype, 1]) -> Compensated[dtype, 1]:
    return Compensated[dtype, 1](x, 0)


def unwrap_compensated_value(c: Compensated[dtype, 1]) -> SIMD[dtype, 1]:
    return c.value


def unwrap_compensated_error(c: Compensated[dtype, 1]) -> SIMD[dtype, 1]:
    return c.error


def main() raises:
    var ctx = DeviceContext()
    print("GPU API:", ctx.api())

    comptime layout = row_major[n]()

    var xs_buf = ctx.enqueue_create_buffer[dtype](n)
    var ys_buf = ctx.enqueue_create_buffer[dtype](n)
    var dydx_buf = ctx.enqueue_create_buffer[dtype](n)
    var precise_values_buf = ctx.enqueue_create_buffer[dtype](n)
    var precise_errors_buf = ctx.enqueue_create_buffer[dtype](n)

    with xs_buf.map_to_host() as h:
        for i in range(n):
            h[i] = Scalar[dtype](i) * 0.001 - 2.0

    var xs = TileTensor(xs_buf, layout)
    var ys = TileTensor(ys_buf, layout)
    var dydx = TileTensor(dydx_buf, layout)
    var precise_values = TileTensor(precise_values_buf, layout)
    var precise_errors = TileTensor(precise_errors_buf, layout)

    comptime block_size = 256
    comptime num_blocks = (n + block_size - 1) // block_size

    # 1. Plain SIMD -- unchanged from the CPU example.
    ctx.enqueue_function[
        map_gpu[
            dtype=dtype,
            T=Plain[dtype, 1],
            width=1,
            LayoutType=type_of(layout),
            kernel=gaussian[Plain[dtype, 1]],
            wrap=wrap_plain,
            unwrap=unwrap_plain,
        ]
    ](xs, ys, grid_dim=num_blocks, block_dim=block_size)

    # 2. The same function, differentiated -- unchanged from the CPU example.
    ctx.enqueue_function[
        map_gpu[
            dtype=dtype,
            T=Dual[Plain[dtype, 1]],
            width=1,
            LayoutType=type_of(layout),
            kernel=gaussian[Dual[Plain[dtype, 1]]],
            wrap=wrap_dual,
            unwrap=unwrap_dual_deriv,
        ]
    ](xs, dydx, grid_dim=num_blocks, block_dim=block_size)

    # 3. The same function again, at extra precision.
    ctx.enqueue_function[
        map_gpu[
            dtype=dtype,
            T=Compensated[dtype, 1],
            width=1,
            LayoutType=type_of(layout),
            kernel=gaussian[Compensated[dtype, 1]],
            wrap=wrap_compensated,
            unwrap=unwrap_compensated_value,
        ]
    ](xs, precise_values, grid_dim=num_blocks, block_dim=block_size)
    ctx.enqueue_function[
        map_gpu[
            dtype=dtype,
            T=Compensated[dtype, 1],
            width=1,
            LayoutType=type_of(layout),
            kernel=gaussian[Compensated[dtype, 1]],
            wrap=wrap_compensated,
            unwrap=unwrap_compensated_error,
        ]
    ](xs, precise_errors, grid_dim=num_blocks, block_dim=block_size)
    ctx.synchronize()

    with xs_buf.map_to_host() as xs_h, ys_buf.map_to_host() as ys_h, dydx_buf.map_to_host() as dydx_h, precise_values_buf.map_to_host() as pv_h, precise_errors_buf.map_to_host() as pe_h:
        for idx in [0, n // 4, n // 2, n - 1]:
            var x0 = Float64(xs_h[idx])
            var expected_y = exp(-(x0 * x0))
            var expected_d = -2.0 * x0 * expected_y
            print(
                "x=",
                x0,
                " y=",
                ys_h[idx],
                " (expected ",
                expected_y,
                ") dy/dx=",
                dydx_h[idx],
                " (expected ",
                expected_d,
                ") compensated=",
                pv_h[idx] + pe_h[idx],
            )

        var max_err = Float64(0)
        var max_err_compensated = Float64(0)
        for idx in range(n):
            var x0 = Float64(xs_h[idx])
            var reference = exp(-(x0 * x0))
            max_err = max(max_err, abs(Float64(ys_h[idx]) - reference))
            max_err_compensated = max(
                max_err_compensated,
                abs((Float64(pv_h[idx]) + Float64(pe_h[idx])) - reference),
            )
        print("max |GPU plain f32 - f64 reference|       =", max_err)
        print(
            "max |GPU compensated - f64 reference|     =", max_err_compensated
        )
