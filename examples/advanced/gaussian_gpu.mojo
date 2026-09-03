"""The same `gaussian` kernel from `examples/gaussian.mojo`, run on the GPU.

`DevicePassable` -- the trait a value must conform to in order to cross the
host/device boundary as a kernel argument -- only constrains what's passed
*into* a kernel function (buffers, pointers, plain scalars, and `TileTensor`
itself). `Plain`, `Dual`, and `Compensated` are all built entirely from
`SIMD` lanes already on the device side, with no pointers or allocations of
their own, so none of them need any changes to be used inside a kernel body:
`gaussian` itself, and every `FloatLike` type it's called with here, are
exactly the same code imported from `numax.special`, `numax.core.plain`,
`numax.core.dual`, and `numax.core.compensated` as the CPU example uses.

The walk itself is `numax.core.tensor.map[gpu=True]`, launched once per element
via `DeviceContext.enqueue_function` -- one GPU thread per element, since
GPU parallelism comes from thread count rather than per-thread SIMD
registers (unlike the CPU example's `map[gpu=False]`, which vectorizes
within a thread). Same function, same `step` shape ("`wrap` -> kernel ->
`unwrap`" composed inline) as the CPU example, over a `DeviceBuffer`-backed
`TileTensor` instead of a `List`-backed one -- picked with one `gpu: Bool`
compile-time parameter rather than a separate name (see
`numax/core/tensor.mojo`'s module docstring for why `step` has to stay `thin` on
both paths for that to work through `enqueue_function`).

`Compensated.exp()` used to be excluded here because its Taylor-series
coefficients were built as an `Array[Float64, ...]` at runtime, and Apple's
Metal backend has no float64 support at all (CUDA does, which is exactly
why testing only there would have hidden this).
That's fixed now: the coefficients are split into `dtype`-native hi/lo pairs
at compile time (see `_split_f64` in `numax/core/compensated.mojo`), so no
float64 arithmetic or storage reaches device code, on Metal or anywhere
else.

The last section reruns the plain pass over a rank-3 `TileTensor` (same
`n` elements, reshaped) to check that `map[gpu=True]`'s internal
`.coalesce()` (see `numax/core/tensor.mojo`) produces the same result on the
device as it does on the rank-1 layout above -- no manual flattening
needed at either the CPU or GPU call site.
"""

from layout import TileTensor
from layout.tile_layout import row_major
from max.gpu.host import DeviceContext
from std.math import exp

from numax import Compensated, Dual, Plain, gaussian
from numax.core.tensor import map

comptime dtype = DType.float32
comptime n = 4096


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


def weighted_step[
    dt: DType, w: Int
](x: SIMD[dt, w], y: SIMD[dt, w]) -> SIMD[dt, w] where dt.is_floating_point():
    """A two-input kernel for the binary `map` below: `x * y`."""
    return (Plain[dt, w](x) * Plain[dt, w](y)).v


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
        map[LayoutType=type_of(layout), step=gaussian_step, gpu=True]
    ](xs, ys, grid_dim=num_blocks, block_dim=block_size)

    # 2. The same function, differentiated -- unchanged from the CPU example.
    ctx.enqueue_function[
        map[LayoutType=type_of(layout), step=gaussian_deriv_step, gpu=True]
    ](xs, dydx, grid_dim=num_blocks, block_dim=block_size)

    # 3. The same function again, at extra precision.
    ctx.enqueue_function[
        map[
            LayoutType=type_of(layout),
            step=gaussian_compensated_value_step,
            gpu=True,
        ]
    ](xs, precise_values, grid_dim=num_blocks, block_dim=block_size)
    ctx.enqueue_function[
        map[
            LayoutType=type_of(layout),
            step=gaussian_compensated_error_step,
            gpu=True,
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

    # Same `n` elements, reshaped to rank-3 (16 = n ** (1/3)) instead of
    # rank-1 -- `map[gpu=True]` takes it directly, no `.coalesce()` needed
    # at this call site either.
    comptime cube = 16
    comptime layout3d = row_major[cube, cube, cube]()
    var xs3_buf = ctx.enqueue_create_buffer[dtype](n)
    var ys3_buf = ctx.enqueue_create_buffer[dtype](n)
    with xs_buf.map_to_host() as h, xs3_buf.map_to_host() as h3:
        for i in range(n):
            h3[i] = h[i]
    var xs3 = TileTensor(xs3_buf, layout3d)
    var ys3 = TileTensor(ys3_buf, layout3d)
    ctx.enqueue_function[
        map[LayoutType=type_of(layout3d), step=gaussian_step, gpu=True]
    ](xs3, ys3, grid_dim=num_blocks, block_dim=block_size)
    ctx.synchronize()

    with ys_buf.map_to_host() as ys_h, ys3_buf.map_to_host() as ys3_h:
        var max_diff = Float64(0)
        for i in range(n):
            max_diff = max(max_diff, abs(Float64(ys_h[i] - ys3_h[i])))
        print("max |rank-1 GPU result - rank-3 GPU result| =", max_diff)

    # The binary `map` overload -- two input tensors, one output -- goes
    # through `enqueue_function` the same way the unary one does. This is
    # the check that overloading `map` on arity didn't cost the GPU path
    # anything: `enqueue_function` has to resolve which overload it's
    # building a kernel signature for before it sees any runtime argument.
    var weighted_buf = ctx.enqueue_create_buffer[dtype](n)
    var weighted = TileTensor(weighted_buf, layout)
    ctx.enqueue_function[
        map[LayoutType=type_of(layout), step=weighted_step[dtype, _], gpu=True]
    ](xs, ys, weighted, grid_dim=num_blocks, block_dim=block_size)
    ctx.synchronize()

    with xs_buf.map_to_host() as xs_h, ys_buf.map_to_host() as ys_h, weighted_buf.map_to_host() as w_h:
        var max_diff = Float64(0)
        for i in range(n):
            max_diff = max(
                max_diff,
                abs(Float64(w_h[i]) - Float64(xs_h[i]) * Float64(ys_h[i])),
            )
        print("max |binary map on GPU - x*y|               =", max_diff)

    # Thread coarsening: `width` on the GPU path gives each thread that many
    # consecutive elements instead of one, which changes the launch geometry
    # (`grid_dim` counts threads, and there are now `ceildiv(n, width)` of
    # them, not `n`). The result must be bit-identical to the `width=1` pass
    # above -- `step` is the same function either way, only the number of
    # lanes it is instantiated at differs.
    #
    # `width=1` is the default precisely because coarsening measured no
    # faster on Metal (see `bench/bench_gpu_roofline.mojo`); this section
    # exists to keep the option correct rather than to recommend it. `odd_n`
    # is deliberately not a multiple of `coarse_width`, so the scalar tail
    # that picks up the final partial group is exercised rather than assumed.
    comptime coarse_width = 4
    comptime coarse_threads = (n + coarse_width - 1) // coarse_width
    comptime coarse_blocks = (coarse_threads + block_size - 1) // block_size
    var coarse_buf = ctx.enqueue_create_buffer[dtype](n)
    var coarse = TileTensor(coarse_buf, layout)
    ctx.enqueue_function[
        map[
            LayoutType=type_of(layout),
            step=gaussian_step,
            width=coarse_width,
            gpu=True,
        ]
    ](xs, coarse, grid_dim=coarse_blocks, block_dim=block_size)
    ctx.synchronize()

    with ys_buf.map_to_host() as ys_h, coarse_buf.map_to_host() as c_h:
        var max_diff = Float64(0)
        for i in range(n):
            max_diff = max(max_diff, abs(Float64(ys_h[i]) - Float64(c_h[i])))
        print("max |coarsened (width=4) - scalar|          =", max_diff)

    comptime odd_n = n + 3
    comptime odd_layout = row_major[odd_n]()
    comptime odd_threads = (odd_n + coarse_width - 1) // coarse_width
    comptime odd_blocks = (odd_threads + block_size - 1) // block_size
    var odd_xs_buf = ctx.enqueue_create_buffer[dtype](odd_n)
    var odd_scalar_buf = ctx.enqueue_create_buffer[dtype](odd_n)
    var odd_coarse_buf = ctx.enqueue_create_buffer[dtype](odd_n)
    with odd_xs_buf.map_to_host() as h:
        for i in range(odd_n):
            h[i] = Scalar[dtype](i) * 0.001 - 2.0
    var odd_xs = TileTensor(odd_xs_buf, odd_layout)
    var odd_scalar = TileTensor(odd_scalar_buf, odd_layout)
    var odd_coarse = TileTensor(odd_coarse_buf, odd_layout)
    ctx.enqueue_function[
        map[LayoutType=type_of(odd_layout), step=gaussian_step, gpu=True]
    ](
        odd_xs,
        odd_scalar,
        grid_dim=(odd_n + block_size - 1) // block_size,
        block_dim=block_size,
    )
    ctx.enqueue_function[
        map[
            LayoutType=type_of(odd_layout),
            step=gaussian_step,
            width=coarse_width,
            gpu=True,
        ]
    ](odd_xs, odd_coarse, grid_dim=odd_blocks, block_dim=block_size)
    ctx.synchronize()

    with odd_scalar_buf.map_to_host() as s_h, odd_coarse_buf.map_to_host() as c_h:
        var max_diff = Float64(0)
        for i in range(odd_n):
            max_diff = max(max_diff, abs(Float64(s_h[i]) - Float64(c_h[i])))
        print("max |coarsened - scalar| with a tail       =", max_diff)

    # The binary overload coarsens the same way, reading two inputs per
    # thread at `width` lanes each.
    var coarse_weighted_buf = ctx.enqueue_create_buffer[dtype](n)
    var coarse_weighted = TileTensor(coarse_weighted_buf, layout)
    ctx.enqueue_function[
        map[
            LayoutType=type_of(layout),
            step=weighted_step[dtype, _],
            width=coarse_width,
            gpu=True,
        ]
    ](xs, ys, coarse_weighted, grid_dim=coarse_blocks, block_dim=block_size)
    ctx.synchronize()

    with weighted_buf.map_to_host() as w_h, coarse_weighted_buf.map_to_host() as cw_h:
        var max_diff = Float64(0)
        for i in range(n):
            max_diff = max(max_diff, abs(Float64(w_h[i]) - Float64(cw_h[i])))
        print("max |coarsened binary map - scalar|        =", max_diff)
