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
`unwrap`" composed inline) as the CPU example, over a `Tensor` built on the
GPU context instead of the CPU one -- picked with one `gpu: Bool`
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

The last section reruns the plain pass over a rank-3 tensor (same
`n` elements, reshaped) to check that `map[gpu=True]`'s internal
`.coalesce()` (see `numax/core/tensor.mojo`) produces the same result on the
device as it does on the rank-1 layout above -- no manual flattening
needed at either the CPU or GPU call site.
"""

from max.gpu.host import DeviceContext
from std.math import exp

from numax import Compensated, Dual, Plain, Shaped, gaussian
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

    comptime Flat = Shaped[dtype, n]

    var host_xs = List[Scalar[dtype]](capacity=n)
    for i in range(n):
        host_xs.append(Scalar[dtype](i) * 0.001 - 2.0)

    var xs = Flat(ctx)
    xs.copy_from_host(host_xs)
    var ys = Flat(ctx)
    var dydx = Flat(ctx)
    var precise_values = Flat(ctx)
    var precise_errors = Flat(ctx)

    comptime block_size = 256
    comptime num_blocks = (n + block_size - 1) // block_size

    # 1. Plain SIMD -- unchanged from the CPU example.
    ctx.enqueue_function[
        map[LayoutType=Flat.LayoutType, step=gaussian_step, gpu=True]
    ](xs.view(), ys.view(), grid_dim=num_blocks, block_dim=block_size)

    # 2. The same function, differentiated -- unchanged from the CPU example.
    ctx.enqueue_function[
        map[LayoutType=Flat.LayoutType, step=gaussian_deriv_step, gpu=True]
    ](xs.view(), dydx.view(), grid_dim=num_blocks, block_dim=block_size)

    # 3. The same function again, at extra precision.
    ctx.enqueue_function[
        map[
            LayoutType=Flat.LayoutType,
            step=gaussian_compensated_value_step,
            gpu=True,
        ]
    ](
        xs.view(),
        precise_values.view(),
        grid_dim=num_blocks,
        block_dim=block_size,
    )
    ctx.enqueue_function[
        map[
            LayoutType=Flat.LayoutType,
            step=gaussian_compensated_error_step,
            gpu=True,
        ]
    ](
        xs.view(),
        precise_errors.view(),
        grid_dim=num_blocks,
        block_dim=block_size,
    )
    ctx.synchronize()

    # One staged read per tensor, rather than a host mapping per element.
    var ys_h = ys.to_host()
    var dydx_h = dydx.to_host()
    var pv_h = precise_values.to_host()
    var pe_h = precise_errors.to_host()

    for idx in [0, n // 4, n // 2, n - 1]:
        var x0 = Float64(host_xs[idx])
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
        var x0 = Float64(host_xs[idx])
        var reference = exp(-(x0 * x0))
        max_err = max(max_err, abs(Float64(ys_h[idx]) - reference))
        max_err_compensated = max(
            max_err_compensated,
            abs((Float64(pv_h[idx]) + Float64(pe_h[idx])) - reference),
        )
    print("max |GPU plain f32 - f64 reference|       =", max_err)
    print("max |GPU compensated - f64 reference|     =", max_err_compensated)

    # Same `n` elements, reshaped to rank-3 (16 = n ** (1/3)) instead of
    # rank-1 -- `map[gpu=True]` takes it directly, no `.coalesce()` needed
    # at this call site either.
    comptime cube = 16
    comptime Cube = Shaped[dtype, cube, cube, cube]
    var xs3 = Cube(ctx)
    xs3.copy_from_host(host_xs)
    var ys3 = Cube(ctx)
    ctx.enqueue_function[
        map[LayoutType=Cube.LayoutType, step=gaussian_step, gpu=True]
    ](xs3.view(), ys3.view(), grid_dim=num_blocks, block_dim=block_size)
    ctx.synchronize()

    var ys3_h = ys3.to_host()
    var max_diff = Float64(0)
    for i in range(n):
        max_diff = max(max_diff, abs(Float64(ys_h[i] - ys3_h[i])))
    print("max |rank-1 GPU result - rank-3 GPU result| =", max_diff)

    # The binary `map` overload -- two input tensors, one output -- goes
    # through `enqueue_function` the same way the unary one does. This is
    # the check that overloading `map` on arity didn't cost the GPU path
    # anything: `enqueue_function` has to resolve which overload it's
    # building a kernel signature for before it sees any runtime argument.
    var weighted = Flat(ctx)
    ctx.enqueue_function[
        map[
            LayoutType=Flat.LayoutType,
            step=weighted_step[dtype, _],
            gpu=True,
        ]
    ](
        xs.view(),
        ys.view(),
        weighted.view(),
        grid_dim=num_blocks,
        block_dim=block_size,
    )
    ctx.synchronize()

    var w_h = weighted.to_host()
    max_diff = 0
    for i in range(n):
        max_diff = max(
            max_diff,
            abs(Float64(w_h[i]) - Float64(host_xs[i]) * Float64(ys_h[i])),
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
    var coarse = Flat(ctx)
    ctx.enqueue_function[
        map[
            LayoutType=Flat.LayoutType,
            step=gaussian_step,
            width=coarse_width,
            gpu=True,
        ]
    ](xs.view(), coarse.view(), grid_dim=coarse_blocks, block_dim=block_size)
    ctx.synchronize()

    var c_h = coarse.to_host()
    max_diff = 0
    for i in range(n):
        max_diff = max(max_diff, abs(Float64(ys_h[i]) - Float64(c_h[i])))
    print("max |coarsened (width=4) - scalar|          =", max_diff)

    comptime odd_n = n + 3
    comptime Odd = Shaped[dtype, odd_n]
    comptime odd_threads = (odd_n + coarse_width - 1) // coarse_width
    comptime odd_blocks = (odd_threads + block_size - 1) // block_size
    var odd_host = List[Scalar[dtype]](capacity=odd_n)
    for i in range(odd_n):
        odd_host.append(Scalar[dtype](i) * 0.001 - 2.0)
    var odd_xs = Odd(ctx)
    odd_xs.copy_from_host(odd_host)
    var odd_scalar = Odd(ctx)
    var odd_coarse = Odd(ctx)
    ctx.enqueue_function[
        map[LayoutType=Odd.LayoutType, step=gaussian_step, gpu=True]
    ](
        odd_xs.view(),
        odd_scalar.view(),
        grid_dim=(odd_n + block_size - 1) // block_size,
        block_dim=block_size,
    )
    ctx.enqueue_function[
        map[
            LayoutType=Odd.LayoutType,
            step=gaussian_step,
            width=coarse_width,
            gpu=True,
        ]
    ](
        odd_xs.view(),
        odd_coarse.view(),
        grid_dim=odd_blocks,
        block_dim=block_size,
    )
    ctx.synchronize()

    var odd_scalar_h = odd_scalar.to_host()
    var odd_coarse_h = odd_coarse.to_host()
    max_diff = 0
    for i in range(odd_n):
        max_diff = max(
            max_diff, abs(Float64(odd_scalar_h[i]) - Float64(odd_coarse_h[i]))
        )
    print("max |coarsened - scalar| with a tail       =", max_diff)

    # The binary overload coarsens the same way, reading two inputs per
    # thread at `width` lanes each.
    var coarse_weighted = Flat(ctx)
    ctx.enqueue_function[
        map[
            LayoutType=Flat.LayoutType,
            step=weighted_step[dtype, _],
            width=coarse_width,
            gpu=True,
        ]
    ](
        xs.view(),
        ys.view(),
        coarse_weighted.view(),
        grid_dim=coarse_blocks,
        block_dim=block_size,
    )
    ctx.synchronize()

    var cw_h = coarse_weighted.to_host()
    max_diff = 0
    for i in range(n):
        max_diff = max(max_diff, abs(Float64(w_h[i]) - Float64(cw_h[i])))
    print("max |coarsened binary map - scalar|        =", max_diff)
