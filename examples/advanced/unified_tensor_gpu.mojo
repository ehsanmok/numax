"""One `Tensor` type, both devices: the same code on CPU and on GPU.

`numax.core.array.Tensor` owns a MAX `DeviceBuffer`, so which device a tensor
lives on is decided by the `DeviceContext` handed to the factory function
and by nothing else. `DeviceContext(api="cpu")` puts it in host memory,
`DeviceContext()` puts it on the accelerator; the type, the shape
parameters, the factory names, and `.view()`'s `TileTensor` type are
identical either way.

It then runs three of the NumPy-named routines on the same device tensor:
`numax.exp`, the comparison `numax.greater`, whose result is a
`DType.bool` tensor written by the launch itself, and the reduction
`numax.stats.sum`, which folds through MAX's `ReduceSum` monoid and brings
back only the scalar. Those took a host copy of the elements and walked
them one at a time until 0.2; `exp[gpu=True]`, `greater[gpu=True]` and
`sum[gpu=True]` now run over the tensor where it already lives, and
`exp(a)` on a device tensor still answers -- on the host, with a line on
`stderr` naming the spelling that would not have.

This example runs one `FloatLike` kernel over the same tensor type on both
devices and checks the results agree elementwise to within one float32 ulp
(they are not bit-identical: `exp` is one of the functions whose host and
device codegen differ in the last place, which is a property of the math
library rather than of anything `numax` does). What differs between the
two halves below is exactly two things -- which context is created, and
whether the walk is `map[gpu=False]` (a host SIMD loop) or `map[gpu=True]`
(one thread per element, launched with `enqueue_function`). Everything
about the tensor itself is shared, which is the point.

Needs real GPU hardware (Metal or CUDA, whichever `DeviceContext` finds),
so this is a local/manual example rather than a CI one -- the same reason
`gaussian_gpu.mojo` is.
"""

from max.gpu.host import DeviceContext

from numax import Plain, exp, gaussian, greater
from numax.core.array import Static, Tensor, linspace, zeros
from numax.core.tensor import map
from numax.stats import sum as tensor_sum

comptime dtype = DType.float32
comptime n = 1024
comptime block_size = 256
comptime num_blocks = (n + block_size - 1) // block_size

comptime TensorType = Static[dtype, n]
comptime LayoutType = TensorType.LayoutType


def gaussian_step[w: Int](x: SIMD[dtype, w]) -> SIMD[dtype, w]:
    """`exp(-x*x)` written once against `FloatLike`, called here at `Plain`."""
    return gaussian(Plain[dtype, w](x)).v


def main() raises:
    # ---- CPU: host context, host walk ----
    var cpu = DeviceContext(api="cpu")
    print("cpu context api:", cpu.api())

    var cpu_xs = linspace[n, dtype](-2.0, 2.0, ctx=cpu)
    var cpu_ys = TensorType(cpu)
    var cpu_xs_view = cpu_xs.view()
    var cpu_ys_view = cpu_ys.view()
    map[step=gaussian_step, width=8](cpu_xs_view, cpu_ys_view)

    # ---- GPU: device context, device launch, same tensor type ----
    var gpu = DeviceContext()
    print("gpu context api:", gpu.api())

    var gpu_xs = linspace[n, dtype](-2.0, 2.0, ctx=gpu)
    var gpu_ys = TensorType(gpu)
    var gpu_xs_view = gpu_xs.view()
    var gpu_ys_view = gpu_ys.view()
    gpu.enqueue_function[
        map[LayoutType=LayoutType, step=gaussian_step, gpu=True]
    ](gpu_xs_view, gpu_ys_view, grid_dim=num_blocks, block_dim=block_size)
    gpu.synchronize()

    # ---- the same values, from two devices ----
    var cpu_values = cpu_ys.to_host()
    var gpu_values = gpu_ys.to_host()
    var max_diff = Scalar[dtype](0)
    for i in range(n):
        var diff = cpu_values[i] - gpu_values[i]
        if diff < 0:
            diff = -diff
        if diff > max_diff:
            max_diff = diff

    for idx in [0, n // 4, n // 2, n - 1]:
        print(
            "i=",
            idx,
            " cpu=",
            cpu_values[idx],
            " gpu=",
            gpu_values[idx],
        )
    # One float32 ulp near 1.0 is ~1.19e-7; `exp`'s host and device
    # implementations differ in the last place, so that -- not equality --
    # is the honest bar for "the same kernel ran on both".
    comptime one_ulp_near_one = Scalar[dtype](1.2e-7)
    print("max |cpu - gpu| =", max_diff)
    print("within one float32 ulp:", max_diff <= one_ulp_near_one)

    # ---- the NumPy-named surface, launched where the tensor lives ----
    var device_exp = exp[gpu=True](gpu_xs).to_host()
    var host_exp = exp(cpu_xs).to_host()
    var max_exp_diff = Scalar[dtype](0)
    for i in range(n):
        var diff = host_exp[i] - device_exp[i]
        if diff < 0:
            diff = -diff
        if diff > max_exp_diff:
            max_exp_diff = diff
    print("max |exp cpu - exp gpu| =", max_exp_diff)

    # ---- a comparison, whose destination is a bool tensor the launch
    # writes on the device: the mask has to come back identical, since a
    # comparison has no last-place freedom the way `exp` does ----
    var gpu_zero = zeros[dtype, n](gpu)
    var cpu_zero = zeros[dtype, n](cpu)
    var device_mask = greater[gpu=True](gpu_xs, gpu_zero).to_host()
    var host_mask = greater(cpu_xs, cpu_zero).to_host()
    var mask_agrees = True
    var positives = 0
    for i in range(n):
        if device_mask[i] != host_mask[i]:
            mask_agrees = False
        if device_mask[i]:
            positives += 1
    print("greater mask positives =", positives, "of", n)
    print("greater mask identical on both devices:", mask_agrees)

    # ---- a reduction, which never downloads the tensor: MAX's `ReduceSum`
    # monoid folds it on the device and only the scalar crosses back. The
    # fold is reassociated on both targets but by different tilings, so a
    # float32 tolerance -- not equality -- is the honest bar ----
    var device_sum = tensor_sum[gpu=True](gpu_xs)
    var host_sum = tensor_sum(cpu_xs)
    var sum_gap = device_sum - host_sum
    if sum_gap < 0:
        sum_gap = -sum_gap
    print("sum cpu =", host_sum, " sum gpu =", device_sum)
    print("|sum cpu - sum gpu| =", sum_gap)
    # `linspace(-2, 2)` sums to zero by symmetry, so the scale to judge the
    # gap against is the sum of magnitudes, about n/2 * 2 = 1024.
    print("sum agrees to 1e-3 of the vector scale:", sum_gap <= 1.0)
