"""`std.math.sin` at float32 on an NVIDIA target lowers to
`sin.approx.ftz.f32`: absolute error about 1e-6, and a tiny argument can
return exactly 0 (`sin(1e-10)` is 0, `sin(1e-5)` is 0.75% off). The host
returns the correctly rounded value. Float64 `sin` is unsupported on the
device. `cos` (~1.1e-6 absolute) and `log` (~2.9e-5 relative) are similarly
approximate at float32.

Consequence downstream: `sin(eps) / eps` is 0 and `ln|sin(pi x)|` is
`-inf` at integer `x` only on the device. Expected: either a precise
default, or a documented fast variant spelled separately.
"""

from std.math import sin

from max.gpu.host import DeviceContext

comptime dtype = DType.float32


def kernel(
    xs: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    ys: UnsafePointer[Scalar[dtype], MutAnyOrigin],
):
    ys[0] = sin(xs[0])
    ys[1] = sin(xs[1])


def main() raises:
    var ctx = DeviceContext()
    var xs = ctx.enqueue_create_buffer[dtype](2)
    var ys = ctx.enqueue_create_buffer[dtype](2)
    with xs.map_to_host() as h:
        h[0] = 1e-10
        h[1] = 1e-5
    ctx.enqueue_function[kernel](xs, ys, grid_dim=1, block_dim=1)
    with ys.map_to_host() as h:
        print("device sin(1e-10) =", h[0], " host", sin(Float32(1e-10)))
        print("device sin(1e-5)  =", h[1], " host", sin(Float32(1e-5)))
