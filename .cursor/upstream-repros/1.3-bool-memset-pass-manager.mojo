from max.gpu.host import DeviceContext


def main() raises:
    var ctx = DeviceContext(api="cpu")
    var buf = ctx.enqueue_create_buffer[DType.bool](16)
    ctx.enqueue_memset(buf, Scalar[DType.bool](0))
    ctx.synchronize()
    print("filled 16 bools")
