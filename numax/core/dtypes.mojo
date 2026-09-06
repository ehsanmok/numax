"""Short names for every `DType` the toolchain has.

This module declares no tier: it holds names, not code.

A dtype is not a conformer -- the same name works at every layer, which is
the point: `linspace[5, f32](...)` and `Shaped[f32, 4, 4](ctx)` on the
tensor side, `Plain[f32]` and `Dual[Plain[f32]]` on the kernel side.
`Plain`'s `width` defaults to 1, so `Plain[f32]` is the single-lane scalar;
name `Plain[f32, 8]` for a vector width.

The naming rule, so a dtype not listed here can still be guessed at: `f` is
a binary float and `bf` a bfloat, `i` is signed and `u` unsigned, and the
digits are the bit width. The float8 encodings keep their upstream suffix
(`f8e4m3fn`), since the exponent/mantissa split is the whole difference
between them and shortening it further would only hide which one is which.

Only `f16`, `bf16`, `f32` and `f64` satisfy `FloatLike`'s
`dtype.is_floating_point()` requirement in a way the hardware computes on
directly. The float8 encodings are storage formats -- MAX kernels convert
through a wider type to do arithmetic -- and the integer and boolean names
are here for tensors (index buffers, masks) rather than for kernels.
"""

comptime f16 = DType.float16
"""`DType.float16`, IEEE half precision."""

comptime bf16 = DType.bfloat16
"""`DType.bfloat16`: `f32`'s exponent range at `f16`'s width."""

comptime f32 = DType.float32
"""`DType.float32`, the default working precision for most of numax."""

comptime f64 = DType.float64
"""`DType.float64`."""

comptime f8e3m4 = DType.float8_e3m4
"""`DType.float8_e3m4`."""

comptime f8e4m3fn = DType.float8_e4m3fn
"""`DType.float8_e4m3fn`: finite only, no infinities."""

comptime f8e4m3fnuz = DType.float8_e4m3fnuz
"""`DType.float8_e4m3fnuz`: finite only, unsigned zero."""

comptime f8e5m2 = DType.float8_e5m2
"""`DType.float8_e5m2`: more exponent, less mantissa than `f8e4m3fn`."""

comptime f8e5m2fnuz = DType.float8_e5m2fnuz
"""`DType.float8_e5m2fnuz`: finite only, unsigned zero."""

comptime i8 = DType.int8
"""`DType.int8`."""

comptime i16 = DType.int16
"""`DType.int16`."""

comptime i32 = DType.int32
"""`DType.int32`."""

comptime i64 = DType.int64
"""`DType.int64`, what `numax.core.sorting`'s index buffers hold."""

comptime u8 = DType.uint8
"""`DType.uint8`."""

comptime u16 = DType.uint16
"""`DType.uint16`."""

comptime u32 = DType.uint32
"""`DType.uint32`."""

comptime u64 = DType.uint64
"""`DType.uint64`."""

comptime bool = DType.bool
"""`DType.bool`, what a `numax.core.logic` comparison returns.

Not a shadow of anything: Mojo's boolean type is `Bool`, and there is no
`bool` builtin for this to displace. A tensor at this dtype has to come
from a comparison or a `List`, since the zero-fill constructor every
factory routes through is unavailable here.
"""
