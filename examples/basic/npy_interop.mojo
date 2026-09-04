"""`numax.io.numpy.load`/`numpy.save`: NumPy's `.npy`, read and written directly.

The realistic first step in porting a NumPy program is not re-exporting
the data -- it is opening the `.npy` files that already exist. This example
does the round trip from Mojo's side:

1. Write a `Tensor` as `.npy` with `numpy.save`. The bytes are what
   `numpy.save` would have written for the same array, so `numpy.load`
   opens it with the right dtype and shape.
2. Read it back with `numpy.load` and check every element survived
   bit-for-bit.
3. Show the header the file actually carries, which is the interchange
   contract: `{'descr': '<f4', 'fortran_order': False, 'shape': (2, 3), }`.
4. Do the same at `float64` and `int32`, since `descr` -- not a guess --
   is what decides how the payload is read.

No Python and no NumPy runs here: `.npy` is a self-contained binary
format, so this works in the default `mojo` + `max` environment. To see
the other half from Python, in an environment that has NumPy:

```python
import numpy as np
print(np.load("/tmp/numax_example_f32.npy"))     # what step 1 wrote
np.save("/tmp/from_numpy.npy", np.arange(6, dtype=np.float32).reshape(2, 3))
```

and that second file loads here with
`numpy.load[DType.float32, 2, 3]("/tmp/from_numpy.npy", ctx=ctx)`.

`numax.io.nmx.save`/`nmx.load` remain the choice for numax-to-numax round trips:
that format (`NMX1`) carries the dtype name in full and has no Python
literal to parse.
"""

from max.gpu.host import DeviceContext

from numax.core.array import Tensor
from numax.io import numpy

comptime N_ROWS = 2
comptime N_COLS = 3


def _header_of(path: String) raises -> String:
    """The `.npy` header text of `path`, for display.

    Format version 1.0 puts a little-endian `UInt16` header length at
    bytes 8-9, after the 6-byte magic and the two version bytes.
    """
    var f = open(path, "r")
    var data = f.read_bytes()
    f.close()
    var header_len = Int(data[8]) | (Int(data[9]) << 8)
    var text = String(
        StringSlice(unsafe_from_utf8=Span(data)[10 : 10 + header_len])
    )
    return String(text.strip())


def main() raises:
    var ctx = DeviceContext(api="cpu")

    # --- float32, rank 2 --------------------------------------------------
    var storage = List[Scalar[DType.float32]](capacity=N_ROWS * N_COLS)
    for i in range(N_ROWS * N_COLS):
        storage.append(Scalar[DType.float32](Float64(i) * 0.5 - 1.0))
    var xs = Tensor[DType.float32, N_ROWS, N_COLS](ctx, storage^)

    var f32_path = String("/tmp/numax_example_f32.npy")
    numpy.save(xs, f32_path)
    print("wrote", f32_path)
    print("  header:", _header_of(f32_path))

    var loaded = numpy.load[DType.float32, N_ROWS, N_COLS](f32_path, ctx=ctx)
    print("  values: ", end="")
    print(loaded)

    var original = xs.to_host()
    var values = loaded.to_host()
    var identical = True
    for i in range(N_ROWS * N_COLS):
        if values[i].to_bits() != original[i].to_bits():
            identical = False
    print("  bit-identical round trip:", identical)

    # --- float64 and int32: `descr` is what changes -----------------------
    var f64_storage = List[Scalar[DType.float64]](capacity=4)
    for i in range(4):
        f64_storage.append(Scalar[DType.float64](Float64(i) / 3.0))
    var f64_path = String("/tmp/numax_example_f64.npy")
    numpy.save(Tensor[DType.float64, 4](ctx, f64_storage^), f64_path)
    print("wrote", f64_path)
    print("  header:", _header_of(f64_path))
    var f64_back = numpy.load[DType.float64, 4](f64_path, ctx=ctx)
    print("  values: ", end="")
    print(f64_back.format(precision=6))

    var i32_storage = List[Scalar[DType.int32]](capacity=4)
    for i in range(4):
        i32_storage.append(Scalar[DType.int32](i * i))
    var i32_path = String("/tmp/numax_example_i32.npy")
    numpy.save(Tensor[DType.int32, 2, 2](ctx, i32_storage^), i32_path)
    print("wrote", i32_path)
    print("  header:", _header_of(i32_path))
    var i32_back = numpy.load[DType.int32, 2, 2](i32_path, ctx=ctx)
    var i32_values = i32_back.to_host()
    print(
        "  values: [",
        i32_values[0],
        i32_values[1],
        i32_values[2],
        i32_values[3],
        "]",
    )

    # --- the load is typed: a wrong dtype or shape raises -----------------
    # `dtype`/`dims` are compile-time parameters, so `numpy.load` checks the
    # file against what the caller asked for instead of inferring a shape.
    try:
        _ = numpy.load[DType.int32, N_ROWS, N_COLS](f32_path, ctx=ctx)
        print("unreachable: loading a float32 file as int32 should raise")
    except e:
        print("asking for the wrong dtype raises:", e)

    try:
        _ = numpy.load[DType.float32, 6](f32_path, ctx=ctx)
        print("unreachable: loading a (2, 3) file as (6,) should raise")
    except e:
        print("asking for the wrong shape raises:", e)
