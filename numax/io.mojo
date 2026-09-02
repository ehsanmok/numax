"""Binary save/load and configurable printing for `numax.array.Tensor`.

`docs/parity.md` picks tensor I/O as a genuine `numax` gap: MAX ships
no binary `TileTensor` I/O at all (verified by direct probe -- there is no
`layout`-level save/load, and `max.algorithm.functional` exports only
`elementwise`), and NumPy's own `.npy` format is a Python-ecosystem
interchange format this project has no dependency on in its default
environment (`pixi.toml`'s default feature is `mojo` + `max` only, no
Python/NumPy). So this module is `numax`'s own round-trip format, not an
interchange one -- `Plain`-only, axis 2, the same shape as `numax.array`.

**Format.** 4-byte magic (`"NMX1"`), a length-prefixed UTF-8 dtype name
(e.g. `"float32"` -- this is what `load` checks against the caller's
requested `dtype`, catching a same-width-different-dtype mistake like
`float32` vs `int32` that a bare byte-size check would miss), a 1-byte
rank, `rank` little-endian `Int64` dims, then the raw row-major payload
bytes, copied out through `Tensor.to_host()` so the format is the same
whichever device the tensor lives on. This is a
*typed* load: `dtype`/`dims` are compile-time parameters the caller
supplies (matching every other `numax.array` factory function's
comptime-shape contract), and `load` raises if the file's header doesn't
match them exactly, rather than inferring a shape from the file the way a
dynamically-shaped reader would.

**`print_tensor`, not `print`.** A function named `print` can be defined
here without a parser error (unlike `def std(...)`/`def var(...)`
elsewhere in `numax`, which collide with a keyword and a package name
respectively) -- `print` is an ordinary overloadable builtin, and adding
another overload compiles fine. It is named `print_tensor` anyway, to
avoid a caller ever wondering whether a `print(...)` call in a file that
imports `numax.io` resolved to this overload or the builtin one.
"""

from std.sys.info import size_of

from max.gpu.host import DeviceContext

from .array import Tensor


comptime _MAGIC = "NMX1"


def _append_i64_le(mut buf: List[UInt8], v: Int64):
    var uv = UInt64(v)
    for i in range(8):
        buf.append(UInt8((uv >> UInt64(8 * i)) & 0xFF))


def _read_i64_le(data: List[UInt8], offset: Int) -> Int64:
    var uv = UInt64(0)
    for i in range(8):
        uv |= UInt64(data[offset + i]) << UInt64(8 * i)
    return Int64(uv)


def save[
    dtype: DType, *dims: Int
](a: Tensor[dtype, *dims], path: String) raises:
    """Write `a` to `path` in `numax`'s own binary tensor format.

    See this module's own docstring for the format and for why this isn't
    NumPy's `.npy`.
    """
    comptime rank = dims.__len__()
    var header = List[UInt8]()
    for c in _MAGIC.as_bytes():
        header.append(c)
    var dtype_name = String(dtype).as_bytes()
    header.append(UInt8(len(dtype_name)))
    for c in dtype_name:
        header.append(c)
    header.append(UInt8(rank))
    comptime for i in range(rank):
        _append_i64_le(header, Int64(dims[i]))

    var f = open(path, "w")
    f.write_bytes(Span(header))

    comptime nbytes = Tensor[dtype, *dims].num_elements * size_of[
        Scalar[dtype]
    ]()
    var values = a.to_host()
    var byte_ptr = values.unsafe_ptr().unsafe_bitcast[UInt8]()
    var payload = Span[UInt8, origin_of(values)](
        unsafe_ptr=byte_ptr, length=nbytes
    )
    f.write_bytes(payload)
    f.close()


def load[
    dtype: DType, *dims: Int
](ctx: DeviceContext, path: String) raises -> Tensor[dtype, *dims]:
    """Read a tensor written by `save`, onto `ctx`'s device.

    Raises if the file's dtype name, rank, or shape doesn't exactly match
    the `dtype`/`dims` requested here. The `DeviceContext` is what every
    `numax.array` root factory takes, for the same reason: the bytes have
    to land on a device, and the file does not name one.
    """
    var f = open(path, "r")
    var data = f.read_bytes()
    f.close()

    comptime rank = dims.__len__()
    var magic_bytes = _MAGIC.as_bytes()
    if len(data) < 4:
        raise Error("numax.io.load: file too short to contain a header")
    for i in range(4):
        if data[i] != magic_bytes[i]:
            raise Error(
                "numax.io.load: bad magic bytes -- not a numax tensor file"
            )

    var name_len = Int(data[4])
    var expected_name = String(dtype).as_bytes()
    if name_len != len(expected_name):
        raise Error("numax.io.load: dtype name length mismatch")
    var offset = 5
    for i in range(name_len):
        if data[offset + i] != expected_name[i]:
            raise Error(
                "numax.io.load: dtype mismatch -- file was saved with a"
                " different dtype"
            )
    offset += name_len

    var file_rank = Int(data[offset])
    offset += 1
    if file_rank != rank:
        raise Error("numax.io.load: rank mismatch")

    comptime for i in range(rank):
        var file_dim = Int(_read_i64_le(data, offset))
        offset += 8
        if file_dim != dims[i]:
            raise Error("numax.io.load: shape mismatch")

    comptime nbytes = Tensor[dtype, *dims].num_elements * size_of[
        Scalar[dtype]
    ]()
    if len(data) - offset != nbytes:
        raise Error("numax.io.load: payload size doesn't match the header")

    var out_storage = List[Scalar[dtype]](
        length=Tensor[dtype, *dims].num_elements, fill=0
    )
    var dst_ptr = out_storage.unsafe_ptr().unsafe_bitcast[UInt8]()
    for i in range(nbytes):
        dst_ptr[unsafe_offset=i] = data[offset + i]
    return Tensor[dtype, *dims](ctx, out_storage^)


def print_tensor[
    dtype: DType, *dims: Int
](
    a: Tensor[dtype, *dims],
    *,
    precision: Int = 4,
    threshold: Int = 1000,
    edge_items: Int = 3,
) raises:
    """Print `a`'s flat contents, NumPy-`arrayprint`-shaped but simplified.

    Every element is printed with `precision` digits after the decimal
    point. If `a` has more than `threshold` elements, only the first and
    last `edge_items` are printed, with a `...` in between -- the same
    truncation NumPy's own default printer applies to large arrays, though
    without NumPy's line-wrapping (`linewidth`): this prints one flat,
    comma-separated line rather than wrapping to a terminal width. Built on
    `_format_tensor` below so the exact string this prints is also directly
    testable (`tests/test_io.mojo` checks it against frozen fixtures).
    """
    print(_format_tensor(a, precision, threshold, edge_items))


def _format_tensor[
    dtype: DType, *dims: Int
](
    a: Tensor[dtype, *dims], precision: Int, threshold: Int, edge_items: Int
) raises -> String:
    comptime n = Tensor[dtype, *dims].num_elements
    var values = a.to_host()
    var out = String("[")
    if n <= threshold:
        for i in range(n):
            out += _format_one(values[i], precision)
            if i != n - 1:
                out += ", "
    else:
        for i in range(edge_items):
            out += _format_one(values[i], precision)
            out += ", "
        out += "..."
        for i in range(n - edge_items, n):
            out += ", "
            out += _format_one(values[i], precision)
    out += "]"
    return out^


def _round_to[dtype: DType](x: Scalar[dtype], precision: Int) -> Scalar[dtype]:
    """Round `x` to `precision` decimal digits, half-away-from-zero.

    An approximation, not `Python`'s exact `%.4f` truncation -- the
    rounded value is still a `dtype` float, and its default `String`
    conversion (shortest round-trippable representation) occasionally
    shows one digit more or fewer than `precision` when the rounded value
    itself isn't exactly representable in binary. Good enough for a
    convenience printer; not a claim of exact fixed-point formatting.
    """
    var scale = Scalar[dtype](10.0) ** Scalar[dtype](precision)
    var sign = Scalar[dtype](1) if x >= 0 else Scalar[dtype](-1)
    var shifted = x * scale + sign * Scalar[dtype](0.5)
    return Scalar[dtype](Int64(shifted)) / scale


def _format_one[dtype: DType](x: Scalar[dtype], precision: Int) -> String:
    return String(_round_to(x, precision))
