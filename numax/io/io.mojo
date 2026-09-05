"""Binary save/load and configurable printing for `numax.core.array.Tensor`.

`docs/parity.md` picks tensor I/O as a genuine `numax` gap: MAX ships
no binary `TileTensor` I/O at all (verified by direct probe -- there is no
`layout`-level save/load, and `max.algorithm.functional` exports only
`elementwise`). This module is `numax`'s own round-trip format -- for
NumPy interchange, `npy.mojo`'s `numpy.save`/`numpy.load` read and write
`.npy` directly (no Python and no NumPy: the format is self-contained).
`NMX1` is the better choice between numax programs, since it carries the
dtype name in full and has no Python literal to parse. `Plain`-only, axis
2, the same shape as `numax.core.array`.

**Format.** 4-byte magic (`"NMX1"`), a length-prefixed UTF-8 dtype name
(e.g. `"float32"` -- this is what `nmx.load` checks against the caller's
requested `dtype`, catching a same-width-different-dtype mistake like
`float32` vs `int32` that a bare byte-size check would miss), a 1-byte
rank, `rank` little-endian `Int64` dims, then the raw row-major payload
bytes, copied out through `Tensor.to_host()` so the format is the same
whichever device the tensor lives on. This is a
*typed* load: `dtype`/`dims` are compile-time parameters the caller
supplies (matching every other `numax.core.array` factory function's
comptime-shape contract), and `nmx.load` raises if the file's header doesn't
match them exactly, rather than inferring a shape from the file the way a
dynamically-shaped reader would.

**No printing lives here.** `Tensor` conforms to `Writable`, so `print(a)`
is the builtin doing its ordinary job, and `a.format(precision=8)` is the
same output with the truncation and precision under the caller's control.
Neither needs an import from this module.
"""

from std.sys.info import size_of

from max.gpu.host import DeviceContext

from layout.tile_layout import TensorLayout
from ..core.array import Shaped, Tensor, _context


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


struct nmx:
    """numax's own `NMX1` binary tensor format.

    `from numax.io import nmx` then `nmx.save(a, path)` / `nmx.load`. The
    format is named in the call, because the alternative -- an unprefixed
    `save` -- silently meant the one format NumPy cannot read. Reach for
    `numax.io.numpy` to interchange, and for this to round-trip between
    numax programs, where carrying the full dtype name and having no Python
    literal to parse is worth more.
    """

    @staticmethod
    def save[
        dtype: DType, LayoutType: TensorLayout
    ](a: Tensor[dtype, LayoutType], path: String) raises:
        """Write `a` to `path` in `numax`'s own binary tensor format.

        See this module's own docstring for the format and for why this isn't
        NumPy's `.npy`.
        """
        comptime rank = LayoutType.rank
        var header = List[UInt8]()
        for c in _MAGIC.as_bytes():
            header.append(c)
        var dtype_name = String(dtype).as_bytes()
        header.append(UInt8(len(dtype_name)))
        for c in dtype_name:
            header.append(c)
        header.append(UInt8(rank))
        comptime for i in range(rank):
            _append_i64_le(header, Int64(a.dim[i]()))

        var f = open(path, "w")
        f.write_bytes(Span(header))

        var nbytes = a.size() * size_of[Scalar[dtype]]()
        var values = a.to_host()
        var byte_ptr = values.unsafe_ptr().unsafe_bitcast[UInt8]()
        var payload = Span[UInt8, origin_of(values)](
            unsafe_ptr=byte_ptr, length=nbytes
        )
        f.write_bytes(payload)
        f.close()

    @staticmethod
    def load[
        dtype: DType, *dims: Int
    ](path: String, ctx: Optional[DeviceContext] = None) raises -> Shaped[
        dtype, *dims
    ]:
        """Read a tensor written by `nmx.save`, onto `ctx`'s device.

        Raises if the file's dtype name, rank, or shape doesn't exactly match
        the `dtype`/`dims` requested here. The `DeviceContext` is what every
        `numax.core.array` root factory takes, for the same reason: the bytes have
        to land on a device, and the file does not name one.
        """
        var f = open(path, "r")
        var data = f.read_bytes()
        f.close()

        comptime rank = dims.__len__()
        var magic_bytes = _MAGIC.as_bytes()
        if len(data) < 4:
            raise Error("numax.io.nmx.load: file too short to contain a header")
        for i in range(4):
            if data[i] != magic_bytes[i]:
                raise Error(
                    "numax.io.nmx.load: bad magic bytes -- not a numax tensor"
                    " file"
                )

        var name_len = Int(data[4])
        var expected_name = String(dtype).as_bytes()
        if name_len != len(expected_name):
            raise Error("numax.io.nmx.load: dtype name length mismatch")
        var offset = 5
        for i in range(name_len):
            if data[offset + i] != expected_name[i]:
                raise Error(
                    "numax.io.nmx.load: dtype mismatch -- file was saved with a"
                    " different dtype"
                )
        offset += name_len

        var file_rank = Int(data[offset])
        offset += 1
        if file_rank != rank:
            raise Error("numax.io.nmx.load: rank mismatch")

        comptime for i in range(rank):
            var file_dim = Int(_read_i64_le(data, offset))
            offset += 8
            if file_dim != dims[i]:
                raise Error("numax.io.nmx.load: shape mismatch")

        comptime nbytes = Shaped[dtype, *dims].num_elements * size_of[
            Scalar[dtype]
        ]()
        if len(data) - offset != nbytes:
            raise Error(
                "numax.io.nmx.load: payload size doesn't match the header"
            )

        var out_storage = List[Scalar[dtype]](
            length=Shaped[dtype, *dims].num_elements, fill=0
        )
        var dst_ptr = out_storage.unsafe_ptr().unsafe_bitcast[UInt8]()
        for i in range(nbytes):
            dst_ptr[unsafe_offset=i] = data[offset + i]
        return Shaped[dtype, *dims](_context(ctx), out_storage^)
