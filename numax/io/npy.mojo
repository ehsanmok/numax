"""NumPy `.npy` interchange: read what `numpy.save` wrote, write what
`numpy.load` reads.

`numax.io`'s own `NMX1` format (`io.mojo`) round-trips a `Tensor` between
numax programs. This module is the other half: the format a user arrives
with. Someone porting a NumPy program to Mojo already has their data in
`.npy` files, and re-exporting all of it is not a reasonable first step --
so `numpy.load` reads those files directly, and `numpy.save` writes files
`numpy.load` opens, with no conversion step on either side.

No Python and no NumPy is involved. `.npy` is a documented, self-contained
binary format, so this is a header parse and a payload copy, available in
the default `mojo` + `max` environment.

**Format** (version 1.0, what `numpy.save` writes unless the header
exceeds 64 KiB): the 6-byte magic `\\x93NUMPY`, a major and a minor version
byte, a little-endian `UInt16` header length, then that many bytes of
ASCII holding a Python dict literal plus padding, for example

```text
{'descr': '<f4', 'fortran_order': False, 'shape': (2, 3), }<spaces>\\n
```

padded with spaces so the whole header (the 10-byte prelude included) is a
multiple of 64 bytes and ends in a newline. The raw C-order payload
follows. `numpy.save` reproduces that layout byte for byte -- NumPy's key
order, its `, }` before the padding, its 64-byte alignment -- so a file
this writes is indistinguishable from one `numpy.save` wrote, which is
what `tests/io/test_npy.mojo` asserts against real NumPy bytes.

**A typed load, like `nmx.load`.** `dtype` and `dims` are compile-time
parameters the caller supplies, matching every other `numax.core.array`
factory, and `numpy.load` raises if the file disagrees. It is not a
shape-inferring reader: a numax `Tensor`'s shape lives in its type.

**What is rejected, loudly.** `fortran_order: True` (column-major, so the
payload order is not numax's), a big-endian `descr` such as `'>f4'`
(nothing in numax byte-swaps), a `descr` naming a dtype other than the one
requested, and `.npz` archives, which are zip containers rather than
`.npy` files -- `numpy.savez` output has to be unzipped, or re-saved per
array with `numpy.save`, first. `bfloat16` and the float8 formats have no
NumPy dtype at all, so both directions raise for them.

Tier 2, `Plain`-only, host-side, like the rest of `numax.io`.
"""

from std.sys.info import size_of

from max.gpu.host import DeviceContext

from ..core.array import Tensor, _context


# The magic's first byte is 0x93, which is not valid UTF-8 on its own: a
# Mojo `String` literal would encode it as two bytes (0xC2 0x93), so the
# byte and the ASCII tail are kept apart and assembled by `_magic_bytes`.
comptime _NPY_MAGIC_BYTE = UInt8(0x93)
comptime _NPY_MAGIC_TAIL = "NUMPY"

# NumPy's `ARRAY_ALIGN`: the prelude plus the dict plus its padding is a
# multiple of this, so the payload starts on a 64-byte boundary.
comptime _ALIGN = 64


def _magic_bytes() -> List[UInt8]:
    """The 6-byte `.npy` magic: `0x93` then `NUMPY`."""
    var out = List[UInt8](capacity=6)
    out.append(_NPY_MAGIC_BYTE)
    for c in _NPY_MAGIC_TAIL.as_bytes():
        out.append(c)
    return out^


def _descr[dtype: DType]() -> String:
    """NumPy's `descr` string for `dtype`, e.g. `'<f4'` for `float32`.

    Single-byte types carry `'|'` rather than `'<'` (byte order is
    meaningless for them), which is what `numpy.save` writes, so a header
    built from this stays byte-identical to NumPy's. Returns `""` for a
    `DType` NumPy has no equivalent for.
    """

    comptime if dtype == DType.float16:
        return "<f2"
    elif dtype == DType.float32:
        return "<f4"
    elif dtype == DType.float64:
        return "<f8"
    elif dtype == DType.int8:
        return "|i1"
    elif dtype == DType.int16:
        return "<i2"
    elif dtype == DType.int32:
        return "<i4"
    elif dtype == DType.int64:
        return "<i8"
    elif dtype == DType.uint8:
        return "|u1"
    elif dtype == DType.uint16:
        return "<u2"
    elif dtype == DType.uint32:
        return "<u4"
    elif dtype == DType.uint64:
        return "<u8"
    elif dtype == DType.bool:
        return "|b1"
    else:
        # No NumPy dtype corresponds to this one (`bfloat16`, the float8
        # formats). Empty is the sentinel both entry points check for,
        # rather than a compile-time constraint, so the error message names
        # the function the caller actually called.
        return ""


def _shape_literal[*dims: Int]() -> String:
    """`dims` as Python's tuple literal: `(3,)` at rank 1, `(2, 3)` above.

    The rank-1 trailing comma is not cosmetic: `(3)` is an `int` in Python,
    not a tuple, and `numpy.load` evaluates this text.
    """
    comptime rank = dims.__len__()
    var out = String("(")

    comptime for i in range(rank):
        out += String(dims[i])
        if i != rank - 1:
            out += ", "
    if rank == 1:
        out += ","
    out += ")"
    return out^


struct numpy:
    """NumPy `.npy` interchange, namespaced so the format is the namespace.

    `from numax.io import numpy` then `numpy.save(a, path)` / `numpy.load`,
    reading and writing exactly what `numpy.save`/`numpy.load` do. A struct
    with two static methods rather than two prefixed free functions: the
    subpackage tree already exists, and `npy_save` was that tree spelled
    into an identifier.
    """

    @staticmethod
    def save[
        dtype: DType, *dims: Int
    ](a: Tensor[dtype, *dims], path: String) raises:
        """Write `a` to `path` as a NumPy `.npy` file (format version 1.0).

        The bytes are what `numpy.save` would have written for the same array,
        so `numpy.load(path)` returns it with the right dtype and shape and no
        conversion in between. The payload comes out through
        `Tensor.to_host()`, so the file is the same whichever device the tensor
        lives on.

        Raises if `dtype` has no NumPy equivalent (`bfloat16`, the float8
        formats).
        """
        comptime descr = _descr[dtype]()
        comptime shape = _shape_literal[*dims]()
        if descr == "":
            raise Error(
                String(
                    "numax.io.numpy.save: ",
                    dtype,
                    " has no NumPy dtype, so it has no .npy representation",
                )
            )

        # NumPy's key order and its `, }` terminator, so a byte-for-byte
        # comparison against `numpy.save` output holds.
        var dict_text = String(
            "{'descr': '",
            descr,
            "', 'fortran_order': False, 'shape': ",
            shape,
            ", }",
        )

        # `numpy.lib.format._write_array_header`: pad so 6 magic + 2 version +
        # 2 length + dict + padding + newline is a multiple of 64. The `+ 1` is
        # the newline; padding a full 64 bytes when the dict already lands on
        # the boundary is what NumPy does too.
        var pad = _ALIGN - ((10 + dict_text.byte_length() + 1) % _ALIGN)
        var header = String(dict_text)
        for _ in range(pad):
            header += " "
        header += "\n"

        var out = _magic_bytes()
        out.append(UInt8(1))  # major version
        out.append(UInt8(0))  # minor version
        var header_len = header.byte_length()
        out.append(UInt8(header_len & 0xFF))
        out.append(UInt8((header_len >> 8) & 0xFF))
        for c in header.as_bytes():
            out.append(c)

        var f = open(path, "w")
        f.write_bytes(Span(out))

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

    @staticmethod
    def load[
        dtype: DType, *dims: Int
    ](path: String, ctx: Optional[DeviceContext] = None) raises -> Tensor[
        dtype, *dims
    ]:
        """Read a NumPy `.npy` file written by `numpy.save`, onto `ctx`'s device.

        Raises if the file's `descr` names a different dtype, if its `shape`
        doesn't match `dims` exactly, if it is Fortran-ordered or big-endian,
        if the payload length disagrees with the header, or if it is not an
        `.npy` file at all -- a `.npz` archive lands here as bad magic bytes,
        since it is a zip container.

        The `DeviceContext` is what every `numax.core.array` root factory
        takes, for the same reason: the bytes have to land on a device, and the
        file does not name one.
        """
        var f = open(path, "r")
        var data = f.read_bytes()
        f.close()

        var magic = _magic_bytes()
        if len(data) < 10:
            raise Error("numax.io.numpy.load: file too short to be a .npy file")
        for i in range(len(magic)):
            if data[i] != magic[i]:
                raise Error(
                    "numax.io.numpy.load: bad magic bytes -- not a .npy file. A"
                    " .npz archive is a zip container, not an .npy file: unzip"
                    " it, or re-save each array with numpy.save"
                )

        var major = Int(data[6])
        if major != 1 and major != 2:
            raise Error(
                "numax.io.numpy.load: unsupported .npy format version -- only"
                " the 1.x and 2.x headers are understood"
            )

        # 1.x stores the header length as a little-endian `UInt16`, 2.x as a
        # `UInt32`; the dict text that follows is identical.
        var header_len: Int
        var header_start: Int
        if major == 1:
            header_len = Int(data[8]) | (Int(data[9]) << 8)
            header_start = 10
        else:
            if len(data) < 12:
                raise Error("numax.io.numpy.load: truncated .npy 2.x header")
            header_len = (
                Int(data[8])
                | (Int(data[9]) << 8)
                | (Int(data[10]) << 16)
                | (Int(data[11]) << 24)
            )
            header_start = 12
        if len(data) < header_start + header_len:
            raise Error("numax.io.numpy.load: truncated .npy header")

        var header = String(
            StringSlice(
                unsafe_from_utf8=Span(data)[
                    header_start : header_start + header_len
                ]
            )
        )

        comptime expected_descr = _descr[dtype]()
        if expected_descr == "":
            raise Error(
                String(
                    "numax.io.numpy.load: ",
                    dtype,
                    " has no NumPy dtype, so no .npy file can hold it",
                )
            )
        var file_descr = _dict_value(header, "descr")
        if not _descr_matches(file_descr, expected_descr):
            if file_descr.startswith(">"):
                raise Error(
                    String(
                        "numax.io.numpy.load: big-endian .npy (descr '",
                        file_descr,
                        (
                            "') -- nothing in numax byte-swaps; re-save it"
                            " native with"
                            " arr.astype(arr.dtype.newbyteorder('<'))"
                        ),
                    )
                )
            raise Error(
                String(
                    "numax.io.numpy.load: dtype mismatch -- the file holds '",
                    file_descr,
                    "' but '",
                    expected_descr,
                    "' was requested",
                )
            )

        if _dict_value(header, "fortran_order") != "False":
            raise Error(
                "numax.io.numpy.load: Fortran-ordered .npy -- numax tensors are"
                " row-major; re-save with numpy.ascontiguousarray(arr)"
            )

        comptime rank = dims.__len__()
        var file_dims = _parse_shape(_dict_value(header, "shape"))
        if len(file_dims) != rank:
            raise Error(
                String(
                    "numax.io.numpy.load: rank mismatch -- the file is rank ",
                    len(file_dims),
                    ", rank ",
                    rank,
                    " was requested",
                )
            )

        comptime for i in range(rank):
            if file_dims[i] != dims[i]:
                raise Error("numax.io.numpy.load: shape mismatch")

        var offset = header_start + header_len
        comptime nbytes = Tensor[dtype, *dims].num_elements * size_of[
            Scalar[dtype]
        ]()
        if len(data) - offset != nbytes:
            raise Error(
                "numax.io.numpy.load: payload size doesn't match the header"
            )

        var out_storage = List[Scalar[dtype]](
            length=Tensor[dtype, *dims].num_elements, fill=0
        )
        var dst_ptr = out_storage.unsafe_ptr().unsafe_bitcast[UInt8]()
        for i in range(nbytes):
            dst_ptr[unsafe_offset=i] = data[offset + i]
        return Tensor[dtype, *dims](_context(ctx), out_storage^)


def _descr_matches(file_descr: String, expected: String) -> Bool:
    """Whether `file_descr` names the same dtype as `expected`.

    `'='` is NumPy's "native byte order", which on every platform numax
    targets is little-endian, so `'=f4'` and `'<f4'` are the same file.
    Single-byte types are spelled `'|u1'` but `'<u1'` is legal too, and
    both mean the same thing, so the byte-order character is compared only
    when it actually carries information.
    """
    if file_descr == expected:
        return True
    var fb = file_descr.as_bytes()
    var eb = expected.as_bytes()
    if len(fb) != 3 or len(eb) != 3:
        return False
    if fb[1] != eb[1] or fb[2] != eb[2]:
        return False
    comptime _LT = UInt8(ord("<"))
    comptime _EQ = UInt8(ord("="))
    comptime _PIPE = UInt8(ord("|"))
    comptime _ONE = UInt8(ord("1"))
    var file_order = fb[0]
    var native = file_order == _LT or file_order == _EQ
    # For a one-byte itemsize, `|` says "order does not apply".
    if fb[2] == _ONE:
        return native or file_order == _PIPE
    return native


def _dict_value(header: String, key: String) raises -> String:
    """The value for `key` in an `.npy` header dict, quotes stripped.

    Enough of a reader for the three keys the format fixes (`descr`,
    `fortran_order`, `shape`), not a Python literal parser: the value runs
    from the colon after the key to the next comma at bracket depth zero,
    so the comma inside `'shape': (2, 3)` does not end it.
    """
    var needle = String("'", key, "'")
    var start = header.find(needle)
    if start < 0:
        raise Error(
            String("numax.io.numpy.load: .npy header has no '", key, "' key")
        )
    var colon = header.find(":", start)
    if colon < 0:
        raise Error(String("numax.io.numpy.load: malformed '", key, "' entry"))

    comptime _OPEN_PAREN = UInt8(ord("("))
    comptime _OPEN_BRACKET = UInt8(ord("["))
    comptime _OPEN_BRACE = UInt8(ord("{"))
    comptime _CLOSE_PAREN = UInt8(ord(")"))
    comptime _CLOSE_BRACKET = UInt8(ord("]"))
    comptime _CLOSE_BRACE = UInt8(ord("}"))
    comptime _COMMA = UInt8(ord(","))
    comptime _NEWLINE = UInt8(ord("\n"))
    comptime _SPACE = UInt8(ord(" "))
    comptime _QUOTE = UInt8(ord("'"))
    comptime _DQUOTE = UInt8(ord('"'))

    var bytes = header.as_bytes()
    var out = List[UInt8]()
    var depth = 0
    for i in range(colon + 1, len(bytes)):
        var c = bytes[i]
        if c == _OPEN_PAREN or c == _OPEN_BRACKET or c == _OPEN_BRACE:
            depth += 1
        elif c == _CLOSE_PAREN or c == _CLOSE_BRACKET or c == _CLOSE_BRACE:
            if depth == 0:
                break
            depth -= 1
        elif (c == _COMMA or c == _NEWLINE) and depth == 0:
            break
        if c == _SPACE or c == _QUOTE or c == _DQUOTE:
            continue
        out.append(c)
    return String(StringSlice(unsafe_from_utf8=Span(out)))


def _parse_shape(text: String) raises -> List[Int]:
    """The dims in an `.npy` header's `shape` tuple text, e.g. `(2, 3)`.

    `()` -- NumPy's rank-0 scalar -- parses to an empty list, which then
    fails the rank check against any numax `Tensor`, whose rank is at
    least 1.
    """
    comptime _ZERO = UInt8(ord("0"))
    comptime _NINE = UInt8(ord("9"))

    var out = List[Int]()
    var digits = List[UInt8]()
    var bytes = text.as_bytes()
    for i in range(len(bytes)):
        var c = bytes[i]
        if c >= _ZERO and c <= _NINE:
            digits.append(c)
        elif len(digits) > 0:
            out.append(Int(String(StringSlice(unsafe_from_utf8=Span(digits)))))
            digits.clear()
    if len(digits) > 0:
        out.append(Int(String(StringSlice(unsafe_from_utf8=Span(digits)))))
    return out^
