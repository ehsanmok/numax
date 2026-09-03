"""Tests for `numax.io.npy_save`/`npy_load` against real NumPy output.

Two directions, both pinned to bytes NumPy actually produced rather than
to this module's own idea of the format:

- **Reading.** `_write_numpy_bytes` reconstructs, byte for byte, files
  `numpy.save` wrote (the headers and payloads below were copied out of a
  `numpy.save` run; see `_NUMPY_F32_HEADER` and friends), and `npy_load`
  has to return the same values NumPy started with. If the header layout
  or the payload order were misread, these fail.
- **Writing.** `npy_save`'s output is compared against those same NumPy
  bytes, so the prelude, the dict text, NumPy's 64-byte padding and the
  payload all have to agree exactly -- which is what makes
  `numpy.load(path)` on a numax-written file a claim rather than a hope.

Plus the failure modes `npy_load` promises to raise on: a Fortran-ordered
file, a big-endian `descr`, a dtype or shape that disagrees with what the
caller asked for, and a file that is not `.npy` at all.
"""

from std.testing import (
    TestSuite,
    assert_almost_equal,
    assert_equal,
    assert_true,
)

from max.gpu.host import DeviceContext

from numax.core.array import Tensor
from numax.io import npy_load, npy_save

comptime _TMP = "/tmp/numax_test_npy"

# Copied verbatim from `numpy.save` output: the dict text, NumPy's space
# padding to a 64-byte total, and the trailing newline. A version-1.0
# header is 118 bytes here, which with the 10-byte prelude is 128.
comptime _NUMPY_F32_HEADER = (
    "{'descr': '<f4', 'fortran_order': False, 'shape': (4,), }             "
    "                                               \n"
)
comptime _NUMPY_F64_HEADER = (
    "{'descr': '<f8', 'fortran_order': False, 'shape': (2, 3), }           "
    "                                               \n"
)
comptime _NUMPY_I32_HEADER = (
    "{'descr': '<i4', 'fortran_order': False, 'shape': (2, 2), }           "
    "                                               \n"
)
comptime _NUMPY_FORTRAN_HEADER = (
    "{'descr': '<f4', 'fortran_order': True, 'shape': (2, 3), }            "
    "                                               \n"
)
comptime _NUMPY_BIG_ENDIAN_HEADER = (
    "{'descr': '>f4', 'fortran_order': False, 'shape': (4,), }             "
    "                                               \n"
)


def _write_numpy_bytes(
    path: String, header: String, payload: List[UInt8]
) raises:
    """Write the exact bytes `numpy.save` would have, for `header`."""
    var out = List[UInt8]()
    out.append(UInt8(0x93))  # not valid UTF-8 alone, hence the raw byte
    for c in String("NUMPY").as_bytes():
        out.append(c)
    out.append(UInt8(1))
    out.append(UInt8(0))
    var header_len = header.byte_length()
    out.append(UInt8(header_len & 0xFF))
    out.append(UInt8((header_len >> 8) & 0xFF))
    for c in header.as_bytes():
        out.append(c)
    for b in payload:
        out.append(b)
    var f = open(path, "w")
    f.write_bytes(Span(out))
    f.close()


def _f32_payload() -> List[UInt8]:
    """`numpy.save`'s payload for `float32([1.5, -2.25, 3.0, 0.125])`."""
    return [
        UInt8(0),
        UInt8(0),
        UInt8(192),
        UInt8(63),
        UInt8(0),
        UInt8(0),
        UInt8(16),
        UInt8(192),
        UInt8(0),
        UInt8(0),
        UInt8(64),
        UInt8(64),
        UInt8(0),
        UInt8(0),
        UInt8(0),
        UInt8(62),
    ]


def _i32_payload() -> List[UInt8]:
    """`numpy.save`'s payload for `int32([[1, 2], [3, 4]])`."""
    var out = List[UInt8]()
    for v in [1, 2, 3, 4]:
        out.append(UInt8(v))
        out.append(UInt8(0))
        out.append(UInt8(0))
        out.append(UInt8(0))
    return out^


def _read_all(path: String) raises -> List[UInt8]:
    var f = open(path, "r")
    var data = f.read_bytes()
    f.close()
    return data^


def test_npy_load_reads_a_numpy_written_rank_1_float32_file() raises:
    var ctx = DeviceContext(api="cpu")
    var path = String(_TMP, "_read_f32.npy")
    _write_numpy_bytes(path, _NUMPY_F32_HEADER, _f32_payload())

    var loaded = npy_load[DType.float32, 4](ctx, path)
    var values = loaded.to_host()
    var expected = [1.5, -2.25, 3.0, 0.125]
    for i in range(4):
        assert_almost_equal(Float64(values[i]), expected[i])


def test_npy_load_reads_a_numpy_written_rank_2_int32_file() raises:
    var ctx = DeviceContext(api="cpu")
    var path = String(_TMP, "_read_i32.npy")
    _write_numpy_bytes(path, _NUMPY_I32_HEADER, _i32_payload())

    var loaded = npy_load[DType.int32, 2, 2](ctx, path)
    var values = loaded.to_host()
    var expected = [1, 2, 3, 4]
    for i in range(4):
        assert_equal(Int(values[i]), expected[i])


def test_npy_save_output_is_byte_identical_to_numpy_save() raises:
    """The whole interop claim: same prelude, same dict, same padding."""
    var ctx = DeviceContext(api="cpu")
    var vals = [1.5, -2.25, 3.0, 0.125]
    var storage = List[Scalar[DType.float32]](capacity=4)
    for i in range(4):
        storage.append(Scalar[DType.float32](vals[i]))
    var xs = Tensor[DType.float32, 4](ctx, storage^)

    var ours = String(_TMP, "_write_f32.npy")
    npy_save(xs, ours)
    var theirs = String(_TMP, "_numpy_f32.npy")
    _write_numpy_bytes(theirs, _NUMPY_F32_HEADER, _f32_payload())

    var a = _read_all(ours)
    var b = _read_all(theirs)
    assert_equal(len(a), len(b))
    assert_equal(len(a), 144)  # 10 prelude + 118 header + 16 payload
    for i in range(len(a)):
        assert_equal(Int(a[i]), Int(b[i]))


def test_npy_save_writes_numpys_rank_2_header_shape() raises:
    """Rank 2 drops the rank-1 trailing comma: `(2, 3)`, not `(2, 3,)`."""
    var ctx = DeviceContext(api="cpu")
    var storage = List[Scalar[DType.float64]](capacity=6)
    for i in range(6):
        storage.append(Scalar[DType.float64](i))
    var xs = Tensor[DType.float64, 2, 3](ctx, storage^)

    var path = String(_TMP, "_write_f64.npy")
    npy_save(xs, path)

    var data = _read_all(path)
    var header = _NUMPY_F64_HEADER.as_bytes()
    assert_equal(len(data), 10 + len(header) + 48)
    for i in range(len(header)):
        assert_equal(Int(data[10 + i]), Int(header[i]))


def test_npy_round_trips_through_numax() raises:
    var ctx = DeviceContext(api="cpu")
    var storage = List[Scalar[DType.float32]](capacity=6)
    for i in range(6):
        storage.append(Scalar[DType.float32](Float64(i) * 0.5 - 1.0))
    var xs = Tensor[DType.float32, 2, 3](ctx, storage^)

    var path = String(_TMP, "_round_trip.npy")
    npy_save(xs, path)
    var loaded = npy_load[DType.float32, 2, 3](ctx, path)

    var original = xs.to_host()
    var values = loaded.to_host()
    for i in range(6):
        assert_equal(Int(values[i].to_bits()), Int(original[i].to_bits()))


def test_npy_load_raises_on_fortran_order() raises:
    var ctx = DeviceContext(api="cpu")
    var path = String(_TMP, "_fortran.npy")
    var payload = List[UInt8]()
    for _ in range(24):
        payload.append(UInt8(0))
    _write_numpy_bytes(path, _NUMPY_FORTRAN_HEADER, payload)

    var raised = False
    try:
        _ = npy_load[DType.float32, 2, 3](ctx, path)
    except:
        raised = True
    assert_true(raised, "expected a Fortran-order .npy to raise")


def test_npy_load_raises_on_big_endian_descr() raises:
    var ctx = DeviceContext(api="cpu")
    var path = String(_TMP, "_big_endian.npy")
    _write_numpy_bytes(path, _NUMPY_BIG_ENDIAN_HEADER, _f32_payload())

    var raised = False
    try:
        _ = npy_load[DType.float32, 4](ctx, path)
    except:
        raised = True
    assert_true(raised, "expected a big-endian .npy to raise")


def test_npy_load_raises_on_dtype_mismatch() raises:
    var ctx = DeviceContext(api="cpu")
    var path = String(_TMP, "_dtype_mismatch.npy")
    _write_numpy_bytes(path, _NUMPY_F32_HEADER, _f32_payload())

    var raised = False
    try:
        _ = npy_load[DType.int32, 4](ctx, path)
    except:
        raised = True
    assert_true(raised, "expected a float32 file loaded as int32 to raise")


def test_npy_load_raises_on_shape_mismatch() raises:
    var ctx = DeviceContext(api="cpu")
    var path = String(_TMP, "_shape_mismatch.npy")
    _write_numpy_bytes(path, _NUMPY_F32_HEADER, _f32_payload())

    var raised = False
    try:
        _ = npy_load[DType.float32, 2, 2](ctx, path)
    except:
        raised = True
    assert_true(raised, "expected a rank-1 file loaded as rank 2 to raise")


def test_npy_load_raises_on_a_non_npy_file() raises:
    var ctx = DeviceContext(api="cpu")
    var path = String(_TMP, "_not_npy.npy")
    var f = open(path, "w")
    f.write_bytes(String("PK\x03\x04 this is a zip, i.e. an .npz").as_bytes())
    f.close()

    var raised = False
    try:
        _ = npy_load[DType.float32, 4](ctx, path)
    except:
        raised = True
    assert_true(raised, "expected a non-.npy file to raise")


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
