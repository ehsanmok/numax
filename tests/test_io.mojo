"""Tests for `numax.io`.

`save`/`load` are checked for byte-identical round trips at rank 1, 2, and
3, plus the header-validation failure modes (`load` raising on a dtype,
rank, or shape mismatch against what the caller requests). `print_tensor`
is checked against frozen expected strings via `_format_tensor` (the
helper it's built on, which returns the formatted `String` instead of
printing it directly) for a small array and an array over `threshold`
(exercising the `edge_items` truncation).
"""

from std.testing import (
    TestSuite,
    assert_almost_equal,
    assert_equal,
    assert_true,
)

from numax.array import Tensor, full
from numax.io import _format_tensor, load, print_tensor, save

comptime dtype = DType.float32
comptime _TMP_DIR = "/tmp/numax_test_io"


def _fixed_1d() -> Tensor[dtype, 4]:
    var xs = full[dtype, 4](Scalar[dtype](0))
    var v = xs.view()
    var vals = [1.5, -2.25, 3.0, 0.125]
    for i in range(4):
        v[i] = Scalar[dtype](vals[i])
    return xs^


def test_save_load_round_trips_a_rank_1_tensor() raises:
    var xs = _fixed_1d()
    var path = String(_TMP_DIR, "_rank1.nmx")
    save(xs, path)
    var loaded = load[dtype, 4](path)
    for i in range(4):
        assert_almost_equal(loaded[i], xs[i])


def test_save_load_round_trips_a_rank_2_tensor() raises:
    var xs = full[dtype, 2, 3](Scalar[dtype](0))
    var v = xs.view()
    var k = 0
    for r in range(2):
        for c in range(3):
            v[r, c] = Scalar[dtype](k)
            k += 1
    var path = String(_TMP_DIR, "_rank2.nmx")
    save(xs, path)
    var loaded = load[dtype, 2, 3](path)
    for i in range(6):
        assert_almost_equal(loaded[i], xs[i])


def test_save_load_round_trips_a_rank_3_tensor() raises:
    var xs = full[dtype, 2, 2, 2](Scalar[dtype](0))
    var v = xs.view()
    var k = 0
    for a in range(2):
        for b in range(2):
            for c in range(2):
                v[a, b, c] = Scalar[dtype](k) * 1.5
                k += 1
    var path = String(_TMP_DIR, "_rank3.nmx")
    save(xs, path)
    var loaded = load[dtype, 2, 2, 2](path)
    for i in range(8):
        assert_almost_equal(loaded[i], xs[i])


def test_load_raises_on_dtype_mismatch() raises:
    var xs = _fixed_1d()
    var path = String(_TMP_DIR, "_dtype_mismatch.nmx")
    save(xs, path)
    var raised = False
    try:
        _ = load[DType.float64, 4](path)
    except:
        raised = True
    assert_true(raised, msg="load should raise on a dtype mismatch")


def test_load_raises_on_shape_mismatch() raises:
    var xs = _fixed_1d()
    var path = String(_TMP_DIR, "_shape_mismatch.nmx")
    save(xs, path)
    var raised = False
    try:
        _ = load[dtype, 5](path)
    except:
        raised = True
    assert_true(raised, msg="load should raise on a shape mismatch")


def test_load_raises_on_rank_mismatch() raises:
    var xs = _fixed_1d()
    var path = String(_TMP_DIR, "_rank_mismatch.nmx")
    save(xs, path)
    var raised = False
    try:
        _ = load[dtype, 2, 2](path)
    except:
        raised = True
    assert_true(raised, msg="load should raise on a rank mismatch")


def test_load_raises_on_bad_magic_bytes() raises:
    var path = String(_TMP_DIR, "_bad_magic.nmx")
    var f = open(path, "w")
    var junk = List[UInt8]()
    for c in String("NOPE-not-a-numax-file").as_bytes():
        junk.append(c)
    f.write_bytes(Span(junk))
    f.close()
    var raised = False
    try:
        _ = load[dtype, 4](path)
    except:
        raised = True
    assert_true(raised, msg="load should raise on bad magic bytes")


def test_print_tensor_matches_a_frozen_small_array_string() raises:
    var xs = _fixed_1d()
    var formatted = _format_tensor(xs, 2, 1000, 3)
    # 0.125 rounds half-away-from-zero to 0.13, not banker's-rounding 0.12.
    assert_equal(formatted, "[1.5, -2.25, 3.0, 0.13]")
    # `print_tensor` calls this same helper before printing it, so exercise
    # the public entry point too, even though its stdout isn't captured.
    print_tensor(xs, precision=2)


def test_print_tensor_truncates_arrays_over_threshold() raises:
    var xs = full[dtype, 20](Scalar[dtype](0))
    var v = xs.view()
    for i in range(20):
        v[i] = Scalar[dtype](i)
    var formatted = _format_tensor(xs, 4, 10, 2)
    assert_equal(formatted, "[0.0, 1.0, ..., 18.0, 19.0]")
    print_tensor(xs, threshold=10, edge_items=2)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
