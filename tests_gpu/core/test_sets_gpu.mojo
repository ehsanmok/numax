"""The set routines at `gpu=True`, against the host.

Values, counts, indices and membership are exact on both targets, so
every comparison is equality: a 400-element sample with many repeats and
scattered NaNs (each its own value, never a member), and the three binary
set operations over two such samples.
"""

from std.testing import TestSuite, assert_equal, assert_false, assert_true

from std.utils.numerics import nan

from max.gpu.host import DeviceContext

from numax.core.tensor import Static
from numax.core.sorting import (
    intersect1d,
    isin,
    setdiff1d,
    union1d,
    unique_counts,
    unique_inverse,
)

comptime f32 = DType.float32
comptime n = 400


def _sample(ctx: DeviceContext, seed: Int) raises -> Static[f32, n]:
    var values = List[Scalar[f32]](capacity=n)
    for i in range(n):
        if i % 89 == 3:
            values.append(nan[f32]())
        else:
            values.append(Float32((i * 37 + seed) % 29) - 14.0)
    return Static[f32, n](values^, ctx)


def _same_values(got: List[Float32], want: List[Float32]) raises:
    assert_equal(len(got), len(want))
    for i in range(len(want)):
        if want[i] != want[i]:
            assert_true(got[i] != got[i])
        else:
            assert_equal(got[i], want[i])


def test_unique_counts_and_inverse_on_the_device() raises:
    var gpu = DeviceContext()
    var cpu = DeviceContext(api="cpu")
    var d = unique_counts[gpu=True](_sample(gpu, 0))
    assert_false(d.values.on_host())
    var h = unique_counts(_sample(cpu, 0))
    _same_values(d.values.to_host(), h.values.to_host())
    var dc = d.counts.to_host()
    var hc = h.counts.to_host()
    for i in range(len(hc)):
        assert_equal(dc[i], hc[i])
    var di = unique_inverse[gpu=True](_sample(gpu, 0))
    var hi = unique_inverse(_sample(cpu, 0))
    _same_values(di.values.to_host(), hi.values.to_host())
    var dinv = di.inverse_indices.to_host()
    var hinv = hi.inverse_indices.to_host()
    for i in range(n):
        assert_equal(dinv[i], hinv[i])


def test_isin_on_the_device() raises:
    var gpu = DeviceContext()
    var cpu = DeviceContext(api="cpu")
    var probe = Static[f32, 5]([-14.0, 0.0, 3.0, 100.0, nan[f32]()], gpu)
    var probe_h = Static[f32, 5]([-14.0, 0.0, 3.0, 100.0, nan[f32]()], cpu)
    var got = isin[gpu=True](_sample(gpu, 5), probe).to_host()
    var want = isin(_sample(cpu, 5), probe_h).to_host()
    var inv = isin[invert=True, gpu=True](_sample(gpu, 5), probe).to_host()
    for i in range(n):
        assert_equal(got[i], want[i])
        assert_equal(inv[i], not want[i])


def test_binary_set_operations_on_the_device() raises:
    var gpu = DeviceContext()
    var cpu = DeviceContext(api="cpu")
    _same_values(
        intersect1d[gpu=True](_sample(gpu, 0), _sample(gpu, 11)).to_host(),
        intersect1d(_sample(cpu, 0), _sample(cpu, 11)).to_host(),
    )
    _same_values(
        setdiff1d[gpu=True](_sample(gpu, 0), _sample(gpu, 11)).to_host(),
        setdiff1d(_sample(cpu, 0), _sample(cpu, 11)).to_host(),
    )
    _same_values(
        union1d[gpu=True](_sample(gpu, 0), _sample(gpu, 11)).to_host(),
        union1d(_sample(cpu, 0), _sample(cpu, 11)).to_host(),
    )


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
