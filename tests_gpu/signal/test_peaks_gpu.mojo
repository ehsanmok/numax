"""`find_peaks` and `peak_prominences` at `gpu=True`, against the host: a
long signal with flat tops, every filter alone and together."""

from std.testing import (
    TestSuite,
    assert_almost_equal,
    assert_equal,
    assert_false,
)

from max.gpu.host import DeviceContext

from numax.core.tensor import Static
from numax.signal import find_peaks, peak_prominences

comptime f32 = DType.float32
comptime n = 3000


def _signal(ctx: DeviceContext) raises -> Static[f32, n]:
    var values = List[Scalar[f32]](capacity=n)
    for i in range(n):
        var base = Float32((i * 7919) % 97) * 0.02
        values.append(base if i % 50 != 7 else base + 0.0)
    for i in range(0, n - 3, 211):
        values[i + 1] = 5.0
        values[i + 2] = 5.0
    return Static[f32, n](values^, ctx)


def _same(
    got: List[Scalar[DType.int64]], want: List[Scalar[DType.int64]]
) raises:
    assert_equal(len(got), len(want))
    for i in range(len(want)):
        assert_equal(got[i], want[i])


def test_find_peaks_on_the_device_matches_the_host() raises:
    var gpu = DeviceContext()
    var cpu = DeviceContext(api="cpu")
    var d = find_peaks[gpu=True](_signal(gpu))
    assert_false(d.on_host())
    _same(d.to_host(), find_peaks(_signal(cpu)).to_host())
    _same(
        find_peaks[gpu=True](_signal(gpu), height=1.0).to_host(),
        find_peaks(_signal(cpu), height=1.0).to_host(),
    )
    _same(
        find_peaks[gpu=True](_signal(gpu), threshold=0.3).to_host(),
        find_peaks(_signal(cpu), threshold=0.3).to_host(),
    )
    _same(
        find_peaks[gpu=True](_signal(gpu), distance=25).to_host(),
        find_peaks(_signal(cpu), distance=25).to_host(),
    )
    _same(
        find_peaks[gpu=True](
            _signal(gpu), height=0.5, threshold=0.1, distance=10
        ).to_host(),
        find_peaks(
            _signal(cpu), height=0.5, threshold=0.1, distance=10
        ).to_host(),
    )


def test_peak_prominences_on_the_device_matches_the_host() raises:
    var gpu = DeviceContext()
    var cpu = DeviceContext(api="cpu")
    var peaks = find_peaks[gpu=True](_signal(gpu))
    var d = peak_prominences[gpu=True](_signal(gpu), peaks)
    assert_false(d.on_host())
    var got = d.to_host()
    var want = peak_prominences(
        _signal(cpu), find_peaks(_signal(cpu))
    ).to_host()
    assert_equal(len(got), len(want))
    for i in range(len(want)):
        assert_almost_equal(got[i], want[i], atol=1e-6)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
