"""The histogram family at `gpu=True`: atomic tallies on the device
against the host's one-pass count. Counts are small integers, so they
compare exactly; densities and weights to float32 rounding."""

from std.testing import TestSuite, assert_almost_equal, assert_equal

from max.gpu.host import DeviceContext

from numax.core.tensor import Static
from numax.stats import bincount, digitize, histogram, histogram2d, histogramdd

comptime f32 = DType.float32
comptime n = 4000


def _samples(ctx: DeviceContext, salt: Int) raises -> Static[f32, n]:
    var values = List[Scalar[f32]](capacity=n)
    for i in range(n):
        var u = Float32(((i + salt) * 7919) % 1009) / 1009.0
        values.append(u * u * 6.0 - 1.0)
    return Static[f32, n](values^, ctx)


def _close(got: List[Scalar[f32]], want: List[Scalar[f32]]) raises:
    assert_equal(len(got), len(want))
    for i in range(len(want)):
        assert_almost_equal(got[i], want[i], atol=1e-5, rtol=1e-5)


def test_the_one_dimensional_histograms_on_the_device() raises:
    var gpu = DeviceContext()
    var cpu = DeviceContext(api="cpu")
    var d = histogram[bins=17, gpu=True](_samples(gpu, 0))
    var h = histogram[bins=17](_samples(cpu, 0))
    _close(d.counts.to_host(), h.counts.to_host())
    _close(d.edges.to_host(), h.edges.to_host())
    _close(
        histogram[bins=9, gpu=True](
            _samples(gpu, 0), _samples(gpu, 3)
        ).counts.to_host(),
        histogram[bins=9](_samples(cpu, 0), _samples(cpu, 3)).counts.to_host(),
    )
    _close(
        histogram[bins=12, gpu=True](
            _samples(gpu, 0), 0.0, 4.0, True
        ).counts.to_host(),
        histogram[bins=12](_samples(cpu, 0), 0.0, 4.0, True).counts.to_host(),
    )
    var edges: List[Scalar[f32]] = [-1.0, 0.0, 0.5, 2.0, 5.0]
    _close(
        histogram[gpu=True](
            _samples(gpu, 0), Static[f32, 5](edges.copy(), gpu)
        ).counts.to_host(),
        histogram(
            _samples(cpu, 0), Static[f32, 5](edges^, cpu)
        ).counts.to_host(),
    )


def test_the_multi_dimensional_histograms_on_the_device() raises:
    var gpu = DeviceContext()
    var cpu = DeviceContext(api="cpu")
    var d = histogram2d[xbins=6, ybins=5, gpu=True](
        _samples(gpu, 0), _samples(gpu, 11)
    )
    var h = histogram2d[xbins=6, ybins=5](_samples(cpu, 0), _samples(cpu, 11))
    _close(d.counts.to_host(), h.counts.to_host())
    var pts = List[Scalar[f32]](capacity=3000)
    for i in range(3000):
        pts.append(Float32((i * 37) % 101) * 0.1)
    var dd = histogramdd[4, 3, 5, gpu=True](
        Static[f32, 1000, 3](pts.copy(), gpu)
    )
    var hd = histogramdd[4, 3, 5](Static[f32, 1000, 3](pts^, cpu))
    _close(dd.counts.to_host(), hd.counts.to_host())


def test_bincount_and_digitize_on_the_device() raises:
    var gpu = DeviceContext()
    var cpu = DeviceContext(api="cpu")
    var ints = List[Scalar[DType.int32]](capacity=n)
    for i in range(n):
        ints.append(Int32((i * 13) % 37))
    var d = bincount[gpu=True](
        Static[DType.int32, n](ints.copy(), gpu), 40
    ).to_host()
    var h = bincount(Static[DType.int32, n](ints.copy(), cpu), 40).to_host()
    assert_equal(len(d), len(h))
    for i in range(len(h)):
        assert_equal(d[i], h[i])
    _close(
        bincount[gpu=True](
            Static[DType.int32, n](ints.copy(), gpu), _samples(gpu, 0)
        ).to_host(),
        bincount(
            Static[DType.int32, n](ints^, cpu), _samples(cpu, 0)
        ).to_host(),
    )
    var bins: List[Scalar[f32]] = [-1.0, 0.0, 1.0, 2.5, 4.0]
    for right in [False, True]:
        var dg = digitize[gpu=True](
            _samples(gpu, 0), Static[f32, 5](bins.copy(), gpu), right
        ).to_host()
        var hg = digitize(
            _samples(cpu, 0), Static[f32, 5](bins.copy(), cpu), right
        ).to_host()
        for i in range(n):
            assert_equal(dg[i], hg[i])


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
