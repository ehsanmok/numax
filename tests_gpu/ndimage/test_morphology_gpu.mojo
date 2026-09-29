"""`label`, `binary_erosion`, `binary_dilation` and
`distance_transform_edt` at `gpu=True`, against the host, at `float32`:
the propagation rounds and the renumbering scan, the footprint lanes, and
the per-line envelope passes, on an image dense enough that its
components percolate across it."""

from std.math import sin
from std.testing import TestSuite, assert_almost_equal, assert_equal
from std.testing import assert_false

from max.gpu.host import DeviceContext

from numax.core.tensor import Static
from numax.ndimage import (
    binary_dilation,
    binary_erosion,
    distance_transform_edt,
    label,
)

comptime f32 = DType.float32


def _img(ctx: DeviceContext) raises -> Static[f32, 64, 64]:
    var values = List[Scalar[f32]](capacity=64 * 64)
    for i in range(64 * 64):
        var v = sin(Float32(i) * 12.9898) * 43758.5453
        values.append(
            Float32(1) if v - Float32(Int(v)) > 0.4
            or v - Float32(Int(v)) < -0.6 else Float32(0)
        )
    return Static[f32, 64, 64](values^, ctx)


def _vol(ctx: DeviceContext) raises -> Static[f32, 12, 13, 11]:
    var values = List[Scalar[f32]](capacity=12 * 13 * 11)
    for i in range(12 * 13 * 11):
        values.append(
            Float32(1) if sin(Float32(i) * 0.37) + sin(Float32(i) * 1.13)
            > -0.3 else Float32(0)
        )
    return Static[f32, 12, 13, 11](values^, ctx)


def test_morphology_on_the_device_matches_the_host() raises:
    var gpu = DeviceContext()
    var cpu = DeviceContext(api="cpu")
    for connectivity in range(1, 3):
        var d = label[gpu=True](_img(gpu), connectivity)
        var h = label(_img(cpu), connectivity)
        assert_equal(d.num_features, h.num_features)
        assert_false(d.labels.on_host())
        var dl = d.labels.to_host()
        var hl = h.labels.to_host()
        for i in range(64 * 64):
            assert_equal(Int(dl[i]), Int(hl[i]))
    var d3 = label[gpu=True](_vol(gpu), 3)
    var h3 = label(_vol(cpu), 3)
    assert_equal(d3.num_features, h3.num_features)
    var de = binary_erosion[gpu=True](_img(gpu), 1, 2).to_host()
    var he = binary_erosion(_img(cpu), 1, 2).to_host()
    var dd = binary_dilation[gpu=True](_vol(gpu), 2).to_host()
    var hd = binary_dilation(_vol(cpu), 2).to_host()
    for i in range(64 * 64):
        assert_equal(de[i], he[i])
    for i in range(12 * 13 * 11):
        assert_equal(dd[i], hd[i])
    var dt = distance_transform_edt[gpu=True](_vol(gpu)).to_host()
    var ht = distance_transform_edt(_vol(cpu)).to_host()
    for i in range(12 * 13 * 11):
        assert_almost_equal(dt[i], ht[i], atol=1e-5)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
