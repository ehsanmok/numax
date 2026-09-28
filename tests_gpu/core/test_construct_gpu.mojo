"""`diag`, `diagonal`, `diagflat`, `vander`, `meshgrid` and `broadcast_to`
at `gpu=True`, against the host. Each builds its result by an index rule
over the input, one launch on the device. The comparisons are exact except
`vander`, whose powers are repeated products in the same order on both
devices and so agree to a float32 ulp at most.
"""

from std.testing import (
    TestSuite,
    assert_almost_equal,
    assert_equal,
    assert_false,
)

from max.gpu.host import DeviceContext

from numax.core.tensor import (
    Static,
    broadcast_to,
    diag,
    diagflat,
    diagonal,
    meshgrid,
    reshape_dyn,
    vander,
)

comptime f32 = DType.float32


def _line[n: Int](ctx: DeviceContext, base: Int) raises -> Static[f32, n]:
    var values = List[Scalar[f32]](capacity=n)
    for i in range(n):
        values.append(Float32(base + i) * 0.5)
    return Static[f32, n](values^, ctx)


def _grid[r: Int, c: Int](ctx: DeviceContext) raises -> Static[f32, r, c]:
    var values = List[Scalar[f32]](capacity=r * c)
    for i in range(r * c):
        values.append(Float32(i) - 3.0)
    return Static[f32, r, c](values^, ctx)


def _assert_same(got: List[Scalar[f32]], want: List[Scalar[f32]]) raises:
    assert_equal(len(got), len(want))
    for i in range(len(want)):
        assert_equal(got[i], want[i])


def test_the_diag_family_on_the_device_matches_the_host() raises:
    var gpu = DeviceContext()
    var cpu = DeviceContext(api="cpu")
    var d = diag[gpu=True](_line[5](gpu, 1))
    assert_false(d.on_host())
    _assert_same(d.to_host(), diag(_line[5](cpu, 1)).to_host())
    _assert_same(
        diagonal[gpu=True](_grid[4, 4](gpu)).to_host(),
        diagonal(_grid[4, 4](cpu)).to_host(),
    )
    _assert_same(
        diagflat[gpu=True](_grid[2, 3](gpu)).to_host(),
        diagflat(_grid[2, 3](cpu)).to_host(),
    )
    var dyn_d = reshape_dyn[rank=1](_grid[2, 3](gpu), 6)
    var dyn_h = reshape_dyn[rank=1](_grid[2, 3](cpu), 6)
    _assert_same(diagflat[gpu=True](dyn_d).to_host(), diagflat(dyn_h).to_host())


def test_vander_on_the_device_matches_the_host() raises:
    var gpu = DeviceContext()
    var cpu = DeviceContext(api="cpu")
    var d = vander[cols=5, gpu=True](_line[4](gpu, -2)).to_host()
    var h = vander[cols=5](_line[4](cpu, -2)).to_host()
    assert_equal(len(d), len(h))
    for i in range(len(h)):
        assert_almost_equal(d[i], h[i], atol=1e-6, rtol=1e-6)


def test_meshgrid_on_the_device_matches_the_host() raises:
    var gpu = DeviceContext()
    var cpu = DeviceContext(api="cpu")
    var d = meshgrid[gpu=True](_line[4](gpu, 0), _line[3](gpu, 10))
    var h = meshgrid(_line[4](cpu, 0), _line[3](cpu, 10))
    assert_false(d[0].on_host())
    _assert_same(d[0].to_host(), h[0].to_host())
    _assert_same(d[1].to_host(), h[1].to_host())


def test_broadcast_to_on_the_device_matches_the_host() raises:
    """A `(3, 1)` column stretched to `(2, 3, 4)`: one new leading axis
    and one stretched trailing axis."""
    var gpu = DeviceContext()
    var cpu = DeviceContext(api="cpu")
    var d = broadcast_to[rank=3, gpu=True](_grid[3, 1](gpu), 2, 3, 4)
    assert_false(d.on_host())
    _assert_same(
        d.to_host(), broadcast_to[rank=3](_grid[3, 1](cpu), 2, 3, 4).to_host()
    )


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
