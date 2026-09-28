"""`solve_banded` at `gpu=True` for a tridiagonal band, against the host.

The device path is parallel cyclic reduction without pivoting, so the
systems here are diagonally dominant -- what it asks for. The sizes are
not powers of two, so a reduction step's partner running off either end
is exercised.
"""

from std.testing import TestSuite, assert_almost_equal, assert_false

from max.gpu.host import DeviceContext

from numax.core.tensor import Static
from numax.linalg import solve_banded

comptime f32 = DType.float32


def _band[n: Int](ctx: DeviceContext) raises -> Static[f32, 3, n]:
    """Rows: superdiagonal (column-aligned, `ab[0, 0]` unused), diagonal,
    subdiagonal (`ab[2, n - 1]` unused); the diagonal dominates."""
    var values = List[Scalar[f32]](capacity=3 * n)
    for j in range(n):
        values.append(
            Float32(0) if j == 0 else Float32((j * 37) % 11) / 11.0 - 0.5
        )
    for j in range(n):
        values.append(Float32(3) + Float32((j * 13) % 7) / 7.0)
    for j in range(n):
        values.append(
            Float32(0) if j == n - 1 else Float32((j * 29) % 13) / 13.0 - 0.5
        )
    return Static[f32, 3, n](ctx, values^)


def _rhs[n: Int](ctx: DeviceContext) raises -> Static[f32, n]:
    var values = List[Scalar[f32]](capacity=n)
    for j in range(n):
        values.append(Float32((j * 17) % 23) / 23.0 - 0.4)
    return Static[f32, n](ctx, values^)


def _check[n: Int]() raises where n >= 1:
    var gpu = DeviceContext()
    var cpu = DeviceContext(api="cpu")
    var d = solve_banded[l=1, u=1, gpu=True](_band[n](gpu), _rhs[n](gpu))
    assert_false(d.on_host())
    var got = d.to_host()
    var want = solve_banded[l=1, u=1](_band[n](cpu), _rhs[n](cpu)).to_host()
    for i in range(n):
        assert_almost_equal(Float64(got[i]), Float64(want[i]), atol=1e-5)


def test_tridiagonal_solve_banded_on_the_device_matches_the_host() raises:
    _check[1]()
    _check[2]()
    _check[7]()
    _check[100]()
    _check[1000]()


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
