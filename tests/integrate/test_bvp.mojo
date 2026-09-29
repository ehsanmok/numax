"""Tests for `solve_bvp` against SciPy on its own documentation example,
Bratu's problem `y'' + exp(y) = 0`, `y(0) = y(1) = 0`, from both of its
initial guesses: the lower solution in one outer iteration on the
initial mesh, the upper one after refining it to the same 25 nodes
SciPy's mesh rule reaches."""

from std.math import exp as exp_f64
from std.testing import (
    TestSuite,
    assert_almost_equal,
    assert_equal,
    assert_true,
)

from layout.tile_layout import row_major
from max.gpu.host import DeviceContext

from numax.core.tensor import Dynamic, Static, _dyn_shape
from numax.integrate import solve_bvp

comptime f64 = DType.float64


def _cpu() raises -> DeviceContext:
    return DeviceContext(api="cpu")


def bratu(
    x: Dynamic[f64, 1], y: Dynamic[f64, 2], ctx: DeviceContext
) raises -> Dynamic[f64, 2]:
    """`(y1, -exp(y0))`, SciPy's example, on the host."""
    var m = y.dim[1]()
    var v = y.to_host()
    var out = List[Float64](capacity=2 * m)
    for j in range(m):
        out.append(v[m + j])
    for j in range(m):
        out.append(-exp_f64(v[j]))
    var t = Dynamic[f64, 2](row_major(_dyn_shape[2](2, m)), ctx)
    t.copy_from_host(out^)
    return t^


def ends_zero(
    ya: Dynamic[f64, 1], yb: Dynamic[f64, 1], ctx: DeviceContext
) raises -> Dynamic[f64, 1]:
    var a = ya.to_host()
    var b = yb.to_host()
    var t = Dynamic[f64, 1](row_major(_dyn_shape[1](2)), ctx)
    t.copy_from_host([a[0], b[0]])
    return t^


def _mesh(count: Int) raises -> Dynamic[f64, 1]:
    var values = List[Float64](capacity=count)
    for i in range(count):
        values.append(Float64(i) / Float64(count - 1))
    var t = Dynamic[f64, 1](row_major(_dyn_shape[1](count)), _cpu())
    t.copy_from_host(values^)
    return t^


def _guess(count: Int, level: Float64) raises -> Dynamic[f64, 2]:
    var values = List[Float64](capacity=2 * count)
    for _ in range(count):
        values.append(level)
    for _ in range(count):
        values.append(0.0)
    var t = Dynamic[f64, 2](row_major(_dyn_shape[2](2, count)), _cpu())
    t.copy_from_host(values^)
    return t^


def _points() raises -> Static[f64, 3]:
    return Static[f64, 3]([0.25, 0.5, 0.75], _cpu())


def test_bratu_lower_solution_matches_scipy() raises:
    var r = solve_bvp[fun=bratu, bc=ends_zero](_mesh(5), _guess(5, 0.0))
    assert_equal(r.status, 0)
    assert_true(r.success)
    assert_equal(r.niter, 1)
    assert_equal(r.x.size(), 5)
    var s = r.sol(_points()).to_host()
    var want: List[Float64] = [
        0.10478414478746345,
        0.140534772543497,
        0.10478414478746347,
    ]
    for i in range(3):
        assert_almost_equal(s[i], want[i], atol=1e-10)


def test_bratu_upper_solution_matches_scipy() raises:
    var r = solve_bvp[fun=bratu, bc=ends_zero](_mesh(5), _guess(5, 3.0))
    assert_equal(r.status, 0)
    assert_equal(r.niter, 4)
    assert_equal(r.x.size(), 25)
    var s = r.sol(_points()).to_host()
    var want: List[Float64] = [
        2.617322800797979,
        4.091478915149736,
        2.617322800798019,
    ]
    for i in range(3):
        assert_almost_equal(s[i], want[i], atol=1e-8)
    var rms = r.rms_residuals.to_host()
    for i in range(len(rms)):
        assert_true(rms[i] < 1e-3)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
