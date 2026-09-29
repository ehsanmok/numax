"""`linprog` and `milp` at `gpu=True`: the scaled product `A diag(d) A^T`,
its LU and every solve against it on the device, at `float32`, against
the host run at the same `dtype` and SciPy's optimum. The problems are
`test_linprog`'s mixed one (inequalities, equalities and the
boxed-variable rows) and `test_milp`'s knapsack."""

from std.math import inf as _inf
from std.testing import TestSuite, assert_almost_equal, assert_equal

from max.gpu.host import DeviceContext

from numax.core.tensor import Static
from numax.optimize import Bounds, LinearConstraint, linprog, milp

comptime f32 = DType.float32


def test_linprog_gpu_matches_host() raises:
    """Device and host runs reach SciPy's optimum at `float32`."""
    var ctx = DeviceContext()
    var c = Static[f32, 5](
        [
            0.6672475608343279,
            1.438522591656152,
            -0.6756622510056528,
            0.20313861038960904,
            -0.46330757653841514,
        ],
        ctx,
    )
    var a_ub = Static[f32, 7, 5](
        [
            0.0012301533574825742,
            0.2987455375084699,
            -0.2741378553622176,
            -0.8905918387572742,
            -0.45467078517172255,
            -0.9916465549964624,
            0.060143602597438485,
            1.3402152455545335,
            -0.49220651855132963,
            -0.6204748998199404,
            0.4898420501851982,
            0.35688700816006075,
            0.10541424899789856,
            -0.9304680447082047,
            -0.02925182246327349,
            0.6953031944582878,
            -1.344214547285082,
            -0.45761576104021817,
            -1.901222739800844,
            -1.289537739784976,
            -1.8417350377917323,
            -0.23509113107468127,
            -1.2674464814437032,
            0.2712643588217015,
            0.15675108662422516,
            -0.18693094462995438,
            -2.516759710820513,
            -0.5386928958466366,
            -0.048500945401071985,
            0.11330898600330756,
            -1.5301357655053935,
            -0.47775327603393064,
            -0.9785190780566395,
            -0.8088372394255993,
            1.0608986233860787,
        ],
        ctx,
    )
    var b_ub = Static[f32, 7](
        [
            -0.4952264968252784,
            0.9630557674458077,
            1.0360926174940466,
            -2.6228989828818694,
            -3.165025747066963,
            -2.1606220299277767,
            -2.6108646062211176,
        ],
        ctx,
    )
    var a_eq = Static[f32, 2, 5](
        [
            0.11935402569658124,
            -0.6414703941072214,
            2.000416546342423,
            0.7622597120847118,
            -1.1992889021052233,
            0.07451622877146342,
            0.5766895836701853,
            -0.1887821253507493,
            0.682910267195206,
            -0.06651732014941557,
        ],
        ctx,
    )
    var b_eq = Static[f32, 2]([2.7690785487712426, 0.48042058562658285], ctx)

    var box = Bounds(0.0, 3.0)
    var device = linprog[gpu=True](c, a_ub, b_ub, a_eq, b_eq, box.copy())
    var host_ctx = DeviceContext(api="cpu")
    var ch = Static[f32, 5](c.to_host(), host_ctx)
    var auh = Static[f32, 7, 5](a_ub.to_host(), host_ctx)
    var buh = Static[f32, 7](b_ub.to_host(), host_ctx)
    var aeh = Static[f32, 2, 5](a_eq.to_host(), host_ctx)
    var beh = Static[f32, 2](b_eq.to_host(), host_ctx)
    var host = linprog(ch, auh, buh, aeh, beh, box^)
    assert_equal(device.status, 0)
    assert_equal(host.status, 0)
    assert_almost_equal(device.fun, -1.6626251514723984, atol=1e-3)
    assert_almost_equal(device.fun, host.fun, atol=1e-3)
    var xd = device.x.to_host()
    var xh = host.x.to_host()
    for i in range(5):
        assert_almost_equal(Float64(xd[i]), Float64(xh[i]), atol=1e-2)


def test_milp_gpu_knapsack() raises:
    """Every node's relaxation on the device reaches HiGHS's knapsack
    optimum at `float32`."""
    var ctx = DeviceContext()
    var c = Static[f32, 8](
        [-21.0, -10.0, -19.0, -37.0, -24.0, -7.0, -23.0, -9.0], ctx
    )
    var a = Static[f32, 1, 8](
        [5.0, 5.0, 16.0, 11.0, 13.0, 13.0, 15.0, 3.0], ctx
    )
    var lo = Static[f32, 1]([-_inf[f32]()], ctx)
    var hi = Static[f32, 1]([36.45], ctx)
    var r = milp[gpu=True](
        c,
        integrality=[1],
        bounds=Bounds(0.0, 1.0),
        constraints=[LinearConstraint(a, lo, hi)],
    )
    assert_equal(r.status, 0)
    assert_almost_equal(r.fun, -92.0, atol=1e-3)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
