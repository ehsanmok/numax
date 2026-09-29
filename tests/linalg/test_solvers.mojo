"""Tests for `solve_sylvester`, `solve_continuous_lyapunov`,
`solve_discrete_lyapunov` and `solve_continuous_are` against SciPy's
solutions, and the residual of each equation on matrices whose Schur forms have `2 x 2` blocks on both
sides, so every one of `trsyl_column`'s block shapes -- `1 x 1` through
`2 x 2` against `2 x 2` -- is exercised."""

from std.math import sin
from max.gpu.host import DeviceContext
from std.testing import TestSuite, assert_almost_equal

from numax.core.tensor import Static, transpose
from numax.linalg import (
    matmul,
    solve_continuous_are,
    solve_continuous_lyapunov,
    solve_discrete_lyapunov,
    solve_sylvester,
)

comptime f64 = DType.float64


def _cpu() raises -> DeviceContext:
    return DeviceContext(api="cpu")


def _a() raises -> Static[f64, 3, 3]:
    return Static[f64, 3, 3](
        [-3.0, 1.0, 0.5, 0.2, -2.0, 1.0, 0.0, -1.0, -4.0], _cpu()
    )


def _qs() raises -> Static[f64, 3, 3]:
    return Static[f64, 3, 3](
        [1.0, 0.2, 0.0, 0.2, 2.0, 0.3, 0.0, 0.3, 1.0], _cpu()
    )


def _close(got: List[Float64], want: List[Float64], atol: Float64) raises:
    for i in range(len(want)):
        assert_almost_equal(got[i], want[i], atol=atol)


def test_sylvester_matches_scipy() raises:
    var b = Static[f64, 2, 2]([1.0, 2.0, -2.0, 1.0], _cpu())
    var q = Static[f64, 3, 2]([1.0, 2.0, 0.0, -1.0, 3.0, 0.5], _cpu())
    var x = solve_sylvester(_a(), b, q).to_host()
    _close(
        x,
        [
            0.22226101122941727,
            -0.9828649560714499,
            -0.22767207773317075,
            -0.15747367196136636,
            -0.5870716239017859,
            -0.5055565252807355,
        ],
        1e-14,
    )


def test_lyapunov_equations_match_scipy() raises:
    var c = solve_continuous_lyapunov(_a(), _qs()).to_host()
    _close(
        c,
        [
            -0.21305382112468985,
            -0.14597911841567493,
            0.013635310083211255,
            -0.1459791184156749,
            -0.5077590248891074,
            0.013677773904919885,
            0.01363531008321128,
            0.01367777390491995,
            -0.12841944347622994,
        ],
        1e-14,
    )
    var ad = Static[f64, 3, 3](
        [0.5, 0.1, 0.0, 0.2, -0.3, 0.4, 0.0, 0.1, 0.6], _cpu()
    )
    var d = solve_discrete_lyapunov(ad, _qs()).to_host()
    _close(
        d,
        [
            1.4020297530804824,
            0.27626059815738013,
            0.10319968329298738,
            0.27626059815738024,
            2.389625499462389,
            0.5755082233766378,
            0.10319968329298738,
            0.5755082233766379,
            1.7077456903122195,
        ],
        1e-13,
    )


def _rotation_heavy[
    n: Int
](seed: Float64, shift: Float64) raises -> Static[f64, n, n]:
    """A non-symmetric matrix with complex eigenvalue pairs -- a skew part
    dominating a small perturbation -- moved by `shift` along the real
    axis, so `a` and `-b` stay apart and the equation well conditioned."""
    var values = List[Float64](capacity=n * n)
    for i in range(n):
        for j in range(n):
            var skew = Float64(j - i) * 0.7
            var diag = shift if i == j else 0.0
            values.append(skew + diag + 0.3 * sin(Float64(i * n + j) * seed))
    return Static[f64, n, n](values^, _cpu())


def test_sylvester_residual_with_blocks_on_both_sides() raises:
    var a = _rotation_heavy[6](1.3, -2.0)
    var b = _rotation_heavy[5](0.7, 1.0)
    var qv = List[Float64](capacity=30)
    for i in range(30):
        qv.append(sin(Float64(i) * 0.9))
    var q = Static[f64, 6, 5](qv.copy(), _cpu())
    var x = solve_sylvester(a, b, q)
    var lhs = matmul(a, x).to_host()
    var right = matmul(x, b).to_host()
    for i in range(30):
        assert_almost_equal(lhs[i] + right[i], qv[i], atol=1e-12)


def test_discrete_residual() raises:
    var a = Static[f64, 3, 3](
        [0.5, 0.1, 0.0, 0.2, -0.3, 0.4, 0.0, 0.1, 0.6], _cpu()
    )
    var x = solve_discrete_lyapunov(a, _qs())
    var axa = matmul(matmul(a, x), transpose(a)).to_host()
    var xv = x.to_host()
    var qv = _qs().to_host()
    for i in range(9):
        assert_almost_equal(xv[i] - axa[i], qv[i], atol=1e-13)


def test_care_matches_scipy() raises:
    # A triple integrator with unit weights, and a two-input system with
    # coupled weights: scipy.linalg.solve_continuous_are.
    var a = Static[f64, 3, 3](
        [0.0, 1.0, 0.0, 0.0, 0.0, 1.0, -1.0, -2.0, -3.0], _cpu()
    )
    var b = Static[f64, 3, 1]([0.0, 0.0, 1.0], _cpu())
    var q = Static[f64, 3, 3](
        [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0], _cpu()
    )
    var r = Static[f64, 1, 1]([1.0], _cpu())
    _close(
        solve_continuous_are(a, b, q, r).to_host(),
        [
            2.1870824521318517,
            1.8829148652376184,
            0.4142135623730948,
            1.8829148652376184,
            3.808370011372004,
            0.9607143952896295,
            0.4142135623730948,
            0.9607143952896295,
            0.4527422131661174,
        ],
        1e-13,
    )
    var a2 = Static[f64, 2, 2]([1.0, 2.0, -3.0, 0.5], _cpu())
    var b2 = Static[f64, 2, 2]([1.0, 0.5, 0.0, 1.0], _cpu())
    var q2 = Static[f64, 2, 2]([2.0, 0.3, 0.3, 1.0], _cpu())
    var r2 = Static[f64, 2, 2]([1.5, 0.2, 0.2, 0.8], _cpu())
    _close(
        solve_continuous_are(a2, b2, q2, r2).to_host(),
        [
            3.005527846867029,
            0.046965179081539514,
            0.046965179081539514,
            1.3959298980547403,
        ],
        1e-13,
    )


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
