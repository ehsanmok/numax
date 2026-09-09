"""Tests for `numax.optimize.array.minimize` and the `cg` method it adds.

Three claims, and they are different in kind.

`minimize[..., method="bfgs"]` must return *exactly* what `bfgs` returns on
the same input, not merely a comparable answer -- it is a dispatcher, so
agreement to the last bit is the specification. Same for `"cg"` and
`"nelder-mead"`.

The per-method `tol` defaults must survive the dispatch. Passing nothing has
to reproduce each method's own documented default, which is the property a
single shared default would silently break.

`cg` is new mathematics rather than a rename, so it gets ordinary
correctness tests plus the one that says why it exists: it converges without
ever forming an `n x n` matrix, on the same problems `bfgs` solves.
"""

from std.collections import Array
from std.testing import (
    TestSuite,
    assert_almost_equal,
    assert_equal,
    assert_true,
)

from numax import FloatLike, Plain
from numax.optimize.array import bfgs, cg, minimize, nelder_mead

comptime P = Plain[DType.float64, 1]


def rosenbrock[U: FloatLike](v: Array[U, 2]) -> U:
    """The standard two-variable Rosenbrock function, minimum `(1, 1)`."""
    var a = U.one() - v[0]
    var b = v[1] - (v[0] * v[0])
    return a * a + U.constant(100.0) * b * b


def quadratic_bowl[U: FloatLike](v: Array[U, 3]) -> U:
    """`(x-1)^2 + 2(y+2)^2 + 3(z-3)^2`, minimum `(1, -2, 3)`."""
    var a = v[0] - U.one()
    var b = v[1] + U.constant(2.0)
    var c = v[2] - U.constant(3.0)
    return a * a + U.constant(2.0) * b * b + U.constant(3.0) * c * c


def _rosenbrock_start() -> Array[Float64, 2]:
    var start = Array[Float64, 2](fill=0)
    start[0] = -1.2
    start[1] = 1.0
    return start^


def _bowl_start() -> Array[Float64, 3]:
    var start = Array[Float64, 3](fill=0)
    start[0] = -3.0
    start[1] = 4.0
    start[2] = -1.0
    return start^


# --- the dispatcher agrees with the method it names -------------------------


def test_minimize_bfgs_is_exactly_bfgs() raises:
    var direct = bfgs[2, rosenbrock](_rosenbrock_start())
    var dispatched = minimize[2, rosenbrock, method="bfgs"](_rosenbrock_start())

    assert_equal(direct.x[0], dispatched.x[0])
    assert_equal(direct.x[1], dispatched.x[1])
    assert_equal(direct.f_x, dispatched.f_x)
    assert_equal(direct.grad_norm, dispatched.grad_norm)
    assert_equal(direct.iterations, dispatched.iterations)
    assert_equal(direct.converged, dispatched.converged)


def test_minimize_defaults_to_bfgs() raises:
    var direct = bfgs[2, rosenbrock](_rosenbrock_start())
    var dispatched = minimize[2, rosenbrock](_rosenbrock_start())

    assert_equal(direct.x[0], dispatched.x[0])
    assert_equal(direct.f_x, dispatched.f_x)
    assert_equal(direct.iterations, dispatched.iterations)


def test_minimize_cg_is_exactly_cg() raises:
    var direct = cg[2, rosenbrock](_rosenbrock_start())
    var dispatched = minimize[2, rosenbrock, method="cg"](_rosenbrock_start())

    assert_equal(direct.x[0], dispatched.x[0])
    assert_equal(direct.x[1], dispatched.x[1])
    assert_equal(direct.f_x, dispatched.f_x)
    assert_equal(direct.grad_norm, dispatched.grad_norm)
    assert_equal(direct.iterations, dispatched.iterations)


def test_minimize_nelder_mead_is_exactly_nelder_mead() raises:
    var direct = nelder_mead[2, rosenbrock](_rosenbrock_start())
    var dispatched = minimize[2, rosenbrock, method="nelder-mead"](
        _rosenbrock_start()
    )

    assert_equal(direct.x[0], dispatched.x[0])
    assert_equal(direct.x[1], dispatched.x[1])
    assert_equal(direct.f_x, dispatched.f_x)
    assert_equal(direct.iterations, dispatched.iterations)


# --- the per-method defaults survive the dispatch ---------------------------


def test_omitted_tol_reproduces_each_methods_own_default() raises:
    """`tol=None` must mean *this* method's default, not a shared one.

    BFGS and CG stop on `max|grad| < 1e-8`; Nelder-Mead stops on a simplex
    spread below `1e-10`. Handing either the other's number changes where it
    stops, so passing the documented default explicitly has to be
    indistinguishable from passing nothing.
    """
    var bfgs_default = minimize[2, rosenbrock, method="bfgs"](
        _rosenbrock_start()
    )
    var bfgs_explicit = minimize[2, rosenbrock, method="bfgs"](
        _rosenbrock_start(), tol=1e-8, max_iter=200
    )
    assert_equal(bfgs_default.iterations, bfgs_explicit.iterations)
    assert_equal(bfgs_default.f_x, bfgs_explicit.f_x)

    var cg_default = minimize[2, rosenbrock, method="cg"](_rosenbrock_start())
    var cg_explicit = minimize[2, rosenbrock, method="cg"](
        _rosenbrock_start(), tol=1e-8, max_iter=200
    )
    assert_equal(cg_default.iterations, cg_explicit.iterations)

    var nm_default = minimize[2, rosenbrock, method="nelder-mead"](
        _rosenbrock_start()
    )
    var nm_explicit = minimize[2, rosenbrock, method="nelder-mead"](
        _rosenbrock_start(), tol=1e-10, max_iter=1000
    )
    assert_equal(nm_default.iterations, nm_explicit.iterations)
    assert_equal(nm_default.f_x, nm_explicit.f_x)


def test_a_tighter_tol_is_honoured_through_the_dispatch() raises:
    """A loose tolerance must stop earlier than a tight one, or `tol` is
    being dropped on the floor somewhere in the dispatch."""
    var loose = minimize[2, rosenbrock, method="bfgs"](
        _rosenbrock_start(), tol=1e-2
    )
    var tight = minimize[2, rosenbrock, method="bfgs"](
        _rosenbrock_start(), tol=1e-10
    )
    assert_true(loose.converged)
    assert_true(tight.converged)
    assert_true(loose.iterations < tight.iterations)
    assert_true(loose.grad_norm > tight.grad_norm)


def test_max_iter_is_honoured_through_the_dispatch() raises:
    var capped = minimize[2, rosenbrock, method="bfgs"](
        _rosenbrock_start(), max_iter=2
    )
    assert_true(not capped.converged)
    assert_true(capped.iterations <= 2)


# --- an unknown method is an error, not a different algorithm ---------------


def test_an_unknown_method_raises_and_names_itself() raises:
    var raised = False
    try:
        _ = minimize[2, rosenbrock, method="neldermead"](_rosenbrock_start())
    except e:
        raised = True
        assert_true("neldermead" in String(e))
        assert_true("unknown method" in String(e))
    assert_true(raised, "a mistyped method must raise, never fall through")


# --- `cg` on its own merits -------------------------------------------------


def test_cg_minimizes_rosenbrock_from_the_standard_start() raises:
    var result = cg[2, rosenbrock](_rosenbrock_start())
    assert_true(result.converged)
    assert_almost_equal(result.x[0], 1.0, atol=1e-5)
    assert_almost_equal(result.x[1], 1.0, atol=1e-5)


def test_cg_minimizes_a_quadratic_bowl() raises:
    var result = cg[3, quadratic_bowl](_bowl_start())
    assert_true(result.converged)
    assert_almost_equal(result.x[0], 1.0, atol=1e-6)
    assert_almost_equal(result.x[1], -2.0, atol=1e-6)
    assert_almost_equal(result.x[2], 3.0, atol=1e-6)


def test_cg_starting_at_the_minimum_converges_immediately() raises:
    var start = Array[Float64, 3](fill=0)
    start[0] = 1.0
    start[1] = -2.0
    start[2] = 3.0
    var result = cg[3, quadratic_bowl](start^)
    assert_true(result.converged)
    assert_equal(result.iterations, 0)


def test_cg_and_bfgs_land_on_the_same_minimum() raises:
    """Two different algorithms on the same exact gradient. They take
    different paths -- that is the point of having both -- so only the
    destination is compared."""
    var conjugate = cg[2, rosenbrock](_rosenbrock_start())
    var quasi_newton = bfgs[2, rosenbrock](_rosenbrock_start())
    assert_true(conjugate.converged)
    assert_true(quasi_newton.converged)
    assert_almost_equal(conjugate.x[0], quasi_newton.x[0], atol=1e-4)
    assert_almost_equal(conjugate.x[1], quasi_newton.x[1], atol=1e-4)


def test_cg_reports_the_gradient_norm_it_converged_on() raises:
    var result = cg[3, quadratic_bowl](_bowl_start())
    assert_true(result.converged)
    assert_true(result.grad_norm < 1e-8)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
