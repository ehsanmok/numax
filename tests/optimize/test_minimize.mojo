"""Tests for `numax.optimize.array`'s SciPy-shaped entry points.

Covers `minimize` and the `cg` method it adds, `minimize_scalar` with the
three one-variable minimizers underneath it, and `root_scalar` with the five
root finders underneath that.

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
from numax.optimize.array import (
    bfgs,
    bisect_tol,
    brent,
    brentq,
    cg,
    fminbound,
    golden,
    halley_tol,
    minimize,
    minimize_scalar,
    nelder_mead,
    newton_tol,
    root_scalar,
    secant,
)

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


# --- the one-variable minimizers -------------------------------------------


def shifted_parabola[U: FloatLike](x: U) -> U:
    """`(x - 3)^2 + 2`, minimum `2` at `x = 3` -- and the minimum is far
    outside the default `(0, 1)` starting pair, which is the point: those
    two numbers are a direction, not a bracket."""
    var d = x - U.constant(3.0)
    return d * d + U.constant(2.0)


def quartic_well[U: FloatLike](x: U) -> U:
    """`x^4 - 3x^2 + x`, whose global minimum is near `-1.30`. Not
    quadratic anywhere near it, so a parabolic step has to be corrected."""
    var x2 = x * x
    return x2 * x2 - U.constant(3.0) * x2 + x


def descending[U: FloatLike](x: U) -> U:
    """`-x`, which has no minimum at all. The bracketing search must give
    up rather than run away."""
    return -x


def test_brent_finds_a_minimum_far_outside_the_starting_pair() raises:
    var result = brent[shifted_parabola]()
    assert_true(result.converged)
    assert_almost_equal(result.x, 3.0, atol=1e-6)
    assert_almost_equal(result.f_x, 2.0, atol=1e-12)


def test_golden_agrees_with_brent() raises:
    """Two different refinements of the same bracket. They take different
    numbers of evaluations -- that is why both exist -- so only the answer
    is compared."""
    var by_brent = brent[quartic_well](-2.0, -1.0)
    var by_golden = golden[quartic_well](-2.0, -1.0)
    assert_true(by_brent.converged)
    assert_true(by_golden.converged)
    assert_almost_equal(by_brent.x, by_golden.x, atol=1e-6)


def test_brent_beats_golden_on_evaluation_count() raises:
    """The reason to default to `brent`: interpolation converges faster
    than a fixed 0.618 shrink on a function smooth near its minimum."""
    var by_brent = brent[shifted_parabola]()
    var by_golden = golden[shifted_parabola]()
    assert_true(by_brent.converged)
    assert_true(by_golden.converged)
    assert_true(by_brent.iterations < by_golden.iterations)


def test_an_unbounded_descent_is_reported_not_run_away_from() raises:
    var result = brent[descending]()
    assert_true(not result.converged)
    assert_equal(result.iterations, 0)


def test_fminbound_keeps_the_answer_inside_the_bounds() raises:
    """The minimum of `(x-3)^2 + 2` is at 3, which is outside `[-1, 1]`.
    A bounded search must return the boundary, not walk to 3."""
    var result = fminbound[shifted_parabola](-1.0, 1.0)
    assert_true(result.converged)
    assert_true(result.x <= 1.0)
    assert_almost_equal(result.x, 1.0, atol=1e-4)


def test_fminbound_finds_an_interior_minimum() raises:
    var result = fminbound[shifted_parabola](0.0, 10.0)
    assert_true(result.converged)
    assert_almost_equal(result.x, 3.0, atol=1e-4)


# --- minimize_scalar dispatches to exactly those ---------------------------


def test_minimize_scalar_brent_is_exactly_brent() raises:
    var direct = brent[quartic_well](-2.0, -1.0)
    var dispatched = minimize_scalar[quartic_well, method="brent"](
        bracket=(-2.0, -1.0)
    )
    assert_equal(direct.x, dispatched.x)
    assert_equal(direct.f_x, dispatched.f_x)
    assert_equal(direct.iterations, dispatched.iterations)


def test_minimize_scalar_defaults_to_brent_with_the_default_bracket() raises:
    var direct = brent[shifted_parabola]()
    var dispatched = minimize_scalar[shifted_parabola]()
    assert_equal(direct.x, dispatched.x)
    assert_equal(direct.iterations, dispatched.iterations)


def test_minimize_scalar_golden_is_exactly_golden() raises:
    var direct = golden[quartic_well](-2.0, -1.0)
    var dispatched = minimize_scalar[quartic_well, method="golden"](
        bracket=(-2.0, -1.0)
    )
    assert_equal(direct.x, dispatched.x)
    assert_equal(direct.iterations, dispatched.iterations)


def test_minimize_scalar_bounded_is_exactly_fminbound() raises:
    var direct = fminbound[shifted_parabola](-1.0, 1.0)
    var dispatched = minimize_scalar[shifted_parabola, method="bounded"](
        bounds=(-1.0, 1.0)
    )
    assert_equal(direct.x, dispatched.x)
    assert_equal(direct.iterations, dispatched.iterations)


def test_bounded_without_bounds_raises() raises:
    var raised = False
    try:
        _ = minimize_scalar[shifted_parabola, method="bounded"]()
    except e:
        raised = True
        assert_true("requires 'bounds'" in String(e))
    assert_true(raised)


def test_a_bracket_is_not_silently_accepted_as_bounds() raises:
    """Passing `bounds` to a bracketing method must raise. Reinterpreting
    one as the other is how a constrained problem quietly returns an answer
    outside its range."""
    var raised = False
    try:
        _ = minimize_scalar[shifted_parabola, method="brent"](
            bounds=(-1.0, 1.0)
        )
    except e:
        raised = True
        assert_true("bounds" in String(e))
    assert_true(raised)


def test_reversed_bounds_raise() raises:
    var raised = False
    try:
        _ = minimize_scalar[shifted_parabola, method="bounded"](
            bounds=(1.0, -1.0)
        )
    except e:
        raised = True
        assert_true("increasing" in String(e))
    assert_true(raised)


def test_minimize_scalar_rejects_an_unknown_method() raises:
    var raised = False
    try:
        _ = minimize_scalar[shifted_parabola, method="Brent"]()
    except e:
        raised = True
        assert_true("unknown method" in String(e))
    assert_true(raised)


def test_scalar_tol_defaults_per_method() raises:
    """`brent` defaults to sqrt(eps); `bounded` to SciPy's looser 1e-5.
    Passing each explicitly must be indistinguishable from passing
    nothing."""
    var brent_default = minimize_scalar[shifted_parabola, method="brent"]()
    var brent_explicit = minimize_scalar[shifted_parabola, method="brent"](
        tol=1.48e-8, max_iter=500
    )
    assert_equal(brent_default.x, brent_explicit.x)
    assert_equal(brent_default.iterations, brent_explicit.iterations)

    var bounded_default = minimize_scalar[shifted_parabola, method="bounded"](
        bounds=(0.0, 10.0)
    )
    var bounded_explicit = minimize_scalar[shifted_parabola, method="bounded"](
        bounds=(0.0, 10.0), tol=1e-5, max_iter=500
    )
    assert_equal(bounded_default.x, bounded_explicit.x)
    assert_equal(bounded_default.iterations, bounded_explicit.iterations)


# --- root_scalar and the tolerance siblings it adds -------------------------


def cos_minus_x[U: FloatLike](x: U) -> U:
    """`cos(x) - x`, whose root is the Dottie number, 0.7390851332151607."""
    return x.cos() - x


comptime DOTTIE = 0.7390851332151607


def test_halley_tol_converges_faster_than_newton_tol() raises:
    """Cubic against quadratic: from the same start, Halley must reach the
    same root in strictly fewer steps. That is the only reason to pay for a
    second derivative."""
    var by_newton = newton_tol[cos_minus_x](0.5)
    var by_halley = halley_tol[cos_minus_x](0.5)
    assert_true(by_newton.converged)
    assert_true(by_halley.converged)
    assert_almost_equal(by_halley.x, DOTTIE, atol=1e-14)
    assert_true(by_halley.iterations < by_newton.iterations)


def test_secant_finds_the_root_without_a_derivative() raises:
    var result = secant[cos_minus_x](0.5)
    assert_true(result.converged)
    assert_almost_equal(result.x, DOTTIE, atol=1e-12)


def test_secant_accepts_an_explicit_second_point() raises:
    var result = secant[cos_minus_x](0.0, x1=2.0)
    assert_true(result.converged)
    assert_almost_equal(result.x, DOTTIE, atol=1e-12)


def test_bisect_tol_halves_to_the_root() raises:
    var result = bisect_tol[cos_minus_x](0.0, 2.0)
    assert_true(result.converged)
    assert_almost_equal(result.x, DOTTIE, atol=1e-11)


def test_bisect_tol_reports_a_bracket_without_a_sign_change() raises:
    """`cos(x) - x` is positive at both 0.0 and 0.5, so there is no
    guarantee to be had and the fixed-iteration sibling could not say so."""
    var result = bisect_tol[cos_minus_x](0.0, 0.5)
    assert_true(not result.converged)
    assert_equal(result.iterations, 0)


def test_brentq_beats_bisect_tol_on_the_same_bracket() raises:
    """The reason `root_scalar` defaults to brentq: same bracket, same
    guarantee, superlinear instead of one bit per iteration."""
    var by_brentq = brentq[cos_minus_x](0.0, 2.0)
    var by_bisect = bisect_tol[cos_minus_x](0.0, 2.0)
    assert_true(by_brentq.converged)
    assert_true(by_bisect.converged)
    assert_true(by_brentq.iterations < by_bisect.iterations)


def test_all_five_methods_find_the_same_root() raises:
    var by_brentq = root_scalar[cos_minus_x](bracket=(0.0, 2.0))
    var by_bisect = root_scalar[cos_minus_x, method="bisect"](
        bracket=(0.0, 2.0)
    )
    var by_newton = root_scalar[cos_minus_x, method="newton"](x0=0.5)
    var by_halley = root_scalar[cos_minus_x, method="halley"](x0=0.5)
    var by_secant = root_scalar[cos_minus_x, method="secant"](x0=0.5)

    assert_almost_equal(by_brentq.x, DOTTIE, atol=1e-11)
    assert_almost_equal(by_bisect.x, DOTTIE, atol=1e-11)
    assert_almost_equal(by_newton.x, DOTTIE, atol=1e-11)
    assert_almost_equal(by_halley.x, DOTTIE, atol=1e-11)
    assert_almost_equal(by_secant.x, DOTTIE, atol=1e-11)


def test_root_scalar_dispatches_to_exactly_the_named_function() raises:
    var direct_brentq = brentq[cos_minus_x](0.0, 2.0)
    var via_brentq = root_scalar[cos_minus_x, method="brentq"](
        bracket=(0.0, 2.0)
    )
    assert_equal(direct_brentq.x, via_brentq.x)
    assert_equal(direct_brentq.iterations, via_brentq.iterations)

    var direct_halley = halley_tol[cos_minus_x](0.5)
    var via_halley = root_scalar[cos_minus_x, method="halley"](x0=0.5)
    assert_equal(direct_halley.x, via_halley.x)
    assert_equal(direct_halley.iterations, via_halley.iterations)

    var direct_secant = secant[cos_minus_x](0.5)
    var via_secant = root_scalar[cos_minus_x, method="secant"](x0=0.5)
    assert_equal(direct_secant.x, via_secant.x)
    assert_equal(direct_secant.iterations, via_secant.iterations)


def test_a_bracketing_method_without_a_bracket_raises() raises:
    var raised = False
    try:
        _ = root_scalar[cos_minus_x, method="brentq"](x0=0.5)
    except e:
        raised = True
        assert_true("requires 'bracket'" in String(e))
    assert_true(raised)


def test_a_guess_method_without_x0_raises() raises:
    var raised = False
    try:
        _ = root_scalar[cos_minus_x, method="newton"](bracket=(0.0, 2.0))
    except e:
        raised = True
        assert_true("requires 'x0'" in String(e))
    assert_true(raised)


def test_a_guess_method_given_a_bracket_raises() raises:
    """Newton may leave any interval it starts in, so accepting a bracket
    would promise a guarantee it does not have."""
    var raised = False
    try:
        _ = root_scalar[cos_minus_x, method="newton"](
            x0=0.5, bracket=(0.0, 2.0)
        )
    except e:
        raised = True
        assert_true("not a 'bracket'" in String(e))
    assert_true(raised)


def test_root_scalar_rejects_an_unknown_method() raises:
    var raised = False
    try:
        _ = root_scalar[cos_minus_x, method="brent"](bracket=(0.0, 2.0))
    except e:
        raised = True
        assert_true("unknown method" in String(e))
    assert_true(raised)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
