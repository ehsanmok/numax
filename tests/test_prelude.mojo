"""Tests for `numax.prelude`.

Two things are checked. That a star-import actually reaches the surface it
promises -- one call per subsystem, which fails to compile if a name is
missing from the prelude. And that the star-import does *not* take the
builtins with it: `min`, `max`, `sum`, `abs`, `all`, `any` and `round` must
still mean what Mojo means by them in a file that wrote
`from numax.prelude import *`, which is why the tensor reductions of those
names are deliberately left out.
"""

from std.testing import (
    TestSuite,
    assert_almost_equal,
    assert_equal,
    assert_true,
)

from numax.prelude import *

comptime dtype = DType.float64
comptime P = Plain[dtype, 1]


def test_the_builtins_survive_a_star_import() raises:
    # The point of the exclusions in `numax/prelude.mojo`.
    assert_equal(min(3, 4), 3)
    assert_equal(max(3, 4), 4)
    assert_equal(abs(-3), 3)
    assert_equal(round(2.5), 2.0)
    var flags = List[Bool]()
    flags.append(True)
    flags.append(False)
    assert_true(any(flags))
    assert_true(not all(flags))


def test_creation_and_manipulation_are_reachable() raises:
    var xs = linspace[dtype, 5](0.0, 1.0)
    assert_almost_equal(xs[0], Scalar[dtype](0.0))
    assert_almost_equal(xs[4], Scalar[dtype](1.0))

    var grid = reshape[rows=2, cols=3](arange[dtype, 6]())
    assert_equal(grid.dim[0](), 2)
    assert_equal(ravel(grid)[5], Scalar[dtype](5.0))


def test_elementwise_and_logic_are_reachable() raises:
    var xs = full[dtype, 3](4.0)
    assert_almost_equal(sqrt(xs)[0], Scalar[dtype](2.0))
    assert_true(allclose(xs, full[dtype, 3](4.0)))


def test_the_conformers_are_reachable() raises:
    var d = gaussian(Dual[P](P(0.5), P.one()))
    assert_almost_equal(d.value.v, Scalar[dtype](0.7788007830714049))


def test_linalg_and_the_bridge_are_reachable() raises:
    var i3 = eye[dtype, 3]()
    assert_almost_equal(det[P, 3](to_array[P](i3)).v, Scalar[dtype](1.0))
    # Frobenius norm of I3 is sqrt(3).
    assert_almost_equal(
        norm[P, 3](to_array[P](i3)).v, Scalar[dtype](1.7320508075688772)
    )


def test_stats_and_io_are_reachable() raises:
    var xs = arange[dtype, 4]()
    assert_almost_equal(mean(xs), Scalar[dtype](1.5))

    var path = String("/tmp/numax_prelude_test.npy")
    numpy.save(xs, path)
    var back = numpy.load[dtype, 4](path)
    assert_almost_equal(back[3], Scalar[dtype](3.0))


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
