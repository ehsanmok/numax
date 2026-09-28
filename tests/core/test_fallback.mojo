"""`set_fallback` and `fallback_policy`: what a residency mismatch does.

The policy is process-wide, and the aggregate binary runs every core test
in one process, so each test here restores `"warn"` before it returns.
The notice itself is exercised through `_notice`, the one place every
routine's fallback reports; a GPU-less runner has no device tensor to
mismatch against, and `tests_gpu/` checks the same policy end to end.
"""

from std.testing import TestSuite, assert_equal, assert_raises

from numax.core import fallback_policy, set_fallback
from numax.core._drive import _notice


def test_the_default_policy_is_warn() raises:
    set_fallback("warn")
    assert_equal(fallback_policy(), "warn")
    _notice[False]("probe")


def test_raise_turns_the_notice_into_an_error() raises:
    set_fallback("raise")
    try:
        _notice[True]("probe")
        set_fallback("warn")
        raise Error("expected a raise")
    except e:
        set_fallback("warn")
        assert_equal(
            String(e),
            (
                "numax: probe ran on the host because gpu=True was asked of a"
                " tensor on a CPU context; build the tensor on a GPU context"
            ),
        )


def test_silent_drops_the_notice() raises:
    set_fallback("silent")
    assert_equal(fallback_policy(), "silent")
    _notice[True]("probe")
    _notice[False]("probe")
    set_fallback("warn")


def test_an_unknown_policy_raises() raises:
    with assert_raises(contains="set_fallback"):
        set_fallback("loud")
    assert_equal(fallback_policy(), "warn")


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
