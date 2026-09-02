"""Mathematical constants, at `Float64` and at any `FloatLike`.

`pi` and `e` only. NuMojo also ships the speed of light and the reduced
Planck constant; a numerics library is not a units library, and a caller
who needs physical constants needs a whole table with units attached rather
than two of them without.

The `_at` forms exist because `T.constant(...)` is how a `FloatLike` kernel
names a literal, and writing `T.constant(3.141592653589793)` inline loses
the name at the call site.
"""

from .numeric import FloatLike

comptime pi: Float64 = 3.141592653589793
"""Ratio of a circle's circumference to its diameter."""

comptime e: Float64 = 2.718281828459045
"""Base of the natural logarithm."""


def pi_at[T: FloatLike]() -> T:
    """`pi` as a `T`, for use inside a `FloatLike` kernel."""
    return T.constant(pi)


def e_at[T: FloatLike]() -> T:
    """`e` as a `T`, for use inside a `FloatLike` kernel."""
    return T.constant(e)
