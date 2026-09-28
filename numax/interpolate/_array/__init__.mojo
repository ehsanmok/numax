"""Private: the register tier of `numax.interpolate`, over `Array[T, n]`
and `FloatLike` scalars.

Nothing imports this package by name. `numax.interpolate` re-exports its
names, and the one it shares with the `Tensor` tier, `horner`, is an
overload in `numax/interpolate/interp.mojo` that forwards here. The
algorithms, their tier and their bounds are documented on the functions
in `interp.mojo` below; every routine is a `FloatLike` kernel with a
compile-time size, so it differentiates at `Dual` and runs per SIMD lane
inside a GPU thread.
"""

from .interp import (
    ArrayChebyshev,
    ArrayCubicSpline,
    chebyshev_eval,
    chebyshev_fit,
    cubic_spline_eval,
    cubic_spline_moments,
    horner,
)
