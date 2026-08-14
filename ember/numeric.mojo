"""The trait every numeric kernel in `ember` is written against.

A kernel bounded on `FloatLike` names no SIMD width, no dtype, and no
instruction set -- only the handful of operations it actually needs. The
_type_ it gets called with then decides what you get out of it: plain SIMD
(`Plain`), a value paired with its derivative (`Dual`), or a value carried
to roughly double precision (`Compensated`). Swap the type, not the kernel.

The trait is intentionally small: enough to build the `special` module's
activations and any similar closed-form kernel on it, and no more. Add an
operation here only when a kernel genuinely needs it -- every method added
is one more thing every future `FloatLike` type must implement.
"""


trait FloatLike(Copyable, Deinitable, Movable):
    """The minimal arithmetic a real-valued vectorized kernel needs."""

    @staticmethod
    def one() -> Self:
        """The multiplicative identity, at whatever width/precision `Self` is.
        """
        ...

    def __add__(self, rhs: Self) -> Self:
        ...

    def __mul__(self, rhs: Self) -> Self:
        ...

    def __neg__(self) -> Self:
        ...

    def __truediv__(self, rhs: Self) -> Self:
        ...

    def exp(self) -> Self:
        ...

    def ln(self) -> Self:
        """The natural logarithm. Only meaningful for `self > 0`."""
        ...

    @staticmethod
    def constant(v: Float64) -> Self:
        """Build a literal coefficient as `Self`.

        Lets a kernel embed constants (polynomial coefficients, etc.)
        without knowing how `Self` represents a value -- `Dual` gives the
        constant a zero derivative, `Compensated` gets to decide how much
        of `v` survives the trip through a single `dtype` lane.
        """
        ...

    def abs(self) -> Self:
        ...

    def copysign(self, sign_source: Self) -> Self:
        """Return `self`'s magnitude with `sign_source`'s sign.

        Lets a kernel derived for `x >= 0` extend to all `x` via
        `result.copysign(x)`, without ever branching on `Self` itself --
        which may hold a SIMD vector with mixed-sign lanes.
        """
        ...
