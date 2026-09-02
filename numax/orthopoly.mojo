"""Hermite, Laguerre, and Chebyshev polynomials.

The other classical orthogonal families next to `numax.legendre`, which
keeps its own file because `numax.quadrature` depends on it specifically
(Gauss-Legendre nodes are roots of `P_n`). All four are the same algorithm
with different coefficients -- a three-term recurrence upward from two
known starting polynomials -- and, like Legendre, they're exact rather than
approximations, so there is no error bound to document and no domain
restriction to observe.

The degree `n` is a scalar control parameter identical in every SIMD lane
and every GPU thread, so looping `n` times is not the per-lane branching
the fixed-work invariant rules out.

Each recurrence carries its loop counter as a running `T` value rather
than converting the `Int` index per iteration. `T.constant` takes a
`Float64`, and converting a *runtime* `Int` into one emits an
int64-to-double instruction Metal rejects ("air.convert.f.f64.s.i64 has
Metal-unsupported instructions"), which would leave these functions
CPU-only. Counting in `T` keeps every value `dtype`-native; the same
applies to `numax.legendre`, `numax.beta`, and `numax.quadrature`.

CUDA has no such restriction, so this reads as unnecessary on an NVIDIA
box -- it is not. The constraint is what makes these kernels launchable on
*every* backend `DeviceContext` supports rather than on the permissive
ones, and counting in `T` costs nothing on the backends that would have
allowed the conversion. Don't "simplify" it back to
`T.constant(Float64(i))` after testing on CUDA alone.

Conventions here are the common ones, and they matter because the
alternatives differ by more than a scale factor:

- `hermite_h` is the *physicists'* Hermite polynomial (`H_2 = 4x^2 - 2`),
  orthogonal against `exp(-x^2)`, not the probabilists' `He_n`.
- `laguerre_l` is the plain Laguerre polynomial `L_n`, not the generalized
  `L_n^(alpha)`.
- `chebyshev_t`/`chebyshev_u` are the first and second kinds. Both are
  computed by recurrence rather than through `cos(n*acos(x))`, which is
  only valid on `[-1, 1]` and would need an `acos` the trait doesn't have.

As with `legendre_p`, no derivative functions are provided: evaluate at
`Dual` for those.
"""

from .numeric import FloatLike


def hermite_h[T: FloatLike](n: Int, x: T) -> T:
    """`H_n(x)`, the physicists' Hermite polynomial, by the recurrence
    `H_{k+1} = 2x*H_k - 2k*H_{k-1}` from `H_0 = 1`, `H_1 = 2x`.

    Grows very fast in both `n` and `x` (`H_10(5)` is already about
    `3.6e9`), so a low-precision `dtype` overflows here sooner than the
    other families in this file.
    """
    if n <= 0:
        return T.one()

    var h_prev = T.one()
    var h_curr = T.constant(2.0) * x
    var kf = T.one()

    for _ in range(1, n):
        var h_next = T.constant(2.0) * x * h_curr + (
            -(T.constant(2.0) * kf * h_prev)
        )
        h_prev = h_curr.copy()
        h_curr = h_next.copy()
        kf = kf + T.one()

    return h_curr^


def laguerre_l[T: FloatLike](n: Int, x: T) -> T:
    """`L_n(x)`, the Laguerre polynomial, by the recurrence
    `(k+1)L_{k+1} = (2k+1-x)L_k - k*L_{k-1}` from `L_0 = 1`, `L_1 = 1-x`.
    """
    if n <= 0:
        return T.one()

    var l_prev = T.one()
    var l_curr = T.one() + (-x)
    var kf = T.one()

    for _ in range(1, n):
        var l_next = (
            (T.constant(2.0) * kf + T.one() + (-x)) * l_curr + (-(kf * l_prev))
        ) / (kf + T.one())
        l_prev = l_curr.copy()
        l_curr = l_next.copy()
        kf = kf + T.one()

    return l_curr^


def chebyshev_t[T: FloatLike](n: Int, x: T) -> T:
    """`T_n(x)`, the Chebyshev polynomial of the first kind, by the
    recurrence `T_{k+1} = 2x*T_k - T_{k-1}` from `T_0 = 1`, `T_1 = x`.

    On `[-1, 1]` this equals `cos(n*acos(x))` and stays within `[-1, 1]`;
    outside that interval it grows quickly, which is correct rather than a
    failure -- the recurrence is the polynomial's definition, and the
    trigonometric form is what stops being valid.
    """
    if n <= 0:
        return T.one()

    var t_prev = T.one()
    var t_curr = x.copy()

    for _ in range(1, n):
        var t_next = T.constant(2.0) * x * t_curr + (-t_prev)
        t_prev = t_curr.copy()
        t_curr = t_next.copy()

    return t_curr^


def chebyshev_u[T: FloatLike](n: Int, x: T) -> T:
    """`U_n(x)`, the Chebyshev polynomial of the second kind -- the same
    recurrence as `chebyshev_t` from a different start (`U_0 = 1`,
    `U_1 = 2x`)."""
    if n <= 0:
        return T.one()

    var u_prev = T.one()
    var u_curr = T.constant(2.0) * x

    for _ in range(1, n):
        var u_next = T.constant(2.0) * x * u_curr + (-u_prev)
        u_prev = u_curr.copy()
        u_curr = u_next.copy()

    return u_curr^
