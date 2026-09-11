"""Initial-value problems over a `Tensor` state: `rk4_system`, `dopri5` and
`dopri5_step` for a system whose state is a `Static[dtype, n]`.

**This module is tier 2**, host-orchestrated: the step loop runs on the
host, and every stage combination -- `y + a*k` -- is one `elementwise`
launch on the state's own device, so a large state never round-trips
between stages. `gpu=True` puts those launches on the accelerator; the
right-hand side `f` is the caller's and runs wherever the caller wrote it
to run. The adaptive `solve_ivp` over a `Tensor` state drives `dopri5_step` from
`numax.integrate.integrate`, beside the scalar one, so the name has one
owning module.

This is the method-of-lines case the `Array` tier cannot reach: a PDE
discretized to ten thousand unknowns is a `Tensor` state, and
`numax.integrate.array.rk4_system`'s `Array[T, n]` lives in registers. The
two tiers share the Dormand-Prince tableau and the same stage structure,
and the tests pin the `Tensor` forms against the `Array` ones component by
component on the same problem.

`f` takes `(t, y, ctx)` and returns `dy/dt` as a new tensor -- the
convention `numax.optimize.minimize` uses for its objective, with the time
first. It is a compile-time parameter, so it cannot capture run-time data;
a parameter goes into the state as a component with zero derivative, as it
does at the `Array` tier.

`ponytail:` each stage argument is built by repeated in-place `axpy`
launches, one per `k`, so a `dopri5` step is about thirty small launches
rather than seven fused ones. Fusing each stage's combination into one
body is the upgrade; nothing in the signatures changes when it lands.
"""

from layout import Coord
from max.algorithm.functional import elementwise
from max.gpu.host import DeviceContext
from std.sys.info import simd_width_of

from ..core.array import Static, copy

from .array.ode import (
    _A21,
    _A31,
    _A32,
    _A41,
    _A42,
    _A43,
    _A51,
    _A52,
    _A53,
    _A54,
    _A61,
    _A62,
    _A63,
    _A64,
    _A65,
    _B1,
    _B3,
    _B4,
    _B5,
    _B6,
    _BH1,
    _BH3,
    _BH4,
    _BH5,
    _BH6,
    _BH7,
    _C2,
    _C3,
    _C4,
    _C5,
)


@always_inline
def _target[gpu: Bool]() -> StaticString:
    return "gpu" if gpu else "cpu"


@always_inline
def _width[dtype: DType, gpu: Bool]() -> Int:
    comptime if gpu:
        return 1
    else:
        return simd_width_of[dtype]()


def _axpy_into[
    dtype: DType, n: Int, gpu: Bool
](
    mut out: Static[dtype, n],
    mut x: Static[dtype, n],
    a: Scalar[dtype],
    ctx: DeviceContext,
) raises where dtype.is_floating_point():
    """`out += a * x`, in place, on `out`'s device -- the one primitive
    every Runge-Kutta stage argument and combination is built from."""
    var o = out.view().coalesce()
    var xs = x.view().coalesce()

    @always_inline
    def body[w: Int, alignment: Int = 1](coord: Coord) {var o, var xs, var a}:
        o.store[w](coord, o.load[w](coord) + a * xs.load[w](coord))

    elementwise[simd_width=_width[dtype, gpu](), target=_target[gpu]()](
        body, Coord(n), ctx
    )
    ctx.synchronize()


def rk4_system[
    dtype: DType,
    n: Int,
    f: def(
        Scalar[dtype], Static[dtype, n], DeviceContext
    ) raises thin -> Static[dtype, n],
    num_steps: Int = 100,
    gpu: Bool = False,
](t0: Float64, mut y0: Static[dtype, n], t1: Float64) raises -> Static[
    dtype, n
] where (dtype.is_floating_point() and num_steps >= 1):
    """Integrate the `n`-component system `dy/dt = f(t, y)` from `t0` to
    `t1` in `num_steps` classical fourth-order Runge-Kutta steps, the
    state a `Tensor`. The `Tensor` form of `numax.integrate.array.rk4_system`.

    `t1 < t0` integrates backwards; the step is `(t1 - t0) / num_steps` and
    nothing here assumes its sign.
    """
    var ctx = y0.context()
    var h = (t1 - t0) / Float64(num_steps)
    var y = copy(y0)

    for step in range(num_steps):
        var t = t0 + Float64(step) * h
        var k1 = f(Scalar[dtype](t), y, ctx)

        var arg = copy(y)
        _axpy_into[gpu=gpu](arg, k1, Scalar[dtype](h / 2), ctx)
        var k2 = f(Scalar[dtype](t + h / 2), arg, ctx)

        arg = copy(y)
        _axpy_into[gpu=gpu](arg, k2, Scalar[dtype](h / 2), ctx)
        var k3 = f(Scalar[dtype](t + h / 2), arg, ctx)

        arg = copy(y)
        _axpy_into[gpu=gpu](arg, k3, Scalar[dtype](h), ctx)
        var k4 = f(Scalar[dtype](t + h), arg, ctx)

        _axpy_into[gpu=gpu](y, k1, Scalar[dtype](h / 6), ctx)
        _axpy_into[gpu=gpu](y, k2, Scalar[dtype](h / 3), ctx)
        _axpy_into[gpu=gpu](y, k3, Scalar[dtype](h / 3), ctx)
        _axpy_into[gpu=gpu](y, k4, Scalar[dtype](h / 6), ctx)

    return y^


struct TensorStep[dtype: DType, n: Int](
    Movable where dtype.is_floating_point()
):
    """One Dormand-Prince step's result: the 5th-order state and the
    embedded 4th-order one, whose disagreement is the local error
    estimate. A struct rather than a tuple because a `Tuple` of two
    `Tensor`s cannot be destructured in Mojo 1.0.

    `ponytail:` neither field can be *moved* out either -- Mojo 1.0 rejects
    moving one field from a struct that still owns another ("destroyed out
    of the middle of a value") -- so `dopri5` and `solve_ivp` `copy` the
    accepted state out, one `n`-element device copy per step. A consuming
    accessor that moves both fields out is the upgrade once the language
    allows it.
    """

    var y: Static[Self.dtype, Self.n]
    """The 5th-order solution after the step."""

    var y_hat: Static[Self.dtype, Self.n]
    """The embedded 4th-order solution; `|y - y_hat|` is the error estimate."""

    def __init__(
        out self,
        var y: Static[Self.dtype, Self.n],
        var y_hat: Static[Self.dtype, Self.n],
    ):
        self.y = y^
        self.y_hat = y_hat^


def dopri5_step[
    dtype: DType,
    n: Int,
    f: def(
        Scalar[dtype], Static[dtype, n], DeviceContext
    ) raises thin -> Static[dtype, n],
    gpu: Bool = False,
](t: Float64, mut y: Static[dtype, n], h: Float64) raises -> TensorStep[
    dtype, n
] where dtype.is_floating_point():
    """One Dormand-Prince 5(4) step of the system, returning both
    embedded solutions. The `Tensor` form of
    `numax.integrate.array.dopri5_step`, from the same tableau.

    Public but low-level: `dopri5` drives it at a fixed step and
    `solve_ivp` with adaptive control, so the tableau lives in one place.
    """
    var ctx = y.context()
    var hs = Scalar[dtype](h)

    var k1 = f(Scalar[dtype](t), y, ctx)

    var arg = copy(y)
    _axpy_into[gpu=gpu](arg, k1, hs * Scalar[dtype](_A21), ctx)
    var k2 = f(Scalar[dtype](t + _C2 * h), arg, ctx)

    arg = copy(y)
    _axpy_into[gpu=gpu](arg, k1, hs * Scalar[dtype](_A31), ctx)
    _axpy_into[gpu=gpu](arg, k2, hs * Scalar[dtype](_A32), ctx)
    var k3 = f(Scalar[dtype](t + _C3 * h), arg, ctx)

    arg = copy(y)
    _axpy_into[gpu=gpu](arg, k1, hs * Scalar[dtype](_A41), ctx)
    _axpy_into[gpu=gpu](arg, k2, hs * Scalar[dtype](_A42), ctx)
    _axpy_into[gpu=gpu](arg, k3, hs * Scalar[dtype](_A43), ctx)
    var k4 = f(Scalar[dtype](t + _C4 * h), arg, ctx)

    arg = copy(y)
    _axpy_into[gpu=gpu](arg, k1, hs * Scalar[dtype](_A51), ctx)
    _axpy_into[gpu=gpu](arg, k2, hs * Scalar[dtype](_A52), ctx)
    _axpy_into[gpu=gpu](arg, k3, hs * Scalar[dtype](_A53), ctx)
    _axpy_into[gpu=gpu](arg, k4, hs * Scalar[dtype](_A54), ctx)
    var k5 = f(Scalar[dtype](t + _C5 * h), arg, ctx)

    arg = copy(y)
    _axpy_into[gpu=gpu](arg, k1, hs * Scalar[dtype](_A61), ctx)
    _axpy_into[gpu=gpu](arg, k2, hs * Scalar[dtype](_A62), ctx)
    _axpy_into[gpu=gpu](arg, k3, hs * Scalar[dtype](_A63), ctx)
    _axpy_into[gpu=gpu](arg, k4, hs * Scalar[dtype](_A64), ctx)
    _axpy_into[gpu=gpu](arg, k5, hs * Scalar[dtype](_A65), ctx)
    var k6 = f(Scalar[dtype](t + h), arg, ctx)

    # The 5th-order solution; its weights are row 7 of the tableau.
    var y5 = copy(y)
    _axpy_into[gpu=gpu](y5, k1, hs * Scalar[dtype](_B1), ctx)
    _axpy_into[gpu=gpu](y5, k3, hs * Scalar[dtype](_B3), ctx)
    _axpy_into[gpu=gpu](y5, k4, hs * Scalar[dtype](_B4), ctx)
    _axpy_into[gpu=gpu](y5, k5, hs * Scalar[dtype](_B5), ctx)
    _axpy_into[gpu=gpu](y5, k6, hs * Scalar[dtype](_B6), ctx)

    var k7 = f(Scalar[dtype](t + h), y5, ctx)
    var y4 = copy(y)
    _axpy_into[gpu=gpu](y4, k1, hs * Scalar[dtype](_BH1), ctx)
    _axpy_into[gpu=gpu](y4, k3, hs * Scalar[dtype](_BH3), ctx)
    _axpy_into[gpu=gpu](y4, k4, hs * Scalar[dtype](_BH4), ctx)
    _axpy_into[gpu=gpu](y4, k5, hs * Scalar[dtype](_BH5), ctx)
    _axpy_into[gpu=gpu](y4, k6, hs * Scalar[dtype](_BH6), ctx)
    _axpy_into[gpu=gpu](y4, k7, hs * Scalar[dtype](_BH7), ctx)

    return TensorStep[dtype, n](y5^, y4^)


def dopri5[
    dtype: DType,
    n: Int,
    f: def(
        Scalar[dtype], Static[dtype, n], DeviceContext
    ) raises thin -> Static[dtype, n],
    num_steps: Int = 100,
    gpu: Bool = False,
](t0: Float64, mut y0: Static[dtype, n], t1: Float64) raises -> Static[
    dtype, n
] where (dtype.is_floating_point() and num_steps >= 1):
    """Integrate the system with fixed-step Dormand-Prince 5(4). The
    `Tensor` form of `numax.integrate.array.dopri5`: fifth order for seven
    stages per step, against `rk4_system`'s fourth for four."""
    var h = (t1 - t0) / Float64(num_steps)
    var y = copy(y0)
    for step in range(num_steps):
        var t = t0 + Float64(step) * h
        var stepped = dopri5_step[dtype, n, f, gpu](t, y, h)
        y = copy(stepped.y)
    return y^
