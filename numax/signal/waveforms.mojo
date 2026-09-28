"""Periodic and swept waveforms over `numax.core.tensor.Tensor`:
`sawtooth`, `square` and `chirp`, `scipy.signal`'s generators.

**Tier 2 in shape, both targets in fact**, like `numax.core.elementwise`:
each is one pass over the sample times `t`, run through `numax.core._drive`
on the target `gpu: Bool` names, and the per-sample formula is shared by
the device body and the host fallback, so the two agree. The phase is
computed at `t`'s own dtype; SciPy computes in `float64` whatever the
input, so at `float32` a long chirp drifts from SciPy's by the phase's
rounding, which grows with `t`.

`square` shares its name with `numax.core.square`, the elementwise
`x * x`, so the flat surface keeps the arithmetic one and the wave is
reached as `numax.signal.square`.

## The MAX gate

Nothing: MAX has no signal generators. An **extend**, one launch each.
"""

from std.math import cos as _cos, floor as _floor
from std.utils.numerics import nan as _nan

from layout import Coord
from layout.tile_layout import TensorLayout

from ..core.libm import exp as _exp, log as _log
from ..core.tensorlike import TensorLike, is_row_major
from ..core.tensor import Tensor
from ..core._drive import _check_device, _flat, _flat_out, _launch, _notice

comptime _PI = 3.141592653589793
comptime _TWO_PI = 6.283185307179586


def _wave_value[
    dtype: DType, kind: StaticString
](
    x: Scalar[dtype],
    p0: Scalar[dtype],
    p1: Scalar[dtype],
    p2: Scalar[dtype],
    p3: Scalar[dtype],
    p4: Scalar[dtype],
) -> Scalar[dtype] where dtype.is_floating_point():
    """The waveform `kind` at time `x`, its constants precomputed on the
    host and passed as `p0..p4`. The one formula both paths run."""
    var two_pi = Scalar[dtype](_TWO_PI)
    var pi = Scalar[dtype](_PI)
    comptime if kind == "sawtooth":
        var w = p0
        if w > 1 or w < 0:
            return _nan[dtype]()
        var tmod = x - two_pi * _floor(x / two_pi)
        if tmod < w * two_pi:
            return tmod / (pi * w) - 1
        return (pi * (w + 1) - tmod) / (pi * (1 - w))
    elif kind == "square":
        var duty = p0
        if duty > 1 or duty < 0:
            return _nan[dtype]()
        var tmod = x - two_pi * _floor(x / two_pi)
        return Scalar[dtype](1) if tmod < duty * two_pi else Scalar[dtype](-1)
    elif kind == "linear":
        # `p0 = f0`, `p1 = beta`, `p2 = phi` in radians.
        return _cos(two_pi * (p0 * x + 0.5 * p1 * x * x) + p2)
    elif kind == "quadratic":
        # `p0 = f0`, `p1 = beta`, `p2 = phi`, `p3 = f1`, `p4 = t1` or `0`
        # for the vertex at zero.
        if p4 == 0:
            return _cos(two_pi * (p0 * x + p1 * x * x * x / 3) + p2)
        var r = p4 - x
        return _cos(
            two_pi * (p3 * x + p1 * (r * r * r - p4 * p4 * p4) / 3) + p2
        )
    elif kind == "logarithmic":
        # `p0 = f0`, `p1 = beta = t1 / ln(f1/f0)`, `p2 = phi`,
        # `p3 = ln(f1/f0) / t1`, `p4 = 1` when `f0 == f1`.
        if p4 != 0:
            return _cos(two_pi * p0 * x + p2)
        return _cos(two_pi * p1 * p0 * (_exp(x * p3) - 1) + p2)
    else:
        # `"hyperbolic"`: `p0 = f0`, `p1 = sing`, `p2 = phi`, `p4 = 1` when
        # `f0 == f1`.
        if p4 != 0:
            return _cos(two_pi * p0 * x + p2)
        return _cos(two_pi * (-p1 * p0) * _log(abs(1 - x / p1)) + p2)


def _waveform[
    T: TensorLike, kind: StaticString, gpu: Bool, name: StaticString
](
    t: T,
    p0: Float64,
    p1: Float64 = 0,
    p2: Float64 = 0,
    p3: Float64 = 0,
    p4: Float64 = 0,
) raises -> Tensor[T.dtype, T.LayoutType] where (
    is_row_major[T] and T.dtype.is_floating_point()
):
    """`kind` at every sample of `t`, on the target `gpu` names."""
    comptime dtype = T.dtype
    comptime LayoutType = T.LayoutType
    var c0 = Scalar[dtype](p0)
    var c1 = Scalar[dtype](p1)
    var c2 = Scalar[dtype](p2)
    var c3 = Scalar[dtype](p3)
    var c4 = Scalar[dtype](p4)
    if not _check_device[gpu=gpu](t):
        _notice[gpu](name)
        var values = t.to_host()
        for i in range(len(values)):
            values[i] = _wave_value[dtype, kind](values[i], c0, c1, c2, c3, c4)
        return Tensor[dtype, LayoutType](t.tile().layout, values^, t.context())
    var ctx = t.context()
    var out = Tensor[dtype, LayoutType]._uninitialized(ctx, t.tile().layout)
    var xs = _flat(t)
    var ys = _flat_out(out)

    @always_inline
    def body[
        width: Int, alignment: Int = 1
    ](coord: Coord) {var xs, var ys, var c0, var c1, var c2, var c3, var c4}:
        var x = xs.load[1](coord)[0]
        ys.store[1](
            coord,
            SIMD[dtype, 1](_wave_value[dtype, kind](x, c0, c1, c2, c3, c4)),
        )

    _launch[gpu=gpu, lanes=1](body, t.size(), ctx)
    return out^


def sawtooth[
    T: TensorLike, gpu: Bool = False
](t: T, width: Float64 = 1.0) raises -> Tensor[T.dtype, T.LayoutType] where (
    is_row_major[T] and T.dtype.is_floating_point()
):
    """A sawtooth wave of period `2 pi` between `-1` and `1`.
    `scipy.signal.sawtooth(t, width)`.

    It rises from `-1` to `1` over the first `width` of each period and
    falls back over the rest: `width = 1` (the default) is a rising ramp,
    `0` a falling one, `0.5` a triangle. A `width` outside `[0, 1]` gives
    NaN, as SciPy's does.

    Parameters:
        T: The `TensorLike` type of `t`, row-major, floating-point.
        gpu: `True` runs on the tensor's device, `False` on the host; a
            residency mismatch falls back to the host with a `stderr` notice.

    Args:
        t: The sample times, in radians of the period.
        width: The rising fraction of each period, in `[0, 1]`.

    Returns:
        A new tensor at `t`'s layout holding the wave at each time.

    Raises:
        If allocating the result or launching the walk fails, or on a
        residency mismatch under the `"raise"` fallback policy.
    """
    return _waveform[T, "sawtooth", gpu, "sawtooth"](t, width)


def square[
    T: TensorLike, gpu: Bool = False
](t: T, duty: Float64 = 0.5) raises -> Tensor[T.dtype, T.LayoutType] where (
    is_row_major[T] and T.dtype.is_floating_point()
):
    """A square wave of period `2 pi`: `1` for the first `duty` of each
    period and `-1` for the rest. `scipy.signal.square(t, duty)`.

    A `duty` outside `[0, 1]` gives NaN, as SciPy's does.

    Parameters:
        T: The `TensorLike` type of `t`, row-major, floating-point.
        gpu: `True` runs on the tensor's device, `False` on the host; a
            residency mismatch falls back to the host with a `stderr` notice.

    Args:
        t: The sample times, in radians of the period.
        duty: The fraction of each period at `1`, in `[0, 1]`.

    Returns:
        A new tensor at `t`'s layout holding the wave at each time.

    Raises:
        If allocating the result or launching the walk fails, or on a
        residency mismatch under the `"raise"` fallback policy.
    """
    return _waveform[T, "square", gpu, "square"](t, duty)


def chirp[
    T: TensorLike, method: StaticString = "linear", gpu: Bool = False
](
    t: T,
    f0: Float64,
    t1: Float64,
    f1: Float64,
    phi: Float64 = 0.0,
    vertex_zero: Bool = True,
) raises -> Tensor[T.dtype, T.LayoutType] where (
    is_row_major[T] and T.dtype.is_floating_point()
):
    """A swept-frequency cosine: frequency `f0` at `t = 0` and `f1` at
    `t = t1`. `scipy.signal.chirp(t, f0, t1, f1, method, phi,
    vertex_zero)`.

    `cos(phase(t) + phi)`, with `phi` in degrees as in SciPy and the phase
    SciPy's for each `method`: `"linear"` (frequency linear in `t`),
    `"quadratic"` (a parabola, its vertex at `t = 0` when `vertex_zero`,
    else at `t1`), `"logarithmic"` (geometric, `f0` and `f1` nonzero and
    of one sign) and `"hyperbolic"` (`f0` and `f1` nonzero). The method is
    a compile-time parameter, so each is its own kernel with no branch on
    it; an unknown one is a compile error.

    Parameters:
        T: The `TensorLike` type of `t`, row-major, floating-point.
        method: `"linear"` (the default), `"quadratic"`, `"logarithmic"` or
            `"hyperbolic"`.
        gpu: `True` runs on the tensor's device, `False` on the host; a
            residency mismatch falls back to the host with a `stderr` notice.

    Args:
        t: The sample times.
        f0: The frequency at `t = 0`.
        t1: The time at which `f1` is reached.
        f1: The frequency at `t = t1`.
        phi: The phase offset in degrees.
        vertex_zero: For `"quadratic"`, whether the parabola's vertex is at
            `t = 0` (else at `t1`).

    Returns:
        A new tensor at `t`'s layout holding the chirp at each time.

    Raises:
        If `f0` and `f1` do not suit `"logarithmic"` or `"hyperbolic"`, if
        allocating the result or launching the walk fails, or on a
        residency mismatch under the `"raise"` fallback policy.
    """
    comptime assert (
        method == "linear"
        or method == "quadratic"
        or method == "logarithmic"
        or method == "hyperbolic"
    ), 'chirp: method is "linear", "quadratic", "logarithmic" or "hyperbolic"'
    var radians = phi * _PI / 180.0
    comptime if method == "linear":
        return _waveform[T, "linear", gpu, "chirp"](
            t, f0, (f1 - f0) / t1, radians
        )
    elif method == "quadratic":
        return _waveform[T, "quadratic", gpu, "chirp"](
            t,
            f0,
            (f1 - f0) / (t1 * t1),
            radians,
            f1,
            0.0 if vertex_zero else t1,
        )
    elif method == "logarithmic":
        if f0 * f1 <= 0.0:
            raise Error(
                "chirp: for a logarithmic chirp, f0 and f1 must be nonzero"
                " and have the same sign"
            )
        if f0 == f1:
            return _waveform[T, "logarithmic", gpu, "chirp"](
                t, f0, 0.0, radians, 0.0, 1.0
            )
        var ln_ratio = _log(Float64(f1 / f0))
        return _waveform[T, "logarithmic", gpu, "chirp"](
            t, f0, t1 / ln_ratio, radians, ln_ratio / t1, 0.0
        )
    else:
        if f0 == 0 or f1 == 0:
            raise Error(
                "chirp: for a hyperbolic chirp, f0 and f1 must be nonzero"
            )
        if f0 == f1:
            return _waveform[T, "hyperbolic", gpu, "chirp"](
                t, f0, 1.0, radians, 0.0, 1.0
            )
        return _waveform[T, "hyperbolic", gpu, "chirp"](
            t, f0, -f1 * t1 / (f0 - f1), radians, 0.0, 0.0
        )
