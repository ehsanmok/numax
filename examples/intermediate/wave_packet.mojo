"""A 2-D Gaussian wave packet, and its width differentiated exactly.

Builds `psi(x, y) = exp(-(x^2 + y^2) / 2 sigma^2) * exp(i k x)` on a grid,
reduces it to the probability density `|psi|^2`, and checks its
normalization. A cut along `y = 0` shows the density is a plain Gaussian --
the carrier cannot appear in it, since `|exp(i k x)|` is 1 -- while counting
the sign changes of `Re psi` along the same cut recovers the carrier. Then it
computes the spread `<x^2>` of that same packet against `FloatLike` and
evaluates it at `Dual`, which returns
`d<x^2>/d(sigma)` from the same call -- checked against the analytic answer,
which for this packet is exactly `sigma`.

Two limitations are visible in the code rather than hidden by it:

- **There is no complex tensor.** `Complex[T]` is a conformer for scalars
  and register-resident `Array`s; `Tensor` is keyed on a `DType`. So the
  real and imaginary parts are two real fields here, and every operation on
  the wavefunction touches both.
- **`numax.fft.fft2` cannot take a `Tensor`.** It works over
  `Array[Complex[T], n*n]` in registers, so a momentum-space step would mean
  copying the grid out element by element. This example stays in real space.
"""

from std.math import cos, exp, sin, sqrt

from max.gpu.host import DeviceContext

from numax import Dual, FloatLike, Plain
from numax.core.array import Tensor, linspace, meshgrid
from numax.core.ops import add, multiply
from numax.stats import sum
from numax.core.tensor import map

comptime dtype = DType.float64
comptime P = Plain[dtype, 1]

comptime n = 48
"""Grid points per axis. Even, so the cut at y = 0 sits between rows."""

comptime extent = 6.0
comptime sigma = 1.5
comptime k_x = 3.0

comptime moment_points = 4096
"""Quadrature points for `<x^2>`. Fixed, so `second_moment` stays tier 1."""

comptime moment_extent = 12.0


def packet_re[
    dtype: DType, w: Int
](x: SIMD[dtype, w], y: SIMD[dtype, w]) -> SIMD[
    dtype, w
] where dtype.is_floating_point():
    """The real part of the packet at `(x, y)`."""
    var envelope = exp(-(x * x + y * y) / SIMD[dtype, w](2 * sigma * sigma))
    return envelope * cos(SIMD[dtype, w](k_x) * x)


def packet_im[
    dtype: DType, w: Int
](x: SIMD[dtype, w], y: SIMD[dtype, w]) -> SIMD[
    dtype, w
] where dtype.is_floating_point():
    """The imaginary part of the packet at `(x, y)`."""
    var envelope = exp(-(x * x + y * y) / SIMD[dtype, w](2 * sigma * sigma))
    return envelope * sin(SIMD[dtype, w](k_x) * x)


def second_moment[T: FloatLike](width: T) -> T:
    """`<x^2>` of the packet whose envelope width is `width`.

    A midpoint sum over a fixed number of points, so the iteration count is
    a compile-time constant and this stays tier 1 -- which is what lets it
    be called at `Dual` and at `Plain` with the same body. The density is
    `|psi|^2 = exp(-x^2 / width^2)`, whose second moment is `width^2 / 2`.
    """
    var step = T.constant(2.0 * moment_extent / Float64(moment_points))
    var weighted = T.constant(0.0)
    var total = T.constant(0.0)
    var inv_width_sq = T.one() / (width * width)
    for i in range(moment_points):
        var x = T.constant(-moment_extent) + step * T.constant(Float64(i) + 0.5)
        var density = (-(x * x * inv_width_sq)).exp()
        weighted = weighted + x * x * density
        total = total + density
    return weighted / total


def main() raises:
    var ctx = DeviceContext(api="cpu")

    # --- the grid: two coordinate vectors, one pair of coordinate fields ---
    var axis_x = linspace[dtype, n](-extent, extent, ctx=ctx)
    var axis_y = linspace[dtype, n](-extent, extent, ctx=ctx)
    var grid = meshgrid(axis_x, axis_y)

    # --- the wavefunction, as two real fields ---
    var re = Tensor[dtype, n, n](ctx)
    var im = Tensor[dtype, n, n](ctx)
    var xs = grid[0].view()
    var ys = grid[1].view()
    var re_view = re.view()
    var im_view = im.view()
    map[step=packet_re[dtype, _], width=4](xs, ys, re_view)
    map[step=packet_im[dtype, _], width=4](xs, ys, im_view)

    # --- |psi|^2, and the norm that makes it a density ---
    var density = add(multiply(re, re), multiply(im, im))
    var norm = sum(density)
    var cell = (2.0 * extent / Float64(n - 1)) ** 2

    print("2-D Gaussian wave packet")
    print("  grid:            ", n, "x", n, "over [-", extent, ",", extent, "]")
    print("  sigma:           ", sigma, "  carrier k_x:", k_x)
    print("  sum |psi|^2 dA:  ", Float64(norm) * cell)

    # A cut along y = 0. The density is a plain Gaussian: the carrier
    # cannot appear in it, because |exp(i k x)| is 1.
    var values = density.to_host()
    var re_values = re.to_host()
    var mid = n // 2
    var axis = axis_x.to_host()

    print()
    print("|psi|^2 along y = 0 (a Gaussian; the carrier is invisible here):")
    for c in range(0, n, n // 6):
        print("   x =", axis[c], "  |psi|^2 =", values[mid * n + c])

    # The carrier only shows in a signed view, so count the sign changes of
    # Re psi along the same cut -- that is the fringe count.
    var crossings = 0
    for c in range(1, n):
        var previous = Float64(re_values[mid * n + c - 1])
        var current = Float64(re_values[mid * n + c])
        if (previous < 0.0) != (current < 0.0):
            crossings += 1
    print()
    print("Re psi changes sign", crossings, "times along that cut, from the")
    print("  carrier exp(i k x) at k_x =", k_x, "across a width of", 2 * extent)

    print()
    print("Spread of the packet, and its sensitivity to sigma:")
    var at_plain = second_moment(P(sigma))
    var at_dual = second_moment(Dual[P](P(sigma), P(1.0)))
    print(
        "  <x^2>            =",
        at_plain.v,
        " (analytic:",
        sigma * sigma / 2.0,
        ")",
    )
    print("  d<x^2>/d(sigma)  =", at_dual.deriv.v, " (analytic:", sigma, ")")
    print()
    print("The derivative came from the same body as the value -- one kernel,")
    print("called at Dual instead of Plain. No second formula, no tape.")
