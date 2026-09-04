"""Two-source interference, drawn in the terminal.

Two point sources a distance `d` apart emit at wavenumber `k`. The field at
a point is the sum of two spherical waves, and the intensity is the square
of that sum -- so the fringes come from the *path difference* between the
sources, not from either wave alone.

The whole field is one `numax.core.tensor.map` over a pair of coordinate fields:
a kernel evaluated at every grid point. The fringes are then *measured* --
every local maximum down the far column, and their mean spacing -- rather
than plotted. Then the far-field phase is differentiated with
respect to the slit separation, which is the measurement an experimentalist
actually wants -- how far the pattern moves when the geometry moves -- and
it comes from the same kernel evaluated at `Dual`.
"""

from std.math import cos, sqrt

from max.gpu.host import DeviceContext

from numax import Dual, FloatLike, Plain
from numax.core.array import Tensor, linspace, meshgrid
from numax.core.tensor import map

comptime dtype = DType.float64
comptime P = Plain[dtype]

comptime rows = 40
comptime cols = 76

comptime k = 12.0
"""Wavenumber. Larger means finer fringes."""

comptime separation = 1.1
"""Distance between the two sources, along y."""


def two_source_field[
    dtype: DType, w: Int
](x: SIMD[dtype, w], y: SIMD[dtype, w]) -> SIMD[
    dtype, w
] where dtype.is_floating_point():
    """Intensity of two interfering spherical waves at `(x, y)`.

    Amplitudes fall as `1/sqrt(r)` in two dimensions; the intensity is the
    square of the summed amplitude, which is where the cross term -- the
    interference -- comes from.
    """
    var half = SIMD[dtype, w](separation * 0.5)
    var r1 = sqrt(x * x + (y - half) * (y - half)) + SIMD[dtype, w](1e-6)
    var r2 = sqrt(x * x + (y + half) * (y + half)) + SIMD[dtype, w](1e-6)
    var a1 = cos(SIMD[dtype, w](k) * r1) / sqrt(r1)
    var a2 = cos(SIMD[dtype, w](k) * r2) / sqrt(r2)
    var total = a1 + a2
    return total * total


def fringe_phase[T: FloatLike](gap: T) -> T:
    """Phase difference between the two paths at a fixed far-field point.

    `k * (r1 - r2)` for a point off the axis. Where this passes a multiple
    of `2 pi` there is a bright fringe, so its derivative with respect to
    `gap` is how fast the pattern shifts as the sources move apart.
    """
    var x = T.constant(9.0)
    var y = T.constant(1.5)
    var half = gap / T.constant(2.0)
    var dy1 = y - half
    var dy2 = y + half
    var r1 = (x * x + dy1 * dy1).sqrt()
    var r2 = (x * x + dy2 * dy2).sqrt()
    return T.constant(k) * (r1 - r2)


def main() raises:
    var ctx = DeviceContext(api="cpu")

    var axis_x = linspace[cols](0.15, 10.0, ctx=ctx)
    var axis_y = linspace[rows](-4.0, 4.0, ctx=ctx)
    var grid = meshgrid(axis_x, axis_y)

    var field = Tensor[dtype, rows, cols](ctx)
    var xs = grid[0].view()
    var ys = grid[1].view()
    var out = field.view()
    map[step=two_source_field[dtype, _], width=4](xs, ys, out)

    var values = field.to_host()
    var axis = axis_y.to_host()

    print("Two-source interference")
    print("  sources:   2, separated by", separation, "along y")
    print("  wavenumber:", k)
    print("  grid:      ", cols, "x", rows, " (sources at the left edge)")

    # Fringes, measured rather than drawn: walk the far column and record
    # every local maximum in y. Their spacing is the fringe spacing.
    var far = cols - 1
    var maxima = List[Float64]()
    for r in range(1, rows - 1):
        var below = Float64(values[(r - 1) * cols + far])
        var here = Float64(values[r * cols + far])
        var above = Float64(values[(r + 1) * cols + far])
        if here > below and here >= above:
            maxima.append(Float64(axis[r]))

    print()
    print(
        "Bright fringes at x =",
        axis_x.to_host()[far],
        ":",
        len(maxima),
        "maxima",
    )
    for i in range(len(maxima)):
        print("   y =", maxima[i])
    if len(maxima) > 1:
        var spacing = 0.0
        for i in range(1, len(maxima)):
            spacing += maxima[i] - maxima[i - 1]
        print("  mean fringe spacing:", spacing / Float64(len(maxima) - 1))

    print()
    print("Sensitivity of the far-field phase to the slit separation:")
    var at_plain = fringe_phase(P(separation))
    var at_dual = fringe_phase(Dual[P](P(separation), P(1.0)))
    print("  phase difference        =", at_plain.v, "rad")
    print("  d(phase)/d(separation)  =", at_dual.deriv.v, "rad per unit")
    print()
    print("Move the sources by 1%, and the fringe phase moves by")
    print(
        " ", at_dual.deriv.v * separation * 0.01, "rad -- from the same kernel."
    )
