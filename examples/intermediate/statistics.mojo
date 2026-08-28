"""`numax.statistics`: NumPy-named statistics, `Plain`-only and `FloatLike`.

The `Plain`-only, `TileTensor`-based half (`sum`/`prod`/`min`/`max`/`mean`/
`median`/`mode`/`argmax`/`argmin`/`cumprod`) is axis-2 parity surface, the
same shape as `basic/array_creation.mojo`. `argmax`/`argmin` route straight
to MAX's own `nn.argmaxmin` rather than reimplementing comparison logic.

The `FloatLike`-generic half (`mean`/`variance`/`stddev`/`cumsum`, over a
`List[T]` rather than a `TileTensor` -- see `numax/statistics.mojo`'s own
docstring for why) is where axis 1 shows up: calling `variance` at
`Compensated` instead of `Plain` keeps a long running sum in extra
precision, which this example demonstrates directly rather than just
asserting.
"""

from std.math import sin

from numax import Compensated, Plain
from numax.array import full
from numax.statistics import (
    argmax,
    argmin,
    cumprod,
    cumsum,
    max,
    mean,
    median,
    min,
    mode,
    prod,
    stddev,
    sum,
    variance,
)

comptime dtype = DType.float32


def main() raises:
    print("--- Plain-only, over TileTensor ---")
    var xs = full[dtype, 6](Scalar[dtype](0))
    var v = xs.view()
    var vals = [3.0, 1.0, 9.0, 2.0, 7.0, 2.0]
    for i in range(6):
        v[i] = Scalar[dtype](vals[i])

    print("xs:", v[0], v[1], v[2], v[3], v[4], v[5])
    print("sum:", sum(v))
    print("prod:", prod(v))
    print("min:", min(v))
    print("max:", max(v))
    print("mean:", mean(v))
    print("median:", median(v))
    print("mode:", mode(v))
    print("argmax:", argmax(v))
    print("argmin:", argmin(v))

    var cp = cumprod(xs)
    print("cumprod:", cp[0], cp[1], cp[2], cp[3], cp[4], cp[5])

    print()
    print("--- FloatLike-generic, over List[T] ---")
    print("The axis-1 win: Compensated recovers precision a long")
    print("summation would otherwise lose, with the exact same kernel.")

    comptime n = 200_000
    var plain_list = List[Plain[dtype, 1]](capacity=n)
    var comp_list = List[Compensated[dtype, 1]](capacity=n)
    for i in range(n):
        var x = Float64(1.0) + Float64(0.01) * Float64(sin(Float64(i)))
        plain_list.append(Plain[dtype, 1](Scalar[dtype](x)))
        comp_list.append(
            Compensated[dtype, 1](Scalar[dtype](x), Scalar[dtype](0))
        )

    var plain_var = variance(plain_list).v
    var comp_var = variance(comp_list).value
    print("Plain variance (float32):     ", plain_var)
    print("Compensated variance (~f64):  ", comp_var)
    print("stddev, Plain:                ", stddev(plain_list).v)
    print("stddev, Compensated:          ", stddev(comp_list).value)
