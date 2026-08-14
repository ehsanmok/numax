"""The `special` kernels, differentiated for free.

`sigmoid`, `swish`, `tanh`, `relu`, `leaky_relu`, and `gelu` in `ember.special`
are each written once, against `FloatLike`, with no derivative logic anywhere
in them. Calling them on `Dual` gets the value and the derivative in the same
pass -- useful for, say, a hand-rolled backward pass that wants `sigmoid'(x)`
without a second formula to keep in sync with the first.
"""

from ember import Dual, Plain, gelu, leaky_relu, relu, sigmoid, swish, tanh


def main():
    comptime dtype = DType.float64
    comptime width = 1
    comptime alpha = 0.1

    for x_raw in [-2.0, -0.5, 0.0, 0.5, 2.0]:
        var x = Plain[dtype, width](SIMD[dtype, width](x_raw))
        var seed = Plain[dtype, width](SIMD[dtype, width](1))

        var s = sigmoid(Dual[Plain[dtype, width]](x.copy(), seed.copy()))
        var sw = swish(Dual[Plain[dtype, width]](x.copy(), seed.copy()))
        var t = tanh(Dual[Plain[dtype, width]](x.copy(), seed.copy()))
        var r = relu(Dual[Plain[dtype, width]](x.copy(), seed.copy()))
        var lr = leaky_relu(
            Dual[Plain[dtype, width]](x.copy(), seed.copy()), alpha
        )
        var g = gelu(Dual[Plain[dtype, width]](x^, seed^))

        print(
            "x=",
            x_raw,
            " sigmoid=",
            s.value.v,
            " sigmoid'=",
            s.deriv.v,
            " swish=",
            sw.value.v,
            " swish'=",
            sw.deriv.v,
            " tanh=",
            t.value.v,
            " tanh'=",
            t.deriv.v,
            " relu=",
            r.value.v,
            " relu'=",
            r.deriv.v,
            " leaky_relu=",
            lr.value.v,
            " leaky_relu'=",
            lr.deriv.v,
            " gelu=",
            g.value.v,
            " gelu'=",
            g.deriv.v,
        )
