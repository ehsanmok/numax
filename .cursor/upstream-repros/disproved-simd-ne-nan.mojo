from std.math import nan


def main():
    var x = SIMD[DType.float32, 2](nan[DType.float32]())

    # NumPy: np.not_equal(nan, nan) is True (unordered).
    print("ne(nan, nan)  =", x.ne(x))
    print("~eq(nan, nan) =", ~x.eq(x))
