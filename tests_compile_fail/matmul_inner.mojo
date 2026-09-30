# `@` between compile-time matrices whose inner extents differ.
# expect: a @ b: the inner extents differ -- a's columns must equal b's rows

from numax.core.tensor import zeros


def main() raises:
    var a = zeros[DType.float32, 2, 3]()
    var b = zeros[DType.float32, 4, 5]()
    print(a @ b)
