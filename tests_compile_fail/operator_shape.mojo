# Two compile-time shapes that cannot broadcast stop at compile time rather
# than compiling a call that can only raise.
# expect: a * b: the shapes do not broadcast

from numax.core.tensor import zeros


def main() raises:
    var a = zeros[DType.float32, 2, 3]()
    var b = zeros[DType.float32, 4, 5]()
    print(a * b)
