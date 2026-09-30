# A view's comparison takes the same check as a tensor's.
# expect: a < b: the operands' dtypes differ

from numax.core.tensor import zeros


def main() raises:
    var a = zeros[DType.float32, 4, 3]()
    var b = zeros[DType.float64, 2, 3]()
    var top = a[0:2]
    print(top < b)
