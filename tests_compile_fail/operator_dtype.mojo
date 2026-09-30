# An operator between two dtypes stops on the conversion to make.
# expect: a + b: the operands' dtypes differ; convert one first with numax.core.ops.astype

from numax.core.tensor import zeros


def main() raises:
    var a = zeros[DType.float32, 2, 3]()
    var b = zeros[DType.float64, 2, 3]()
    print(a + b)
