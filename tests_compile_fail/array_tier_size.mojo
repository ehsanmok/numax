# An `Array`-tier matrix past the size the register tier unrolls in
# reasonable time names the `Tensor` spelling instead of hanging the build.
# expect: n is past the 64 the register tier compiles in reasonable time; use numax.linalg.cholesky over a Tensor

from std.collections import Array

from numax import Plain
from numax.linalg import cholesky

comptime P = Plain[DType.float64, 1]


def main():
    var a = Array[P, 65 * 65](fill=P(0.0))
    _ = cholesky[P, 65](a)
