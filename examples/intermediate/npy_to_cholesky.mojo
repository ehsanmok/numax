"""From a `.npy` file to a Cholesky factor and back, in one program.

This is the composition numax could not express before `to_array` and
`to_tensor`. Two array types live here for good reasons -- `Tensor` owns
device storage and carries its shape, `Array[T, n]` is register-resident
and generic over the `FloatLike` conformer, which is what makes
`numax.linalg` differentiable and GPU-launchable -- but a program that
loads data and factors it has to cross between them.

The steps:

1. Write a symmetric positive-definite matrix as `.npy`, the way a NumPy
   program would have handed it over.
2. Read it back with `numax.io.numpy.load`.
3. `to_array` it into the conformer layer and factor it with `cholesky`.
4. `to_tensor` the factor back and save it as `.npy` for NumPy to read.
5. Do step 3 again at `Dual`, seeding one entry, to get `d(det)/dA[0,0]`
   out of the same factorization code.

Nothing here is Python: `.npy` is a self-contained format.
"""

from max.gpu.host import DeviceContext

from numax import Dual, Plain
from numax.core.array import Tensor, to_array, to_tensor
from numax.io import numpy
from numax.linalg import cholesky, det

comptime dtype = DType.float64
comptime P = Plain[dtype, 1]
comptime N = 3


def main() raises:
    var ctx = DeviceContext(api="cpu")

    # A well-conditioned symmetric positive-definite matrix.
    var entries = [4.0, 2.0, 0.6, 2.0, 5.0, 1.2, 0.6, 1.2, 3.0]
    var values = List[Scalar[dtype]](capacity=N * N)
    for v in entries:
        values.append(Scalar[dtype](v))
    var a = Tensor[dtype, N, N](ctx, values^)

    var path = String("/tmp/numax_spd.npy")
    numpy.save(a, path)
    print("wrote", path)

    # --- 1. back off disk, into the conformer layer ----------------------
    var loaded = numpy.load[dtype, N, N](path, ctx=ctx)
    print("loaded:  ", loaded)

    var lifted = to_array[P](loaded)
    var lower = cholesky[P, N](lifted)

    # --- 2. and back down to a tensor, which is what gets saved ----------
    var factor = to_tensor(lower, ctx)
    print("cholesky:", factor)

    var factor_path = String("/tmp/numax_spd_chol.npy")
    numpy.save(factor, factor_path)
    print("wrote", factor_path, "-- numpy.load opens it")

    # L @ L.T should reproduce A; check the corner that is easiest to read.
    var l = factor.to_host()
    print("  L[0,0]**2 =", l[0] * l[0], "and A[0,0] =", a[0])

    # --- 3. the same lift at Dual, for a derivative ----------------------
    # `to_array` lifts through `T.constant`, so every derivative starts at
    # zero and the caller seeds the one it wants. Seeding A[0,0] gives
    # d(det A)/dA[0,0], which is the cofactor of that entry.
    var seeded = to_array[Dual[P]](loaded)
    seeded[0] = Dual[P](P.constant(entries[0]), P.one())
    var d = det[Dual[P], N](seeded)
    print("det A =", d.value.v, "  d(det A)/dA[0,0] =", d.deriv.v)
