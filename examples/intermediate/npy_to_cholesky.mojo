"""From a NumPy array to a Cholesky factor and back, in one program.

This is the composition numax could not express before `to_array` and
`to_tensor`. Two array types live here for good reasons -- `Tensor` owns
device storage and carries its shape, `Array[T, n]` is register-resident
and generic over the `FloatLike` conformer, which is what makes
`numax.linalg` differentiable and GPU-launchable -- but a program that
loads data and factors it has to cross between them.

The data comes from NumPy itself, called from Mojo through `std.python`,
so this is the handoff a real caller has rather than a matrix typed into
the source:

1. Build a symmetric positive-definite matrix with `numpy`, and `np.save`
   it the way the program upstream of yours would have.
2. Read it back with `numax.io.numpy.load`.
3. `to_array` it into the conformer layer and factor it with `cholesky`,
   checking the result against `np.linalg.cholesky` entry by entry.
4. `to_tensor` the factor back, save it, and have NumPy load it -- so the
   round trip is closed at both ends rather than asserted.
5. Do step 3 again at `Dual`, seeding one entry, to get `d(det)/dA[0,0]`
   out of the same factorization code. This is the step NumPy has no
   answer for: it factors numbers, and numax factors whatever conforms.

The file is the bridge because it has to be. The pinned MAX ships no
in-memory NumPy conversion (there is no `layout/numpy` module in 26.5), so
an array crosses as bytes on disk.

Only this example needs Python. `numax.io.numpy` parses `.npy` itself, so
a program that reads a file someone else wrote pulls in nothing -- the
interpreter here is what plays the part of that someone else. It comes
from the environment MAX already installs, which is why numax declares no
NumPy dependency of its own.
"""

from std.python import Python

from max.gpu.host import DeviceContext

from numax import Dual, Plain
from numax.core.array import Tensor, to_array, to_tensor
from numax.io import numpy  # numax's .npy reader; `np` below is NumPy itself
from numax.linalg import cholesky, det

comptime dtype = DType.float64
comptime P = Plain[dtype]
comptime N = 3


def main() raises:
    var ctx = DeviceContext(api="cpu")
    var np = Python.import_module("numpy")

    # --- 1. NumPy's array, NumPy's file ----------------------------------
    var a = np.array(
        [[4.0, 2.0, 0.6], [2.0, 5.0, 1.2], [0.6, 1.2, 3.0]], dtype="float64"
    )
    var path = String("/tmp/numax_spd.npy")
    np.save(path, a)
    print("numpy wrote", path)
    print(a)

    # --- 2. back off disk, into the conformer layer ----------------------
    var loaded = numpy.load[dtype, N, N](path, ctx=ctx)
    print("numax loaded:")
    print(loaded)

    var lifted = to_array[P](loaded)
    var lower = cholesky[P, N](lifted)

    # --- 3. and back down to a tensor, which is what gets saved ----------
    var factor = to_tensor[dtype, N, N](lower, ctx)
    print("cholesky:")
    print(factor)

    var factor_path = String("/tmp/numax_spd_chol.npy")
    numpy.save(factor, factor_path)

    # NumPy reads what numax wrote, and agrees with its own factorization.
    var reference = np.linalg.cholesky(a)
    var reloaded = np.load(factor_path)
    print("numpy reloaded numax's factor, max |difference| vs its own:")
    print(np.abs(np.subtract(reloaded, reference)).max())

    # --- 4. the same lift at Dual, for a derivative ----------------------
    # `to_array` lifts through `T.constant`, so every derivative starts at
    # zero and the caller seeds the one it wants. Seeding A[0,0] gives
    # d(det A)/dA[0,0], which is the cofactor of that entry.
    var seeded = to_array[Dual[P]](loaded)
    var a00 = seeded[0].value.copy()
    seeded[0] = Dual[P](a00^, P.one())
    var d = det[Dual[P], N](seeded)
    print("det A =", d.value, "  d(det A)/dA[0,0] =", d.deriv)
