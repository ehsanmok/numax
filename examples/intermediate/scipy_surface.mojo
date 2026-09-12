"""The SciPy entry points, side by side with the SciPy they mirror.

Everything here is a name a `scipy.optimize` or `scipy.linalg` user already
knows, spelled the way SciPy spells it -- including the `method=` strings.
The point is that the transliteration is nearly mechanical:

```python
# SciPy                                    # numax
minimize(f, x0, method="nelder-mead")      minimize[2, f, method="nelder-mead"](x0)
root_scalar(f, bracket=(0, 2))             root_scalar[f](bracket=(0.0, 2.0))
minimize_scalar(f, method="bounded",       minimize_scalar[f, method="bounded"](
                bounds=(-1, 1))                bounds=(-1.0, 1.0))
solve_banded((1, 1), ab, b)                solve_banded[dtype, 1, 1, n](ab, b)
expm(A)                                    expm[dtype, n](A)
eigvalsh(A), svdvals(A), sqrtm(A)          eigvalsh[dtype, n](A), and so on
nnls(A, b)                                 nnls(A, b)
minimize(f, x0, method="L-BFGS-B",         minimize[dtype, n, f, jac,
         bounds=bounds)                        method="l-bfgs"](x0, lo, hi)
```

Three differences are real and worth seeing rather than reading about.

**No derivative is ever passed.** SciPy's `minimize` takes `jac=`, its
`root_scalar` takes `fprime=` and `fprime2=`. Nothing here does, at the
`Array` tier: the objective is an ordinary `FloatLike` kernel, so the
optimizer evaluates it at `Gradient` or `Dual` and gets exact derivatives
from the same function the caller already wrote.

**The shapes are in the types.** `minimize[2, ...]` says two variables at
compile time, which is what lets the result carry an `Array[Float64, 2]`
rather than a heap vector.

**Which tier a name comes from is a choice, not a limitation.** The
factorizations, the spectra and the matrix functions all exist twice, and
the two are different algorithms rather than two spellings: over `Tensor`
they are blocked and device-resident, over `Array` they are register-
resident and differentiate at `Dual`. The linalg section below calls the
`Tensor` tier, which is what `from numax.linalg import ...` gives you; the
optimize section above calls the `Array` tier, which is where an objective
can be an ordinary `FloatLike` kernel with no `jac` argument. Both are the
SciPy name.
"""

from std.collections import Array
from std.math import exp as exp_f64

from max.gpu.host import DeviceContext

from numax import FloatLike, Plain
from numax.core.array import Static
from numax.linalg import (
    eigvalsh,
    expm,
    logm,
    solve_banded,
    sqrtm,
    svdvals,
    toeplitz,
)
from numax.optimize import nnls
from numax.optimize.array import (
    minimize,
    minimize_scalar,
    root,
    root_scalar,
)

comptime dtype = DType.float64
comptime P = Plain[dtype]


def rosenbrock[U: FloatLike](v: Array[U, 2]) -> U:
    """The standard test objective. Minimum 0 at `(1, 1)`, in a curved
    valley that defeats naive descent."""
    var a = U.one() - v[0]
    var b = v[1] - (v[0] * v[0])
    return a * a + U.constant(100.0) * b * b


def spd() -> Array[P, 9]:
    """A symmetric positive definite matrix, built fresh per call because
    each of the three entry points below consumes the one it is given.
    Its eigenvalues are `3 - sqrt(2)`, `5` and `3 + sqrt(2)`, so they sum to
    the trace, 11, and the printed spectrum can be checked by eye."""
    var out = Array[P, 9](fill=P.constant(0.0))
    var entries = [4.0, -1.0, -1.0, -1.0, 4.0, -1.0, -1.0, -1.0, 3.0]
    for i in range(9):
        out[i] = P.constant(entries[i])
    return out^


def cos_minus_x[U: FloatLike](x: U) -> U:
    """`cos(x) - x`. Its root is the Dottie number, 0.739085..."""
    return x.cos() - x


def shifted_parabola[U: FloatLike](x: U) -> U:
    """`(x - 3)^2 + 2`. Minimum 2 at `x = 3`."""
    var d = x - U.constant(3.0)
    return d * d + U.constant(2.0)


def circle_and_line[U: FloatLike](v: Array[U, 2]) -> Array[U, 2]:
    """`x^2 + y^2 = 4`, `x = y`. Root at `(sqrt(2), sqrt(2))`."""
    var out = Array[U, 2](fill=U.constant(0.0))
    out[0] = v[0] * v[0] + v[1] * v[1] - U.constant(4.0)
    out[1] = v[0] - v[1]
    return out^


def main() raises:
    print("scipy.optimize")
    print("--------------")

    var start = Array[Float64, 2](fill=0)
    start[0] = -1.2
    start[1] = 1.0

    # minimize(rosenbrock, x0)                     -- BFGS is the default
    var bfgs_run = minimize[2, rosenbrock](start)
    print(
        "  minimize(default=bfgs)   x =",
        bfgs_run.x[0],
        bfgs_run.x[1],
        " in",
        bfgs_run.iterations,
        "iterations",
    )

    var start_cg = Array[Float64, 2](fill=0)
    start_cg[0] = -1.2
    start_cg[1] = 1.0
    var cg_run = minimize[2, rosenbrock, method="cg"](start_cg)
    print(
        "  minimize(method='cg')    x =",
        cg_run.x[0],
        cg_run.x[1],
        " in",
        cg_run.iterations,
        "iterations",
    )

    var start_nm = Array[Float64, 2](fill=0)
    start_nm[0] = -1.2
    start_nm[1] = 1.0
    var nm_run = minimize[2, rosenbrock, method="nelder-mead"](start_nm)
    print(
        "  minimize('nelder-mead')  x =",
        nm_run.x[0],
        nm_run.x[1],
        " in",
        nm_run.iterations,
        "iterations",
    )
    print("  (no `jac=` anywhere: the gradient comes from `Gradient`)")

    # minimize_scalar(f) and minimize_scalar(f, method="bounded", bounds=...)
    var free = minimize_scalar[shifted_parabola]()
    print("\n  minimize_scalar()        x =", free.x, " f =", free.f_x)
    var bounded = minimize_scalar[shifted_parabola, method="bounded"](
        bounds=(-1.0, 1.0)
    )
    print(
        "  minimize_scalar(bounded) x =",
        bounded.x,
        " -- held inside [-1, 1]",
    )

    # root_scalar(f, bracket=...) and root_scalar(f, x0=..., method=...)
    var bracketed = root_scalar[cos_minus_x](bracket=(0.0, 2.0))
    var guessed = root_scalar[cos_minus_x, method="halley"](x0=0.5)
    print("\n  root_scalar(brentq)      x =", bracketed.x)
    print("  root_scalar(halley)      x =", guessed.x)
    print("  (no `fprime`/`fprime2`: Halley reads both off `Dual[Dual]`)")

    # root(f, x0)
    var vector_start = Array[Float64, 2](fill=0)
    vector_start[0] = 1.0
    vector_start[1] = 3.0
    var vector_root = root[2, circle_and_line](vector_start^)
    print(
        "\n  root(method='lm')        x =",
        vector_root.x[0],
        vector_root.x[1],
        " cost =",
        vector_root.f_x,
    )

    print("\nscipy.linalg")
    print("------------")
    var ctx = DeviceContext(api="cpu")

    # solve_banded((l, u), ab, b), in SciPy's diagonal-ordered storage.
    comptime n = 5
    var ab_entries = List[Scalar[dtype]](length=3 * n, fill=0)
    for j in range(n):
        ab_entries[1 * n + j] = 4.0
    for j in range(1, n):
        ab_entries[0 * n + j] = -1.0
    for j in range(n - 1):
        ab_entries[2 * n + j] = -1.0
    var ab = Static[dtype, 3, n](ctx, ab_entries^)
    var rhs = Static[dtype, n](ctx, [1.0, 2.0, 3.0, 4.0, 5.0])
    var banded = solve_banded[dtype, 1, 1, n](ab, rhs).to_host()
    print("  solve_banded((1,1), ab, b)")
    print(
        "    x =",
        banded[0],
        banded[1],
        banded[2],
        banded[3],
        banded[4],
    )

    # toeplitz(c) -- the structured constructor SciPy has and NumPy does not.
    var column = Static[dtype, 3](ctx, [2.0, 1.0, 0.5])
    var structured = toeplitz[dtype, 3](column).to_host()
    print("\n  toeplitz([2, 1, 0.5])")
    for i in range(3):
        print(
            "    ",
            structured[i * 3],
            structured[i * 3 + 1],
            structured[i * 3 + 2],
        )

    # expm(A) -- a rotation generator, whose exponential is the rotation.
    var generator = Static[dtype, 2, 2](ctx, [0.0, -0.7, 0.7, 0.0])
    var rotated = expm[dtype, 2](generator).to_host()
    print("\n  expm([[0, -0.7], [0.7, 0]])  -- the rotation by 0.7 rad")
    print("    ", rotated[0], rotated[1])
    print("    ", rotated[2], rotated[3])

    # eigvalsh / svdvals / sqrtm over `Tensor`, so the first parameter is
    # a `DType` and the argument is a device-resident matrix. Same SciPy
    # names, and `numax.linalg.array` has the same three for the conformer
    # tier one import away.
    var symmetric = Static[dtype, 3, 3](
        ctx, [4.0, -1.0, -1.0, -1.0, 4.0, -1.0, -1.0, -1.0, 3.0]
    )
    var spectrum = eigvalsh[dtype, 3](symmetric).to_host()
    print("\n  eigvalsh(A)   w =", spectrum[0], spectrum[1], spectrum[2])
    print("    ascending, and they sum to the trace, 11")

    var for_svd = Static[dtype, 3, 3](
        ctx, [4.0, -1.0, -1.0, -1.0, 4.0, -1.0, -1.0, -1.0, 3.0]
    )
    var singular = svdvals[dtype, 3, 3](for_svd).to_host()
    print("  svdvals(A)    s =", singular[0], singular[1], singular[2])
    print("    descending, which the Array tier's is not")

    # sqrtm over `Tensor` is the general Bjorck-Hammarling recurrence on
    # the real Schur form, not the SPD-only Array version -- so a matrix
    # with complex eigenvalues has a real square root here.
    var for_root = Static[dtype, 2, 2](ctx, [1.0, -3.0, 2.0, 1.0])
    var half = sqrtm[dtype, 2](for_root).to_host()
    print("\n  sqrtm([[1, -3], [2, 1]])  -- eigenvalues 1 +- 2.449i")
    print("    ", half[0], half[1])
    print("    ", half[2], half[3])

    var for_log = Static[dtype, 2, 2](ctx, [1.0, -3.0, 2.0, 1.0])
    var logarithm = logm[dtype, 2](for_log).to_host()
    print("  logm(same)")
    print("    ", logarithm[0], logarithm[1])
    print("    ", logarithm[2], logarithm[3])

    # nnls(A, b) -- least squares with the answer held at or above zero,
    # which is the constraint a physical quantity usually carries.
    var design = Static[dtype, 5, 3](
        ctx,
        [
            1.0,
            2.0,
            0.5,
            2.0,
            1.0,
            1.0,
            0.5,
            1.0,
            3.0,
            1.0,
            0.0,
            2.0,
            0.0,
            1.0,
            0.0,
        ],
    )
    var observations = Static[dtype, 5](ctx, [1.0, 2.0, -1.0, 0.5, 3.0])
    var constrained = nnls(design, observations)
    var coefficients = constrained.x.to_host()
    print(
        "\n  nnls(A, b)    x =",
        coefficients[0],
        coefficients[1],
        coefficients[2],
    )
    print("    residual =", constrained.residual_norm)
    print("    the third coefficient is held at zero by the constraint")
