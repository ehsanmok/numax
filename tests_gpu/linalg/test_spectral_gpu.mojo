"""The spectral decompositions at `gpu=True`, against the host answer.

Each routine was refused at `gpu=True` until the `latrd`/`labrd`/`lahr2`
panels stopped forwarding by-reference closures into device launches. The
checks are the ones that do not depend on a sign or ordering convention:
eigenvalues and singular values against the host, and every returned basis
through its reconstruction. `float32`, since Metal has no `double`.
"""

from std.testing import (
    TestSuite,
    assert_almost_equal,
    assert_equal,
    assert_true,
)

from max.gpu.host import DeviceContext

from numax.core.tensor import Static
from numax.interpolate import roots
from numax.linalg import (
    cond,
    cosm,
    fractional_matrix_power,
    logm,
    null_space,
    orth,
    polar,
    eigh,
    eigvals,
    eigvalsh,
    hessenberg,
    lstsq,
    matrix_rank,
    pinv,
    schur,
    sinm,
    sqrtm,
    svd,
    svdvals,
    tanm,
)

comptime f32 = DType.float32
comptime n = 48
comptime m = 40
comptime k = 24


def _entry(i: Int) -> Float32:
    """A fixed pseudo-random value in `[-0.5, 0.5)`: splitmix64 of `i`, so
    the matrices below have full rank. An affine pattern in `i` would not,
    and `pinv` of a rank-deficient matrix only amplifies rounding."""
    var z = UInt64(i + 1) * 0x9E3779B97F4A7C15
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9
    z = (z ^ (z >> 27)) * 0x94D049BB133111EB
    z = z ^ (z >> 31)
    return Float32(z >> 40) / Float32(1 << 24) - 0.5


def _general[r: Int, c: Int](ctx: DeviceContext) raises -> Static[f32, r, c]:
    var values = List[Scalar[f32]](capacity=r * c)
    for i in range(r * c):
        values.append(_entry(i))
    return Static[f32, r, c](ctx, values^)


def _symmetric(ctx: DeviceContext) raises -> Static[f32, n, n]:
    """`M + M^T + n I`: symmetric, positive definite, well separated."""
    var values = List[Scalar[f32]](capacity=n * n)
    for r in range(n):
        for c in range(n):
            var x = _entry(r * n + c) + _entry(c * n + r)
            values.append(x + (Float32(n) if r == c else Float32(0)))
    return Static[f32, n, n](ctx, values^)


def _max_abs_diff(a: List[Scalar[f32]], b: List[Scalar[f32]]) -> Float32:
    var worst = Float32(0)
    for i in range(len(a)):
        worst = max(worst, abs(a[i] - b[i]))
    return worst


def _product(
    a: List[Scalar[f32]], b: List[Scalar[f32]], r: Int, inner: Int, c: Int
) -> List[Scalar[f32]]:
    var out = List[Scalar[f32]](capacity=r * c)
    for i in range(r):
        for j in range(c):
            var acc = Float32(0)
            for t in range(inner):
                acc += a[i * inner + t] * b[t * c + j]
            out.append(acc)
    return out^


def _transpose(a: List[Scalar[f32]], r: Int, c: Int) -> List[Scalar[f32]]:
    var out = List[Scalar[f32]](capacity=r * c)
    for j in range(c):
        for i in range(r):
            out.append(a[i * c + j])
    return out^


def test_eigvalsh_on_the_device_matches_the_host() raises:
    var gpu = DeviceContext()
    var cpu = DeviceContext(api="cpu")
    var d = eigvalsh[gpu=True](_symmetric(gpu)).to_host()
    var h = eigvalsh(_symmetric(cpu)).to_host()
    assert_true(_max_abs_diff(d, h) < 1e-3)


def test_eigh_on_the_device_reconstructs_its_matrix() raises:
    """`A V = V diag(w)` for the device's `V` and `w`, and `w` is the
    host's."""
    var gpu = DeviceContext()
    var cpu = DeviceContext(api="cpu")
    var e = eigh[gpu=True](_symmetric(gpu))
    var w = e.values.to_host()
    var v = e.vectors.to_host()
    assert_true(_max_abs_diff(w, eigvalsh(_symmetric(cpu)).to_host()) < 1e-3)
    var av = _product(_symmetric(cpu).to_host(), v, n, n, n)
    var worst = Float32(0)
    for i in range(n):
        for j in range(n):
            worst = max(worst, abs(av[i * n + j] - v[i * n + j] * w[j]))
    assert_true(worst < 1e-3)


def test_svd_on_the_device_reconstructs_its_matrix() raises:
    var gpu = DeviceContext()
    var cpu = DeviceContext(api="cpu")
    var s_dev = svdvals[gpu=True](_general[m, k](gpu)).to_host()
    var s_host = svdvals(_general[m, k](cpu)).to_host()
    assert_true(_max_abs_diff(s_dev, s_host) < 1e-4)
    var f = svd[gpu=True](_general[m, k](gpu))
    var u = f.u.to_host()
    var s = f.s.to_host()
    var v = f.v.to_host()
    for i in range(m):
        for j in range(k):
            u[i * k + j] *= s[j]
    var back = _product(u, _transpose(v, k, k), m, k, k)
    assert_true(_max_abs_diff(back, _general[m, k](cpu).to_host()) < 1e-4)


def test_schur_on_the_device_reconstructs_its_matrix() raises:
    """`Z T Z^T = A`, and `hessenberg`'s `H` is similar to `A`. Neither
    factor is compared entrywise: a real Schur form is not unique."""
    var gpu = DeviceContext()
    var cpu = DeviceContext(api="cpu")
    var f = schur[gpu=True](_general[n, n](gpu))
    var z = f.z.to_host()
    var back = _product(
        _product(z, f.t.to_host(), n, n, n), _transpose(z, n, n), n, n, n
    )
    assert_true(_max_abs_diff(back, _general[n, n](cpu).to_host()) < 1e-3)
    # `H` is fixed only up to the signs of the reflectors, so check what a
    # similarity preserves: the trace, the Frobenius norm, and the zeros
    # below the subdiagonal.
    var h = hessenberg[gpu=True](_general[n, n](gpu)).h.to_host()
    var a = _general[n, n](cpu).to_host()
    var trace_h = Float32(0)
    var trace_a = Float32(0)
    var fro_h = Float32(0)
    var fro_a = Float32(0)
    for i in range(n):
        trace_h += h[i * n + i]
        trace_a += a[i * n + i]
        for j in range(n):
            fro_h += h[i * n + j] * h[i * n + j]
            fro_a += a[i * n + j] * a[i * n + j]
            if i > j + 1:
                assert_true(abs(h[i * n + j]) < 1e-5)
    assert_almost_equal(trace_h, trace_a, atol=1e-3)
    assert_almost_equal(fro_h, fro_a, rtol=1e-4)


def test_eigvals_on_the_device_matches_the_host() raises:
    """The same spectrum, compared as sums of the real and imaginary parts
    and of their squares, which do not depend on the order returned."""
    var gpu = DeviceContext()
    var cpu = DeviceContext(api="cpu")
    var d = eigvals[gpu=True](_general[n, n](gpu))
    var h = eigvals(_general[n, n](cpu))
    var dre = d.re.to_host()
    var dim_ = d.im.to_host()
    var hre = h.re.to_host()
    var him = h.im.to_host()
    var sd = Float32(0)
    var sh = Float32(0)
    var qd = Float32(0)
    var qh = Float32(0)
    for i in range(n):
        sd += dre[i]
        sh += hre[i]
        qd += dre[i] * dre[i] + dim_[i] * dim_[i]
        qh += hre[i] * hre[i] + him[i] * him[i]
    assert_almost_equal(sd, sh, atol=1e-3)
    assert_almost_equal(qd, qh, atol=1e-3)


def test_the_svd_dependents_on_the_device_match_the_host() raises:
    """`pinv`, `matrix_rank`, `cond`, `lstsq(method="svd")` and `sqrtm`, the
    routines that were refused because they reach `svd` or `schur`."""
    var gpu = DeviceContext()
    var cpu = DeviceContext(api="cpu")
    assert_true(
        _max_abs_diff(
            pinv[gpu=True](_general[m, k](gpu)).to_host(),
            pinv(_general[m, k](cpu)).to_host(),
        )
        < 1e-3
    )
    assert_equal(
        matrix_rank[gpu=True](_general[m, k](gpu)),
        matrix_rank(_general[m, k](cpu)),
    )
    var c_dev = Float32(cond[gpu=True](_symmetric(gpu)))
    var c_host = Float32(cond(_symmetric(cpu)))
    assert_almost_equal(c_dev, c_host, rtol=1e-3)
    var rhs_values = List[Scalar[f32]](capacity=m)
    for i in range(m):
        rhs_values.append(Float32(i % 5) - 2.0)
    var rhs_d = Static[f32, m](gpu, rhs_values.copy())
    var rhs_h = Static[f32, m](cpu, rhs_values^)
    assert_true(
        _max_abs_diff(
            lstsq[gpu=True, method="svd"](_general[m, k](gpu), rhs_d).to_host(),
            lstsq[method="svd"](_general[m, k](cpu), rhs_h).to_host(),
        )
        < 1e-3
    )
    var root = sqrtm[gpu=True](_symmetric(gpu)).to_host()
    var square = _product(root, root, n, n, n)
    assert_true(_max_abs_diff(square, _symmetric(cpu).to_host()) < 1e-2)


def test_the_matrix_functions_on_the_device_match_the_host() raises:
    """`logm`, `cosm`, `fractional_matrix_power`, `polar` and `orth` are
    unique answers (the principal branch, the positive polar factor, the
    range's projector), so they are compared with the host entrywise."""
    var gpu = DeviceContext()
    var cpu = DeviceContext(api="cpu")
    var scale = Float32(n)
    assert_true(
        _max_abs_diff(
            logm[gpu=True](_symmetric(gpu)).to_host(),
            logm(_symmetric(cpu)).to_host(),
        )
        < 1e-3
    )
    assert_true(
        _max_abs_diff(
            cosm[gpu=True](_general[n, n](gpu)).to_host(),
            cosm(_general[n, n](cpu)).to_host(),
        )
        < 1e-3
    )
    assert_true(
        _max_abs_diff(
            fractional_matrix_power[gpu=True](_symmetric(gpu), 0.5).to_host(),
            fractional_matrix_power(_symmetric(cpu), 0.5).to_host(),
        )
        < 1e-3 * scale
    )
    assert_true(
        _max_abs_diff(
            polar[gpu=True](_general[n, n](gpu)).p.to_host(),
            polar(_general[n, n](cpu)).p.to_host(),
        )
        < 1e-3
    )
    assert_true(
        _max_abs_diff(
            sinm[gpu=True](_general[n, n](gpu)).to_host(),
            sinm(_general[n, n](cpu)).to_host(),
        )
        < 1e-3
    )
    assert_true(
        _max_abs_diff(
            tanm[gpu=True](_general[n, n](gpu)).to_host(),
            tanm(_general[n, n](cpu)).to_host(),
        )
        < 1e-3
    )
    # `orth`'s basis is fixed only up to a rotation; its projector is not.
    var q = orth[gpu=True](_general[m, k](gpu)).to_host()
    var qh = orth(_general[m, k](cpu)).to_host()
    assert_true(
        _max_abs_diff(
            _product(q, _transpose(q, m, k), m, k, m),
            _product(qh, _transpose(qh, m, k), m, k, m),
        )
        < 1e-4
    )


def test_roots_on_the_device_finds_the_host_roots() raises:
    """`(x - 1)(x - 2)(x - 3)(x + 0.5)`: four real roots, found through
    `eigvals` of the companion matrix on either device."""
    var gpu = DeviceContext()
    var coeffs: List[Scalar[f32]] = [1.0, -5.5, 7.0, 0.5, -3.0]
    var r = roots[gpu=True](Static[f32, 5](gpu, coeffs^))
    var re = r.re.to_host()
    var total = Float32(0)
    var product = Float32(1)
    for i in range(4):
        total += re[i]
        product *= re[i]
    assert_almost_equal(total, 5.5, atol=1e-4)
    assert_almost_equal(product, -3.0, atol=1e-4)


def _repeated(ctx: DeviceContext) raises -> Static[f32, m, k]:
    """`m x k` with its last 8 columns repeating its first 8: rank
    `k - 8`, so an 8-dimensional null space."""
    var values = List[Scalar[f32]](capacity=m * k)
    for r in range(m):
        for c in range(k):
            values.append(_entry(r * k + (c - 8 if c >= k - 8 else c)))
    return Static[f32, m, k](ctx, values^)


def test_null_space_on_the_device_is_a_null_space() raises:
    """The device's basis has the right width, is orthonormal, and `A`
    sends it to zero."""
    var gpu = DeviceContext()
    var cpu = DeviceContext(api="cpu")
    var basis = null_space[gpu=True](_repeated(gpu))
    comptime w = 8
    assert_equal(basis.dim[1](), w)
    var z = basis.to_host()
    var az = _product(_repeated(cpu).to_host(), z, m, k, w)
    var worst = Float32(0)
    for i in range(len(az)):
        worst = max(worst, abs(az[i]))
    assert_true(worst < 1e-4)
    var gram = _product(_transpose(z, k, w), z, w, k, w)
    for i in range(w):
        for j in range(w):
            var want = Float32(1) if i == j else Float32(0)
            assert_true(abs(gram[i * w + j] - want) < 1e-4)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
