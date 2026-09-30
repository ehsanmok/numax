"""The run-time-shape `eigvalsh`, `eigh`, `svdvals`, `svd`, `eigvals` and
`schur` at `gpu=True`: a `Dynamic` matrix on the device, reduced and
diagonalized there at `float32`, against the host run at the same
`dtype`."""

from std.math import sin
from std.builtin.sort import sort
from std.testing import TestSuite, assert_almost_equal, assert_equal

from layout.tile_layout import row_major
from max.gpu.host import DeviceContext

from numax.core.tensor import Dynamic, _dyn_shape
from numax.linalg import eigh, eigvals, eigvalsh, schur, svd, svdvals

comptime f32 = DType.float32


def _symmetric(n: Int, ctx: DeviceContext) raises -> Dynamic[f32, 2]:
    var values = List[Scalar[f32]](capacity=n * n)
    for i in range(n):
        for j in range(n):
            var v = sin(Float64(min(i, j) * 31 + max(i, j) * 17) * 0.37)
            if i == j:
                v += Float64(i % 7) * 0.5
            values.append(Scalar[f32](v))
    return Dynamic[f32, 2](row_major(_dyn_shape[2](n, n)), values^, ctx)


def test_dynamic_eigh_gpu_matches_host() raises:
    """`n = 70`: two `sytrd` panels and a remainder. The values against
    the host's; the vectors up to sign, since each column is determined
    only up to one."""
    var n = 70
    var gpu = DeviceContext()
    var cpu = DeviceContext(api="cpu")
    var dev = eigh[gpu=True](_symmetric(n, gpu))
    var host = eigh(_symmetric(n, cpu))
    var dw = dev.values.to_host()
    var hw = host.values.to_host()
    var dv = dev.vectors.to_host()
    var hv = host.vectors.to_host()
    for j in range(n):
        assert_almost_equal(Float64(dw[j]), Float64(hw[j]), atol=1e-4)
        var sign = 1.0
        var dotted = 0.0
        for i in range(n):
            dotted += Float64(dv[i * n + j]) * Float64(hv[i * n + j])
        if dotted < 0:
            sign = -1.0
        for i in range(n):
            assert_almost_equal(
                sign * Float64(dv[i * n + j]), Float64(hv[i * n + j]), atol=2e-3
            )
    var only = eigvalsh[gpu=True](_symmetric(n, gpu)).to_host()
    for j in range(n):
        assert_almost_equal(Float64(only[j]), Float64(hw[j]), atol=1e-4)


def _tall(m: Int, n: Int, ctx: DeviceContext) raises -> Dynamic[f32, 2]:
    var values = List[Scalar[f32]](capacity=m * n)
    for i in range(m):
        for j in range(n):
            var v = sin(Float64(i * 13 + j * 29) * 0.41)
            if i == j:
                v += 1.0
            values.append(Scalar[f32](v))
    return Dynamic[f32, 2](row_major(_dyn_shape[2](m, n)), values^, ctx)


def test_dynamic_svd_gpu_matches_host() raises:
    """`70 x 45`: two `gebrd` panels and a remainder for `svd`, the
    two-stage band for `svdvals`. The values against the host's; the
    vectors through `U diag(s) V^T = A` and `V^T V = I` rather than
    entrywise, since near-equal singular values leave them free to rotate
    within their span at `float32`, as the static device test does."""
    var m = 70
    var n = 45
    var gpu = DeviceContext()
    var cpu = DeviceContext(api="cpu")
    var dev = svd[gpu=True](_tall(m, n, gpu))
    var hs = svd(_tall(m, n, cpu)).s.to_host()
    var a = _tall(m, n, cpu).to_host()
    var ds = dev.s.to_host()
    var du = dev.u.to_host()
    var dv = dev.v.to_host()
    for j in range(n):
        assert_almost_equal(Float64(ds[j]), Float64(hs[j]), atol=1e-4)
    for i in range(m):
        for j in range(n):
            var acc = 0.0
            for k in range(n):
                acc += (
                    Float64(du[i * n + k])
                    * Float64(ds[k])
                    * Float64(dv[j * n + k])
                )
            assert_almost_equal(acc, Float64(a[i * n + j]), atol=1e-4)
    for i in range(n):
        for j in range(n):
            var vtv = 0.0
            for k in range(n):
                vtv += Float64(dv[k * n + i]) * Float64(dv[k * n + j])
            assert_almost_equal(vtv, 1.0 if i == j else 0.0, atol=1e-4)
    var only = svdvals[gpu=True](_tall(m, n, gpu)).to_host()
    for j in range(n):
        assert_almost_equal(Float64(only[j]), Float64(hs[j]), atol=1e-4)


def test_dynamic_schur_gpu_matches_host() raises:
    """Both QR iterations: `_hqr` at `n = 48`, the multishift QR at
    `n = 90`. `Z T Z^T = A` and `Z^T Z = I` from the device factors, and
    the device spectrum against the host's as two multisets -- neither
    factor is unique, and the deflation order need not agree."""
    var gpu = DeviceContext()
    var cpu = DeviceContext(api="cpu")
    for n in [48, 90]:
        var a = _tall(n, n, cpu).to_host()
        var f = schur[gpu=True](_tall(n, n, gpu))
        var t = f.t.to_host()
        var z = f.z.to_host()
        var tzt = List[Float64](length=n * n, fill=0.0)
        for k in range(n):
            for j in range(n):
                var acc = 0.0
                for l in range(n):
                    acc += Float64(t[k * n + l]) * Float64(z[j * n + l])
                tzt[k * n + j] = acc
        for i in range(n):
            for j in range(n):
                var back = 0.0
                var ztz = 0.0
                for k in range(n):
                    back += Float64(z[i * n + k]) * tzt[k * n + j]
                    ztz += Float64(z[k * n + i]) * Float64(z[k * n + j])
                assert_almost_equal(back, Float64(a[i * n + j]), atol=2e-3)
                assert_almost_equal(ztz, 1.0 if i == j else 0.0, atol=1e-4)

        var dw = eigvals[gpu=True](_tall(n, n, gpu))
        var hw = eigvals(_tall(n, n, cpu))
        var dre = dw.re.to_host()
        var hre = hw.re.to_host()
        var dim_ = dw.im.to_host()
        var him = hw.im.to_host()
        assert_equal(len(dre), n)
        sort(dre)
        sort(hre)
        sort(dim_)
        sort(him)
        for i in range(n):
            assert_almost_equal(Float64(dre[i]), Float64(hre[i]), atol=1e-3)
            assert_almost_equal(Float64(dim_[i]), Float64(him[i]), atol=1e-3)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
