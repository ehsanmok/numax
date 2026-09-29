"""The two-stage reductions behind the values-only `eigvalsh` and
`svdvals`: dense to band by block reflectors (LAPACK's `dsytrd_sy2sb`
shape), then band to tridiagonal or bidiagonal by Givens bulge chasing
(`dsbtrd` and `dgbbrd`, transcribed from the reference implementation, BSD
licensed).

**Tier 2, private.** `sytrd` reduces a column at a time and every column
costs a whole-matrix `matvec` of the trailing block -- `2 n^3 / 3` flops at
memory speed, which no panel width removes. Stage 1 here reduces `b`
columns at a time to a band of half-width `b` instead: the panel is a QR
(`qr_factor`'s run-time overload), and the trailing block becomes `Q^T
A22 Q` by two block-reflector applications with an in-place transpose
between them, so its `O(n^3)` is `linalg.matmul` on the tensor's device.
Stage 2 is `O(n^2 b)` scalar work on the band, on the host, and runs at
`float64` whatever the input's `dtype`: its many rotations per element are
where a `float32` reduction loses digits, and on the host the wider type
costs a few tens of milliseconds at `n = 1024`.

The same two stages give the singular values: `_ge2gb` alternates QR
panels (applied on the left) with LQ panels (the QR of the transposed row
block, applied on the right) to reach an upper band, and `dgbbrd`
(transcribed) chases it to bidiagonal form.

Only the values are formed. Stage 2's Givens rotations would have to be
accumulated into an `n x n` product to give eigenvectors, which is why
`eigh` keeps the one-stage `sytrd`.
"""

from std.math import sqrt as _sqrt
from std.sys.info import simd_width_of

from layout import Coord, TileTensor, coord_to_index_list
from layout.tile_layout import row_major
from max.algorithm.functional import elementwise
from max.gpu.host import DeviceContext
from linalg.matmul import matmul as _max_matmul
from std.sys.info import align_of
from std.utils import IndexList

from ..core.tensor import Dynamic, _dyn_shape
from .blas import _target, matmul
from .common import _Dense
from .panel import pack_block
from ._multishift import _dlartg


def _panel_copy[
    dtype: DType, gpu: Bool
](
    mut a: Dynamic[dtype, 2],
    n: Int,
    r0: Int,
    c0: Int,
    m: Int,
    p: Int,
    ctx: DeviceContext,
) raises -> Dynamic[dtype, 2]:
    """`A[r0 : r0 + m, c0 : c0 + p]` as a dense `m x p` tensor, through
    `pack_block`'s vectorized copy."""
    var out = Dynamic[dtype, 2](row_major(_dyn_shape[2](m, p)), ctx)
    var av = a.tile()
    var ov = out.tile()
    pack_block[target=_target[gpu]()](av, ov, r0, c0, m, p, ctx)
    ctx.synchronize()
    return out^


def _write_upper[
    dtype: DType, gpu: Bool
](
    mut a: Dynamic[dtype, 2],
    n: Int,
    r0: Int,
    c0: Int,
    src: Dynamic[dtype, 2],
    ld: Int,
    p: Int,
    ctx: DeviceContext,
) raises:
    """`A[r0 + i, c0 + j] = src[i, j]` for `i <= j < p`: the panel's `R`."""
    var ap = a.tile().ptr.unsafe_origin_cast[MutAnyOrigin]()
    var sp = src.tile().ptr.unsafe_origin_cast[MutAnyOrigin]()

    @always_inline
    def write[
        w: Int, alignment: Int = 1
    ](coord: Coord) {var ap, var sp, var n, var r0, var c0, var ld, var p}:
        var e = coord_to_index_list(coord)[0]
        var i = e // p
        var j = e % p
        if i <= j:
            ap[unsafe_offset=(r0 + i) * n + c0 + j] = sp[
                unsafe_offset=i * ld + j
            ]

    elementwise[simd_width=1, target=_target[gpu]()](write, Coord(p * p), ctx)
    ctx.synchronize()


struct _Reflectors[dtype: DType](Movable):
    """One panel's block reflector `Q = I - V T V^T` in dense form: `V`
    (`m x p`, unit lower trapezoidal), `V^T` and `T` (`p x p` upper
    triangular), the three operands every product below takes."""

    var v: Dynamic[Self.dtype, 2]
    var vt: Dynamic[Self.dtype, 2]
    var t: Dynamic[Self.dtype, 2]
    var m: Int
    var p: Int

    def __init__(
        out self,
        var v: Dynamic[Self.dtype, 2],
        var vt: Dynamic[Self.dtype, 2],
        var t: Dynamic[Self.dtype, 2],
        m: Int,
        p: Int,
    ):
        self.v = v^
        self.vt = vt^
        self.t = t^
        self.m = m
        self.p = p


def _upload[
    dtype: DType
](
    var values: List[Scalar[dtype]], rows: Int, cols: Int, ctx: DeviceContext
) raises -> Dynamic[dtype, 2]:
    var out = Dynamic[dtype, 2](row_major(_dyn_shape[2](rows, cols)), ctx)
    out.copy_from_host(values^)
    return out^


@always_inline
def _dot[
    dtype: DType
](
    x: Pointer[Scalar[dtype], MutUntrackedOrigin],
    xo: Int,
    y: Pointer[Scalar[dtype], MutUntrackedOrigin],
    yo: Int,
    count: Int,
) -> Scalar[dtype]:
    """`sum x[xo + i] y[yo + i]` over `count` entries, vectorized."""
    comptime w = 2 * simd_width_of[dtype]()
    var acc = SIMD[dtype, w](0)
    var i = 0
    while i + w <= count:
        acc += x.unsafe_load[width=w](xo + i) * y.unsafe_load[width=w](yo + i)
        i += w
    var total = acc.reduce_add()
    while i < count:
        total += x[unsafe_offset=xo + i] * y[unsafe_offset=yo + i]
        i += 1
    return total


@always_inline
def _axpy[
    dtype: DType
](
    y: Pointer[Scalar[dtype], MutUntrackedOrigin],
    yo: Int,
    x: Pointer[Scalar[dtype], MutUntrackedOrigin],
    xo: Int,
    c: Scalar[dtype],
    count: Int,
):
    """`y[yo + i] += c x[xo + i]` over `count` entries, vectorized."""
    comptime w = 2 * simd_width_of[dtype]()
    var cv = SIMD[dtype, w](c)
    var i = 0
    while i + w <= count:
        y.unsafe_store(
            yo + i,
            y.unsafe_load[width=w](yo + i)
            + cv * x.unsafe_load[width=w](xo + i),
        )
        i += w
    while i < count:
        y[unsafe_offset=yo + i] = (
            y[unsafe_offset=yo + i] + c * x[unsafe_offset=xo + i]
        )
        i += 1


def _reflectors[
    dtype: DType, gpu: Bool
](
    mut panel: Dynamic[dtype, 2], m: Int, p: Int, ctx: DeviceContext
) raises -> _Reflectors[dtype] where dtype.is_floating_point():
    """QR-factor the dense `m x p` `panel` and return its block reflector
    of `r = min(m, p)` reflections; the panel is replaced by `R` (`r x p`,
    upper trapezoidal). LAPACK's `geqr2` and `larft` on a host copy held
    column-major, so every inner loop is a contiguous SIMD dot or axpy: the
    panel is `m x p` with `p` the band width, `O(m p^2)` of work, cheaper
    round-tripped than the workspace a device QR allocates per panel."""
    var h = panel.to_host()
    var r = min(m, p)
    # Column-major copy: column j is col[j * m : (j + 1) * m].
    var col = List[Scalar[dtype]](length=m * p, fill=0)
    for i in range(m):
        for j in range(p):
            col[j * m + i] = h[i * p + j]
    var cp = col.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin]()
    var taus = List[Scalar[dtype]](length=r, fill=0)
    for j in range(r):
        var base = j * m
        var alpha = col[base + j]
        var tail = m - j - 1
        var xnorm = Scalar[dtype](0)
        if tail > 0:
            var ssq = _dot(cp, base + j + 1, cp, base + j + 1, tail)
            xnorm = _sqrt(ssq)
        var tau = Scalar[dtype](0)
        if xnorm > 0:
            var norm = _sqrt(alpha * alpha + xnorm * xnorm)
            var beta = -norm if alpha >= 0 else norm
            tau = (beta - alpha) / beta
            var inv = 1 / (alpha - beta)
            for i in range(j + 1, m):
                col[base + i] = col[base + i] * inv
            col[base + j] = beta
            for c in range(j + 1, p):
                var cb = c * m
                var w = col[cb + j] + _dot(
                    cp, base + j + 1, cp, cb + j + 1, tail
                )
                w = w * tau
                col[cb + j] = col[cb + j] - w
                _axpy(cp, cb + j + 1, cp, base + j + 1, -w, tail)
        taus[j] = tau
    # V column-major (unit diagonal), for the larft dots.
    var vcol = List[Scalar[dtype]](length=m * r, fill=0)
    for j in range(r):
        vcol[j * m + j] = 1
        for i in range(j + 1, m):
            vcol[j * m + i] = col[j * m + i]
    var vp = vcol.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin]()
    var v = List[Scalar[dtype]](length=m * r, fill=0)
    var vt = List[Scalar[dtype]](length=r * m, fill=0)
    var upper = List[Scalar[dtype]](length=r * p, fill=0)
    for j in range(r):
        for i in range(m):
            v[i * r + j] = vcol[j * m + i]
            vt[j * m + i] = vcol[j * m + i]
        for c in range(j, p):
            upper[j * p + c] = col[c * m + j]
    # larft, forward and columnwise: T[:j, j] = -tau_j T[:j, :j] V[:, :j]^T v_j.
    var t = List[Scalar[dtype]](length=r * r, fill=0)
    var dots = List[Scalar[dtype]](length=r, fill=0)
    for j in range(r):
        t[j * r + j] = taus[j]
        for q in range(j):
            dots[q] = _dot(vp, q * m + j, vp, j * m + j, m - j)
        for q in range(j):
            var s = Scalar[dtype](0)
            for l in range(q, j):
                s += t[q * r + l] * dots[l]
            t[q * r + j] = -taus[j] * s
    _ = len(col)
    _ = len(vcol)
    panel = _upload[dtype](upper^, r, p, ctx)
    return _Reflectors[dtype](
        _upload[dtype](v^, m, r, ctx),
        _upload[dtype](vt^, r, m, ctx),
        _upload[dtype](t^, r, r, ctx),
        m,
        r,
    )


def _subtract_into[
    dtype: DType, gpu: Bool
](
    mut a: Dynamic[dtype, 2],
    n: Int,
    r0: Int,
    c0: Int,
    src: Dynamic[dtype, 2],
    rows: Int,
    cols: Int,
    ctx: DeviceContext,
) raises:
    """`A[r0 + i, c0 + j] -= src[i, j]` over a `rows x cols` block."""
    var ap = a.tile().ptr.unsafe_origin_cast[MutAnyOrigin]()
    var sp = src.tile().ptr.unsafe_origin_cast[MutAnyOrigin]()

    @always_inline
    def sub[
        w: Int, alignment: Int = 1
    ](coord: Coord) {var ap, var sp, var n, var r0, var c0, var cols}:
        var e = coord_to_index_list(coord)[0]
        var i = e // cols
        var j = e % cols
        var x = (r0 + i) * n + c0 + j
        ap[unsafe_offset=x] = ap[unsafe_offset=x] - sp[unsafe_offset=e]

    elementwise[simd_width=1, target=_target[gpu]()](
        sub, Coord(rows * cols), ctx
    )
    ctx.synchronize()


def _subtract_product[
    dtype: DType, gpu: Bool
](
    mut a: Dynamic[dtype, 2],
    r0: Int,
    c0: Int,
    mut x: Dynamic[dtype, 2],
    mut y: Dynamic[dtype, 2],
    rows: Int,
    inner: Int,
    cols: Int,
    ctx: DeviceContext,
) raises:
    """`A[r0 :, c0 :] -= x y` for dense `x` (`rows x inner`) and `y`
    (`inner x cols`), subtracted by `matmul`'s epilogue as it is formed, so
    the product is never staged back."""
    var av = a.tile()
    var product = Dynamic[dtype, 2](row_major(_dyn_shape[2](rows, cols)), ctx)
    var pv: _Dense[dtype] = TileTensor(
        product.tile().ptr_at_offset(Coord(0, 0)), row_major(Coord(rows, cols))
    )
    var xv: _Dense[dtype] = TileTensor(
        x.tile().ptr_at_offset(Coord(0, 0)), row_major(Coord(rows, inner))
    )
    var yv: _Dense[dtype] = TileTensor(
        y.tile().ptr_at_offset(Coord(0, 0)), row_major(Coord(inner, cols))
    )

    @__parameter
    @always_inline
    @__copy_capture(av, r0, c0)
    def subtract[
        _dtype: DType,
        width: SIMDLength,
        *,
        alignment: Int = align_of[SIMD[_dtype, width]](),
    ](idx: IndexList[2], value: SIMD[_dtype, width]) capturing -> None:
        var at = Coord(r0 + idx[0], c0 + idx[1])
        av.store[width](
            at, av.load[width](at) - rebind[SIMD[dtype, width]](value)
        )

    _max_matmul[elementwise_lambda_fn=subtract, target=_target[gpu]()](
        pv, xv, yv, ctx
    )
    ctx.synchronize()
    _ = product^


def _concat_columns[
    dtype: DType, gpu: Bool
](
    x: Dynamic[dtype, 2],
    y: Dynamic[dtype, 2],
    rows: Int,
    p: Int,
    ctx: DeviceContext,
) raises -> Dynamic[dtype, 2]:
    """`[x | y]` for two dense `rows x p` operands."""
    var out = Dynamic[dtype, 2](row_major(_dyn_shape[2](rows, 2 * p)), ctx)
    var xp = x.tile().ptr.unsafe_origin_cast[MutAnyOrigin]()
    var yp = y.tile().ptr.unsafe_origin_cast[MutAnyOrigin]()
    var op = out.tile().ptr.unsafe_origin_cast[MutAnyOrigin]()

    @always_inline
    def cat[
        w: Int, alignment: Int = 1
    ](coord: Coord) {var xp, var yp, var op, var p}:
        var e = coord_to_index_list(coord)[0]
        var i = e // (2 * p)
        var j = e % (2 * p)
        op[unsafe_offset=e] = (
            xp[unsafe_offset=i * p + j] if j
            < p else yp[unsafe_offset=i * p + j - p]
        )

    elementwise[simd_width=1, target=_target[gpu]()](
        cat, Coord(rows * 2 * p), ctx
    )
    ctx.synchronize()
    return out^


def _concat_rows[
    dtype: DType, gpu: Bool
](
    x: Dynamic[dtype, 2],
    y: Dynamic[dtype, 2],
    p: Int,
    cols: Int,
    ctx: DeviceContext,
) raises -> Dynamic[dtype, 2]:
    """`[x; y]` for two dense `p x cols` operands."""
    var out = Dynamic[dtype, 2](row_major(_dyn_shape[2](2 * p, cols)), ctx)
    var xp = x.tile().ptr.unsafe_origin_cast[MutAnyOrigin]()
    var yp = y.tile().ptr.unsafe_origin_cast[MutAnyOrigin]()
    var op = out.tile().ptr.unsafe_origin_cast[MutAnyOrigin]()
    var half = p * cols

    @always_inline
    def cat[
        w: Int, alignment: Int = 1
    ](coord: Coord) {var xp, var yp, var op, var half}:
        var e = coord_to_index_list(coord)[0]
        op[unsafe_offset=e] = (
            xp[unsafe_offset=e] if e < half else yp[unsafe_offset=e - half]
        )

    elementwise[simd_width=1, target=_target[gpu]()](cat, Coord(2 * half), ctx)
    ctx.synchronize()
    return out^


def _transposed[
    dtype: DType, gpu: Bool
](
    x: Dynamic[dtype, 2], rows: Int, cols: Int, ctx: DeviceContext
) raises -> Dynamic[dtype, 2]:
    """`x^T` for a dense `rows x cols` operand."""
    var out = Dynamic[dtype, 2](row_major(_dyn_shape[2](cols, rows)), ctx)
    var xp = x.tile().ptr.unsafe_origin_cast[MutAnyOrigin]()
    var op = out.tile().ptr.unsafe_origin_cast[MutAnyOrigin]()

    @always_inline
    def tr[
        w: Int, alignment: Int = 1
    ](coord: Coord) {var xp, var op, var rows, var cols}:
        var e = coord_to_index_list(coord)[0]
        var i = e // rows
        var j = e % rows
        op[unsafe_offset=e] = xp[unsafe_offset=j * cols + i]

    elementwise[simd_width=1, target=_target[gpu]()](
        tr, Coord(rows * cols), ctx
    )
    ctx.synchronize()
    return out^


def _sy2sb[
    dtype: DType, gpu: Bool
](
    mut a: Dynamic[dtype, 2], n: Int, b: Int, ctx: DeviceContext
) raises where dtype.is_floating_point():
    """Stage 1: reduce the symmetric `a` (both triangles held) in place to
    a band of half-width `b` in its lower triangle, keeping the trailing
    block symmetric. LAPACK's `dsytrd_sy2sb` update per panel: with `Q = I
    - V T V^T`, `X = A22 V T`, `S = T^T V^T X` and `W = X - V S / 2`,
    `Q^T A22 Q = A22 - V W^T - W V^T`, the last as one `K = 2p` product."""
    var k = 0
    while k < n - b - 1:
        var r0 = k + b
        var m = n - r0
        var p = min(m, b)
        var panel = _panel_copy[dtype, gpu](a, n, r0, k, m, p, ctx)
        var q = _reflectors[dtype, gpu](panel, m, p, ctx)
        _write_upper[dtype, gpu](a, n, r0, k, panel, p, p, ctx)
        # A narrower last panel leaves columns `k + p .. r0 - 1` inside the
        # band, and `Q^T` mixes their rows: `C -= V (T^T (V^T C))`.
        if k + p < r0:
            var c = _panel_copy[dtype, gpu](a, n, r0, k + p, m, r0 - k - p, ctx)
            var w1 = matmul[gpu=gpu](q.vt, c)
            var w2 = matmul[gpu=gpu](
                _transposed[dtype, gpu](q.t, p, p, ctx), w1
            )
            var w3 = matmul[gpu=gpu](q.v, w2)
            _subtract_into[dtype, gpu](a, n, r0, k + p, w3, m, r0 - k - p, ctx)
        var a22 = _panel_copy[dtype, gpu](a, n, r0, r0, m, m, ctx)
        var x = matmul[gpu=gpu](matmul[gpu=gpu](a22, q.v), q.t)
        var s = matmul[gpu=gpu](
            _transposed[dtype, gpu](q.t, p, p, ctx), matmul[gpu=gpu](q.vt, x)
        )
        var vs = matmul[gpu=gpu](q.v, s)
        var xp = x.tile().ptr.unsafe_origin_cast[MutAnyOrigin]()
        var vsp = vs.tile().ptr.unsafe_origin_cast[MutAnyOrigin]()

        @always_inline
        def half_sub[
            w: Int, alignment: Int = 1
        ](coord: Coord) {var xp, var vsp}:
            var e = coord_to_index_list(coord)[0]
            xp[unsafe_offset=e] = (
                xp[unsafe_offset=e] - Scalar[dtype](0.5) * vsp[unsafe_offset=e]
            )

        elementwise[simd_width=1, target=_target[gpu]()](
            half_sub, Coord(m * p), ctx
        )
        ctx.synchronize()
        # A22 -= [V | W] [W^T ; V^T].
        var left = _concat_columns[dtype, gpu](q.v, x, m, p, ctx)
        var right = _concat_rows[dtype, gpu](
            _transposed[dtype, gpu](x, m, p, ctx), q.vt, p, m, ctx
        )
        _subtract_product[dtype, gpu](a, r0, r0, left, right, m, 2 * p, m, ctx)
        _ = a22^
        _ = panel^
        _ = vs^
        _ = q^
        k += b


def _dsbtrd_lower[
    dtype: DType
](n: Int, kd: Int, mut ab: List[Scalar[dtype]], ldab: Int) -> Tuple[
    List[Scalar[dtype]], List[Scalar[dtype]]
]:
    """`DSBTRD('N', 'L')` on the column-major lower band `ab` (`ldab >= kd +
    1`, `AB(1 + i - j, j) = A(i, j)`): the tridiagonal `(d, e)`. Every
    index below is the Fortran text's, 1-based into `ab` as a flat array,
    because the routine walks it with strides (`INCA`, `LDAB - 1`) that
    cut across columns."""
    var d = List[Scalar[dtype]](length=n + 1, fill=0)
    var e = List[Scalar[dtype]](length=n + 1, fill=0)
    var work = List[Scalar[dtype]](length=n + 1, fill=0)
    if n == 0:
        return (List[Scalar[dtype]](), List[Scalar[dtype]]())

    @always_inline
    def at(i: Int, j: Int) {var ldab} -> Int:
        """The flat 0-based index of `AB(i, j)`."""
        return (i - 1) + (j - 1) * ldab

    var kd1 = kd + 1
    var kdm1 = kd - 1
    var incx = ldab - 1
    var inca = kd1 * ldab
    var kdn = min(n - 1, kd)

    # DROT over `count` pairs `(x, y)` of flat indices, strides `incx`, `incy`.
    @always_inline
    def rot(
        mut v: List[Scalar[dtype]],
        count: Int,
        x0: Int,
        incx_: Int,
        y0: Int,
        incy_: Int,
        c: Scalar[dtype],
        s: Scalar[dtype],
    ):
        var ix = x0
        var iy = y0
        for _ in range(count):
            var xi = v[ix]
            var yi = v[iy]
            v[ix] = c * xi + s * yi
            v[iy] = c * yi - s * xi
            ix += incx_
            iy += incy_

    if kd > 1:
        var nr = 0
        var j1 = kdn + 2
        var j2 = 1
        for i in range(1, n - 1):
            var k = kdn + 1
            while k >= 2:
                j1 += kdn
                j2 += kdn
                if nr > 0:
                    # DLARGV(NR, AB(KD1, J1-KD1), INCA, WORK(J1), KD1,
                    #        D(J1), KD1)
                    var ix = at(kd1, j1 - kd1)
                    var iy = j1
                    for _ in range(nr):
                        var f = ab[ix]
                        var g = work[iy]
                        if g == 0:
                            d[iy] = 1
                        elif f == 0:
                            d[iy] = 0
                            work[iy] = 1
                            ab[ix] = g
                        elif abs(f) > abs(g):
                            var t = g / f
                            var tt = _sqrt(1 + t * t)
                            d[iy] = 1 / tt
                            work[iy] = t * d[iy]
                            ab[ix] = f * tt
                        else:
                            var t = f / g
                            var tt = _sqrt(1 + t * t)
                            work[iy] = 1 / tt
                            d[iy] = t * work[iy]
                            ab[ix] = g * tt
                        ix += inca
                        iy += kd1
                    if nr > 2 * kd - 1:
                        for l in range(1, kd):
                            # DLARTV(NR, AB(KD1-L, J1-KD1+L), INCA,
                            #        AB(KD1-L+1, J1-KD1+L), INCA, D(J1),
                            #        WORK(J1), KD1)
                            var x = at(kd1 - l, j1 - kd1 + l)
                            var y = at(kd1 - l + 1, j1 - kd1 + l)
                            var ic = j1
                            for _ in range(nr):
                                var xi = ab[x]
                                var yi = ab[y]
                                ab[x] = d[ic] * xi + work[ic] * yi
                                ab[y] = d[ic] * yi - work[ic] * xi
                                x += inca
                                y += inca
                                ic += kd1
                    else:
                        var jend = j1 + kd1 * (nr - 1)
                        var jinc = j1
                        while jinc <= jend:
                            rot(
                                ab,
                                kdm1,
                                at(kd, jinc - kd),
                                incx,
                                at(kd1, jinc - kd),
                                incx,
                                d[jinc],
                                work[jinc],
                            )
                            jinc += kd1
                if k > 2:
                    if k <= n - i + 1:
                        var g = _dlartg(ab[at(k - 1, i)], ab[at(k, i)])
                        d[i + k - 1] = g[0]
                        work[i + k - 1] = g[1]
                        ab[at(k - 1, i)] = g[2]
                        rot(
                            ab,
                            k - 3,
                            at(k - 2, i + 1),
                            ldab - 1,
                            at(k - 1, i + 1),
                            ldab - 1,
                            d[i + k - 1],
                            work[i + k - 1],
                        )
                    nr += 1
                    j1 = j1 - kdn - 1
                if nr > 0:
                    # DLAR2V(NR, AB(1, J1-1), AB(1, J1), AB(2, J1-1), INCA,
                    #        D(J1), WORK(J1), KD1)
                    var x = at(1, j1 - 1)
                    var y = at(1, j1)
                    var z = at(2, j1 - 1)
                    var ic = j1
                    for _ in range(nr):
                        var xi = ab[x]
                        var yi = ab[y]
                        var zi = ab[z]
                        var ci = d[ic]
                        var si = work[ic]
                        var t1 = si * zi
                        var t2 = ci * zi
                        var t3 = t2 - si * xi
                        var t4 = t2 + si * yi
                        var t5 = ci * xi + t1
                        var t6 = ci * yi - t1
                        ab[x] = ci * t5 + si * t4
                        ab[y] = ci * t6 - si * t3
                        ab[z] = ci * t4 - si * t5
                        x += inca
                        y += inca
                        z += inca
                        ic += kd1
                if nr > 0:
                    if nr > 2 * kd - 1:
                        for l in range(1, kd):
                            var nrt = nr - 1 if j2 + l > n else nr
                            if nrt > 0:
                                # DLARTV(NRT, AB(L+2, J1-1), INCA,
                                #        AB(L+1, J1), INCA, D(J1),
                                #        WORK(J1), KD1)
                                var x = at(l + 2, j1 - 1)
                                var y = at(l + 1, j1)
                                var ic = j1
                                for _ in range(nrt):
                                    var xi = ab[x]
                                    var yi = ab[y]
                                    ab[x] = d[ic] * xi + work[ic] * yi
                                    ab[y] = d[ic] * yi - work[ic] * xi
                                    x += inca
                                    y += inca
                                    ic += kd1
                    else:
                        var j1end = j1 + kd1 * (nr - 2)
                        if j1end >= j1:
                            var j1inc = j1
                            while j1inc <= j1end:
                                rot(
                                    ab,
                                    kdm1,
                                    at(3, j1inc - 1),
                                    1,
                                    at(2, j1inc),
                                    1,
                                    d[j1inc],
                                    work[j1inc],
                                )
                                j1inc += kd1
                        var lend = min(kdm1, n - j2)
                        var last = j1end + kd1
                        if lend > 0:
                            rot(
                                ab,
                                lend,
                                at(3, last - 1),
                                1,
                                at(2, last),
                                1,
                                d[last],
                                work[last],
                            )
                if j2 + kdn > n:
                    nr -= 1
                    j2 = j2 - kdn - 1
                var j = j1
                while j <= j2:
                    work[j + kd] = work[j] * ab[at(kd1, j)]
                    ab[at(kd1, j)] = d[j] * ab[at(kd1, j)]
                    j += kd1
                k -= 1
    var dout = List[Scalar[dtype]](capacity=n)
    var eout = List[Scalar[dtype]](capacity=n)
    for i in range(1, n + 1):
        dout.append(ab[at(1, i)])
    for i in range(1, n):
        eout.append(ab[at(2, i)] if kd > 0 else Scalar[dtype](0))
    eout.append(0)
    return (dout^, eout^)


def _two_stage_tridiagonal[
    dtype: DType, gpu: Bool
](var a: Dynamic[dtype, 2], n: Int, b: Int) raises -> Tuple[
    List[Scalar[dtype]], List[Scalar[dtype]]
] where dtype.is_floating_point():
    """The tridiagonal `(d, e)` similar to the symmetric `a` (both
    triangles held, row-major, run-time order `n`), by the two stages.
    `e` has `n` entries, the last zero, the layout `_tql` reads."""
    var ctx = a.context()
    var kd = min(b, n - 1)
    _sy2sb[dtype, gpu](a, n, kd, ctx)
    var host = a.to_host()
    var ldab = kd + 1
    # Stage 2 runs in `float64` whatever `dtype` is: it is host scalar
    # work, and its `O(n^2 / kd)` rotations per element are where a
    # `float32` reduction would lose its digits.
    var ab = List[Float64](length=ldab * n, fill=0)
    for j in range(n):
        for i in range(j, min(n, j + kd + 1)):
            ab[(i - j) + j * ldab] = Float64(host[i * n + j])
    var tri = _dsbtrd_lower[DType.float64](n, kd, ab, ldab)
    var d = List[Scalar[dtype]](capacity=n)
    var e = List[Scalar[dtype]](capacity=n)
    for i in range(n):
        d.append(Scalar[dtype](tri[0][i]))
        e.append(Scalar[dtype](tri[1][i]))
    return (d^, e^)


def _left_apply[
    dtype: DType, gpu: Bool
](
    mut a: Dynamic[dtype, 2],
    n: Int,
    r0: Int,
    c0: Int,
    rows: Int,
    cols: Int,
    mut q: _Reflectors[dtype],
    ctx: DeviceContext,
) raises where dtype.is_floating_point():
    """`A[r0 :, c0 :] := Q^T A[...]` over `rows x cols`: `C -= V (T^T (V^T
    C))`."""
    if cols <= 0:
        return
    var c = _panel_copy[dtype, gpu](a, n, r0, c0, rows, cols, ctx)
    var y = matmul[gpu=gpu](q.vt, c)
    var z = matmul[gpu=gpu](_transposed[dtype, gpu](q.t, q.p, q.p, ctx), y)
    _subtract_product[dtype, gpu](a, r0, c0, q.v, z, rows, q.p, cols, ctx)
    _ = c^


def _right_apply[
    dtype: DType, gpu: Bool
](
    mut a: Dynamic[dtype, 2],
    n: Int,
    r0: Int,
    c0: Int,
    rows: Int,
    cols: Int,
    mut q: _Reflectors[dtype],
    ctx: DeviceContext,
) raises where dtype.is_floating_point():
    """`A[r0 :, c0 :] := A[...] Q` over `rows x cols`: `C -= ((C V) T)
    V^T`."""
    if rows <= 0:
        return
    var c = _panel_copy[dtype, gpu](a, n, r0, c0, rows, cols, ctx)
    var w = matmul[gpu=gpu](matmul[gpu=gpu](c, q.v), q.t)
    _subtract_product[dtype, gpu](a, r0, c0, w, q.vt, rows, q.p, cols, ctx)
    _ = c^


def _ge2gb[
    dtype: DType, gpu: Bool
](
    mut a: Dynamic[dtype, 2], m: Int, n: Int, b: Int, ctx: DeviceContext
) raises where dtype.is_floating_point():
    """Stage 1 for the singular values: reduce the row-major `m x n` `a`
    (`m >= n`) in place to an upper band of width `b` by alternating QR
    panels (applied on the left) and LQ panels (the QR of the transposed
    row block, applied on the right) -- the dense-to-band step of a
    two-stage `gebrd`."""
    var k = 0
    while k < n:
        var p = min(b, n - k)
        var panel = _panel_copy[dtype, gpu](a, n, k, k, m - k, p, ctx)
        var q = _reflectors[dtype, gpu](panel, m - k, p, ctx)
        _write_upper[dtype, gpu](a, n, k, k, panel, p, p, ctx)
        _left_apply[dtype, gpu](a, n, k, k + p, m - k, n - k - p, q, ctx)
        var w = n - k - p
        if w > 0:
            # The row block `A[k : k + p, k + p :]`, transposed to `w x p`.
            var row = Dynamic[dtype, 2](row_major(_dyn_shape[2](w, p)), ctx)
            var av = a.tile()
            var rv = row.tile()
            pack_block[trans=True, target=_target[gpu]()](
                av, rv, k + p, k, w, p, ctx
            )
            ctx.synchronize()
            var q2 = _reflectors[dtype, gpu](row, w, p, ctx)
            # A[k + i, k + p + c] = R2[c, i] for c <= i: the lower `L`.
            var rows_r = min(w, p)
            var r2 = row.to_host()
            var host_l = List[Scalar[dtype]](length=p * rows_r, fill=0)
            for c in range(rows_r):
                for i in range(c, p):
                    host_l[i * rows_r + c] = r2[c * p + i]
            var lower = _upload[dtype](host_l^, p, rows_r, ctx)
            _write_lower[dtype, gpu](a, n, k, k + p, lower, p, rows_r, ctx)
            _right_apply[dtype, gpu](a, n, k + p, k + p, m - k - p, w, q2, ctx)
            _ = q2^
            _ = row^
            _ = lower^
        _ = q^
        _ = panel^
        k += p


def _write_lower[
    dtype: DType, gpu: Bool
](
    mut a: Dynamic[dtype, 2],
    n: Int,
    r0: Int,
    c0: Int,
    src: Dynamic[dtype, 2],
    rows: Int,
    cols: Int,
    ctx: DeviceContext,
) raises:
    """`A[r0 + i, c0 + j] = src[i, j]` for `j <= i`, `j < cols`."""
    var ap = a.tile().ptr.unsafe_origin_cast[MutAnyOrigin]()
    var sp = src.tile().ptr.unsafe_origin_cast[MutAnyOrigin]()

    @always_inline
    def write[
        w: Int, alignment: Int = 1
    ](coord: Coord) {var ap, var sp, var n, var r0, var c0, var cols}:
        var e = coord_to_index_list(coord)[0]
        var i = e // cols
        var j = e % cols
        if j <= i:
            ap[unsafe_offset=(r0 + i) * n + c0 + j] = sp[unsafe_offset=e]

    elementwise[simd_width=1, target=_target[gpu]()](
        write, Coord(rows * cols), ctx
    )
    ctx.synchronize()


def _dgbbrd_upper[
    dtype: DType
](m: Int, n: Int, ku: Int, mut ab: List[Scalar[dtype]], ldab: Int) -> Tuple[
    List[Scalar[dtype]], List[Scalar[dtype]]
]:
    """`DGBBRD('N', m, n, 0, 0, ku)` on the column-major upper band `ab`
    (`AB(ku + 1 + i - j, j) = A(i, j)`, `m >= n`): the upper bidiagonal
    `(d, e)`, `e` of length `n` with the last entry zero. Fortran indices,
    1-based into `ab` as a flat array, as in `_dsbtrd_lower`."""
    var kl = 0
    var mn = max(m, n)
    var minmn = min(m, n)
    var work = List[Scalar[dtype]](length=2 * mn + 2, fill=0)
    var klu1 = kl + ku + 1

    @always_inline
    def at(i: Int, j: Int) {var ldab} -> Int:
        return (i - 1) + (j - 1) * ldab

    @always_inline
    def rot(
        mut v: List[Scalar[dtype]],
        count: Int,
        x0: Int,
        incx_: Int,
        y0: Int,
        incy_: Int,
        c: Scalar[dtype],
        s: Scalar[dtype],
    ):
        var ix = x0
        var iy = y0
        for _ in range(count):
            var xi = v[ix]
            var yi = v[iy]
            v[ix] = c * xi + s * yi
            v[iy] = c * yi - s * xi
            ix += incx_
            iy += incy_

    @always_inline
    def largv(
        mut ab: List[Scalar[dtype]],
        mut work: List[Scalar[dtype]],
        nr: Int,
        x0: Int,
        inca: Int,
        y0: Int,
        c0: Int,
        inc: Int,
    ):
        """`DLARGV(nr, AB(x0), inca, WORK(y0), inc, WORK(c0), inc)`."""
        var ix = x0
        var iy = y0
        var ic = c0
        for _ in range(nr):
            var f = ab[ix]
            var g = work[iy]
            if g == 0:
                work[ic] = 1
            elif f == 0:
                work[ic] = 0
                work[iy] = 1
                ab[ix] = g
            elif abs(f) > abs(g):
                var t = g / f
                var tt = _sqrt(1 + t * t)
                work[ic] = 1 / tt
                work[iy] = t * work[ic]
                ab[ix] = f * tt
            else:
                var t = f / g
                var tt = _sqrt(1 + t * t)
                work[iy] = 1 / tt
                work[ic] = t * work[iy]
                ab[ix] = g * tt
            ix += inca
            iy += inc
            ic += inc

    @always_inline
    def lartv(
        mut ab: List[Scalar[dtype]],
        work: List[Scalar[dtype]],
        nr: Int,
        x0: Int,
        y0: Int,
        inca: Int,
        c0: Int,
        s0: Int,
        inc: Int,
    ):
        """`DLARTV(nr, AB(x0), inca, AB(y0), inca, WORK(c0), WORK(s0), inc)`."""
        var x = x0
        var y = y0
        var ic = c0
        var is_ = s0
        for _ in range(nr):
            var xi = ab[x]
            var yi = ab[y]
            ab[x] = work[ic] * xi + work[is_] * yi
            ab[y] = work[ic] * yi - work[is_] * xi
            x += inca
            y += inca
            ic += inc
            is_ += inc

    if kl + ku > 1:
        var ml0 = 1
        var mu0 = 2
        var klm = min(m - 1, kl)
        var kun = min(n - 1, ku)
        var kb = klm + kun
        var kb1 = kb + 1
        var inca = kb1 * ldab
        var nr = 0
        var j1 = klm + 2
        var j2 = 1 - kun
        for i in range(1, minmn + 1):
            var ml = klm + 1
            var mu = kun + 1
            for _kk in range(1, kb + 1):
                j1 += kb
                j2 += kb
                if nr > 0:
                    largv(
                        ab,
                        work,
                        nr,
                        at(klu1, j1 - klm - 1),
                        inca,
                        j1,
                        mn + j1,
                        kb1,
                    )
                for l in range(1, kb + 1):
                    var nrt = nr - 1 if j2 - klm + l - 1 > n else nr
                    if nrt > 0:
                        lartv(
                            ab,
                            work,
                            nrt,
                            at(klu1 - l, j1 - klm + l - 1),
                            at(klu1 - l + 1, j1 - klm + l - 1),
                            inca,
                            mn + j1,
                            j1,
                            kb1,
                        )
                if ml > ml0:
                    if ml <= m - i + 1:
                        var g = _dlartg(
                            ab[at(ku + ml - 1, i)], ab[at(ku + ml, i)]
                        )
                        work[mn + i + ml - 1] = g[0]
                        work[i + ml - 1] = g[1]
                        ab[at(ku + ml - 1, i)] = g[2]
                        if i < n:
                            rot(
                                ab,
                                min(ku + ml - 2, n - i),
                                at(ku + ml - 2, i + 1),
                                ldab - 1,
                                at(ku + ml - 1, i + 1),
                                ldab - 1,
                                work[mn + i + ml - 1],
                                work[i + ml - 1],
                            )
                    nr += 1
                    j1 -= kb1
                if j2 + kun > n:
                    nr -= 1
                    j2 -= kb1
                var j = j1
                while j <= j2:
                    work[j + kun] = work[j] * ab[at(1, j + kun)]
                    ab[at(1, j + kun)] = work[mn + j] * ab[at(1, j + kun)]
                    j += kb1
                if nr > 0:
                    largv(
                        ab,
                        work,
                        nr,
                        at(1, j1 + kun - 1),
                        inca,
                        j1 + kun,
                        mn + j1 + kun,
                        kb1,
                    )
                for l in range(1, kb + 1):
                    var nrt = nr - 1 if j2 + l - 1 > m else nr
                    if nrt > 0:
                        lartv(
                            ab,
                            work,
                            nrt,
                            at(l + 1, j1 + kun - 1),
                            at(l, j1 + kun),
                            inca,
                            mn + j1 + kun,
                            j1 + kun,
                            kb1,
                        )
                if ml == ml0 and mu > mu0:
                    if mu <= n - i + 1:
                        var g = _dlartg(
                            ab[at(ku - mu + 3, i + mu - 2)],
                            ab[at(ku - mu + 2, i + mu - 1)],
                        )
                        work[mn + i + mu - 1] = g[0]
                        work[i + mu - 1] = g[1]
                        ab[at(ku - mu + 3, i + mu - 2)] = g[2]
                        rot(
                            ab,
                            min(kl + mu - 2, m - i),
                            at(ku - mu + 4, i + mu - 2),
                            1,
                            at(ku - mu + 3, i + mu - 1),
                            1,
                            work[mn + i + mu - 1],
                            work[i + mu - 1],
                        )
                    nr += 1
                    j1 -= kb1
                if j2 + kb > m:
                    nr -= 1
                    j2 -= kb1
                j = j1
                while j <= j2:
                    work[j + kb] = work[j + kun] * ab[at(klu1, j + kun)]
                    ab[at(klu1, j + kun)] = (
                        work[mn + j + kun] * ab[at(klu1, j + kun)]
                    )
                    j += kb1
                if ml > ml0:
                    ml -= 1
                else:
                    mu -= 1
    var d = List[Scalar[dtype]](capacity=n)
    var e = List[Scalar[dtype]](capacity=n)
    for i in range(1, minmn + 1):
        d.append(ab[at(ku + 1, i)])
    for i in range(1, minmn):
        e.append(ab[at(ku, i + 1)] if ku > 0 else Scalar[dtype](0))
    e.append(0)
    return (d^, e^)


def _two_stage_bidiagonal[
    dtype: DType, gpu: Bool
](var a: Dynamic[dtype, 2], m: Int, n: Int, b: Int) raises -> Tuple[
    List[Scalar[dtype]], List[Scalar[dtype]]
] where dtype.is_floating_point():
    """The upper bidiagonal `(d, e)` with the singular values of the
    row-major `m x n` `a` (`m >= n`), by the two stages; `e` has `n`
    entries, the last zero, the layout `_bdsqr` reads."""
    var ctx = a.context()
    var ku = min(b, n - 1)
    if ku < 1:
        ku = 1
    _ge2gb[dtype, gpu](a, m, n, ku, ctx)
    var host = a.to_host()
    var ldab = ku + 1
    # Stage 2 in `float64`, as for the tridiagonal reduction.
    var ab = List[Float64](length=ldab * n, fill=0)
    for j in range(n):
        for i in range(max(0, j - ku), min(m, j + 1)):
            ab[(ku + i - j) + j * ldab] = Float64(host[i * n + j])
    var bd = _dgbbrd_upper[DType.float64](m, n, ku, ab, ldab)
    var d = List[Scalar[dtype]](capacity=n)
    var e = List[Scalar[dtype]](capacity=n)
    for i in range(n):
        d.append(Scalar[dtype](bd[0][i]))
        e.append(Scalar[dtype](bd[1][i]))
    return (d^, e^)
