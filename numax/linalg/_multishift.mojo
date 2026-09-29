"""The small-bulge multishift QR algorithm with aggressive early
deflation: LAPACK's `dhseqr` path (`dlaqr0`, `dlaqr3`, `dlaqr5`, with
`dlahqr`, `dtrexc`, `dlaexc`, `dlanv2`, `dlaqr1` beneath), transcribed
from the reference implementation (Reference-LAPACK, BSD licensed) onto
host memory at the matrix's `dtype`.

**Tier 2, private, host-side.** `_hqr` in `eigen.mojo` is the
double-shift iteration and stays for small matrices; above `_NMIN` this
is what `eigvals` runs. The reason it is faster is structural, not
arithmetic: `dlaqr5` chases a chain of `ns / 2` tightly packed `3 x 3`
bulges through a `2 ns`-wide window, accumulating the window's
orthogonal transformation, and applies it to the rest of `H` as one
matrix product rather than `ns / 2` sweeps of rank-3 updates;
`dlaqr3`'s early deflation finds converged eigenvalues in a trailing
window by a Schur decomposition of that window, so many deflate per
iteration instead of one or two.

Indices below are 1-based and follow the Fortran text line for line,
through `_M` (a 1-based row-major view onto a host buffer) so that a
reader can hold the two side by side. Two routines are simplified where
numax's use is narrower than LAPACK's: `dlasy2`'s `1 x 1`/`2 x 2`
Sylvester solve is a direct Kronecker solve with complete pivoting (the
same perturbation of a tiny pivot, no overflow scaling), and `dlaqr3`
always uses `dlahqr` for its window where LAPACK switches to the
recursive `dlaqr4` past 75.
"""

from std.math import log as _log, sqrt as _sqrt
from std.sys.info import simd_width_of


struct _M[dtype: DType](ImplicitlyCopyable):
    """A 1-based view `A(i, j)` onto row-major host storage: element
    `(i, j)` is `p[off + (i - 1) ld + (j - 1)]`."""

    var p: Pointer[Scalar[Self.dtype], MutUntrackedOrigin]
    var off: Int
    var ld: Int

    @always_inline
    def __init__(
        out self,
        p: Pointer[Scalar[Self.dtype], MutUntrackedOrigin],
        off: Int,
        ld: Int,
    ):
        self.p = p
        self.off = off
        self.ld = ld

    @always_inline
    def __getitem__(self, i: Int, j: Int) -> Scalar[Self.dtype]:
        return self.p[unsafe_offset=self.off + (i - 1) * self.ld + (j - 1)]

    @always_inline
    def __setitem__(self, i: Int, j: Int, v: Scalar[Self.dtype]):
        self.p[unsafe_offset=self.off + (i - 1) * self.ld + (j - 1)] = v

    @always_inline
    def sub(self, i: Int, j: Int) -> Self:
        """The view whose `(1, 1)` is this one's `(i, j)`."""
        return Self(self.p, self.off + (i - 1) * self.ld + (j - 1), self.ld)


def _view[dtype: DType](mut buffer: List[Scalar[dtype]], ld: Int) -> _M[dtype]:
    return _M[dtype](
        buffer.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin](), 0, ld
    )


@always_inline
def _keep[dtype: DType](buffer: List[Scalar[dtype]]):
    """A visible use of `buffer`. An `_M` view's untracked pointer does
    not keep its `List` alive, and ASAP destruction frees a buffer after
    its last *visible* use -- so every path out of a function that reads a
    local buffer through a view touches it here first."""
    _ = len(buffer)


@always_inline
def _sign[dtype: DType](a: Scalar[dtype], b: Scalar[dtype]) -> Scalar[dtype]:
    """Fortran's `SIGN(a, b)`: `|a|` with `b`'s sign (`+` at `b = 0`)."""
    return abs(a) if b >= 0 else -abs(a)


@always_inline
def _ulp[dtype: DType]() -> Scalar[dtype]:
    """`DLAMCH('P')`, `eps * base`."""
    comptime if dtype == DType.float64:
        return Scalar[dtype](2.220446049250313e-16)
    else:
        return Scalar[dtype](1.1920929e-07)


@always_inline
def _safmin[dtype: DType]() -> Scalar[dtype]:
    """`DLAMCH('S')`, the smallest normal number."""
    comptime if dtype == DType.float64:
        return Scalar[dtype](2.2250738585072014e-308)
    else:
        return Scalar[dtype](1.1754944e-38)


# ------------------------------------------------------------ elementary


@always_inline
def _dlapy2[dtype: DType](x: Scalar[dtype], y: Scalar[dtype]) -> Scalar[dtype]:
    var a = abs(x)
    var b = abs(y)
    var w = max(a, b)
    var z = min(a, b)
    if z == 0 or w == 0:
        return w
    var r = z / w
    return w * _sqrt(1 + r * r)


@always_inline
def _dlartg[
    dtype: DType
](f: Scalar[dtype], g: Scalar[dtype]) -> Tuple[
    Scalar[dtype], Scalar[dtype], Scalar[dtype]
]:
    """`(cs, sn, r)` with `[cs sn; -sn cs] [f; g] = [r; 0]`."""
    if g == 0:
        return (Scalar[dtype](1), Scalar[dtype](0), f)
    if f == 0:
        return (Scalar[dtype](0), _sign(Scalar[dtype](1), g), abs(g))
    var d = _dlapy2(f, g)
    var r = _sign(d, f)
    return (abs(f) / d, g / r, r)


@always_inline
def _drot_rows[
    dtype: DType
](
    a: _M[dtype],
    i1: Int,
    i2: Int,
    j0: Int,
    count: Int,
    cs: Scalar[dtype],
    sn: Scalar[dtype],
):
    """`DROT` on rows `i1`, `i2` over `count` columns from `j0`."""
    for k in range(count):
        var j = j0 + k
        var x = a[i1, j]
        var y = a[i2, j]
        a[i1, j] = cs * x + sn * y
        a[i2, j] = cs * y - sn * x


@always_inline
def _drot_cols[
    dtype: DType
](
    a: _M[dtype],
    j1: Int,
    j2: Int,
    i0: Int,
    count: Int,
    cs: Scalar[dtype],
    sn: Scalar[dtype],
):
    """`DROT` on columns `j1`, `j2` over `count` rows from `i0`."""
    for k in range(count):
        var i = i0 + k
        var x = a[i, j1]
        var y = a[i, j2]
        a[i, j1] = cs * x + sn * y
        a[i, j2] = cs * y - sn * x


def _dlarfg[
    dtype: DType
](n: Int, alpha: Scalar[dtype], x: _M[dtype], incx_row: Bool) -> Tuple[
    Scalar[dtype], Scalar[dtype]
]:
    """`DLARFG(n, alpha, x, 1, tau)` with `x` the `n - 1` entries at
    `x(1..n-1, 1)` (`incx_row=False`, down a column) or `x(1, 1..n-1)`
    (along a row). Scales `x` in place; returns `(beta, tau)`."""
    if n <= 1:
        return (alpha, Scalar[dtype](0))
    var xnorm = Scalar[dtype](0)
    var scale = Scalar[dtype](0)
    for k in range(1, n):
        var v = x[1, k] if incx_row else x[k, 1]
        scale = max(scale, abs(v))
    if scale > 0:
        var ssq = Scalar[dtype](0)
        for k in range(1, n):
            var v = (x[1, k] if incx_row else x[k, 1]) / scale
            ssq += v * v
        xnorm = scale * _sqrt(ssq)
    if xnorm == 0:
        return (alpha, Scalar[dtype](0))
    var beta = -_sign(_dlapy2(alpha, xnorm), alpha)
    var tau = (beta - alpha) / beta
    var inv = 1 / (alpha - beta)
    for k in range(1, n):
        if incx_row:
            x[1, k] = x[1, k] * inv
        else:
            x[k, 1] = x[k, 1] * inv
    return (beta, tau)


struct _Nv2[dtype: DType](ImplicitlyCopyable):
    var a: Scalar[Self.dtype]
    var b: Scalar[Self.dtype]
    var c: Scalar[Self.dtype]
    var d: Scalar[Self.dtype]
    var rt1r: Scalar[Self.dtype]
    var rt1i: Scalar[Self.dtype]
    var rt2r: Scalar[Self.dtype]
    var rt2i: Scalar[Self.dtype]
    var cs: Scalar[Self.dtype]
    var sn: Scalar[Self.dtype]

    def __init__(out self):
        self.a = 0
        self.b = 0
        self.c = 0
        self.d = 0
        self.rt1r = 0
        self.rt1i = 0
        self.rt2r = 0
        self.rt2i = 0
        self.cs = 1
        self.sn = 0


def _dlanv2[
    dtype: DType
](
    a_in: Scalar[dtype],
    b_in: Scalar[dtype],
    c_in: Scalar[dtype],
    d_in: Scalar[dtype],
) -> _Nv2[dtype]:
    """`DLANV2`: the Schur factorization of a real `2 x 2` in standard
    form, `[a b; c d] = [cs -sn; sn cs] [aa bb; cc dd] [cs sn; -sn cs]`."""
    comptime multpl = 4.0
    var r = _Nv2[dtype]()
    var a = a_in
    var b = b_in
    var c = c_in
    var d = d_in
    var eps = _ulp[dtype]()
    var cs = Scalar[dtype](1)
    var sn = Scalar[dtype](0)
    if c == 0:
        cs = 1
        sn = 0
    elif b == 0:
        cs = 0
        sn = 1
        var temp = d
        d = a
        a = temp
        b = -c
        c = 0
    elif (a - d) == 0 and _sign(Scalar[dtype](1), b) != _sign(
        Scalar[dtype](1), c
    ):
        cs = 1
        sn = 0
    else:
        var temp = a - d
        var p = Scalar[dtype](0.5) * temp
        var bcmax = max(abs(b), abs(c))
        var bcmis = (
            min(abs(b), abs(c))
            * _sign(Scalar[dtype](1), b)
            * _sign(Scalar[dtype](1), c)
        )
        var scale = max(abs(p), bcmax)
        var z = (p / scale) * p + (bcmax / scale) * bcmis
        if z >= Scalar[dtype](multpl) * eps:
            z = p + _sign(_sqrt(scale) * _sqrt(z), p)
            a = d + z
            d = d - (bcmax / z) * bcmis
            var tau = _dlapy2(c, z)
            cs = z / tau
            sn = c / tau
            b = b - c
            c = 0
        else:
            var sigma = b + c
            p = Scalar[dtype](0.5) * temp
            var tau = _dlapy2(sigma, temp)
            cs = _sqrt(Scalar[dtype](0.5) * (1 + abs(sigma) / tau))
            sn = -(p / (tau * cs)) * _sign(Scalar[dtype](1), sigma)
            var aa = a * cs + b * sn
            var bb = -a * sn + b * cs
            var cc = c * cs + d * sn
            var dd = -c * sn + d * cs
            a = aa * cs + cc * sn
            b = (bb * cs) + (dd * sn)
            c = -(aa * sn) + (cc * cs)
            d = -bb * sn + dd * cs
            temp = Scalar[dtype](0.5) * (a + d)
            a = temp
            d = temp
            if c != 0:
                if b != 0:
                    if _sign(Scalar[dtype](1), b) == _sign(Scalar[dtype](1), c):
                        var sab = _sqrt(abs(b))
                        var sac = _sqrt(abs(c))
                        p = _sign(sab * sac, c)
                        tau = 1 / _sqrt(abs(b + c))
                        a = temp + p
                        d = temp - p
                        b = b - c
                        c = 0
                        var cs1 = sab * tau
                        var sn1 = sac * tau
                        temp = cs * cs1 - sn * sn1
                        sn = cs * sn1 + sn * cs1
                        cs = temp
                else:
                    b = -c
                    c = 0
                    temp = cs
                    cs = -sn
                    sn = temp
    r.a = a
    r.b = b
    r.c = c
    r.d = d
    r.rt1r = a
    r.rt2r = d
    if c == 0:
        r.rt1i = 0
        r.rt2i = 0
    else:
        r.rt1i = _sqrt(abs(b)) * _sqrt(abs(c))
        r.rt2i = -r.rt1i
    r.cs = cs
    r.sn = sn
    return r


# ------------------------------------------------------------- dlahqr


def _dlahqr[
    dtype: DType
](
    wantt: Bool,
    wantz: Bool,
    n: Int,
    ilo: Int,
    ihi: Int,
    h: _M[dtype],
    wr: _M[dtype],
    wi: _M[dtype],
    iloz: Int,
    ihiz: Int,
    z: _M[dtype],
) -> Int:
    """`DLAHQR`: the double-shift QR on `H(ilo:ihi, ilo:ihi)`. `wr`/`wi`
    are indexed `(i, 1)`. Returns `INFO`."""
    comptime dat1 = 0.75
    comptime dat2 = -0.4375
    comptime kexsh = 10
    if n == 0:
        return 0
    if ilo == ihi:
        wr[ilo, 1] = h[ilo, ilo]
        wi[ilo, 1] = 0
        return 0
    for j in range(ilo, ihi - 2):
        h[j + 2, j] = 0
        h[j + 3, j] = 0
    if ilo <= ihi - 2:
        h[ihi, ihi - 2] = 0
    var nh = ihi - ilo + 1
    var nz = ihiz - iloz + 1
    var safmin = _safmin[dtype]()
    var ulp = _ulp[dtype]()
    var smlnum = safmin * (Scalar[dtype](nh) / ulp)
    var i1 = 1
    var i2 = n
    var itmax = 30 * max(10, nh)
    var kdefl = 0
    var i = ihi
    while True:
        var l = ilo
        if i < ilo:
            return 0
        var converged = False
        for _its in range(itmax + 1):
            var k = i
            while k >= l + 1:
                if abs(h[k, k - 1]) <= smlnum:
                    break
                var tst = abs(h[k - 1, k - 1]) + abs(h[k, k])
                if tst == 0:
                    if k - 2 >= ilo:
                        tst += abs(h[k - 1, k - 2])
                    if k + 1 <= ihi:
                        tst += abs(h[k + 1, k])
                if abs(h[k, k - 1]) <= ulp * tst:
                    var ab = max(abs(h[k, k - 1]), abs(h[k - 1, k]))
                    var ba = min(abs(h[k, k - 1]), abs(h[k - 1, k]))
                    var aa = max(abs(h[k, k]), abs(h[k - 1, k - 1] - h[k, k]))
                    var bb = min(abs(h[k, k]), abs(h[k - 1, k - 1] - h[k, k]))
                    var s = aa + ab
                    if ba * (ab / s) <= max(smlnum, ulp * (bb * (aa / s))):
                        break
                k -= 1
            l = k
            if l > ilo:
                h[l, l - 1] = 0
            if l >= i - 1:
                converged = True
                break
            kdefl += 1
            if not wantt:
                i1 = l
                i2 = i
            var h11: Scalar[dtype]
            var h12: Scalar[dtype]
            var h21: Scalar[dtype]
            var h22: Scalar[dtype]
            if kdefl % (2 * kexsh) == 0:
                var s = abs(h[i, i - 1]) + abs(h[i - 1, i - 2])
                h11 = Scalar[dtype](dat1) * s + h[i, i]
                h12 = Scalar[dtype](dat2) * s
                h21 = s
                h22 = h11
            elif kdefl % kexsh == 0:
                var s = abs(h[l + 1, l]) + abs(h[l + 2, l + 1])
                h11 = Scalar[dtype](dat1) * s + h[l, l]
                h12 = Scalar[dtype](dat2) * s
                h21 = s
                h22 = h11
            else:
                h11 = h[i - 1, i - 1]
                h21 = h[i, i - 1]
                h12 = h[i - 1, i]
                h22 = h[i, i]
            var s = abs(h11) + abs(h12) + abs(h21) + abs(h22)
            var rt1r = Scalar[dtype](0)
            var rt1i = Scalar[dtype](0)
            var rt2r = Scalar[dtype](0)
            var rt2i = Scalar[dtype](0)
            if s != 0:
                h11 = h11 / s
                h21 = h21 / s
                h12 = h12 / s
                h22 = h22 / s
                var tr = (h11 + h22) / 2
                var det = (h11 - tr) * (h22 - tr) - h12 * h21
                var rtdisc = _sqrt(abs(det))
                if det >= 0:
                    rt1r = tr * s
                    rt2r = rt1r
                    rt1i = rtdisc * s
                    rt2i = -rt1i
                else:
                    rt1r = tr + rtdisc
                    rt2r = tr - rtdisc
                    if abs(rt1r - h22) <= abs(rt2r - h22):
                        rt1r = rt1r * s
                        rt2r = rt1r
                    else:
                        rt2r = rt2r * s
                        rt1r = rt2r
                    rt1i = 0
                    rt2i = 0
            var v1 = Scalar[dtype](0)
            var v2 = Scalar[dtype](0)
            var v3 = Scalar[dtype](0)
            var m = i - 2
            while m >= l:
                var h21s = h[m + 1, m]
                s = abs(h[m, m] - rt2r) + abs(rt2i) + abs(h21s)
                h21s = h[m + 1, m] / s
                v1 = (
                    h21s * h[m, m + 1]
                    + (h[m, m] - rt1r) * ((h[m, m] - rt2r) / s)
                    - rt1i * (rt2i / s)
                )
                v2 = h21s * (h[m, m] + h[m + 1, m + 1] - rt1r - rt2r)
                v3 = h21s * h[m + 2, m + 1]
                s = abs(v1) + abs(v2) + abs(v3)
                v1 = v1 / s
                v2 = v2 / s
                v3 = v3 / s
                if m == l:
                    break
                if abs(h[m, m - 1]) * (abs(v2) + abs(v3)) <= ulp * abs(v1) * (
                    abs(h[m - 1, m - 1]) + abs(h[m, m]) + abs(h[m + 1, m + 1])
                ):
                    break
                m -= 1
            for kk in range(m, i):
                var nr = min(3, i - kk + 1)
                if kk > m:
                    v1 = h[kk, kk - 1]
                    v2 = h[kk + 1, kk - 1]
                    v3 = h[kk + 2, kk - 1] if nr == 3 else Scalar[dtype](0)
                # DLARFG(nr, v(1), v(2), 1, t1)
                var xnorm: Scalar[dtype]
                if nr == 3:
                    xnorm = _dlapy2(v2, v3)
                else:
                    xnorm = abs(v2)
                var t1 = Scalar[dtype](0)
                if xnorm != 0:
                    var beta = -_sign(_dlapy2(v1, xnorm), v1)
                    t1 = (beta - v1) / beta
                    var inv = 1 / (v1 - beta)
                    v2 = v2 * inv
                    v3 = v3 * inv
                    v1 = beta
                if kk > m:
                    h[kk, kk - 1] = v1
                    h[kk + 1, kk - 1] = 0
                    if kk < i - 1:
                        h[kk + 2, kk - 1] = 0
                elif m > l:
                    h[kk, kk - 1] = h[kk, kk - 1] * (1 - t1)
                var t2 = t1 * v2
                if nr == 3:
                    var t3 = t1 * v3
                    for j in range(kk, i2 + 1):
                        var sm = (
                            h[kk, j] + v2 * h[kk + 1, j] + v3 * h[kk + 2, j]
                        )
                        h[kk, j] = h[kk, j] - sm * t1
                        h[kk + 1, j] = h[kk + 1, j] - sm * t2
                        h[kk + 2, j] = h[kk + 2, j] - sm * t3
                    for j in range(i1, min(kk + 3, i) + 1):
                        var sm = (
                            h[j, kk] + v2 * h[j, kk + 1] + v3 * h[j, kk + 2]
                        )
                        h[j, kk] = h[j, kk] - sm * t1
                        h[j, kk + 1] = h[j, kk + 1] - sm * t2
                        h[j, kk + 2] = h[j, kk + 2] - sm * t3
                    if wantz:
                        for j in range(iloz, ihiz + 1):
                            var sm = (
                                z[j, kk] + v2 * z[j, kk + 1] + v3 * z[j, kk + 2]
                            )
                            z[j, kk] = z[j, kk] - sm * t1
                            z[j, kk + 1] = z[j, kk + 1] - sm * t2
                            z[j, kk + 2] = z[j, kk + 2] - sm * t3
                elif nr == 2:
                    for j in range(kk, i2 + 1):
                        var sm = h[kk, j] + v2 * h[kk + 1, j]
                        h[kk, j] = h[kk, j] - sm * t1
                        h[kk + 1, j] = h[kk + 1, j] - sm * t2
                    for j in range(i1, i + 1):
                        var sm = h[j, kk] + v2 * h[j, kk + 1]
                        h[j, kk] = h[j, kk] - sm * t1
                        h[j, kk + 1] = h[j, kk + 1] - sm * t2
                    if wantz:
                        for j in range(iloz, ihiz + 1):
                            var sm = z[j, kk] + v2 * z[j, kk + 1]
                            z[j, kk] = z[j, kk] - sm * t1
                            z[j, kk + 1] = z[j, kk + 1] - sm * t2
        if not converged:
            return i
        if l == i:
            wr[i, 1] = h[i, i]
            wi[i, 1] = 0
        elif l == i - 1:
            var r = _dlanv2(h[i - 1, i - 1], h[i - 1, i], h[i, i - 1], h[i, i])
            h[i - 1, i - 1] = r.a
            h[i - 1, i] = r.b
            h[i, i - 1] = r.c
            h[i, i] = r.d
            wr[i - 1, 1] = r.rt1r
            wi[i - 1, 1] = r.rt1i
            wr[i, 1] = r.rt2r
            wi[i, 1] = r.rt2i
            if wantt:
                if i2 > i:
                    _drot_rows(h, i - 1, i, i + 1, i2 - i, r.cs, r.sn)
                _drot_cols(h, i - 1, i, i1, i - i1 - 1, r.cs, r.sn)
            if wantz:
                _drot_cols(z, i - 1, i, iloz, nz, r.cs, r.sn)
        kdefl = 0
        i = l - 1


# ---------------------------------------------------- reordering (dtrexc)


def _dlasy2[
    dtype: DType
](
    n1: Int, n2: Int, tl: _M[dtype], tr: _M[dtype], b: _M[dtype], x: _M[dtype]
) -> Scalar[dtype]:
    """`TL X - X TR = scale B` for `n1, n2` in `{1, 2}` (`DLASY2` with
    `ISGN = -1`, no transposes), by Gaussian elimination with complete
    pivoting on the Kronecker form; a pivot below `smin` is replaced by
    `smin`, as `DLASY2` does. Returns `scale`, always `1` here."""
    var eps = _ulp[dtype]()
    var smlnum = _safmin[dtype]() / eps
    var nn = n1 * n2
    # Unknown index for X(i, j) is (j - 1) * n1 + (i - 1).
    var a = List[Scalar[dtype]](length=16, fill=0)
    var rhs = List[Scalar[dtype]](length=4, fill=0)
    var big = Scalar[dtype](0)
    for i in range(1, n1 + 1):
        for j in range(1, n1 + 1):
            big = max(big, abs(tl[i, j]))
    for i in range(1, n2 + 1):
        for j in range(1, n2 + 1):
            big = max(big, abs(tr[i, j]))
    var smin = max(eps * big, smlnum)
    for j in range(1, n2 + 1):
        for i in range(1, n1 + 1):
            var row = (j - 1) * n1 + (i - 1)
            rhs[row] = b[i, j]
            for k in range(1, n1 + 1):
                a[row * 4 + (j - 1) * n1 + (k - 1)] += tl[i, k]
            for k in range(1, n2 + 1):
                a[row * 4 + (k - 1) * n1 + (i - 1)] -= tr[k, j]
    var perm = List[Int](length=4, fill=0)
    for q in range(nn):
        perm[q] = q
    for col in range(nn):
        var pr = col
        var pc = col
        var best = Scalar[dtype](-1)
        for r in range(col, nn):
            for c in range(col, nn):
                if abs(a[r * 4 + c]) > best:
                    best = abs(a[r * 4 + c])
                    pr = r
                    pc = c
        if pr != col:
            for c in range(nn):
                var t = a[col * 4 + c]
                a[col * 4 + c] = a[pr * 4 + c]
                a[pr * 4 + c] = t
            var t = rhs[col]
            rhs[col] = rhs[pr]
            rhs[pr] = t
        if pc != col:
            for r in range(nn):
                var t = a[r * 4 + col]
                a[r * 4 + col] = a[r * 4 + pc]
                a[r * 4 + pc] = t
            var tp = perm[col]
            perm[col] = perm[pc]
            perm[pc] = tp
        if abs(a[col * 4 + col]) < smin:
            a[col * 4 + col] = smin
        for r in range(col + 1, nn):
            var f = a[r * 4 + col] / a[col * 4 + col]
            if f != 0:
                for c in range(col, nn):
                    a[r * 4 + c] -= f * a[col * 4 + c]
                rhs[r] -= f * rhs[col]
    var sol = List[Scalar[dtype]](length=4, fill=0)
    var r = nn - 1
    while r >= 0:
        var s = rhs[r]
        for c in range(r + 1, nn):
            s -= a[r * 4 + c] * sol[c]
        sol[r] = s / a[r * 4 + r]
        r -= 1
    for q in range(nn):
        var unknown = perm[q]
        x[unknown % n1 + 1, unknown // n1 + 1] = sol[q]
    return Scalar[dtype](1)


def _house3_left[
    dtype: DType
](
    a: _M[dtype],
    ncols: Int,
    v1: Scalar[dtype],
    v2: Scalar[dtype],
    v3: Scalar[dtype],
    tau: Scalar[dtype],
):
    """`DLARFX('L', 3, ncols, v, tau, A)`: `A(1:3, 1:ncols) := (I - tau v
    v^T) A`."""
    if tau == 0:
        return
    for j in range(1, ncols + 1):
        var s = v1 * a[1, j] + v2 * a[2, j] + v3 * a[3, j]
        s = s * tau
        a[1, j] = a[1, j] - s * v1
        a[2, j] = a[2, j] - s * v2
        a[3, j] = a[3, j] - s * v3


def _house3_right[
    dtype: DType
](
    a: _M[dtype],
    nrows: Int,
    v1: Scalar[dtype],
    v2: Scalar[dtype],
    v3: Scalar[dtype],
    tau: Scalar[dtype],
):
    """`DLARFX('R', nrows, 3, v, tau, A)`: `A(1:nrows, 1:3) := A (I - tau v
    v^T)`."""
    if tau == 0:
        return
    for i in range(1, nrows + 1):
        var s = a[i, 1] * v1 + a[i, 2] * v2 + a[i, 3] * v3
        s = s * tau
        a[i, 1] = a[i, 1] - s * v1
        a[i, 2] = a[i, 2] - s * v2
        a[i, 3] = a[i, 3] - s * v3


def _dlarfg3[
    dtype: DType
](alpha: Scalar[dtype], x2: Scalar[dtype], x3: Scalar[dtype]) -> Tuple[
    Scalar[dtype], Scalar[dtype], Scalar[dtype], Scalar[dtype]
]:
    """`DLARFG(3, alpha, x, 1, tau)`: `(beta, v2, v3, tau)`."""
    var xnorm = _dlapy2(x2, x3)
    if xnorm == 0:
        return (alpha, x2, x3, Scalar[dtype](0))
    var beta = -_sign(_dlapy2(alpha, xnorm), alpha)
    var tau = (beta - alpha) / beta
    var inv = 1 / (alpha - beta)
    return (beta, x2 * inv, x3 * inv, tau)


def _dlaexc[
    dtype: DType
](n: Int, t: _M[dtype], q: _M[dtype], j1: Int, n1: Int, n2: Int) -> Int:
    """`DLAEXC(WANTQ=.TRUE.)`: swap the adjacent diagonal blocks of order
    `n1` and `n2` at `j1`. Returns `INFO` (`1`: the swap was rejected as
    too ill-conditioned; `T` is unchanged)."""
    if n == 0 or n1 == 0 or n2 == 0:
        return 0
    if j1 + n1 > n:
        return 0
    var j2 = j1 + 1
    var j3 = j1 + 2
    var j4 = j1 + 3
    if n1 == 1 and n2 == 1:
        var t11 = t[j1, j1]
        var t22 = t[j2, j2]
        var g = _dlartg(t[j1, j2], t22 - t11)
        if j3 <= n:
            _drot_rows(t, j1, j2, j3, n - j1 - 1, g[0], g[1])
        _drot_cols(t, j1, j2, 1, j1 - 1, g[0], g[1])
        t[j1, j1] = t22
        t[j2, j2] = t11
        _drot_cols(q, j1, j2, 1, n, g[0], g[1])
        return 0
    var nd = n1 + n2
    var dbuf = List[Scalar[dtype]](length=16, fill=0)
    var d = _view(dbuf, 4)
    for i in range(1, nd + 1):
        for j in range(1, nd + 1):
            d[i, j] = t[j1 + i - 1, j1 + j - 1]
    var dnorm = Scalar[dtype](0)
    for i in range(1, nd + 1):
        for j in range(1, nd + 1):
            dnorm = max(dnorm, abs(d[i, j]))
    var eps = _ulp[dtype]()
    var smlnum = _safmin[dtype]() / eps
    var thresh = max(10 * eps * dnorm, smlnum)
    var xbuf = List[Scalar[dtype]](length=4, fill=0)
    var x = _view(xbuf, 2)
    var scale = _dlasy2(n1, n2, d, d.sub(n1 + 1, n1 + 1), d.sub(1, n1 + 1), x)
    var k = n1 + n1 + n2 - 3
    if k == 1:
        # n1 = 1, n2 = 2.
        var hv = _dlarfg3(x[1, 2], scale, x[1, 1])
        # DLARFG(3, U(3), U, 1, TAU) with U = (scale, x11, x12): the
        # reflector's pivot is the third entry.
        var u1 = hv[1]
        var u2 = hv[2]
        var tau = hv[3]
        var t11 = t[j1, j1]
        _house3_left(d, 3, u1, u2, Scalar[dtype](1), tau)
        _house3_right(d, 3, u1, u2, Scalar[dtype](1), tau)
        if max(abs(d[3, 1]), max(abs(d[3, 2]), abs(d[3, 3] - t11))) > thresh:
            _keep(dbuf)
            _keep(xbuf)
            return 1
        _house3_left(t.sub(j1, j1), n - j1 + 1, u1, u2, Scalar[dtype](1), tau)
        _house3_right(t.sub(1, j1), j2, u1, u2, Scalar[dtype](1), tau)
        t[j3, j1] = 0
        t[j3, j2] = 0
        t[j3, j3] = t11
        _house3_right(q.sub(1, j1), n, u1, u2, Scalar[dtype](1), tau)
    elif k == 2:
        # n1 = 2, n2 = 1.
        var hv = _dlarfg3(-x[1, 1], -x[2, 1], scale)
        var u2 = hv[1]
        var u3 = hv[2]
        var tau = hv[3]
        var t33 = t[j3, j3]
        _house3_left(d, 3, Scalar[dtype](1), u2, u3, tau)
        _house3_right(d, 3, Scalar[dtype](1), u2, u3, tau)
        if max(abs(d[2, 1]), max(abs(d[3, 1]), abs(d[1, 1] - t33))) > thresh:
            _keep(dbuf)
            _keep(xbuf)
            return 1
        _house3_right(t.sub(1, j1), j3, Scalar[dtype](1), u2, u3, tau)
        _house3_left(t.sub(j1, j2), n - j1, Scalar[dtype](1), u2, u3, tau)
        t[j1, j1] = t33
        t[j2, j1] = 0
        t[j3, j1] = 0
        _house3_right(q.sub(1, j1), n, Scalar[dtype](1), u2, u3, tau)
    else:
        # n1 = 2, n2 = 2.
        var h1 = _dlarfg3(-x[1, 1], -x[2, 1], scale)
        var a2 = h1[1]
        var a3 = h1[2]
        var tau1 = h1[3]
        var temp = -tau1 * (x[1, 2] + a2 * x[2, 2])
        var h2 = _dlarfg3(-temp * a2 - x[2, 2], -temp * a3, scale)
        var b2 = h2[1]
        var b3 = h2[2]
        var tau2 = h2[3]
        _house3_left(d, 4, Scalar[dtype](1), a2, a3, tau1)
        _house3_right(d, 4, Scalar[dtype](1), a2, a3, tau1)
        _house3_left(d.sub(2, 1), 4, Scalar[dtype](1), b2, b3, tau2)
        _house3_right(d.sub(1, 2), 4, Scalar[dtype](1), b2, b3, tau2)
        if (
            max(
                max(abs(d[3, 1]), abs(d[3, 2])), max(abs(d[4, 1]), abs(d[4, 2]))
            )
            > thresh
        ):
            _keep(dbuf)
            _keep(xbuf)
            return 1
        _house3_left(t.sub(j1, j1), n - j1 + 1, Scalar[dtype](1), a2, a3, tau1)
        _house3_right(t.sub(1, j1), j4, Scalar[dtype](1), a2, a3, tau1)
        _house3_left(t.sub(j2, j1), n - j1 + 1, Scalar[dtype](1), b2, b3, tau2)
        _house3_right(t.sub(1, j2), j4, Scalar[dtype](1), b2, b3, tau2)
        t[j3, j1] = 0
        t[j3, j2] = 0
        t[j4, j1] = 0
        t[j4, j2] = 0
        _house3_right(q.sub(1, j1), n, Scalar[dtype](1), a2, a3, tau1)
        _house3_right(q.sub(1, j2), n, Scalar[dtype](1), b2, b3, tau2)
    if n2 == 2:
        var r = _dlanv2(t[j1, j1], t[j1, j2], t[j2, j1], t[j2, j2])
        t[j1, j1] = r.a
        t[j1, j2] = r.b
        t[j2, j1] = r.c
        t[j2, j2] = r.d
        _drot_rows(t, j1, j2, j1 + 2, n - j1 - 1, r.cs, r.sn)
        _drot_cols(t, j1, j2, 1, j1 - 1, r.cs, r.sn)
        _drot_cols(q, j1, j2, 1, n, r.cs, r.sn)
    if n1 == 2:
        var j3b = j1 + n2
        var j4b = j3b + 1
        var r = _dlanv2(t[j3b, j3b], t[j3b, j4b], t[j4b, j3b], t[j4b, j4b])
        t[j3b, j3b] = r.a
        t[j3b, j4b] = r.b
        t[j4b, j3b] = r.c
        t[j4b, j4b] = r.d
        if j3b + 2 <= n:
            _drot_rows(t, j3b, j4b, j3b + 2, n - j3b - 1, r.cs, r.sn)
        _drot_cols(t, j3b, j4b, 1, j3b - 1, r.cs, r.sn)
        _drot_cols(q, j3b, j4b, 1, n, r.cs, r.sn)
    _keep(dbuf)
    _keep(xbuf)
    return 0


def _dtrexc[
    dtype: DType
](n: Int, t: _M[dtype], q: _M[dtype], ifst_in: Int, ilst_in: Int) -> Tuple[
    Int, Int
]:
    """`DTREXC('V')`: move the block at `ifst` to `ilst`. Returns `(ilst,
    info)`."""
    if n <= 1:
        return (ilst_in, 0)
    var ifst = ifst_in
    var ilst = ilst_in
    if ifst > 1:
        if t[ifst, ifst - 1] != 0:
            ifst -= 1
    var nbf = 1
    if ifst < n:
        if t[ifst + 1, ifst] != 0:
            nbf = 2
    if ilst > 1:
        if t[ilst, ilst - 1] != 0:
            ilst -= 1
    var nbl = 1
    if ilst < n:
        if t[ilst + 1, ilst] != 0:
            nbl = 2
    if ifst == ilst:
        return (ilst, 0)
    var here: Int
    if ifst < ilst:
        if nbf == 2 and nbl == 1:
            ilst -= 1
        if nbf == 1 and nbl == 2:
            ilst += 1
        here = ifst
        while True:
            if nbf == 1 or nbf == 2:
                var nbnext = 1
                if here + nbf + 1 <= n:
                    if t[here + nbf + 1, here + nbf] != 0:
                        nbnext = 2
                if _dlaexc(n, t, q, here, nbf, nbnext) != 0:
                    return (here, 1)
                here += nbnext
                if nbf == 2:
                    if t[here + 1, here] == 0:
                        nbf = 3
            else:
                var nbnext = 1
                if here + 3 <= n:
                    if t[here + 3, here + 2] != 0:
                        nbnext = 2
                if _dlaexc(n, t, q, here + 1, 1, nbnext) != 0:
                    return (here, 1)
                if nbnext == 1:
                    _ = _dlaexc(n, t, q, here, 1, nbnext)
                    here += 1
                else:
                    if t[here + 2, here + 1] == 0:
                        nbnext = 1
                    if nbnext == 2:
                        if _dlaexc(n, t, q, here, 1, nbnext) != 0:
                            return (here, 1)
                        here += 2
                    else:
                        _ = _dlaexc(n, t, q, here, 1, 1)
                        _ = _dlaexc(n, t, q, here + 1, 1, 1)
                        here += 2
            if here >= ilst:
                break
    else:
        here = ifst
        while True:
            if nbf == 1 or nbf == 2:
                var nbnext = 1
                if here >= 3:
                    if t[here - 1, here - 2] != 0:
                        nbnext = 2
                if _dlaexc(n, t, q, here - nbnext, nbnext, nbf) != 0:
                    return (here, 1)
                here -= nbnext
                if nbf == 2:
                    if t[here + 1, here] == 0:
                        nbf = 3
            else:
                var nbnext = 1
                if here >= 3:
                    if t[here - 1, here - 2] != 0:
                        nbnext = 2
                if _dlaexc(n, t, q, here - nbnext, nbnext, 1) != 0:
                    return (here, 1)
                if nbnext == 1:
                    _ = _dlaexc(n, t, q, here, nbnext, 1)
                    here -= 1
                else:
                    if t[here, here - 1] == 0:
                        nbnext = 1
                    if nbnext == 2:
                        if _dlaexc(n, t, q, here - 1, 2, 1) != 0:
                            return (here, 1)
                        here -= 2
                    else:
                        _ = _dlaexc(n, t, q, here, 1, 1)
                        _ = _dlaexc(n, t, q, here - 1, 1, 1)
                        here -= 2
            if here <= ilst:
                break
    return (here, 0)


# --------------------------------------------------------- dlaqr1, gemm


@always_inline
def _tdiv(a: Int, b: Int) -> Int:
    """Fortran integer division, truncating toward zero."""
    var q = a // b
    if (a % b != 0) and ((a < 0) != (b < 0)):
        q += 1
    return q


def _dlaqr1[
    dtype: DType
](
    n: Int,
    h: _M[dtype],
    sr1: Scalar[dtype],
    si1: Scalar[dtype],
    sr2: Scalar[dtype],
    si2: Scalar[dtype],
) -> Tuple[Scalar[dtype], Scalar[dtype], Scalar[dtype]]:
    """`DLAQR1`: a multiple of the first column of `(H - s1)(H - s2)`."""
    var zero = Scalar[dtype](0)
    if n == 2:
        var s = abs(h[1, 1] - sr2) + abs(si2) + abs(h[2, 1])
        if s == 0:
            return (zero, zero, zero)
        var h21s = h[2, 1] / s
        return (
            h21s * h[1, 2]
            + (h[1, 1] - sr1) * ((h[1, 1] - sr2) / s)
            - si1 * (si2 / s),
            h21s * (h[1, 1] + h[2, 2] - sr1 - sr2),
            zero,
        )
    var s = abs(h[1, 1] - sr2) + abs(si2) + abs(h[2, 1]) + abs(h[3, 1])
    if s == 0:
        return (zero, zero, zero)
    var h21s = h[2, 1] / s
    var h31s = h[3, 1] / s
    return (
        (h[1, 1] - sr1) * ((h[1, 1] - sr2) / s)
        - si1 * (si2 / s)
        + h[1, 2] * h21s
        + h[1, 3] * h31s,
        h21s * (h[1, 1] + h[2, 2] - sr1 - sr2) + h[2, 3] * h31s,
        h31s * (h[1, 1] + h[3, 3] - sr1 - sr2) + h21s * h[3, 2],
    )


@always_inline
def _axpy_row[
    dtype: DType
](
    dst: Pointer[Scalar[dtype], MutUntrackedOrigin],
    d0: Int,
    src: Pointer[Scalar[dtype], MutUntrackedOrigin],
    s0: Int,
    c: Scalar[dtype],
    count: Int,
):
    """`dst[d0 : d0 + count] += c * src[s0 : s0 + count]`, vectorized."""
    comptime w = 2 * simd_width_of[dtype]()
    var j = 0
    var cv = SIMD[dtype, w](c)
    while j + w <= count:
        var x = dst.unsafe_load[width=w](d0 + j) + cv * src.unsafe_load[
            width=w
        ](s0 + j)
        dst.unsafe_store(d0 + j, x)
        j += w
    while j < count:
        dst[unsafe_offset=d0 + j] = (
            dst[unsafe_offset=d0 + j] + c * src[unsafe_offset=s0 + j]
        )
        j += 1


def _left_multiply_t[
    dtype: DType
](u: _M[dtype], nu: Int, a: _M[dtype], ncols: Int):
    """`A(1:nu, 1:ncols) := U(1:nu, 1:nu)^T A`: each output row a sum of
    `A`'s rows, so the inner loop runs along contiguous rows; columns go in
    chunks that keep the output block in cache."""
    if nu <= 0 or ncols <= 0:
        return
    comptime chunk = 256
    var out = List[Scalar[dtype]](length=nu * chunk, fill=0)
    var op = out.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin]()
    var c0 = 1
    while c0 <= ncols:
        var width = min(chunk, ncols - c0 + 1)
        for e in range(nu * width):
            out[e] = 0
        for k in range(1, nu + 1):
            var arow = a.off + (k - 1) * a.ld + (c0 - 1)
            for i in range(1, nu + 1):
                var uki = u[k, i]
                if uki != 0:
                    _axpy_row(op, (i - 1) * width, a.p, arow, uki, width)
        for i in range(1, nu + 1):
            var dst = a.off + (i - 1) * a.ld + (c0 - 1)
            for j in range(width):
                a.p[unsafe_offset=dst + j] = out[(i - 1) * width + j]
        c0 += chunk
    _keep(out)


def _right_multiply[
    dtype: DType
](a: _M[dtype], nrows: Int, u: _M[dtype], nu: Int):
    """`A(1:nrows, 1:nu) := A U(1:nu, 1:nu)`: each output row a sum of `U`'s
    rows, vectorized along them."""
    if nu <= 0 or nrows <= 0:
        return
    var row = List[Scalar[dtype]](length=nu, fill=0)
    var rp = row.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin]()
    for i in range(1, nrows + 1):
        for e in range(nu):
            row[e] = 0
        var base = a.off + (i - 1) * a.ld
        for k in range(1, nu + 1):
            var aik = a.p[unsafe_offset=base + k - 1]
            if aik != 0:
                _axpy_row(rp, 0, u.p, u.off + (k - 1) * u.ld, aik, nu)
        for e in range(nu):
            a.p[unsafe_offset=base + e] = row[e]
    _keep(row)


# ------------------------------------------------------------- dlaqr5


@always_inline
def _small_subdiagonal[
    dtype: DType
](
    h: _M[dtype],
    k: Int,
    ktop: Int,
    kbot: Int,
    smlnum: Scalar[dtype],
    ulp: Scalar[dtype],
):
    """`dlaqr5`'s Ahues-Tisseur test at `H(k+1, k)`, zeroing it when it
    passes."""
    if h[k + 1, k] == 0:
        return
    var tst1 = abs(h[k, k]) + abs(h[k + 1, k + 1])
    if tst1 == 0:
        if k >= ktop + 1:
            tst1 += abs(h[k, k - 1])
        if k >= ktop + 2:
            tst1 += abs(h[k, k - 2])
        if k >= ktop + 3:
            tst1 += abs(h[k, k - 3])
        if k <= kbot - 2:
            tst1 += abs(h[k + 2, k + 1])
        if k <= kbot - 3:
            tst1 += abs(h[k + 3, k + 1])
        if k <= kbot - 4:
            tst1 += abs(h[k + 4, k + 1])
    if abs(h[k + 1, k]) <= max(smlnum, ulp * tst1):
        var h12 = max(abs(h[k + 1, k]), abs(h[k, k + 1]))
        var h21 = min(abs(h[k + 1, k]), abs(h[k, k + 1]))
        var h11 = max(abs(h[k + 1, k + 1]), abs(h[k, k] - h[k + 1, k + 1]))
        var h22 = min(abs(h[k + 1, k + 1]), abs(h[k, k] - h[k + 1, k + 1]))
        var scl = h11 + h12
        var tst2 = h22 * (h11 / scl)
        if tst2 == 0 or h21 * (h12 / scl) <= max(smlnum, ulp * tst2):
            h[k + 1, k] = 0


def _dlaqr5[
    dtype: DType
](
    wantt: Bool,
    wantz: Bool,
    n: Int,
    ktop: Int,
    kbot: Int,
    nshfts: Int,
    sr: _M[dtype],
    si: _M[dtype],
    h: _M[dtype],
    iloz: Int,
    ihiz: Int,
    z: _M[dtype],
):
    """`DLAQR5` with `KACC22 = 1`: one multishift sweep of `nshfts`
    shifts `sr(i, 1) + i si(i, 1)` over `H(ktop:kbot, ktop:kbot)`,
    accumulated per window and applied off the window by products."""
    if nshfts < 2:
        return
    if ktop >= kbot:
        return
    var i = 1
    while i <= nshfts - 2:
        if si[i, 1] != -si[i + 1, 1]:
            var swap = sr[i, 1]
            sr[i, 1] = sr[i + 1, 1]
            sr[i + 1, 1] = sr[i + 2, 1]
            sr[i + 2, 1] = swap
            swap = si[i, 1]
            si[i, 1] = si[i + 1, 1]
            si[i + 1, 1] = si[i + 2, 1]
            si[i + 2, 1] = swap
        i += 2
    var ns = nshfts - nshfts % 2
    var safmin = _safmin[dtype]()
    var ulp = _ulp[dtype]()
    var smlnum = safmin * (Scalar[dtype](n) / ulp)
    if ktop + 2 <= kbot:
        h[ktop + 2, ktop] = 0
    var nbmps = ns // 2
    var kdu = 4 * nbmps
    var vbuf = List[Scalar[dtype]](length=3 * nbmps, fill=0)
    var v = _view(vbuf, nbmps)
    var ubuf = List[Scalar[dtype]](length=kdu * kdu, fill=0)
    var u = _view(ubuf, kdu)
    var incol = ktop - 2 * nbmps + 1
    while incol <= kbot - 2:
        var jtop = max(ktop, incol)
        var ndcol = incol + kdu
        for e in range(kdu * kdu):
            ubuf[e] = 0
        for e in range(kdu):
            ubuf[e * kdu + e] = 1
        for krcol in range(incol, min(incol + 2 * nbmps - 1, kbot - 2) + 1):
            var mtop = max(1, _tdiv(ktop - krcol, 2) + 1)
            var mbot = min(nbmps, _tdiv(kbot - krcol - 1, 2))
            var m22 = mbot + 1
            var bmp22 = (mbot < nbmps) and (krcol + 2 * (m22 - 1)) == (kbot - 2)
            if bmp22:
                var k = krcol + 2 * (m22 - 1)
                if k == ktop - 1:
                    var col = _dlaqr1(
                        2,
                        h.sub(k + 1, k + 1),
                        sr[2 * m22 - 1, 1],
                        si[2 * m22 - 1, 1],
                        sr[2 * m22, 1],
                        si[2 * m22, 1],
                    )
                    var beta = col[0]
                    var x2 = col[1]
                    # DLARFG(2, beta, v(2), 1, v(1))
                    var tau = Scalar[dtype](0)
                    if x2 != 0:
                        var bnew = -_sign(_dlapy2(beta, abs(x2)), beta)
                        tau = (bnew - beta) / bnew
                        x2 = x2 / (beta - bnew)
                    v[2, m22] = x2
                    v[1, m22] = tau
                else:
                    var beta = h[k + 1, k]
                    var x2 = h[k + 2, k]
                    var tau = Scalar[dtype](0)
                    if x2 != 0:
                        var bnew = -_sign(_dlapy2(beta, abs(x2)), beta)
                        tau = (bnew - beta) / bnew
                        x2 = x2 / (beta - bnew)
                        beta = bnew
                    v[2, m22] = x2
                    v[1, m22] = tau
                    h[k + 1, k] = beta
                    h[k + 2, k] = 0
                var t1 = v[1, m22]
                var t2 = t1 * v[2, m22]
                for j in range(jtop, min(kbot, k + 3) + 1):
                    var refsum = h[j, k + 1] + v[2, m22] * h[j, k + 2]
                    h[j, k + 1] = h[j, k + 1] - refsum * t1
                    h[j, k + 2] = h[j, k + 2] - refsum * t2
                var jbot = min(ndcol, kbot)
                for j in range(k + 1, jbot + 1):
                    var refsum = h[k + 1, j] + v[2, m22] * h[k + 2, j]
                    h[k + 1, j] = h[k + 1, j] - refsum * t1
                    h[k + 2, j] = h[k + 2, j] - refsum * t2
                if k >= ktop:
                    _small_subdiagonal(h, k, ktop, kbot, smlnum, ulp)
                var kms = k - incol
                for j in range(max(1, ktop - incol), kdu + 1):
                    var refsum = u[j, kms + 1] + v[2, m22] * u[j, kms + 2]
                    u[j, kms + 1] = u[j, kms + 1] - refsum * t1
                    u[j, kms + 2] = u[j, kms + 2] - refsum * t2
            var m = mbot
            while m >= mtop:
                var k = krcol + 2 * (m - 1)
                if k == ktop - 1:
                    var col = _dlaqr1(
                        3,
                        h.sub(ktop, ktop),
                        sr[2 * m - 1, 1],
                        si[2 * m - 1, 1],
                        sr[2 * m, 1],
                        si[2 * m, 1],
                    )
                    var hv = _dlarfg3(col[0], col[1], col[2])
                    v[1, m] = hv[3]
                    v[2, m] = hv[1]
                    v[3, m] = hv[2]
                else:
                    var t1 = v[1, m]
                    var t2 = t1 * v[2, m]
                    var t3 = t1 * v[3, m]
                    var refsum = v[3, m] * h[k + 3, k + 2]
                    h[k + 3, k] = -refsum * t1
                    h[k + 3, k + 1] = -refsum * t2
                    h[k + 3, k + 2] = h[k + 3, k + 2] - refsum * t3
                    var hv = _dlarfg3(h[k + 1, k], h[k + 2, k], h[k + 3, k])
                    var beta = hv[0]
                    v[1, m] = hv[3]
                    v[2, m] = hv[1]
                    v[3, m] = hv[2]
                    if (
                        h[k + 3, k] != 0
                        or h[k + 3, k + 1] != 0
                        or h[k + 3, k + 2] == 0
                    ):
                        h[k + 1, k] = beta
                        h[k + 2, k] = 0
                        h[k + 3, k] = 0
                    else:
                        var col = _dlaqr1(
                            3,
                            h.sub(k + 1, k + 1),
                            sr[2 * m - 1, 1],
                            si[2 * m - 1, 1],
                            sr[2 * m, 1],
                            si[2 * m, 1],
                        )
                        var vt = _dlarfg3(col[0], col[1], col[2])
                        var vt1 = vt[3]
                        var vt2 = vt[1]
                        var vt3 = vt[2]
                        t1 = vt1
                        t2 = t1 * vt2
                        t3 = t1 * vt3
                        refsum = h[k + 1, k] + vt2 * h[k + 2, k]
                        if abs(h[k + 2, k] - refsum * t2) + abs(
                            refsum * t3
                        ) > ulp * (
                            abs(h[k, k])
                            + abs(h[k + 1, k + 1])
                            + abs(h[k + 2, k + 2])
                        ):
                            h[k + 1, k] = beta
                            h[k + 2, k] = 0
                            h[k + 3, k] = 0
                        else:
                            h[k + 1, k] = h[k + 1, k] - refsum * t1
                            h[k + 2, k] = 0
                            h[k + 3, k] = 0
                            v[1, m] = vt1
                            v[2, m] = vt2
                            v[3, m] = vt3
                var t1 = v[1, m]
                var t2 = t1 * v[2, m]
                var t3 = t1 * v[3, m]
                for j in range(jtop, min(kbot, k + 3) + 1):
                    var refsum = (
                        h[j, k + 1]
                        + v[2, m] * h[j, k + 2]
                        + v[3, m] * h[j, k + 3]
                    )
                    h[j, k + 1] = h[j, k + 1] - refsum * t1
                    h[j, k + 2] = h[j, k + 2] - refsum * t2
                    h[j, k + 3] = h[j, k + 3] - refsum * t3
                var refsum = (
                    h[k + 1, k + 1]
                    + v[2, m] * h[k + 2, k + 1]
                    + v[3, m] * h[k + 3, k + 1]
                )
                h[k + 1, k + 1] = h[k + 1, k + 1] - refsum * t1
                h[k + 2, k + 1] = h[k + 2, k + 1] - refsum * t2
                h[k + 3, k + 1] = h[k + 3, k + 1] - refsum * t3
                if k >= ktop:
                    _small_subdiagonal(h, k, ktop, kbot, smlnum, ulp)
                m -= 1
            var jbot = min(ndcol, kbot)
            m = mbot
            while m >= mtop:
                var k = krcol + 2 * (m - 1)
                var t1 = v[1, m]
                var t2 = t1 * v[2, m]
                var t3 = t1 * v[3, m]
                for j in range(max(ktop, krcol + 2 * m), jbot + 1):
                    var refsum = (
                        h[k + 1, j]
                        + v[2, m] * h[k + 2, j]
                        + v[3, m] * h[k + 3, j]
                    )
                    h[k + 1, j] = h[k + 1, j] - refsum * t1
                    h[k + 2, j] = h[k + 2, j] - refsum * t2
                    h[k + 3, j] = h[k + 3, j] - refsum * t3
                m -= 1
            m = mbot
            while m >= mtop:
                var k = krcol + 2 * (m - 1)
                var kms = k - incol
                var i2 = max(1, ktop - incol)
                i2 = max(i2, kms - (krcol - incol) + 1)
                var i4 = min(kdu, krcol + 2 * (mbot - 1) - incol + 5)
                var t1 = v[1, m]
                var t2 = t1 * v[2, m]
                var t3 = t1 * v[3, m]
                for j in range(i2, i4 + 1):
                    var refsum = (
                        u[j, kms + 1]
                        + v[2, m] * u[j, kms + 2]
                        + v[3, m] * u[j, kms + 3]
                    )
                    u[j, kms + 1] = u[j, kms + 1] - refsum * t1
                    u[j, kms + 2] = u[j, kms + 2] - refsum * t2
                    u[j, kms + 3] = u[j, kms + 3] - refsum * t3
                m -= 1
        # The window's accumulated transformation, off the window.
        var jtop2 = 1 if wantt else ktop
        var jbot2 = n if wantt else kbot
        var k1 = max(1, ktop - incol)
        var nu = (kdu - max(0, ndcol - kbot)) - k1 + 1
        var jcol = min(ndcol, kbot) + 1
        if jcol <= jbot2:
            _left_multiply_t(
                u.sub(k1, k1), nu, h.sub(incol + k1, jcol), jbot2 - jcol + 1
            )
        var jrow_end = max(ktop, incol) - 1
        if jtop2 <= jrow_end:
            _right_multiply(
                h.sub(jtop2, incol + k1),
                jrow_end - jtop2 + 1,
                u.sub(k1, k1),
                nu,
            )
        if wantz:
            _right_multiply(
                z.sub(iloz, incol + k1), ihiz - iloz + 1, u.sub(k1, k1), nu
            )
        incol += 2 * nbmps
    _keep(vbuf)
    _keep(ubuf)


# ------------------------------------------------------------- dlaqr3


def _apply_reflector_left[
    dtype: DType
](a: _M[dtype], m: Int, ncols: Int, w: _M[dtype], tau: Scalar[dtype]):
    """`A(1:m, 1:ncols) := (I - tau w w^T) A` for `w(1:m, 1)`, `w(1) = 1`."""
    if tau == 0:
        return
    for j in range(1, ncols + 1):
        var s = a[1, j]
        for i in range(2, m + 1):
            s += w[i, 1] * a[i, j]
        s = s * tau
        a[1, j] = a[1, j] - s
        for i in range(2, m + 1):
            a[i, j] = a[i, j] - s * w[i, 1]


def _apply_reflector_right[
    dtype: DType
](a: _M[dtype], nrows: Int, m: Int, w: _M[dtype], tau: Scalar[dtype]):
    """`A(1:nrows, 1:m) := A (I - tau w w^T)` for `w(1:m, 1)`, `w(1) = 1`."""
    if tau == 0:
        return
    for i in range(1, nrows + 1):
        var s = a[i, 1]
        for j in range(2, m + 1):
            s += a[i, j] * w[j, 1]
        s = s * tau
        a[i, 1] = a[i, 1] - s
        for j in range(2, m + 1):
            a[i, j] = a[i, j] - s * w[j, 1]


struct _AED(ImplicitlyCopyable):
    var ns: Int
    var nd: Int

    def __init__(out self, ns: Int, nd: Int):
        self.ns = ns
        self.nd = nd


def _dlaqr3[
    dtype: DType
](
    wantt: Bool,
    wantz: Bool,
    n: Int,
    ktop: Int,
    kbot: Int,
    nw: Int,
    h: _M[dtype],
    iloz: Int,
    ihiz: Int,
    z: _M[dtype],
    sr: _M[dtype],
    si: _M[dtype],
) -> _AED:
    """`DLAQR3`: aggressive early deflation on the trailing `nw` window of
    `H(ktop:kbot, ktop:kbot)`. Returns `(ns, nd)`: the undeflated
    eigenvalues left as shifts in `sr/si(kbot-nd-ns+1 : kbot-nd)` and the
    number deflated."""
    if ktop > kbot or nw < 1:
        return _AED(0, 0)
    var safmin = _safmin[dtype]()
    var ulp = _ulp[dtype]()
    var smlnum = safmin * (Scalar[dtype](n) / ulp)
    var jw = min(nw, kbot - ktop + 1)
    var kwtop = kbot - jw + 1
    var s: Scalar[dtype]
    if kwtop == ktop:
        s = 0
    else:
        s = h[kwtop, kwtop - 1]
    if kbot == kwtop:
        sr[kwtop, 1] = h[kwtop, kwtop]
        si[kwtop, 1] = 0
        if abs(s) <= max(smlnum, ulp * abs(h[kwtop, kwtop])):
            if kwtop > ktop:
                h[kwtop, kwtop - 1] = 0
            return _AED(0, 1)
        return _AED(1, 0)
    var tbuf = List[Scalar[dtype]](length=jw * jw, fill=0)
    var t = _view(tbuf, jw)
    for i in range(1, jw + 1):
        for j in range(max(1, i - 1), jw + 1):
            t[i, j] = h[kwtop + i - 1, kwtop + j - 1]
    var vbuf = List[Scalar[dtype]](length=jw * jw, fill=0)
    var v = _view(vbuf, jw)
    for i in range(1, jw + 1):
        v[i, i] = 1
    var infqr = _dlahqr(
        True, True, jw, 1, jw, t, sr.sub(kwtop, 1), si.sub(kwtop, 1), 1, jw, v
    )
    for j in range(1, jw - 2):
        t[j + 2, j] = 0
        t[j + 3, j] = 0
    if jw > 2:
        t[jw, jw - 2] = 0
    var ns = jw
    var ilst = infqr + 1
    while ilst <= ns:
        var bulge: Bool
        if ns == 1:
            bulge = False
        else:
            bulge = t[ns, ns - 1] != 0
        if not bulge:
            var foo = abs(t[ns, ns])
            if foo == 0:
                foo = abs(s)
            if abs(s * v[1, ns]) <= max(smlnum, ulp * foo):
                ns -= 1
            else:
                var r = _dtrexc(jw, t, v, ns, ilst)
                ilst = r[0] + 1
        else:
            var foo = abs(t[ns, ns]) + _sqrt(abs(t[ns, ns - 1])) * _sqrt(
                abs(t[ns - 1, ns])
            )
            if foo == 0:
                foo = abs(s)
            if max(abs(s * v[1, ns]), abs(s * v[1, ns - 1])) <= max(
                smlnum, ulp * foo
            ):
                ns -= 2
            else:
                var r = _dtrexc(jw, t, v, ns, ilst)
                ilst = r[0] + 2
    if ns == 0:
        s = 0
    if ns < jw:
        # Sort the diagonal blocks of the undeflatable part by decreasing
        # magnitude, as `dlaqr3`'s bubble sort does.
        var sorted = False
        var i = ns + 1
        while not sorted:
            sorted = True
            var kend = i - 1
            i = infqr + 1
            var k: Int
            if i == ns:
                k = i + 1
            elif t[i + 1, i] == 0:
                k = i + 1
            else:
                k = i + 2
            while k <= kend:
                var evi: Scalar[dtype]
                if k == i + 1:
                    evi = abs(t[i, i])
                else:
                    evi = abs(t[i, i]) + _sqrt(abs(t[i + 1, i])) * _sqrt(
                        abs(t[i, i + 1])
                    )
                var evk: Scalar[dtype]
                if k == kend:
                    evk = abs(t[k, k])
                elif t[k + 1, k] == 0:
                    evk = abs(t[k, k])
                else:
                    evk = abs(t[k, k]) + _sqrt(abs(t[k + 1, k])) * _sqrt(
                        abs(t[k, k + 1])
                    )
                if evi >= evk:
                    i = k
                else:
                    sorted = False
                    var r = _dtrexc(jw, t, v, i, k)
                    if r[1] == 0:
                        i = r[0]
                    else:
                        i = k
                if i == kend:
                    k = i + 1
                elif t[i + 1, i] == 0:
                    k = i + 1
                else:
                    k = i + 2
    var i = jw
    while i >= infqr + 1:
        if i == infqr + 1:
            sr[kwtop + i - 1, 1] = t[i, i]
            si[kwtop + i - 1, 1] = 0
            i -= 1
        elif t[i, i - 1] == 0:
            sr[kwtop + i - 1, 1] = t[i, i]
            si[kwtop + i - 1, 1] = 0
            i -= 1
        else:
            var r = _dlanv2(t[i - 1, i - 1], t[i - 1, i], t[i, i - 1], t[i, i])
            sr[kwtop + i - 2, 1] = r.rt1r
            si[kwtop + i - 2, 1] = r.rt1i
            sr[kwtop + i - 1, 1] = r.rt2r
            si[kwtop + i - 1, 1] = r.rt2i
            i -= 2
    if ns < jw or s == 0:
        if ns > 1 and s != 0:
            # The spike's reflector, then back to Hessenberg form.
            var wbuf = List[Scalar[dtype]](length=jw, fill=0)
            var w = _view(wbuf, 1)
            for j in range(1, ns + 1):
                w[j, 1] = v[1, j]
            var rg = _dlarfg(ns, w[1, 1], w.sub(2, 1), False)
            var tau = rg[1]
            w[1, 1] = 1
            for i2 in range(3, jw + 1):
                for j in range(1, min(i2 - 2, jw - 2) + 1):
                    t[i2, j] = 0
            _apply_reflector_left(t, ns, jw, w, tau)
            _apply_reflector_right(t, ns, ns, w, tau)
            _apply_reflector_right(v, jw, ns, w, tau)
            # DGEHRD(JW, 1, NS, T) with DORMHR's V := V Q folded in.
            var hbuf = List[Scalar[dtype]](length=jw, fill=0)
            var hv = _view(hbuf, 1)
            for col in range(1, ns - 1):
                var len = ns - col
                hv[1, 1] = 1
                for r2 in range(2, len + 1):
                    hv[r2, 1] = t[col + r2, col]
                var g = _dlarfg(len, t[col + 1, col], hv.sub(2, 1), False)
                t[col + 1, col] = g[0]
                for r2 in range(2, len + 1):
                    t[col + r2, col] = 0
                _apply_reflector_right(t.sub(1, col + 1), ns, len, hv, g[1])
                _apply_reflector_left(
                    t.sub(col + 1, col + 1), len, jw - col, hv, g[1]
                )
                _apply_reflector_right(v.sub(1, col + 1), jw, len, hv, g[1])
            _keep(wbuf)
            _keep(hbuf)
        if kwtop > 1:
            h[kwtop, kwtop - 1] = s * v[1, 1]
        for i2 in range(1, jw + 1):
            for j in range(max(1, i2 - 1), jw + 1):
                h[kwtop + i2 - 1, kwtop + j - 1] = t[i2, j]
        var ltop = 1 if wantt else ktop
        if ltop <= kwtop - 1:
            _right_multiply(h.sub(ltop, kwtop), kwtop - ltop, v, jw)
        if wantt and kbot < n:
            _left_multiply_t(v, jw, h.sub(kwtop, kbot + 1), n - kbot)
        if wantz:
            _right_multiply(z.sub(iloz, kwtop), ihiz - iloz + 1, v, jw)
    _keep(tbuf)
    _keep(vbuf)
    return _AED(ns - infqr, jw - ns)


# ------------------------------------------------------------- dlaqr0


def _iparmq_ns(nh: Int) -> Int:
    """`IPARMQ(ISPEC=15)`: the shift count for an active block of `nh`."""
    var ns = 2
    if nh >= 30:
        ns = 4
    if nh >= 60:
        ns = 10
    if nh >= 150:
        var lg = Int(_log(Float64(nh)) / _log(2.0) + 0.5)
        ns = max(10, nh // lg)
    if nh >= 590:
        ns = 64
    if nh >= 3000:
        ns = 128
    if nh >= 6000:
        ns = 256
    return max(2, ns - ns % 2)


def _dlaqr0[
    dtype: DType
](
    wantt: Bool,
    wantz: Bool,
    n: Int,
    ilo: Int,
    ihi: Int,
    h: _M[dtype],
    wr: _M[dtype],
    wi: _M[dtype],
    iloz: Int,
    ihiz: Int,
    z: _M[dtype],
) -> Int:
    """`DLAQR0`: the multishift QR with aggressive early deflation on
    `H(ilo:ihi, ilo:ihi)`. Returns `INFO`."""
    comptime ntiny = 15
    comptime kexnw = 5
    comptime kexsh = 6
    comptime wilk1 = 0.75
    comptime wilk2 = -0.4375
    comptime nmin = 75
    comptime nibble = 14
    if n == 0:
        return 0
    if n <= ntiny:
        return _dlahqr(wantt, wantz, n, ilo, ihi, h, wr, wi, iloz, ihiz, z)
    var nh_all = ihi - ilo + 1
    var nsr_raw = _iparmq_ns(nh_all)
    var nwr = nsr_raw if nh_all <= 500 else 3 * nsr_raw // 2
    nwr = max(2, nwr)
    nwr = min(ihi - ilo + 1, min((n - 1) // 3, nwr))
    var nsr = min(nsr_raw, min((n - 3) // 6, ihi - ilo))
    nsr = max(2, nsr - nsr % 2)
    var nwmax = (n - 1) // 3
    var nw = nwmax
    var nsmax = (n - 3) // 6
    nsmax = nsmax - nsmax % 2
    var ndfl = 1
    var ndec = -1
    var itmax = max(30, 2 * kexsh) * max(10, ihi - ilo + 1)
    var kbot = ihi
    for _it in range(1, itmax + 1):
        if kbot < ilo:
            return 0
        var k = kbot
        while k >= ilo + 1:
            if h[k, k - 1] == 0:
                break
            k -= 1
        var ktop = k
        var nh = kbot - ktop + 1
        var nwupbd = min(nh, nwmax)
        if ndfl < kexnw:
            nw = min(nwupbd, nwr)
        else:
            nw = min(nwupbd, 2 * nw)
        if nw < nwmax:
            if nw >= nh - 1:
                nw = nh
            else:
                var kwtop = kbot - nw + 1
                if abs(h[kwtop, kwtop - 1]) > abs(h[kwtop - 1, kwtop - 2]):
                    nw += 1
        if ndfl < kexnw:
            ndec = -1
        elif ndec >= 0 or nw >= nwupbd:
            ndec += 1
            if nw - ndec < 2:
                ndec = 0
            nw -= ndec
        var aed = _dlaqr3(
            wantt, wantz, n, ktop, kbot, nw, h, iloz, ihiz, z, wr, wi
        )
        var ls = aed.ns
        var ld = aed.nd
        kbot -= ld
        var ks = kbot - ls + 1
        if (ld == 0) or (
            (100 * ld <= nw * nibble) and (kbot - ktop + 1 > min(nmin, nwmax))
        ):
            var ns = min(nsmax, min(nsr, max(2, kbot - ktop)))
            ns = ns - ns % 2
            if ndfl % kexsh == 0:
                ks = kbot - ns + 1
                var ii = kbot
                while ii >= max(ks + 1, ktop + 2):
                    var ss = abs(h[ii, ii - 1]) + abs(h[ii - 1, ii - 2])
                    var aa = Scalar[dtype](wilk1) * ss + h[ii, ii]
                    var bb = ss
                    var cc = Scalar[dtype](wilk2) * ss
                    var dd = aa
                    var r = _dlanv2(aa, bb, cc, dd)
                    wr[ii - 1, 1] = r.rt1r
                    wi[ii - 1, 1] = r.rt1i
                    wr[ii, 1] = r.rt2r
                    wi[ii, 1] = r.rt2i
                    ii -= 2
                if ks == ktop:
                    wr[ks + 1, 1] = h[ks + 1, ks + 1]
                    wi[ks + 1, 1] = 0
                    wr[ks, 1] = wr[ks + 1, 1]
                    wi[ks, 1] = wi[ks + 1, 1]
            else:
                if kbot - ks + 1 <= ns // 2:
                    ks = kbot - ns + 1
                    var cbuf = List[Scalar[dtype]](length=ns * ns, fill=0)
                    var c = _view(cbuf, ns)
                    for i2 in range(1, ns + 1):
                        for j in range(1, ns + 1):
                            c[i2, j] = h[ks + i2 - 1, ks + j - 1]
                    var inf = _dlahqr(
                        False,
                        False,
                        ns,
                        1,
                        ns,
                        c,
                        wr.sub(ks, 1),
                        wi.sub(ks, 1),
                        1,
                        1,
                        c,
                    )
                    _keep(cbuf)
                    ks += inf
                    if ks >= kbot:
                        var r = _dlanv2(
                            h[kbot - 1, kbot - 1],
                            h[kbot - 1, kbot],
                            h[kbot, kbot - 1],
                            h[kbot, kbot],
                        )
                        wr[kbot - 1, 1] = r.rt1r
                        wi[kbot - 1, 1] = r.rt1i
                        wr[kbot, 1] = r.rt2r
                        wi[kbot, 1] = r.rt2i
                        ks = kbot - 1
                if kbot - ks + 1 > ns:
                    var sorted = False
                    var kk = kbot
                    while kk >= ks + 1:
                        if sorted:
                            break
                        sorted = True
                        for i2 in range(ks, kk):
                            if abs(wr[i2, 1]) + abs(wi[i2, 1]) < abs(
                                wr[i2 + 1, 1]
                            ) + abs(wi[i2 + 1, 1]):
                                sorted = False
                                var swap = wr[i2, 1]
                                wr[i2, 1] = wr[i2 + 1, 1]
                                wr[i2 + 1, 1] = swap
                                swap = wi[i2, 1]
                                wi[i2, 1] = wi[i2 + 1, 1]
                                wi[i2 + 1, 1] = swap
                        kk -= 1
                var ii = kbot
                while ii >= ks + 2:
                    if wi[ii, 1] != -wi[ii - 1, 1]:
                        var swap = wr[ii, 1]
                        wr[ii, 1] = wr[ii - 1, 1]
                        wr[ii - 1, 1] = wr[ii - 2, 1]
                        wr[ii - 2, 1] = swap
                        swap = wi[ii, 1]
                        wi[ii, 1] = wi[ii - 1, 1]
                        wi[ii - 1, 1] = wi[ii - 2, 1]
                        wi[ii - 2, 1] = swap
                    ii -= 2
            if kbot - ks + 1 == 2:
                if wi[kbot, 1] == 0:
                    if abs(wr[kbot, 1] - h[kbot, kbot]) < abs(
                        wr[kbot - 1, 1] - h[kbot, kbot]
                    ):
                        wr[kbot - 1, 1] = wr[kbot, 1]
                    else:
                        wr[kbot, 1] = wr[kbot - 1, 1]
            ns = min(ns, kbot - ks + 1)
            ns = ns - ns % 2
            ks = kbot - ns + 1
            _dlaqr5(
                wantt,
                wantz,
                n,
                ktop,
                kbot,
                ns,
                wr.sub(ks, 1),
                wi.sub(ks, 1),
                h,
                iloz,
                ihiz,
                z,
            )
        if ld > 0:
            ndfl = 1
        else:
            ndfl += 1
    return kbot


comptime _NMIN = 75
"""`dhseqr`'s crossover: below it the double-shift `dlahqr` runs."""


def _hseqr[
    dtype: DType
](
    wantt: Bool,
    wantz: Bool,
    n: Int,
    mut h: List[Scalar[dtype]],
    mut wr: List[Scalar[dtype]],
    mut wi: List[Scalar[dtype]],
    mut z: List[Scalar[dtype]],
) -> Int:
    """`DHSEQR` on the whole row-major `n x n` Hessenberg `h`: `dlahqr`
    below `_NMIN`, `dlaqr0` above. `z` is `n x n` when `wantz`, and is
    multiplied by the Schur vectors on the right. Returns `INFO`."""
    var hv = _view(h, n)
    var rv = _view(wr, 1)
    var iv = _view(wi, 1)
    var zv = _view(z, n) if wantz else hv
    var info: Int
    if n <= _NMIN:
        info = _dlahqr(wantt, wantz, n, 1, n, hv, rv, iv, 1, n, zv)
    else:
        info = _dlaqr0(wantt, wantz, n, 1, n, hv, rv, iv, 1, n, zv)
    # `dhseqr`'s "clear out the trash": the chase leaves stale entries
    # below the first subdiagonal that are zero in exact arithmetic.
    if wantt:
        for i in range(2, n):
            for j in range(i - 1):
                h[i * n + j] = 0
    return info
