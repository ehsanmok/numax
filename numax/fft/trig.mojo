"""Real trigonometric transforms over `numax.core.array.Tensor`: `dct`/`idct`
and `dst`/`idst`, types I-IV, with `scipy.fft`'s definitions and its three
normalizations.

**Tier 2**, like `numax.fft.fft`, and built on it: each of the eight
transforms is one complex DFT of length `M` -- `2N` for types II-IV,
`2(N-1)` for DCT-I and `2(N+1)` for DST-I -- with a gather-and-weight pass
before it and a twiddle-and-project pass after. The eight types and three
norms differ only in three host-built tables (where each DFT input comes
from and what it is multiplied by; what each output bin is multiplied by
and which DFT bin it reads), so there is one kernel pair and one driver,
and a new type would be a new table rather than a new kernel. `M` is a
power of two exactly when `N` is (types II-IV) or `N -+ 1` is (type I);
every other length takes `numax.fft`'s Bluestein path, which is what lets
these accept any `N`.

## The MAX gate

Nothing to delegate to: MAX has no forward DFT at all (`numax.fft.fft`
records the search), so no cosine or sine transform either. An **extend**
on numax's own engine.

## The identities, briefly

Every type is a real projection of a complex DFT. Writing `e(a) =
exp(-i*pi*a)` and `U = fft_M(u)`:

| Type | `u` (length `M`) | `y[k]` |
|---|---|---|
| DCT-I | the even extension `x_0..x_{N-1}, x_{N-2}..x_1` | `Re U[k]` |
| DCT-II | `x`, zero-padded to `2N` | `Re 2 e(k/2N) U[k]` |
| DCT-III | `c_j x_j e(j/2N)`, `c_0 = 1`, else `2` | `Re U[k]` |
| DCT-IV | `2 x_j e(j/2N)` | `Re e((2k+1)/4N) U[k]` |
| DST-I | the odd extension `0, x_0..x_{N-1}, 0, -x_{N-1}..-x_0` | `-Im U[k+1]` |
| DST-II | `x`, zero-padded | `-Im 2 e((k+1)/2N) U[k+1]` |
| DST-III | `c_{j-1} x_{j-1} e(j/2N)` at `j = 1..N`, `c_{N-1} = 1`, else `2` | `-Im U[k]` |
| DST-IV | `2 x_j e(j/2N)` | `-Im e((2k+1)/4N) U[k]` |

`-Im z` is `Re(i z)`, so the sine transforms fold an `i` into their
post-multiplier and the project kernel takes a real part in every case.

## Normalization

`norm="backward"` (SciPy's default) leaves the forward transform as the
table above and puts the `1/M` on the inverse; `"forward"` puts it on the
forward transform instead; `"ortho"` splits it as `1/sqrt(M)` into both,
plus the `sqrt(2)` on the boundary terms that make each matrix orthogonal,
so `idct[norm="ortho"]` is `dct[norm="ortho"]` of the paired type with no
further scale. The inverse of type II is type III and vice versa; I and IV
are their own.

## What is not here

`dctn`/`idctn` and an `axis` argument: these are 1-D over a rank-1 tensor.
The row-column form is `fft2`'s shape, on the same lane engine, and would
be added the way `fft2` was when a caller has one.
"""

from std.math import cos as _cos, sin as _sin, sqrt as _sqrt

from layout import Coord, coord_to_index_list
from max.algorithm.functional import elementwise

from ..core.array import Static
from .fft import _PI, _as_matrix, _dft

comptime _SQRT2 = 1.4142135623730951


def _inverse_type(type: Int) -> Int:
    """The type whose forward transform inverts this one: II and III are
    each other's, I and IV their own."""
    if type == 2:
        return 3
    if type == 3:
        return 2
    return type


def _dft_length(cosine: Bool, type: Int, n: Int) -> Int:
    """The complex DFT length behind a length-`n` transform of this type."""
    if type == 1:
        return 2 * (n - 1) if cosine else 2 * (n + 1)
    return 2 * n


def _trig[
    dtype: DType,
    n: Int,
    gpu: Bool,
    cosine: Bool,
    type: Int,
    norm: StaticString,
    inverse: Bool,
](var x: Static[dtype, n]) raises -> Static[dtype, n]:
    """The one driver behind all eight transforms and both directions.

    Builds the three tables on the host for the effective type (the paired
    type when `inverse`), runs the gather kernel, one length-`M` `_dft`,
    and the project kernel. The module docstring has the tables; the code
    below is those tables written out, with the normalization folded in:
    a uniform `1/M`, `1/sqrt(M)` or `1` on the post-multiplier, and the
    `sqrt(2)` boundary tweaks `"ortho"` needs on whichever side SciPy puts
    them.
    """
    comptime assert (
        norm == "backward" or norm == "ortho" or norm == "forward"
    ), "dct/dst: norm must be 'backward', 'ortho' or 'forward'"
    comptime t = _inverse_type(type) if inverse else type
    comptime m = _dft_length(cosine, t, n)
    comptime ortho = norm == "ortho"
    # `"backward"` scales the inverse, `"forward"` the forward transform.
    comptime scaled = (norm == "backward" and inverse) or (
        norm == "forward" and not inverse
    )
    # Bins `1..N` of the DFT for the two sine types whose extension starts
    # with a zero; bins `0..N-1` for everything else.
    comptime offset = 1 if (not cosine and (t == 1 or t == 2)) else 0
    var ctx = x.context()

    var uniform = 1.0
    if scaled:
        uniform = 1.0 / Float64(m)
    elif ortho:
        uniform = 1.0 / _sqrt(Float64(m))

    var source = List[Int32](length=m, fill=Int32(-1))
    var wr = List[Scalar[dtype]](length=m, fill=Scalar[dtype](0))
    var wi = List[Scalar[dtype]](length=m, fill=Scalar[dtype](0))
    var pr = List[Scalar[dtype]](capacity=n)
    var pi = List[Scalar[dtype]](capacity=n)

    comptime if cosine:
        comptime if t == 1:
            for j in range(m):
                var s = j if j < n else m - j
                source[j] = Int32(s)
                var w = 1.0
                if ortho and (s == 0 or s == n - 1):
                    w = _SQRT2
                wr[j] = Scalar[dtype](w)
            for k in range(n):
                var p = uniform
                if ortho and (k == 0 or k == n - 1):
                    p /= _SQRT2
                pr.append(Scalar[dtype](p))
                pi.append(Scalar[dtype](0))
        elif t == 2:
            for j in range(n):
                source[j] = Int32(j)
                wr[j] = Scalar[dtype](1)
            for k in range(n):
                var angle = -_PI * Float64(k) / Float64(2 * n)
                var p = 2.0 * uniform
                if ortho and k == 0:
                    p /= _SQRT2
                pr.append(Scalar[dtype](p * _cos(angle)))
                pi.append(Scalar[dtype](p * _sin(angle)))
        elif t == 3:
            for j in range(n):
                source[j] = Int32(j)
                var angle = -_PI * Float64(j) / Float64(2 * n)
                var c = 1.0 if j == 0 else 2.0
                if ortho and j == 0:
                    c = _SQRT2
                wr[j] = Scalar[dtype](c * _cos(angle))
                wi[j] = Scalar[dtype](c * _sin(angle))
            for _ in range(n):
                pr.append(Scalar[dtype](uniform))
                pi.append(Scalar[dtype](0))
        else:
            for j in range(n):
                source[j] = Int32(j)
                var angle = -_PI * Float64(j) / Float64(2 * n)
                wr[j] = Scalar[dtype](2.0 * _cos(angle))
                wi[j] = Scalar[dtype](2.0 * _sin(angle))
            for k in range(n):
                var angle = -_PI * Float64(2 * k + 1) / Float64(4 * n)
                pr.append(Scalar[dtype](uniform * _cos(angle)))
                pi.append(Scalar[dtype](uniform * _sin(angle)))
    else:
        comptime if t == 1:
            # `0, x_0 .. x_{N-1}, 0, -x_{N-1} .. -x_0`; `source` is already
            # `-1` at the two zeros.
            for j in range(1, n + 1):
                source[j] = Int32(j - 1)
                wr[j] = Scalar[dtype](1)
            for j in range(n + 2, m):
                source[j] = Int32(m - 1 - j)
                wr[j] = Scalar[dtype](-1)
            for _ in range(n):
                # `i * uniform`
                pr.append(Scalar[dtype](0))
                pi.append(Scalar[dtype](uniform))
        elif t == 2:
            for j in range(n):
                source[j] = Int32(j)
                wr[j] = Scalar[dtype](1)
            for k in range(n):
                var angle = -_PI * Float64(k + 1) / Float64(2 * n)
                var p = 2.0 * uniform
                if ortho and k == n - 1:
                    p /= _SQRT2
                # `i * p * e(angle)`
                pr.append(Scalar[dtype](-p * _sin(angle)))
                pi.append(Scalar[dtype](p * _cos(angle)))
        elif t == 3:
            for j in range(1, n + 1):
                source[j] = Int32(j - 1)
                var angle = -_PI * Float64(j) / Float64(2 * n)
                var c = 1.0 if j == n else 2.0
                if ortho and j == n:
                    c = _SQRT2
                wr[j] = Scalar[dtype](c * _cos(angle))
                wi[j] = Scalar[dtype](c * _sin(angle))
            for _ in range(n):
                pr.append(Scalar[dtype](0))
                pi.append(Scalar[dtype](uniform))
        else:
            for j in range(n):
                source[j] = Int32(j)
                var angle = -_PI * Float64(j) / Float64(2 * n)
                wr[j] = Scalar[dtype](2.0 * _cos(angle))
                wi[j] = Scalar[dtype](2.0 * _sin(angle))
            for k in range(n):
                var angle = -_PI * Float64(2 * k + 1) / Float64(4 * n)
                pr.append(Scalar[dtype](-uniform * _sin(angle)))
                pi.append(Scalar[dtype](uniform * _cos(angle)))

    var index = Static[DType.int32, m](ctx, source^)
    var weight_re = Static[dtype, m](ctx, wr^)
    var weight_im = Static[dtype, m](ctx, wi^)
    var post_re = Static[dtype, n](ctx, pr^)
    var post_im = Static[dtype, n](ctx, pi^)

    # `u[j] = w[j] * x[source[j]]`, or zero where `source[j] < 0`.
    var u_re = Static[dtype, m]._uninitialized(ctx)
    var u_im = Static[dtype, m]._uninitialized(ctx)
    var xs = x.view()
    var ix = index.view()
    var wre = weight_re.view()
    var wim = weight_im.view()
    var ure = u_re.view()
    var uim = u_im.view()

    @always_inline
    def gather[
        w: Int, alignment: Int = 1
    ](coord: Coord) {var xs, var ix, var wre, var wim, var ure, var uim}:
        var j = coord_to_index_list(coord)[0]
        var s = Int(ix[Coord(j)])
        if s >= 0:
            var v = xs[Coord(s)]
            ure.store[1](Coord(j), v * wre[Coord(j)])
            uim.store[1](Coord(j), v * wim[Coord(j)])
        else:
            ure.store[1](Coord(j), Scalar[dtype](0))
            uim.store[1](Coord(j), Scalar[dtype](0))

    elementwise[simd_width=1, target="gpu" if gpu else "cpu"](
        gather, Coord(m), ctx
    )

    var big_re = Static[dtype, m]._uninitialized(ctx)
    var big_im = Static[dtype, m]._uninitialized(ctx)
    _dft[dtype, 1, m, gpu, False](
        _as_matrix[dtype, 1, m](u_re),
        _as_matrix[dtype, 1, m](u_im),
        _as_matrix[dtype, 1, m](big_re),
        _as_matrix[dtype, 1, m](big_im),
        ctx,
    )

    # `y[k] = Re(p[k] * U[k + offset])`.
    var out = Static[dtype, n]._uninitialized(ctx)
    var bre = big_re.view()
    var bim = big_im.view()
    var pre = post_re.view()
    var pim = post_im.view()
    var ys = out.view()

    @always_inline
    def project[
        w: Int, alignment: Int = 1
    ](coord: Coord) {var bre, var bim, var pre, var pim, var ys}:
        var k = coord_to_index_list(coord)[0]
        var c = Coord(k + offset)
        ys.store[1](Coord(k), pre[Coord(k)] * bre[c] - pim[Coord(k)] * bim[c])

    elementwise[simd_width=1, target="gpu" if gpu else "cpu"](
        project, Coord(n), ctx
    )
    ctx.synchronize()

    # Everything the kernels read went through an origin-erased view.
    _ = x^
    _ = index^
    _ = weight_re^
    _ = weight_im^
    _ = post_re^
    _ = post_im^
    _ = u_re^
    _ = u_im^
    _ = big_re^
    _ = big_im^
    return out^


def dct[
    dtype: DType,
    n: Int,
    gpu: Bool = False,
    type: Int = 2,
    norm: StaticString = "backward",
](var x: Static[dtype, n]) raises -> Static[dtype, n] where (
    dtype.is_floating_point()
    and n > 0
    and (type >= 1 and type <= 4)
    and (type != 1 or n >= 2)
):
    """The discrete cosine transform of `x`. `scipy.fft.dct(x, type,
    norm=norm)`, types I-IV, with SciPy's definitions:

    - I: `y[k] = x[0] + (-1)^k x[N-1] + 2 sum_{j=1}^{N-2} x[j] cos(pi j k /
      (N-1))`, which needs `N >= 2`
    - II (the default, and "the DCT"): `y[k] = 2 sum_j x[j] cos(pi (2j+1) k
      / 2N)`
    - III: `y[k] = x[0] + 2 sum_{j=1}^{N-1} x[j] cos(pi j (2k+1) / 2N)`
    - IV: `y[k] = 2 sum_j x[j] cos(pi (2j+1)(2k+1) / 4N)`

    `norm` is `"backward"` (unscaled forward, `1/(2N)`-shaped scale on
    `idct`), `"ortho"` (each matrix orthonormal, so `idct` is the paired
    type's `dct`) or `"forward"` (the scale on this side). Any `n`: the
    length-`2N` DFT underneath takes the radix-2 or Bluestein path as `2N`
    is or is not a power of two.

    One complex DFT plus two `elementwise` passes; the module docstring has
    the reduction and the tables.
    """
    return _trig[dtype, n, gpu, True, type, norm, False](x^)


def idct[
    dtype: DType,
    n: Int,
    gpu: Bool = False,
    type: Int = 2,
    norm: StaticString = "backward",
](var x: Static[dtype, n]) raises -> Static[dtype, n] where (
    dtype.is_floating_point()
    and n > 0
    and (type >= 1 and type <= 4)
    and (type != 1 or n >= 2)
):
    """The inverse of `dct` of the same `type` and `norm`.
    `scipy.fft.idct`.

    Type II's inverse is type III's forward transform and vice versa; I
    and IV invert themselves. Under `"backward"` the `1/(2N)` (`1/(2(N-1))`
    for type I) lands here, under `"forward"` it was already applied, and
    under `"ortho"` nothing further is needed -- so `idct(dct(x))` is `x`
    to rounding for every type and every `norm`.
    """
    return _trig[dtype, n, gpu, True, type, norm, True](x^)


def dst[
    dtype: DType,
    n: Int,
    gpu: Bool = False,
    type: Int = 2,
    norm: StaticString = "backward",
](var x: Static[dtype, n]) raises -> Static[dtype, n] where (
    dtype.is_floating_point() and n > 0 and (type >= 1 and type <= 4)
):
    """The discrete sine transform of `x`. `scipy.fft.dst(x, type,
    norm=norm)`, types I-IV, with SciPy's definitions:

    - I: `y[k] = 2 sum_j x[j] sin(pi (j+1)(k+1) / (N+1))`
    - II (the default): `y[k] = 2 sum_j x[j] sin(pi (2j+1)(k+1) / 2N)`
    - III: `y[k] = (-1)^k x[N-1] + 2 sum_{j=0}^{N-2} x[j] sin(pi (j+1)(2k+1)
      / 2N)`
    - IV: `y[k] = 2 sum_j x[j] sin(pi (2j+1)(2k+1) / 4N)`

    `norm` as for `dct`. Type I is the one whose DFT is `2(N+1)` long;
    every other type's is `2N`.
    """
    return _trig[dtype, n, gpu, False, type, norm, False](x^)


def idst[
    dtype: DType,
    n: Int,
    gpu: Bool = False,
    type: Int = 2,
    norm: StaticString = "backward",
](var x: Static[dtype, n]) raises -> Static[dtype, n] where (
    dtype.is_floating_point() and n > 0 and (type >= 1 and type <= 4)
):
    """The inverse of `dst` of the same `type` and `norm`. `scipy.fft.idst`.
    The same type pairing and scale placement as `idct`, with `1/(2(N+1))`
    for type I."""
    return _trig[dtype, n, gpu, False, type, norm, True](x^)
