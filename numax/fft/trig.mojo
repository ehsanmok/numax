"""Real trigonometric transforms over `numax.core.tensor.Tensor`: `dct`/`idct`
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

An `axis` argument. `dctn`/`idctn`/`dstn`/`idstn` transform every axis,
one batched pass per axis on the same lane engine; each pass writes its
output bin-major, which rotates the axes so the next pass finds its axis
last, the way `fftn` does.
"""

from std.math import cos as _cos, sin as _sin, sqrt as _sqrt

from layout import Coord, coord_to_index_list
from max.algorithm.functional import elementwise

from layout.tile_layout import TensorLayout

from ..core.tensor import Static, Tensor
from ..core.tensorlike import TensorLike
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
    """One transform of a rank-1 `x`: `_trig_pass` over a single lane."""
    var out = Static[dtype, n]._uninitialized(x.context())
    _trig_pass[
        dtype=dtype,
        batch=1,
        n=n,
        gpu=gpu,
        cosine=cosine,
        type=type,
        norm=norm,
        inverse=inverse,
    ](x, out)
    _ = x^
    return out^


def _trig_pass[
    dtype: DType,
    batch: Int,
    n: Int,
    gpu: Bool,
    cosine: Bool,
    type: Int,
    norm: StaticString,
    inverse: Bool,
    S: TensorLike,
    D: TensorLike,
](x: S, mut out: D) raises where S.dtype == dtype and D.dtype == dtype:
    """The one driver behind all eight transforms and both directions, over
    `batch` lanes of length `n`: lane `b` is `x[b*n .. b*n + n)`, and bin
    `k` of lane `b` is written to `out[k*batch + b]` -- bin-major, which is
    the ordinary layout at `batch = 1` and an axis rotation otherwise, the
    one `dctn` uses to reach every axis with no transpose pass.

    Builds the three tables on the host for the effective type (the paired
    type when `inverse`), runs the gather kernel, one length-`M` `_dft`
    per lane, and the project kernel. The module docstring has the tables; the code
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

    var index = Static[DType.int32, m](source^, ctx)
    var weight_re = Static[dtype, m](wr^, ctx)
    var weight_im = Static[dtype, m](wi^, ctx)
    var post_re = Static[dtype, n](pr^, ctx)
    var post_im = Static[dtype, n](pi^, ctx)

    # `u[b, j] = w[j] * x[b, source[j]]`, or zero where `source[j] < 0`.
    var u_re = Static[dtype, batch, m]._uninitialized(ctx)
    var u_im = Static[dtype, batch, m]._uninitialized(ctx)
    var xs = x.tile().ptr.unsafe_origin_cast[MutAnyOrigin]()
    var ix = index.tile()
    var wre = weight_re.tile()
    var wim = weight_im.tile()
    var ure = u_re.tile().ptr.unsafe_origin_cast[MutAnyOrigin]()
    var uim = u_im.tile().ptr.unsafe_origin_cast[MutAnyOrigin]()

    @always_inline
    def gather[
        w: Int, alignment: Int = 1
    ](coord: Coord) {var xs, var ix, var wre, var wim, var ure, var uim}:
        var f = coord_to_index_list(coord)[0]
        var b = f // m
        var j = f % m
        var s = Int(ix[Coord(j)])
        if s >= 0:
            var v = rebind[Scalar[dtype]](xs[unsafe_offset=b * n + s])
            ure[unsafe_offset=f] = v * rebind[Scalar[dtype]](wre[Coord(j)])
            uim[unsafe_offset=f] = v * rebind[Scalar[dtype]](wim[Coord(j)])
        else:
            ure[unsafe_offset=f] = Scalar[dtype](0)
            uim[unsafe_offset=f] = Scalar[dtype](0)

    elementwise[simd_width=1, target="gpu" if gpu else "cpu"](
        gather, Coord(batch * m), ctx
    )

    var big_re = Static[dtype, batch, m]._uninitialized(ctx)
    var big_im = Static[dtype, batch, m]._uninitialized(ctx)
    _dft[dtype=dtype, batch=batch, n=m, gpu=gpu, inverse=False](
        _as_matrix[rows=batch, cols=m](u_re),
        _as_matrix[rows=batch, cols=m](u_im),
        _as_matrix[rows=batch, cols=m](big_re),
        _as_matrix[rows=batch, cols=m](big_im),
        ctx,
    )

    # `y[k, b] = Re(p[k] * U[b, k + offset])`, bin-major.
    var bre = big_re.tile().ptr.unsafe_origin_cast[MutAnyOrigin]()
    var bim = big_im.tile().ptr.unsafe_origin_cast[MutAnyOrigin]()
    var pre = post_re.tile()
    var pim = post_im.tile()
    var ys = out.tile().ptr.unsafe_origin_cast[MutAnyOrigin]()

    @always_inline
    def project[
        w: Int, alignment: Int = 1
    ](coord: Coord) {var bre, var bim, var pre, var pim, var ys}:
        var f = coord_to_index_list(coord)[0]
        var b = f // n
        var k = f % n
        var at = b * m + k + offset
        var y = (
            rebind[Scalar[dtype]](pre[Coord(k)]) * bre[unsafe_offset=at]
            - rebind[Scalar[dtype]](pim[Coord(k)]) * bim[unsafe_offset=at]
        )
        ys[unsafe_offset=k * batch + b] = rebind[Scalar[D.dtype]](y)

    elementwise[simd_width=1, target="gpu" if gpu else "cpu"](
        project, Coord(batch * n), ctx
    )
    ctx.synchronize()

    # Everything the kernels read went through an origin-erased view.
    _ = index^
    _ = weight_re^
    _ = weight_im^
    _ = post_re^
    _ = post_im^
    _ = u_re^
    _ = u_im^
    _ = big_re^
    _ = big_im^


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

    Parameters:
        dtype: The floating-point element type of `x`.
        n: The length of `x`, any `n > 0` (`n >= 2` for type I).
        gpu: When `True`, every kernel launches with `target="gpu"` on the
            input's device; otherwise they run on the CPU.
        type: The transform type, `1` through `4`; `2` is the default.
        norm: The scaling mode, `"backward"`, `"ortho"` or `"forward"`.

    Args:
        x: The real length-`n` sequence to transform.

    Returns:
        The real length-`n` transform of `x`.

    Raises:
        If allocating a buffer or launching a kernel on the input's device
        fails.
    """
    return _trig[gpu=gpu, cosine=True, type=type, norm=norm, inverse=False](x^)


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

    Parameters:
        dtype: The floating-point element type of `x`.
        n: The length of `x`, any `n > 0` (`n >= 2` for type I).
        gpu: When `True`, every kernel launches with `target="gpu"` on the
            input's device; otherwise they run on the CPU.
        type: The transform type, `1` through `4`; `2` is the default.
        norm: The scaling mode, `"backward"`, `"ortho"` or `"forward"`.

    Args:
        x: The real length-`n` sequence to transform.

    Returns:
        The real length-`n` inverse transform of `x`.

    Raises:
        If allocating a buffer or launching a kernel on the input's device
        fails.
    """
    return _trig[gpu=gpu, cosine=True, type=type, norm=norm, inverse=True](x^)


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

    Parameters:
        dtype: The floating-point element type of `x`.
        n: The length of `x`, any `n > 0`.
        gpu: When `True`, every kernel launches with `target="gpu"` on the
            input's device; otherwise they run on the CPU.
        type: The transform type, `1` through `4`; `2` is the default.
        norm: The scaling mode, `"backward"`, `"ortho"` or `"forward"`.

    Args:
        x: The real length-`n` sequence to transform.

    Returns:
        The real length-`n` transform of `x`.

    Raises:
        If allocating a buffer or launching a kernel on the input's device
        fails.
    """
    return _trig[gpu=gpu, cosine=False, type=type, norm=norm, inverse=False](x^)


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
    for type I.

    Parameters:
        dtype: The floating-point element type of `x`.
        n: The length of `x`, any `n > 0`.
        gpu: When `True`, every kernel launches with `target="gpu"` on the
            input's device; otherwise they run on the CPU.
        type: The transform type, `1` through `4`; `2` is the default.
        norm: The scaling mode, `"backward"`, `"ortho"` or `"forward"`.

    Args:
        x: The real length-`n` sequence to transform.

    Returns:
        The real length-`n` inverse transform of `x`.

    Raises:
        If allocating a buffer or launching a kernel on the input's device
        fails.
    """
    return _trig[gpu=gpu, cosine=False, type=type, norm=norm, inverse=True](x^)


def _trign[
    dtype: DType,
    L: TensorLayout,
    gpu: Bool,
    cosine: Bool,
    type: Int,
    norm: StaticString,
    inverse: Bool,
](var x: Tensor[dtype, L]) raises -> Tensor[dtype, L]:
    """The transform along every axis: one `_trig_pass` per axis, each on
    the current last axis and writing bin-major, which rotates the axes so
    the next is last; after one pass per axis the order is the original."""
    comptime rank = L.rank
    comptime total = L.static_product
    var ctx = x.context()
    var layout = x.tile().layout
    var a = Tensor[dtype, L]._uninitialized(ctx, layout)
    var b = Tensor[dtype, L]._uninitialized(ctx, layout)
    comptime for s in range(rank):
        comptime n = L.static_shape[rank - 1 - s]
        comptime lanes = total // n
        comptime if s == 0:
            _trig_pass[
                dtype=dtype,
                batch=lanes,
                n=n,
                gpu=gpu,
                cosine=cosine,
                type=type,
                norm=norm,
                inverse=inverse,
            ](x, a)
        else:
            _trig_pass[
                dtype=dtype,
                batch=lanes,
                n=n,
                gpu=gpu,
                cosine=cosine,
                type=type,
                norm=norm,
                inverse=inverse,
            ](a, b)
            swap(a, b)
    _ = x^
    _ = b^
    return a^


def dctn[
    dtype: DType,
    L: TensorLayout,
    //,
    gpu: Bool = False,
    type: Int = 2,
    norm: StaticString = "backward",
](var x: Tensor[dtype, L]) raises -> Tensor[dtype, L] where (
    dtype.is_floating_point()
    and L.all_dims_known
    and L.static_product > 0
    and (type >= 1 and type <= 4)
):
    """The discrete cosine transform along every axis of `x`. `scipy.fft.dctn(x, type,
    norm=norm)`.

    `dct` of the same type and norm applied along each axis in turn --
    the transform separates, so this is the definition, not an
    approximation. One pass per axis, each over every lane of the current
    last axis and writing bin-major, so the axes rotate into place with no
    transpose pass; at rank 1 it is `dct`. Type I needs every extent at
    least 2.

    Parameters:
        dtype: The floating-point element type of `x`, inferred.
        L: The compile-time layout of `x`, inferred; any rank.
        gpu: When `True`, every kernel launches with `target="gpu"` on the
            input's device; otherwise they run on the CPU.
        type: The transform type, `1` through `4`; `2` is the default.
        norm: The scaling mode, `"backward"`, `"ortho"` or `"forward"`.

    Args:
        x: The real array to transform.

    Returns:
        The real transform at `x`'s shape.

    Raises:
        If allocating a buffer or launching a kernel on the input's device
        fails.
    """
    return _trign[dtype, L, gpu, True, type, norm, False](x^)


def idctn[
    dtype: DType,
    L: TensorLayout,
    //,
    gpu: Bool = False,
    type: Int = 2,
    norm: StaticString = "backward",
](var x: Tensor[dtype, L]) raises -> Tensor[dtype, L] where (
    dtype.is_floating_point()
    and L.all_dims_known
    and L.static_product > 0
    and (type >= 1 and type <= 4)
):
    """The inverse discrete cosine transform along every axis of `x`. `scipy.fft.idctn(x, type,
    norm=norm)`.

    `idct` of the same type and norm applied along each axis in turn --
    the transform separates, so this is the definition, not an
    approximation. One pass per axis, each over every lane of the current
    last axis and writing bin-major, so the axes rotate into place with no
    transpose pass; at rank 1 it is `idct`. Type I needs every extent at
    least 2.

    Parameters:
        dtype: The floating-point element type of `x`, inferred.
        L: The compile-time layout of `x`, inferred; any rank.
        gpu: When `True`, every kernel launches with `target="gpu"` on the
            input's device; otherwise they run on the CPU.
        type: The transform type, `1` through `4`; `2` is the default.
        norm: The scaling mode, `"backward"`, `"ortho"` or `"forward"`.

    Args:
        x: The real array to transform.

    Returns:
        The real transform at `x`'s shape.

    Raises:
        If allocating a buffer or launching a kernel on the input's device
        fails.
    """
    return _trign[dtype, L, gpu, True, type, norm, True](x^)


def dstn[
    dtype: DType,
    L: TensorLayout,
    //,
    gpu: Bool = False,
    type: Int = 2,
    norm: StaticString = "backward",
](var x: Tensor[dtype, L]) raises -> Tensor[dtype, L] where (
    dtype.is_floating_point()
    and L.all_dims_known
    and L.static_product > 0
    and (type >= 1 and type <= 4)
):
    """The discrete sine transform along every axis of `x`. `scipy.fft.dstn(x, type,
    norm=norm)`.

    `dst` of the same type and norm applied along each axis in turn --
    the transform separates, so this is the definition, not an
    approximation. One pass per axis, each over every lane of the current
    last axis and writing bin-major, so the axes rotate into place with no
    transpose pass; at rank 1 it is `dst`. Type I needs every extent at
    least 2 (and SciPy's DST-I at least 1).

    Parameters:
        dtype: The floating-point element type of `x`, inferred.
        L: The compile-time layout of `x`, inferred; any rank.
        gpu: When `True`, every kernel launches with `target="gpu"` on the
            input's device; otherwise they run on the CPU.
        type: The transform type, `1` through `4`; `2` is the default.
        norm: The scaling mode, `"backward"`, `"ortho"` or `"forward"`.

    Args:
        x: The real array to transform.

    Returns:
        The real transform at `x`'s shape.

    Raises:
        If allocating a buffer or launching a kernel on the input's device
        fails.
    """
    return _trign[dtype, L, gpu, False, type, norm, False](x^)


def idstn[
    dtype: DType,
    L: TensorLayout,
    //,
    gpu: Bool = False,
    type: Int = 2,
    norm: StaticString = "backward",
](var x: Tensor[dtype, L]) raises -> Tensor[dtype, L] where (
    dtype.is_floating_point()
    and L.all_dims_known
    and L.static_product > 0
    and (type >= 1 and type <= 4)
):
    """The inverse discrete sine transform along every axis of `x`. `scipy.fft.idstn(x, type,
    norm=norm)`.

    `idst` of the same type and norm applied along each axis in turn --
    the transform separates, so this is the definition, not an
    approximation. One pass per axis, each over every lane of the current
    last axis and writing bin-major, so the axes rotate into place with no
    transpose pass; at rank 1 it is `idst`. Type I needs every extent at
    least 2 (and SciPy's DST-I at least 1).

    Parameters:
        dtype: The floating-point element type of `x`, inferred.
        L: The compile-time layout of `x`, inferred; any rank.
        gpu: When `True`, every kernel launches with `target="gpu"` on the
            input's device; otherwise they run on the CPU.
        type: The transform type, `1` through `4`; `2` is the default.
        norm: The scaling mode, `"backward"`, `"ortho"` or `"forward"`.

    Args:
        x: The real array to transform.

    Returns:
        The real transform at `x`'s shape.

    Raises:
        If allocating a buffer or launching a kernel on the input's device
        fails.
    """
    return _trign[dtype, L, gpu, False, type, norm, True](x^)
