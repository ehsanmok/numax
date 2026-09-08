"""Cholesky factorization and the solves that reuse it.
`scipy.linalg`'s `cholesky`/`cho_solve` pair.

**The `Tensor` tier.** MAX ships no Cholesky at any size -- its only
factorization is `qr_factorization`, on the older `LayoutTensor`, which
numax denies rather than bridges -- so this is numax's, as is the
`FloatLike`-generic tier in `numax.linalg.array.cholesky`.

`cholesky` here is right-looking and blocked: each step factors
one `block x block` diagonal panel, solves the panel below it, then
subtracts `L21 @ L21.T` from what remains. That subtraction is the entire
cubic cost and it is a matrix product, so it goes to `blas.matmul` and
inherits MAX's dispatch. Tier 2, host-orchestrated, and it raises on a
matrix that is not positive definite.

The `Array` tier is where a Cholesky differentiates -- calling it at
`Dual` gives the derivative of a factorization with no adjoint rule
written anywhere -- and it floors the diagonal rather than raising,
because a tier-1 kernel cannot branch on a value.

Neither tier pivots, and neither needs to: a symmetric positive definite
matrix does not require it. That is a theorem, not luck.
"""

from layout import Coord, TileTensor
from layout.tile_layout import row_major
from linalg.matmul import matmul as _max_matmul
from std.sys.info import align_of
from std.utils import IndexList

from ..core.array import Static, tril, zeros, zeros_dyn

from .blas import _target
from .common import _Dense
from .panel import _PANEL_THREADS, pack_block, potrf_diag, trsm_right_lower_t
from .triangular import solve_triangular


def cholesky[
    dtype: DType, n: Int, gpu: Bool = False, block: Int = 32
](mut a: Static[dtype, n, n]) raises -> Static[
    dtype, n, n
] where dtype.is_floating_point():
    """**Tier 2.** The lower-triangular `L` with `L @ L.T == a`, blocked
    and device-resident.

    MAX has no Cholesky -- `qr_factorization` is its only factorization,
    and that one is on the older `LayoutTensor` -- so this is numax filling
    a gap rather than calling into one. What it does not do is fill it from
    scratch: this is the right-looking blocked algorithm, so each step
    factors one `block x block` diagonal block, solves the panel below it,
    and then subtracts `L21 @ L21.T` from everything remaining. That last
    subtraction is the whole cubic cost of a Cholesky, and it is a matrix
    product, so it goes to `linalg.matmul` and inherits MAX's dispatch. The
    panel work is `O(n * block^2)` and the GEMM is `O(n^3)`.

    **Nothing leaves the device.** The matrix is copied once on the way in
    and once on the way out and is otherwise never touched by the host: the
    diagonal block and the panel solve are `numax.linalg.panel` kernels
    addressing the matrix in place, and the trailing update's result goes
    straight back into the strided trailing block through `matmul`'s
    epilogue. Only the GEMM's *operand* is copied, into an `n x block`
    scratch reused every step, because MAX's `matmul` ignores the row
    stride of its arguments and so cannot read `L21` where it lies. One
    `L21` serves as both operands under `transpose_b`.

    The cost is two `n x n` workspaces beside the matrix: one for the
    result and one MAX writes the GEMM into and nothing reads, which has to
    exist because `matmul` stores to `c` whether an epilogue is supplied or
    not. LAPACK's `potrf` takes a workspace argument for the same reason.

    The three round trips per block step that the previous version needed
    -- panel down, operands down, update back -- are gone, and with them
    the `O(n^2)` copy band they put on every step.

    `block` is a parameter so a caller can tune it or set it to `n` to get
    the unblocked algorithm back. The default is `32`, measured rather than
    guessed: the cost is `A * n^2 * block` for the panel solve plus
    `B * n^3 / block` for the GEMM's own writes to `c`, so it has an
    interior minimum, and `32` sat at or next to it at every size and on
    both targets (`bench/bench_linalg.mojo` sweeps it). No pivoting, and
    none is needed: a symmetric positive definite matrix does not require
    it.

    Raises when a diagonal entry comes out non-positive, which is what a
    matrix that is not positive definite looks like from in here. The check
    happens **once, at the end**: `potrf_diag` records the first bad pivot
    in a device-side `info` tensor and floors it, so the factorization runs
    to completion either way rather than putting a device synchronization
    in the loop. The sibling `cholesky` over `Array[T, n*n]` floors the
    diagonal too but has no way to report it, because a tier-1 kernel
    cannot branch on a value.
    """
    var ctx = a.context()
    var work = Static[dtype, n, n](ctx)
    var info = zeros[DType.int32, 1](ctx)
    # `L21` made dense for the GEMM. `n x block` covers every step's panel,
    # so it is allocated once rather than per step.
    var operand = zeros_dyn[dtype, 2](n, block, ctx=ctx)
    # The GEMM's own output, which nothing here ever reads: the epilogue
    # takes each tile as it is computed and subtracts it into the trailing
    # block. It still has to exist and it still has to be the product's
    # full size, because `matmul` writes `c` whether an epilogue is given
    # or not -- measured, and the reason this is not overlaid on `work`.
    var scratch = zeros_dyn[dtype, 2](n, n, ctx=ctx)

    var wv = work.view()
    var iv = info.view()
    var ov = operand.view()
    var sv = scratch.view()

    pack_block[target=_target[gpu]()](a.view(), wv, 0, 0, n, n, ctx)

    var k = 0
    while k < n:
        var nb = min(block, n - k)

        comptime if gpu:
            ctx.enqueue_function[
                potrf_diag[
                    dtype,
                    ALayout=type_of(wv).LayoutType,
                    ILayout=type_of(iv).LayoutType,
                    gpu=True,
                ]
            ](
                wv,
                iv,
                Int32(k),
                Int32(nb),
                grid_dim=1,
                block_dim=_PANEL_THREADS,
            )
        else:
            potrf_diag(wv, iv, Int32(k), Int32(nb))

        trsm_right_lower_t[target=_target[gpu]()](wv, k, nb, n, ctx)

        var m = n - k - nb
        if m > 0:
            var base = k + nb

            # Dense `m x nb` over the head of the scratch, and a second
            # view of it, because `matmul` takes `a` and `b` mutably and
            # rejects two live views that share an origin.
            var left: _Dense[dtype] = TileTensor(
                ov.ptr_at_offset(Coord(0, 0)), row_major(Coord(m, nb))
            )
            var right: _Dense[dtype] = TileTensor(
                ov.ptr_at_offset(Coord(0, 0)), row_major(Coord(m, nb))
            )
            pack_block[target=_target[gpu]()](wv, left, base, k, m, nb, ctx)

            var product: _Dense[dtype] = TileTensor(
                sv.ptr_at_offset(Coord(0, 0)), row_major(Coord(m, m))
            )

            @parameter
            @always_inline
            @__copy_capture(wv, base)
            def subtract[
                _dtype: DType,
                width: SIMDLength,
                *,
                alignment: Int = align_of[SIMD[_dtype, width]](),
            ](idx: IndexList[2], value: SIMD[_dtype, width]) capturing -> None:
                var at = Coord(base + idx[0], base + idx[1])
                wv.store[width](
                    at,
                    wv.load[width](at) - rebind[SIMD[dtype, width]](value),
                )

            _max_matmul[
                transpose_b=True,
                elementwise_lambda_fn=subtract,
                target=_target[gpu](),
            ](product, left, right, ctx)

        k += nb

    ctx.synchronize()

    var flag = Int(info.to_host()[0])
    if flag != 0:
        raise Error(
            (
                "cholesky: matrix is not positive definite (non-positive pivot"
                " at index "
            ),
            flag - 1,
            ")",
        )

    return tril[dtype, n, n, gpu](work)


def cholesky_solve[
    dtype: DType, n: Int, gpu: Bool = False, block: Int = 16
](mut lower: Static[dtype, n, n], mut b: Static[dtype, n]) raises -> Static[
    dtype, n
] where dtype.is_floating_point():
    """**Tier 2.** Solve `A @ x = b` given `A`'s Cholesky factor `L`,
    blocked and device-resident. `scipy.linalg.cho_solve`.

    Two triangular solves, `L @ y = b` then `L.T @ x = y`, both through
    `solve_triangular`. The second one passes `trans=True` rather than
    transposing `L`: an `n x n` transpose would cost more than the solve
    it feeds, and the `Array` sibling only materializes one because a
    tier-1 kernel has no way to read an index pair conditionally.

    Takes the factor rather than `A` for the reason the `Array` version
    does -- one factorization, many right-hand sides -- so pair it with
    `cholesky`, whose output is exactly this input.

    MAX ships neither the factorization nor the solve, so both halves are
    numax's; the cubic work inside them is still MAX's `matmul`.
    """
    var y = solve_triangular[dtype, n, False, False, False, gpu, block](
        lower, b
    )
    return solve_triangular[dtype, n, True, False, True, gpu, block](lower, y)


def cholesky_solve[
    dtype: DType, n: Int, rhs: Int, gpu: Bool = False, block: Int = 16
](
    mut lower: Static[dtype, n, n], mut b: Static[dtype, n, rhs]
) raises -> Static[dtype, n, rhs] where dtype.is_floating_point():
    """**Tier 2.** Solve `A @ X = B` given `A`'s Cholesky factor `L`, for
    a matrix `B`. `scipy.linalg.cho_solve` with a two-dimensional
    right-hand side.

    The vector overload's two solves against the matrix `solve_triangular`,
    so the update between diagonal blocks is a GEMM. This is the spelling
    a Gaussian process wants when it has a batch of right-hand sides
    rather than one.
    """
    var y = solve_triangular[dtype, n, rhs, False, False, False, gpu, block](
        lower, b
    )
    return solve_triangular[dtype, n, rhs, True, False, True, gpu, block](
        lower, y
    )
