"""Helpers shared across `numax.optimize`'s `Tensor` tier, none of them
public.

There is one, and it exists because of a defect rather than for tidiness.
`Tensor.context()` hands back a handle that does not outlive the tensor it
came from, so an iterative driver cannot hold a `DeviceContext` across a
reassignment of the tensor it took the context from -- doing so faults at
run time with no diagnostic. `findings.mdc` carries the twelve-line
reproducer.

Every driver here therefore keeps its running vector in a `List[Float64]`
and materializes a fresh tensor per evaluation, and `_as_tensor` is that
materialization. `least_squares` and `minimize` both need it, so it lives
here rather than once in each.

Nothing is re-exported.
"""

from max.gpu.host import DeviceContext

from ..core.array import Static


def _as_tensor[
    dtype: DType, n: Int
](values: List[Float64], ctx: DeviceContext) raises -> Static[dtype, n]:
    """The running vector, materialized on `ctx` for one evaluation.

    Fresh each call rather than a reassigned one, for the reason this
    module's docstring gives: reassigning the tensor a live `DeviceContext`
    was taken from is a use-after-free.
    """
    var scalars = List[Scalar[dtype]](capacity=n)
    for i in range(n):
        scalars.append(Scalar[dtype](values[i]))
    return Static[dtype, n](ctx, scalars^)
