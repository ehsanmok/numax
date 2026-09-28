"""`concatenate`, `stack`, `split`, `array_split` and `slice` at
`gpu=True`, against the host. Each is one gather launch per result on the
device; a reorder does no arithmetic, so every comparison is exact.
"""

from std.testing import TestSuite, assert_equal, assert_false

from max.gpu.host import DeviceContext

from numax.core.tensor import (
    Static,
    array_split,
    concatenate,
    concatenate_dyn,
    dstack,
    hstack,
    slice,
    split,
    split_dyn,
    stack,
    stack_dyn,
    vstack,
)

comptime f32 = DType.float32


def _grid[
    r: Int, c: Int
](ctx: DeviceContext, base: Int) raises -> Static[f32, r, c]:
    var values = List[Scalar[f32]](capacity=r * c)
    for i in range(r * c):
        values.append(Float32(base + i))
    return Static[f32, r, c](ctx, values^)


def _assert_same(got: List[Scalar[f32]], want: List[Scalar[f32]]) raises:
    assert_equal(len(got), len(want))
    for i in range(len(want)):
        assert_equal(got[i], want[i])


def test_concatenate_on_the_device_matches_the_host() raises:
    var gpu = DeviceContext()
    var cpu = DeviceContext(api="cpu")
    var d = concatenate[axis=1, gpu=True](
        _grid[3, 4](gpu, 0), _grid[3, 2](gpu, 100)
    )
    assert_false(d.on_host())
    _assert_same(
        d.to_host(),
        concatenate[axis=1](
            _grid[3, 4](cpu, 0), _grid[3, 2](cpu, 100)
        ).to_host(),
    )
    _assert_same(
        concatenate[axis=0, gpu=True](
            _grid[3, 4](gpu, 0), _grid[2, 4](gpu, 100)
        ).to_host(),
        concatenate[axis=0](
            _grid[3, 4](cpu, 0), _grid[2, 4](cpu, 100)
        ).to_host(),
    )


def test_stack_on_the_device_matches_the_host() raises:
    """Every insertion axis of a rank-2 pair, spelled out: the `where`
    prover cannot see `axis <= rank` through a `comptime for`."""
    var gpu = DeviceContext()
    var cpu = DeviceContext(api="cpu")
    var d0 = stack[axis=0, gpu=True](_grid[3, 4](gpu, 0), _grid[3, 4](gpu, 100))
    assert_false(d0.on_host())
    _assert_same(
        d0.to_host(),
        stack[axis=0](_grid[3, 4](cpu, 0), _grid[3, 4](cpu, 100)).to_host(),
    )
    _assert_same(
        stack[axis=1, gpu=True](
            _grid[3, 4](gpu, 0), _grid[3, 4](gpu, 100)
        ).to_host(),
        stack[axis=1](_grid[3, 4](cpu, 0), _grid[3, 4](cpu, 100)).to_host(),
    )
    _assert_same(
        stack[axis=2, gpu=True](
            _grid[3, 4](gpu, 0), _grid[3, 4](gpu, 100)
        ).to_host(),
        stack[axis=2](_grid[3, 4](cpu, 0), _grid[3, 4](cpu, 100)).to_host(),
    )


def test_split_and_array_split_on_the_device_match_the_host() raises:
    var gpu = DeviceContext()
    var cpu = DeviceContext(api="cpu")
    var d = split[axis=1, gpu=True](_grid[3, 5](gpu, 0), 2)
    var h = split[axis=1](_grid[3, 5](cpu, 0), 2)
    assert_false(d[0].on_host())
    _assert_same(d[0].to_host(), h[0].to_host())
    _assert_same(d[1].to_host(), h[1].to_host())
    var dp = array_split[axis=0, gpu=True](_grid[7, 2](gpu, 0), 3)
    var hp = array_split[axis=0](_grid[7, 2](cpu, 0), 3)
    assert_equal(len(dp), 3)
    for i in range(3):
        _assert_same(dp[i].to_host(), hp[i].to_host())


def test_slice_on_the_device_matches_the_host() raises:
    var gpu = DeviceContext()
    var cpu = DeviceContext(api="cpu")
    var starts: List[Int] = [1, 2]
    var stops: List[Int] = [3, 5]
    var d = slice[gpu=True](_grid[4, 6](gpu, 0), starts.copy(), stops.copy())
    assert_false(d.on_host())
    assert_equal(d.dim_at(0), 2)
    assert_equal(d.dim_at(1), 3)
    _assert_same(
        d.to_host(), slice(_grid[4, 6](cpu, 0), starts, stops).to_host()
    )


def _line[n: Int](ctx: DeviceContext, base: Int) raises -> Static[f32, n]:
    var values = List[Scalar[f32]](capacity=n)
    for i in range(n):
        values.append(Float32(base + i))
    return Static[f32, n](ctx, values^)


def test_the_fixed_shape_joins_on_the_device_match_the_host() raises:
    """`vstack`, `hstack`, `dstack` and the rank-1 `concatenate`/`stack`,
    whose shapes stay in the type."""
    var gpu = DeviceContext()
    var cpu = DeviceContext(api="cpu")
    var v = vstack[gpu=True](_grid[2, 4](gpu, 0), _grid[3, 4](gpu, 100))
    assert_false(v.on_host())
    _assert_same(
        v.to_host(),
        vstack(_grid[2, 4](cpu, 0), _grid[3, 4](cpu, 100)).to_host(),
    )
    _assert_same(
        hstack[gpu=True](_grid[3, 2](gpu, 0), _grid[3, 4](gpu, 100)).to_host(),
        hstack(_grid[3, 2](cpu, 0), _grid[3, 4](cpu, 100)).to_host(),
    )
    _assert_same(
        dstack[gpu=True](_grid[3, 4](gpu, 0), _grid[3, 4](gpu, 100)).to_host(),
        dstack(_grid[3, 4](cpu, 0), _grid[3, 4](cpu, 100)).to_host(),
    )
    _assert_same(
        concatenate[gpu=True](_line[5](gpu, 0), _line[3](gpu, 100)).to_host(),
        concatenate(_line[5](cpu, 0), _line[3](cpu, 100)).to_host(),
    )
    _assert_same(
        stack[gpu=True](_line[5](gpu, 0), _line[5](gpu, 100)).to_host(),
        stack(_line[5](cpu, 0), _line[5](cpu, 100)).to_host(),
    )


def test_the_fixed_shape_and_run_time_splits_on_the_device() raises:
    """The rank-1 `split[at]` and the flat `_dyn` forms, which flatten a
    matrix first."""
    var gpu = DeviceContext()
    var cpu = DeviceContext(api="cpu")
    var d = split[at=3, gpu=True](_line[8](gpu, 0))
    var h = split[at=3](_line[8](cpu, 0))
    assert_false(d[1].on_host())
    _assert_same(d[0].to_host(), h[0].to_host())
    _assert_same(d[1].to_host(), h[1].to_host())
    var dd = split_dyn[gpu=True](_grid[3, 4](gpu, 0), 5)
    var hh = split_dyn(_grid[3, 4](cpu, 0), 5)
    _assert_same(dd[0].to_host(), hh[0].to_host())
    _assert_same(dd[1].to_host(), hh[1].to_host())
    _assert_same(
        concatenate_dyn[gpu=True](
            _grid[2, 3](gpu, 0), _line[4](gpu, 100)
        ).to_host(),
        concatenate_dyn(_grid[2, 3](cpu, 0), _line[4](cpu, 100)).to_host(),
    )
    _assert_same(
        stack_dyn[gpu=True](_grid[2, 3](gpu, 0), _line[6](gpu, 100)).to_host(),
        stack_dyn(_grid[2, 3](cpu, 0), _line[6](cpu, 100)).to_host(),
    )


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
