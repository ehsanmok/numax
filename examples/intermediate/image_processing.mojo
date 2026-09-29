"""`scipy.ndimage` over a `Tensor`: smooth, segment, measure, resample.

```mojo
from numax.ndimage import gaussian_filter, label, distance_transform_edt
```

`numax.ndimage` is imported by its subpackage rather than from `numax`,
because its `convolve`/`correlate` are the n-dimensional image filters
and `numax.signal`'s are the 1-D sequence ones; the two cannot share the
flat namespace.

The image is two disks on a striped background. A Gaussian filter
washes the stripes out, a threshold turns the result into a mask, `label`
counts the disks (numbered in raster order, as SciPy numbers them), the
Euclidean distance transform reads each disk's radius back off its
deepest pixel, a binary erosion peels one ring of pixels, and `zoom`
resamples the whole image by cubic splines. Every step is one call with a
`gpu=True` spelling for a device-resident image.

Run: `pixi run example-image-processing`
"""

from std.math import sin, sqrt

from numax.core.tensor import Static
from numax.ndimage import (
    binary_erosion,
    distance_transform_edt,
    gaussian_filter,
    label,
    zoom,
)

comptime dtype = DType.float64
comptime side = 48


def image() raises -> Static[dtype, side, side]:
    """Disks of radius 8 at `(14, 14)` and 6 at `(32, 34)`, value `1`, on
    stripes of amplitude `0.3` and period 3 pixels."""
    var values = List[Scalar[dtype]](capacity=side * side)
    for r in range(side):
        for c in range(side):
            var v = 0.3 * sin(2.0943951 * Float64(c))
            var d1 = sqrt(Float64((r - 14) ** 2 + (c - 14) ** 2))
            var d2 = sqrt(Float64((r - 32) ** 2 + (c - 34) ** 2))
            if d1 <= 8.0 or d2 <= 6.0:
                v += 1.0
            values.append(Scalar[dtype](v))
    return Static[dtype, side, side](values^)


def count(mask: List[Scalar[dtype]]) -> Int:
    var total = 0
    for i in range(len(mask)):
        if mask[i] != 0:
            total += 1
    return total


def main() raises:
    var img = image()

    # --- Smooth the stripes away, then threshold.
    var smooth = gaussian_filter(img, 1.5).to_host()
    var mask_values = List[Scalar[dtype]](capacity=side * side)
    for i in range(side * side):
        mask_values.append(Scalar[dtype](1.0) if smooth[i] > 0.5 else 0.0)
    var mask = Static[dtype, side, side](mask_values.copy())
    print("foreground pixels after smoothing:", count(mask_values))

    # --- Count and number the disks.
    var labeled = label(mask)
    print("components:", labeled.num_features)
    var lab = labeled.labels.to_host()
    print(
        "label at (14, 14):",
        lab[14 * side + 14],
        " at (32, 34):",
        lab[32 * side + 34],
    )

    # --- Each disk's radius is its deepest pixel's distance to background.
    var edt = distance_transform_edt(mask).to_host()
    print("depth at the first center: ", edt[14 * side + 14], "(radius 8)")
    print("depth at the second center:", edt[32 * side + 34], "(radius 6)")

    # --- One ring of pixels off every disk.
    var eroded = binary_erosion(mask).to_host()
    print("foreground pixels after one erosion:", count(eroded))

    # --- Resample the original image to twice the size.
    var big = zoom(img, 2.0)
    print("zoomed shape:", big.dim_at(0), "x", big.dim_at(1))
