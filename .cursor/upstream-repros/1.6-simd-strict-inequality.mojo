def main():
    var a = SIMD[DType.float32, 4](1.0, 2.0, 3.0, 4.0)
    var b = SIMD[DType.float32, 4](2.0, 2.0, 2.0, 2.0)

    # What a NumPy user writes for a lane-wise mask.
    print(a > b)
