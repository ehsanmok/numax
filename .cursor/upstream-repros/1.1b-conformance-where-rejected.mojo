trait Tiny(Copyable, Movable):
    def twice(self) -> Self:
        ...


__extension SIMD(Tiny where dtype.is_floating_point()):
    def twice(self) -> Self:
        return self + self


def main():
    print(SIMD[DType.float32, 4](1.5).twice())
