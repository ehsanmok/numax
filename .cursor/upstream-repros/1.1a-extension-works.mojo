trait Tiny(Copyable, Movable):
    def twice(self) -> Self:
        ...


__extension SIMD(Tiny):
    def twice(self) -> Self:
        return self + self


def use[T: Tiny](x: T) -> T:
    return x.twice()


def main():
    print(use(SIMD[DType.float32, 4](1.5)))
