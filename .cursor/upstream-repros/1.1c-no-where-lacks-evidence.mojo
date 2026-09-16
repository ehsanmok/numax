from std.math import exp as std_exp


trait HasExp(Copyable, Movable):
    def exp(self) -> Self:
        ...


__extension SIMD(HasExp):
    def exp(self) -> Self:
        return std_exp(self)


def use[T: HasExp](x: T) -> T:
    return x.exp()


def main():
    print(use(SIMD[DType.float32, 4](1.0)))
