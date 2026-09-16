from std.math import exp


trait HasExp(Copyable, Movable):
    def exp(self) -> Self:
        ...


__extension SIMD(HasExp):
    def exp(self) -> Self:
        return exp(self)


def main():
    print(SIMD[DType.float32, 4](1.0).exp())
