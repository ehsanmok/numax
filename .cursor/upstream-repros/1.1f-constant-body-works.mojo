trait HasConstant(Copyable, Movable):
    @staticmethod
    def constant(v: Float64) -> Self:
        ...


__extension SIMD(HasConstant):
    @staticmethod
    def constant(v: Float64) -> Self:
        return Self(v)


def main():
    print(SIMD[DType.float32, 4].constant(2.5))
