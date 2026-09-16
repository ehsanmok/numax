trait HasExp(Copyable, Movable):
    def exp(self) -> Self:
        ...


# Attempt to restrict the extension to floating-point dtypes by naming
# parameters on the extension itself.
__extension SIMD[dtype: DType, length: SIMDLength](HasExp):
    def exp(self) -> Self:
        return self


def main():
    print("unreachable")
