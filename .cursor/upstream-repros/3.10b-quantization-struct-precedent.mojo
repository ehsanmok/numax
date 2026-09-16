"""MAX already stores composite elements in tensors -- by declaring the
tensor `uint8` and bitcasting the pointer to a struct.

`Q4sym` is a real composite element: two scale bytes plus nibble-packed
quants. `quantization/qmatmul_k.mojo`'s `matmul_Q4_K` takes its weights as
`TileTensor[..., .uint8, ...]` and decodes them through
`ptr.bitcast[Q4sym[...]]()`, because there is no way to declare a tensor
*of* `Q4sym`.

So the element-generic request in `max-feedback.md` 3.10 is not asking for
a capability MAX lacks internally. It is asking for the one it already
needs, with the type system telling the truth about it.
"""

from quantization.per_channel_grouped_4bit import Q4sym
from std.sys import size_of

comptime group = 32


def main():
    comptime Block = Q4sym[group]

    print("Q4sym[", group, "] is a composite element of", size_of[Block](), "bytes")
    print("  scale: StaticTuple[UInt8, 2]")
    print("  bits : StaticTuple[UInt8,", group // 2, "]")
    print()
    print("A tensor of these is spelled `TileTensor[..., DType.uint8, ...]`")
    print("and recovered with `ptr.bitcast[Q4sym[...]]()` -- the element type")
    print("is erased to bytes, so the layout is carried by convention rather")
    print("than by the type.")
