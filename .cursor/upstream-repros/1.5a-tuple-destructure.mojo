@fieldwise_init
struct Owned(Movable):
    """A move-only resource, the shape `numax`'s `Tensor` has."""

    var tag: Int


def pair() -> Tuple[Owned, Owned]:
    return (Owned(1), Owned(2))


def main():
    # The spelling a SciPy user reaches for: `Q, R = qr(a)`.
    var q, r = pair()
    print(q.tag, r.tag)
