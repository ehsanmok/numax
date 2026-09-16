@fieldwise_init
struct Owned(Movable):
    var tag: Int


def pair() -> Tuple[Owned, Owned]:
    return (Owned(1), Owned(2))


def main():
    # Following the compiler's own suggestion: transfer with '^'.
    var got = pair()
    var q = got[0]^
    var r = got[1]^
    print(q.tag, r.tag)
