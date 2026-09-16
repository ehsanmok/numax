def halve[n: Int]() -> Int where n % 2 == 0:
    return n // 2


def main():
    print(halve[8]())
