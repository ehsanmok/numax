"""Generate `reference_data.mojo`: mpmath reference values for the accuracy
harness.

Run this with `pixi run accuracy-gen` when a domain changes or a function is
added. The output is checked in, so `pixi run accuracy` itself needs only
the default Mojo environment -- no Python, no mpmath, and nothing for CI to
install.

Everything is evaluated at 50 decimal digits and then rounded once to
float64. That is deliberate: the harness compares `numax`'s float64 result
against a value that is correct to well past float64, so the error it
reports is `numax`'s own approximation error and not a property of the
reference.
"""

from __future__ import annotations

import math
from pathlib import Path

import mpmath as mp

mp.mp.dps = 50

OUT = Path(__file__).parent / "reference_data.mojo"

# Sample counts are modest on purpose: these tables are compiled into the
# harness as literals, and the max error of a smooth function over an
# interval is found perfectly well by a few dozen points. Chebyshev-ish
# clustering toward the endpoints is where approximations usually fray.
N = 48


def linear(lo: float, hi: float, n: int = N) -> list[float]:
    return [lo + (hi - lo) * i / (n - 1) for i in range(n)]


def clustered(lo: float, hi: float, n: int = N) -> list[float]:
    """Chebyshev points, denser near both endpoints."""
    mid = 0.5 * (lo + hi)
    half = 0.5 * (hi - lo)
    pts = [mid - half * math.cos(math.pi * i / (n - 1)) for i in range(n)]
    pts[0], pts[-1] = lo, hi
    return pts


def logarithmic(lo: float, hi: float, n: int = N) -> list[float]:
    lg_lo, lg_hi = math.log(lo), math.log(hi)
    return [math.exp(lg_lo + (lg_hi - lg_lo) * i / (n - 1)) for i in range(n)]


def between_poles(first: int, last: int, margin: float = 0.06) -> list[float]:
    """Points on the negative axis, spread evenly *inside* each gap between
    consecutive poles at the non-positive integers.

    Filtering a uniform sweep would be the obvious way to dodge the poles,
    but it yields a different count depending on where the samples land,
    and the harness wants every case the same length. Sampling each gap
    directly gives exactly `N` points and distributes them better besides.
    """
    gaps = list(range(first, last))
    per = N // len(gaps)
    pts: list[float] = []
    for g in gaps:
        lo, hi = g + margin, g + 1 - margin
        pts.extend(lo + (hi - lo) * i / (per - 1) for i in range(per))
    # Any remainder from the integer division lands in the last gap.
    while len(pts) < N:
        pts.append(0.5 * (pts[-1] + pts[-2]))
    return sorted(pts[:N])


class Case:
    """One function over one domain, with its reference values."""

    def __init__(self, name: str, label: str, xs: list[float], fn) -> None:
        self.name = name
        self.label = label
        self.xs = xs
        self.refs = [float(fn(mp.mpf(x))) for x in xs]


def build() -> list[Case]:
    cases: list[Case] = []

    # --- special functions ---------------------------------------------
    cases.append(Case("erf_mid", "erf, [-3, 3]", clustered(-3.0, 3.0), mp.erf))
    cases.append(
        Case("erf_small", "erf, [1e-8, 1e-2]", logarithmic(1e-8, 1e-2), mp.erf)
    )
    cases.append(
        Case("erfc_tail", "erfc, [1, 6]", clustered(1.0, 6.0), mp.erfc)
    )

    cases.append(
        Case("gamma_pos", "gamma, [0.5, 8]", clustered(0.5, 8.0), mp.gamma)
    )
    cases.append(
        Case(
            "gamma_neg",
            "gamma, (-5, 0) off the poles",
            between_poles(-5, 0),
            mp.gamma,
        )
    )
    cases.append(
        Case("lgamma_pos", "lgamma, [0.5, 40]", clustered(0.5, 40.0), mp.loggamma)
    )
    cases.append(
        Case("digamma_pos", "digamma, [0.2, 12]", clustered(0.2, 12.0), mp.digamma)
    )
    cases.append(
        Case(
            "digamma_neg",
            "digamma, (-5, 0) off the poles",
            between_poles(-5, 0),
            mp.digamma,
        )
    )

    cases.append(
        Case("bessel_j0", "bessel_j0, [-15, 15]", clustered(-15.0, 15.0),
             lambda x: mp.besselj(0, x))
    )
    cases.append(
        Case("bessel_j1", "bessel_j1, [-15, 15]", clustered(-15.0, 15.0),
             lambda x: mp.besselj(1, x))
    )
    cases.append(
        Case("bessel_y0", "bessel_y0, [0.1, 15]", clustered(0.1, 15.0),
             lambda x: mp.bessely(0, x))
    )
    cases.append(
        Case("bessel_y1", "bessel_y1, [0.1, 15]", clustered(0.1, 15.0),
             lambda x: mp.bessely(1, x))
    )

    inv_e = float(1.0 / mp.e)
    cases.append(
        Case("lambertw_0", "lambertw, [-1/e+1e-9, 20]",
             clustered(-inv_e + 1e-9, 20.0), lambda x: mp.lambertw(x, 0))
    )
    cases.append(
        Case("lambertw_m1", "lambertw_m1, [-1/e+1e-9, -1e-12]",
             [-t for t in logarithmic(1e-12, inv_e - 1e-9)],
             lambda x: mp.lambertw(x, -1))
    )

    cases.append(
        Case("elliptic_k", "elliptic_k, m in [0, 0.999]",
             clustered(0.0, 0.999), mp.ellipk)
    )
    cases.append(
        Case("elliptic_e", "elliptic_e, m in [0, 1]",
             clustered(0.0, 1.0), mp.ellipe)
    )

    # --- Two-argument functions, at fixed parameters ------------------
    # `gammainc` here is the *regularized* lower incomplete gamma, which is
    # mpmath's `gammainc(a, 0, x, regularized=True)`.
    for a in (0.5, 1.0, 2.5, 10.0):
        tag = str(a).replace(".", "p")
        cases.append(
            Case(
                f"gammainc_a{tag}",
                f"gammainc(a={a}, x), x in [1e-3, 30]",
                logarithmic(1e-3, 30.0),
                lambda x, a=a: mp.gammainc(a, 0, x, regularized=True),
            )
        )

    for (a, b) in ((0.5, 0.5), (2.0, 3.0), (5.0, 1.5), (20.0, 20.0)):
        tag = f"{a}_{b}".replace(".", "p")
        cases.append(
            Case(
                f"betainc_{tag}",
                f"betainc(x, a={a}, b={b}), x in [0.001, 0.999]",
                clustered(0.001, 0.999),
                lambda x, a=a, b=b: mp.betainc(a, b, 0, x, regularized=True),
            )
        )

    # --- Orthogonal polynomials and quadrature-adjacent ---------------
    for n in (4, 12):
        cases.append(
            Case(
                f"legendre_p{n}",
                f"legendre_p(n={n}, x), x in [-1, 1]",
                clustered(-1.0, 1.0),
                lambda x, n=n: mp.legendre(n, x),
            )
        )
    for n in (4, 10):
        cases.append(
            Case(
                f"chebyshev_t{n}",
                f"chebyshev_t(n={n}, x), x in [-1, 1]",
                clustered(-1.0, 1.0),
                lambda x, n=n: mp.chebyt(n, x),
            )
        )
    cases.append(
        Case("hermite_h6", "hermite_h(n=6, x), x in [-3, 3]",
             clustered(-3.0, 3.0), lambda x: mp.hermite(6, x))
    )
    cases.append(
        Case("laguerre_l6", "laguerre_l(n=6, x), x in [0, 12]",
             clustered(0.0, 12.0), lambda x: mp.laguerre(6, 0, x))
    )

    # --- Elementary, to confirm the delegating conformers -------------
    cases.append(Case("exp_mid", "exp, [-10, 10]", clustered(-10.0, 10.0), mp.exp))
    cases.append(
        Case("ln_pos", "ln, [1e-6, 1e6]", logarithmic(1e-6, 1e6), mp.log)
    )
    cases.append(
        Case("sin_mid", "sin, [-8, 8]", clustered(-8.0, 8.0), mp.sin)
    )
    cases.append(
        Case("sqrt_pos", "sqrt, [1e-8, 1e8]", logarithmic(1e-8, 1e8), mp.sqrt)
    )

    return cases


def emit(cases: list[Case]) -> str:
    lines = [
        '"""mpmath reference values for `accuracy.mojo`. Generated -- do not',
        "edit by hand.",
        "",
        "Regenerate with `pixi run accuracy-gen` (see `gen_reference.py` in",
        "this directory). Every value was computed at 50 decimal digits and",
        "rounded once to float64, so it is correct to well past float64 and",
        "the harness measures `numax`'s error rather than the reference's.",
        '"""',
        "",
        "from std.collections import Array",
        "",
    ]
    for case in cases:
        n = len(case.xs)
        lines.append(f"# {case.label}")
        lines.append(f"comptime {case.name.upper()}_N = {n}")
        lines.append(
            f"comptime {case.name.upper()}_X: Array[Float64, {n}] = ["
        )
        for x in case.xs:
            lines.append(f"    {x!r},")
        lines.append("]")
        lines.append(
            f"comptime {case.name.upper()}_REF: Array[Float64, {n}] = ["
        )
        for r in case.refs:
            lines.append(f"    {r!r},")
        lines.append("]")
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    cases = build()
    OUT.write_text(emit(cases))
    total = sum(len(c.xs) for c in cases)
    print(f"wrote {OUT} -- {len(cases)} cases, {total} reference points")
    for c in cases:
        print(f"  {c.name:<22} {len(c.xs):>3} pts   {c.label}")


if __name__ == "__main__":
    main()
