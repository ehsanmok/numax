"""Max error per function per domain, against mpmath references.

Every approximation in `numax` documents an error bound in its own module
docstring -- "max absolute error ~1.5e-7" and so on. Those numbers came
from the literature the formula was taken from, not from this
implementation of it, so until now nothing checked that the transcription,
the branchless blends, and the fixed iteration counts actually deliver
them. This does.

The references are mpmath at 50 decimal digits, rounded once to float64
(`gen_reference.py` in this directory generates them; the result is checked
in, so this harness needs only the default Mojo environment). Everything is
evaluated at `Plain[float64, 1]` so that what gets measured is the
approximation's own error rather than the rounding of a narrower `dtype`.

Three numbers per domain, because no single one is honest for every
function:

- **ULP** is the right measure for a function that should be nearly exact
  (`sqrt`, `sin`, the polynomial recurrences). It is useless for a
  deliberately low-order approximation, where it reads in the billions and
  says only "not bit-exact", which was never the claim.
- **Max relative error** is the right measure for those approximations, and
  it is what their docstrings should be checked against -- except where the
  true value passes through zero, which makes relative error meaningless
  through no fault of the implementation.
- **Max absolute error** covers that last case, and is the form most of the
  literature's own bounds are quoted in.

Read the column that matches the claim being checked; the others are
context, not indictments.

## Read the primitives section first

The primitives are printed first because they are not context -- they are a
floor under everything below them. This harness found that Mojo's
`std.math` `exp`, `log`, and `erf` are *not* correctly rounded at float64:
`exp(1.0)` comes back as `2.718281828459813` against a true
`2.718281828459045`, wrong from the 13th significant digit. `sin` and
`sqrt`, by contrast, are exact to the last bit.

That matters for reading every other row. `gamma` is a 9-term Lanczos
approximation that would be good to nearly full float64 in exact
arithmetic, but it is built from `ln` and `exp`, so the ~4e-9 relative
error reported for it is mostly inherited rather than its own. The same
goes for `lgamma`, `lambertw`, the Bessel and elliptic families, and the
incomplete gamma and beta functions. Their true algorithmic error is
somewhere at or below what is printed, and this harness cannot separate the
two without a correctly-rounded `exp`/`ln` to build on.

None of this is a problem at the `dtype` `numax` is normally used at.
Float32 has an epsilon of about 1.2e-7, so a 2e-9 primitive error is two
orders of magnitude below the representable resolution and cannot be
observed at all. It only becomes the limiting factor at float64, which is
exactly where this harness operates in order to see the algorithms.
"""

from std.math import sqrt as std_sqrt
from std.memory import bitcast

from numax import (
    Plain,
    j0,
    j1,
    y0,
    y1,
    betainc,
    chebyshev_t,
    digamma,
    elliptic_e,
    elliptic_k,
    erf,
    erfc,
    gamma,
    gammainc,
    hermite_h,
    laguerre_l,
    lambertw,
    legendre_p,
    lgamma,
)
from numax.special.lambertw import lambertw_m1
from numax.core.numeric import default_erf_approx

from reference_data import *

comptime P = Plain[DType.float64, 1]


def p(x: Float64) -> P:
    return P(SIMD[DType.float64, 1](x))


def ulps_between(a: Float64, b: Float64) -> Float64:
    """Distance in representable float64 steps.

    Ordinary integer distance between the bit patterns, which works because
    IEEE-754 lays consecutive floats out as consecutive integers within a
    sign. Across the sign boundary the two magnitudes are added instead,
    which counts the steps through zero.
    """
    if a == b:
        return 0.0
    var ia = Int(bitcast[DType.int64](a))
    var ib = Int(bitcast[DType.int64](b))
    if (ia < 0) != (ib < 0):
        return Float64(abs(ia) + abs(ib))
    return Float64(abs(ia - ib))


struct Stats(Copyable, Movable):
    var max_abs: Float64
    var max_abs_at: Float64
    var max_rel: Float64
    var max_rel_at: Float64
    var max_ulp: Float64
    var count: Int
    var skipped: Int

    def __init__(out self):
        self.max_abs = 0.0
        self.max_abs_at = 0.0
        self.max_rel = 0.0
        self.max_rel_at = 0.0
        self.max_ulp = 0.0
        self.count = 0
        self.skipped = 0

    def observe(mut self, x: Float64, got: Float64, want: Float64):
        # A non-finite result is a failure of a different kind than an
        # inaccurate one, so it is counted separately rather than folded
        # into a maximum it would dominate.
        if not (got == got) or (got - got) != 0.0:
            self.skipped += 1
            return
        self.count += 1

        var err = abs(got - want)
        if err > self.max_abs:
            self.max_abs = err
            self.max_abs_at = x

        # Relative error is only meaningful away from a zero of the
        # function; near one, every implementation looks terrible by that
        # measure and the absolute column is the one to read.
        if abs(want) > 1e-12:
            var rel = err / abs(want)
            if rel > self.max_rel:
                self.max_rel = rel
                self.max_rel_at = x

        var u = ulps_between(got, want)
        if u > self.max_ulp:
            self.max_ulp = u

    def report(self, label: String):
        var ulp = String(Int(self.max_ulp)) if self.max_ulp < 1e15 else ">1e15"
        var note = String("")
        if self.skipped > 0:
            note = " [" + String(self.skipped) + " non-finite]"
        print(
            label,
            "\t",
            self.max_abs,
            "\t",
            self.max_rel,
            "\t",
            ulp,
            "\t@x=",
            self.max_abs_at,
            note,
        )


def section(title: String):
    print("")
    print("== " + title + " ==")
    print("function / domain \t max abs \t max rel \t max ULP \t worst x")


def main():
    print(
        "numax accuracy vs mpmath (50-digit references), at Plain[float64, 1]"
    )

    # --- Primitives ---------------------------------------------------
    # First, because everything below inherits their error. See this
    # module's docstring: `exp`/`ln` are not correctly rounded here, so the
    # special functions built on them cannot be measured below that floor.
    section("Primitives -- the floor under everything below")

    var s = Stats()
    var xs = materialize[EXP_MID_X]()
    var refs = materialize[EXP_MID_REF]()
    for i in range(EXP_MID_N):
        s.observe(xs[i], p(xs[i]).exp().v[0], refs[i])
    s.report("Plain.exp, [-10,10]")

    s = Stats()
    xs = materialize[LN_POS_X]()
    refs = materialize[LN_POS_REF]()
    for i in range(LN_POS_N):
        s.observe(xs[i], p(xs[i]).ln().v[0], refs[i])
    s.report("Plain.ln, [1e-6,1e6]")

    s = Stats()
    xs = materialize[SIN_MID_X]()
    refs = materialize[SIN_MID_REF]()
    for i in range(SIN_MID_N):
        s.observe(xs[i], p(xs[i]).sin().v[0], refs[i])
    s.report("Plain.sin, [-8,8]")

    s = Stats()
    xs = materialize[SQRT_POS_X]()
    refs = materialize[SQRT_POS_REF]()
    for i in range(SQRT_POS_N):
        s.observe(xs[i], p(xs[i]).sqrt().v[0], refs[i])
    s.report("Plain.sqrt, [1e-8,1e8]")

    # --- erf/erfc -----------------------------------------------------
    section("Error function")
    # `Plain.erf` delegates to `std.math.erf`, so this measures libm rather
    # than any `numax` approximation -- worth reporting precisely because
    # near-zero ULP is the evidence that the delegation is real.
    s = Stats()
    xs = materialize[ERF_MID_X]()
    refs = materialize[ERF_MID_REF]()
    for i in range(ERF_MID_N):
        s.observe(xs[i], erf(p(xs[i])).v[0], refs[i])
    s.report("erf (Plain->std.math), [-3,3]")

    s = Stats()
    xs = materialize[ERF_SMALL_X]()
    refs = materialize[ERF_SMALL_REF]()
    for i in range(ERF_SMALL_N):
        s.observe(xs[i], erf(p(xs[i])).v[0], refs[i])
    s.report("erf (Plain->std.math), [1e-8,1e-2]")

    # The A&S 7.1.26 approximation `Compensated`/`Decimal` fall back to.
    # Its documented bound is ~1.5e-7 absolute; this is where that gets
    # checked rather than taken on faith.
    var s2 = Stats()
    var xs2 = materialize[ERF_MID_X]()
    var refs2 = materialize[ERF_MID_REF]()
    for i in range(ERF_MID_N):
        s2.observe(xs2[i], default_erf_approx(p(xs2[i])).v[0], refs2[i])
    s2.report("default_erf_approx (A&S 7.1.26), [-3,3]")

    s = Stats()
    xs = materialize[ERFC_TAIL_X]()
    refs = materialize[ERFC_TAIL_REF]()
    for i in range(ERFC_TAIL_N):
        s.observe(xs[i], erfc(p(xs[i])).v[0], refs[i])
    s.report("erfc (Plain->std.math), [1,6]")

    # --- gamma family -------------------------------------------------
    section("Gamma family")
    s = Stats()
    xs = materialize[GAMMA_POS_X]()
    refs = materialize[GAMMA_POS_REF]()
    for i in range(GAMMA_POS_N):
        s.observe(xs[i], gamma(p(xs[i])).v[0], refs[i])
    s.report("gamma (9-term Lanczos), [0.5,8]")

    s = Stats()
    xs = materialize[GAMMA_NEG_X]()
    refs = materialize[GAMMA_NEG_REF]()
    for i in range(GAMMA_NEG_N):
        s.observe(xs[i], gamma(p(xs[i])).v[0], refs[i])
    s.report("gamma (reflected), [-4.5,-0.5]")

    s = Stats()
    xs = materialize[LGAMMA_POS_X]()
    refs = materialize[LGAMMA_POS_REF]()
    for i in range(LGAMMA_POS_N):
        s.observe(xs[i], lgamma(p(xs[i])).v[0], refs[i])
    s.report("lgamma, [0.5,40]")

    s = Stats()
    xs = materialize[DIGAMMA_POS_X]()
    refs = materialize[DIGAMMA_POS_REF]()
    for i in range(DIGAMMA_POS_N):
        s.observe(xs[i], digamma(p(xs[i])).v[0], refs[i])
    s.report("digamma (Dual of lgamma), [0.2,12]")

    s = Stats()
    xs = materialize[DIGAMMA_NEG_X]()
    refs = materialize[DIGAMMA_NEG_REF]()
    for i in range(DIGAMMA_NEG_N):
        s.observe(xs[i], digamma(p(xs[i])).v[0], refs[i])
    s.report("digamma (reflected), (-5,0) off poles")

    # --- Bessel -------------------------------------------------------
    section("Bessel")
    s = Stats()
    xs = materialize[BESSEL_J0_X]()
    refs = materialize[BESSEL_J0_REF]()
    for i in range(BESSEL_J0_N):
        s.observe(xs[i], j0(p(xs[i])).v[0], refs[i])
    s.report("j0 (A&S 9.4.1/9.4.3), [-15,15]")

    s = Stats()
    xs = materialize[BESSEL_J1_X]()
    refs = materialize[BESSEL_J1_REF]()
    for i in range(BESSEL_J1_N):
        s.observe(xs[i], j1(p(xs[i])).v[0], refs[i])
    s.report("j1, [-15,15]")

    s = Stats()
    xs = materialize[BESSEL_Y0_X]()
    refs = materialize[BESSEL_Y0_REF]()
    for i in range(BESSEL_Y0_N):
        s.observe(xs[i], y0(p(xs[i])).v[0], refs[i])
    s.report("y0, [0.1,15]")

    s = Stats()
    xs = materialize[BESSEL_Y1_X]()
    refs = materialize[BESSEL_Y1_REF]()
    for i in range(BESSEL_Y1_N):
        s.observe(xs[i], y1(p(xs[i])).v[0], refs[i])
    s.report("y1 (fitted near branch), [0.1,15]")

    # --- Lambert W ----------------------------------------------------
    section("Lambert W")
    s = Stats()
    xs = materialize[LAMBERTW_0_X]()
    refs = materialize[LAMBERTW_0_REF]()
    for i in range(LAMBERTW_0_N):
        s.observe(xs[i], lambertw(p(xs[i])).v[0], refs[i])
    s.report("lambertw (20 Halley), [-1/e,20]")

    s = Stats()
    xs = materialize[LAMBERTW_M1_X]()
    refs = materialize[LAMBERTW_M1_REF]()
    for i in range(LAMBERTW_M1_N):
        s.observe(xs[i], lambertw_m1(p(xs[i])).v[0], refs[i])
    s.report("lambertw_m1 (blended seed), [-1/e,-1e-12]")

    # --- Elliptic -----------------------------------------------------
    section("Elliptic integrals")
    s = Stats()
    xs = materialize[ELLIPTIC_K_X]()
    refs = materialize[ELLIPTIC_K_REF]()
    for i in range(ELLIPTIC_K_N):
        s.observe(xs[i], elliptic_k(p(xs[i])).v[0], refs[i])
    s.report("elliptic_k (A&S 17.3.34), m in [0,0.999]")

    s = Stats()
    xs = materialize[ELLIPTIC_E_X]()
    refs = materialize[ELLIPTIC_E_REF]()
    for i in range(ELLIPTIC_E_N):
        s.observe(xs[i], elliptic_e(p(xs[i])).v[0], refs[i])
    s.report("elliptic_e (A&S 17.3.36), m in [0,1]")

    # --- Incomplete gamma, at four shape parameters -------------------
    section("Incomplete gamma")
    s = Stats()
    xs = materialize[GAMMAINC_A0P5_X]()
    refs = materialize[GAMMAINC_A0P5_REF]()
    for i in range(GAMMAINC_A0P5_N):
        s.observe(xs[i], gammainc(p(0.5), p(xs[i])).v[0], refs[i])
    s.report("gammainc a=0.5 (100-term series), x in [1e-3,30]")

    s = Stats()
    xs = materialize[GAMMAINC_A1P0_X]()
    refs = materialize[GAMMAINC_A1P0_REF]()
    for i in range(GAMMAINC_A1P0_N):
        s.observe(xs[i], gammainc(p(1.0), p(xs[i])).v[0], refs[i])
    s.report("gammainc a=1.0, x in [1e-3,30]")

    s = Stats()
    xs = materialize[GAMMAINC_A2P5_X]()
    refs = materialize[GAMMAINC_A2P5_REF]()
    for i in range(GAMMAINC_A2P5_N):
        s.observe(xs[i], gammainc(p(2.5), p(xs[i])).v[0], refs[i])
    s.report("gammainc a=2.5, x in [1e-3,30]")

    s = Stats()
    xs = materialize[GAMMAINC_A10P0_X]()
    refs = materialize[GAMMAINC_A10P0_REF]()
    for i in range(GAMMAINC_A10P0_N):
        s.observe(xs[i], gammainc(p(10.0), p(xs[i])).v[0], refs[i])
    s.report("gammainc a=10.0, x in [1e-3,30]")

    # --- Incomplete beta, at four parameter pairs ---------------------
    section("Incomplete beta")
    s = Stats()
    xs = materialize[BETAINC_0P5_0P5_X]()
    refs = materialize[BETAINC_0P5_0P5_REF]()
    for i in range(BETAINC_0P5_0P5_N):
        s.observe(xs[i], betainc(p(xs[i]), p(0.5), p(0.5)).v[0], refs[i])
    s.report("betainc a=b=0.5 (100-iter Lentz), x in [0.001,0.999]")

    s = Stats()
    xs = materialize[BETAINC_2P0_3P0_X]()
    refs = materialize[BETAINC_2P0_3P0_REF]()
    for i in range(BETAINC_2P0_3P0_N):
        s.observe(xs[i], betainc(p(xs[i]), p(2.0), p(3.0)).v[0], refs[i])
    s.report("betainc a=2,b=3, x in [0.001,0.999]")

    s = Stats()
    xs = materialize[BETAINC_5P0_1P5_X]()
    refs = materialize[BETAINC_5P0_1P5_REF]()
    for i in range(BETAINC_5P0_1P5_N):
        s.observe(xs[i], betainc(p(xs[i]), p(5.0), p(1.5)).v[0], refs[i])
    s.report("betainc a=5,b=1.5, x in [0.001,0.999]")

    s = Stats()
    xs = materialize[BETAINC_20P0_20P0_X]()
    refs = materialize[BETAINC_20P0_20P0_REF]()
    for i in range(BETAINC_20P0_20P0_N):
        s.observe(xs[i], betainc(p(xs[i]), p(20.0), p(20.0)).v[0], refs[i])
    s.report("betainc a=b=20, x in [0.001,0.999]")

    # --- Orthogonal polynomials (exact recurrences) -------------------
    section("Orthogonal polynomials -- exact recurrences, no approximation")
    s = Stats()
    xs = materialize[LEGENDRE_P4_X]()
    refs = materialize[LEGENDRE_P4_REF]()
    for i in range(LEGENDRE_P4_N):
        s.observe(xs[i], legendre_p(4, p(xs[i])).v[0], refs[i])
    s.report("legendre_p n=4 (Bonnet), [-1,1]")

    s = Stats()
    xs = materialize[LEGENDRE_P12_X]()
    refs = materialize[LEGENDRE_P12_REF]()
    for i in range(LEGENDRE_P12_N):
        s.observe(xs[i], legendre_p(12, p(xs[i])).v[0], refs[i])
    s.report("legendre_p n=12, [-1,1]")

    s = Stats()
    xs = materialize[CHEBYSHEV_T4_X]()
    refs = materialize[CHEBYSHEV_T4_REF]()
    for i in range(CHEBYSHEV_T4_N):
        s.observe(xs[i], chebyshev_t(4, p(xs[i])).v[0], refs[i])
    s.report("chebyshev_t n=4, [-1,1]")

    s = Stats()
    xs = materialize[CHEBYSHEV_T10_X]()
    refs = materialize[CHEBYSHEV_T10_REF]()
    for i in range(CHEBYSHEV_T10_N):
        s.observe(xs[i], chebyshev_t(10, p(xs[i])).v[0], refs[i])
    s.report("chebyshev_t n=10, [-1,1]")

    s = Stats()
    xs = materialize[HERMITE_H6_X]()
    refs = materialize[HERMITE_H6_REF]()
    for i in range(HERMITE_H6_N):
        s.observe(xs[i], hermite_h(6, p(xs[i])).v[0], refs[i])
    s.report("hermite_h n=6, [-3,3]")

    s = Stats()
    xs = materialize[LAGUERRE_L6_X]()
    refs = materialize[LAGUERRE_L6_REF]()
    for i in range(LAGUERRE_L6_N):
        s.observe(xs[i], laguerre_l(6, p(xs[i])).v[0], refs[i])
    s.report("laguerre_l n=6, [0,12]")
