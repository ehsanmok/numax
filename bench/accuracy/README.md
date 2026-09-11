# Accuracy

Every approximation in `numax` documents an error bound in its module
docstring. Those bounds came from the literature each formula was taken
from, not from this implementation of it, so nothing checked that the
transcription, the branchless blends, and the fixed iteration counts
actually deliver them. This does.

```
pixi run accuracy
```

References are [mpmath](https://mpmath.org/) at 50 decimal digits, rounded
once to float64 and checked in as `reference_data.mojo`, so the harness
needs only the default Mojo environment. Regenerate after changing a
domain:

```
pixi run -e bench-python accuracy-gen
```

Everything is measured at `Plain[float64, 1]`, to see the approximation's
own error rather than the rounding of a narrower `dtype`. Numbers below are
from an Apple M3 Pro; they are deterministic, so any machine running the
same Mojo version should reproduce them exactly. Checked: an x86_64 Linux
run reproduces them to the last digit, ULP counts included (`erf` over
[-3,3] gives 2.1674617767253324e-08 and 195,227,601 ULP on both), so these
really are properties of the approximations and not of one machine's libm.

## Read this first: the primitives are a floor, not context

This harness found that Mojo's `std.math` `exp`, `log`, and `erf` are not
correctly rounded at float64 — `exp(1.0)` returns `2.718281828459813`
against a true `2.718281828459045`, wrong from the 13th significant digit.
`sin` and `sqrt` are exact to the last bit, so this is specific to those
three rather than a property of `std.math` generally.

| primitive | max abs | max rel | max ULP |
| --- | --- | --- | --- |
| `Plain.exp`, [-10, 10] | 8.71e-08 | 3.95e-12 | 33,553 |
| `Plain.ln`, [1e-6, 1e6] | 8.01e-10 | 2.12e-09 | 11,217,732 |
| `Plain.sin`, [-8, 8] | 1.11e-16 | 1.55e-16 | 1 |
| `Plain.sqrt`, [1e-8, 1e8] | 0 | 0 | 0 |

`gamma`, `lgamma`, `lambertw`, the Bessel family, the elliptic integrals,
and the incomplete gamma and beta functions are all built on `exp` and
`ln`. Their measured errors sit right at what `ln` alone contributes, so
those rows are mostly an inherited floor rather than each algorithm's own
error. Their true accuracy is somewhere at or below what is printed, and
this harness cannot separate the two without a correctly-rounded `exp`/`ln`
to measure against.

None of this is visible at the `dtype` `numax` is normally used at: float32
epsilon is about 1.2e-7, two orders of magnitude above the worst of these.
It only becomes the limiting factor at float64, which is where the harness
runs precisely in order to see the algorithms at all.

## Which column to read

No single metric is honest for every function here.

- **ULP** is right for anything that should be nearly exact — `sqrt`,
  `sin`, and the polynomial recurrences. It is useless for a deliberately
  low-order approximation, where it reads in the billions and says only
  "not bit-exact", which was never the claim.
- **Max relative error** is right for those approximations, and is what
  their docstrings should be checked against — except where the true value
  passes through zero, which makes relative error meaningless through no
  fault of the implementation.
- **Max absolute error** covers that case, and is how most of the
  literature's own bounds are quoted.

## Results

### Error function

| function / domain | max abs | max rel | max ULP |
| --- | --- | --- | --- |
| `erf` (Plain → `std.math`), [-3, 3] | 2.17e-08 | 2.18e-08 | 195,227,601 |
| `erf` (Plain → `std.math`), [1e-8, 1e-2] | 1.59e-10 | 1.43e-08 | 127,720,638 |
| `default_erf_approx` (A&S 7.1.26), [-3, 3] | 1.38e-07 | 4.35e-07 | 3,530,608,388 |
| `erfc` (Plain → `std.math`), [1, 6] | 2.78e-17 | 5.69e-16 | 4 |
| `erfinv` (guess + 3 Newton), [-0.999, 0.999] | 4.77e-07 | 2.61e-07 | 2,147,351,100 |
| `erfinv` (guess + 3 Newton), 1 - [1e-12, 1e-1] | 1.19e-06 | 5.84e-07 | 3,362,763,826 |

`erfinv`'s two rows are `erf`'s floor seen through a derivative. Three
Newton steps against `erf`/`erfc` converge to the root of the *library's*
`erf`, so the result inherits `std.math.erf`'s 2.2e-08 absolute error
multiplied by `d erfinv/dy = sqrt(pi)/2 * exp(x^2)`: 24 at `y = 0.99`
(giving the 4.77e-07) and 59 at the `w = 5` region split near `y = 0.996`
(the 1.19e-06, the worst point of the tail sweep). Past the split the
residual is read off `erfc`, whose 4-ULP relative accuracy makes the deep
tail -- `y` within 1e-12 of 1 -- accurate to about 1e-16, so the tail row's
maximum sits at its *near* end. A correctly rounded `erf` would collapse both
rows to that level with no change here.

Two things worth pulling out. `default_erf_approx`'s 1.38e-07 confirms the
~1.5e-7 bound A&S 7.1.26 documents, so that transcription is correct — this
is the approximation `Compensated` and `Decimal` fall back to. And
`std.math.erf` beats it by about 6x on the same domain, which is the
measured version of the argument for promoting `erf` to a trait method so
`Plain` could delegate.

### Combinatorics

| function / domain | max abs | max rel | max ULP |
| --- | --- | --- | --- |
| `factorial` (`gamma(n+1)`), [0.5, 20] | 1.10e+09 | 1.17e-08 | 104,856,286 |
| `comb(n, 3)` (`exp` of `lgamma` differences), n in [3, 60] | 3.28e-04 | 2.92e-08 | 180,535,060 |
| `poch(z, 2.5)` (`exp` of `lgamma` differences), z in [0.5, 20] | 1.33e-05 | 1.22e-08 | 95,236,033 |

Read the relative column. All three are `exp` of one or two `lgamma`s, so
they inherit `lgamma`'s floor (itself `ln`'s, below) multiplied out by the
exponential: the 1.1e+09 absolute error on `factorial` is `20!`'s 2.4e+18
magnitude times a 1.2e-08 relative error, not a defect of its own. The
`lgamma` differences in `comb` and `poch` cancel most of the shared
magnitude before the `exp`, which is why they land at the same relative
level as `factorial` rather than above it.

### Exponential and trigonometric integrals

| function / domain | max abs | max rel | max ULP |
| --- | --- | --- | --- |
| `exp1` (series \| fraction at 1.5), [0.05, 30] | 6.09e-10 | 2.94e-09 | 14,126,675 |
| `expi` (series \| asymptotic at 40), [0.05, 60] | 4.75e+13 | 2.70e-09 | 13,179,031 |
| `expi` (`-exp1(-x)`), -[0.05, 30] | 6.09e-10 | 2.94e-09 | 14,126,675 |
| `Si` (series \| `E1(ix)` fraction at 2), [0.05, 60] | 4.44e-16 | 1.39e-15 | 10 |
| `Ci` (series \| `E1(ix)` fraction at 2), [0.05, 60] | 6.84e-10 | 4.66e-09 | 28,390,179 |
| Fresnel `S` (series \| `erfc` fraction at 2), [0.01, 10] | 1.22e-15 | 3.09e-14 | 153 |
| Fresnel `C` (series \| `erfc` fraction at 2), [0.01, 10] | 1.33e-15 | 1.02e-14 | 59 |

Two rows at `1e-15` and four at `3e-9`, and the split is exactly which
functions take a logarithm. `Si` and the Fresnel pair are series and
continued fractions in the four arithmetic operations and `exp`, and land
at the algorithms' own error. `exp1`, `expi` and `Ci` carry `gamma + ln x`
in their series region, and their worst points (`x = 0.66`, `x = 1.38`)
sit there, at `std.math.log`'s error for those arguments -- `log(0.1)` is
off by `1.85e-10` on its own. `expi`'s 4.75e+13 absolute is `Ei(60) ~
2e+24` times the same `2.7e-9` relative, inherited from the `ln` in the
series the asymptotic sum takes over from at `x = 40`. The
`expn(n, x)` family shares `exp1`'s route and floor.

### Zeta and hypergeometric

| function / domain | max abs | max rel | max ULP |
| --- | --- | --- | --- |
| `zeta` (Euler-Maclaurin, N=10, M=8), [1.1, 30] | 9.97e-12 | 9.42e-13 | 5,611 |
| `zeta` (Euler-Maclaurin, N=10, M=8), [-2.5, 0.9] | 5.24e-10 | 2.07e-07 | 1,077,665,923 |
| `hyp1f1(1.5, 2.5, x)` (series \| Kummer), [-40, 40] | 1.10e+01 | 1.64e-11 | 110,721 |
| `hyp2f1(0.5, 1.5, 2.5, x)` (series \| Pfaff), [-5, 0.9] | 2.68e-10 | 4.94e-10 | 2,415,788 |

The two `zeta` rows are one formula measured on two sides of a
cancellation. Above the pole the answer is the sum's size and the error is
`exp`'s floor on each `k^{-s} = exp(-s ln k)` term, with the logarithms
taken from an exact compile-time table -- taking them from `std.math.log`
instead read `3.3e-10` here and `3.9e-4` below, which is what the table
buys. Below the pole `zeta(-2.5) = 0.0085` is a difference of terms near
`300`, so the same `4e-12` relative floor on each term becomes `2e-7`
relative on the answer; the formula itself is at `1e-18` in mpmath. The
`hyp1f1` row's 11.0 absolute is `1.6e-11` relative on an answer of
`7e+11` at `x = 40`; `hyp2f1`'s worst point sits in the Pfaff region,
where `(1 - x)^{-a}` is `exp(-a ln(1 - x))` and `ln`'s floor comes back.

### Gamma family

| function / domain | max abs | max rel | max ULP |
| --- | --- | --- | --- |
| `gamma` (9-term Lanczos), [0.5, 8] | 6.56e-07 | 4.05e-09 | 31,604,283 |
| `gamma` (reflected), (-5, 0) off poles | 9.80e-09 | 3.86e-09 | 34,225,423 |
| `lgamma`, [0.5, 40] | 3.35e-08 | 3.36e-09 | 16,844,630 |
| `digamma` (`Dual` of `lgamma`), [0.2, 12] | 6.06e-10 | 4.19e-10 | 2,726,969 |
| `digamma` (reflected), (-5, 0) off poles | 9.51e-10 | 2.85e-08 | 128,295,437 |

`gamma`'s 6.56e-07 absolute is the wrong column to read: it occurs at
`x = 7.70`, where `gamma` is about 2075, so the relative error is 4e-09.
The reflected rows confirm that extending these three functions through
`Gamma(x)Gamma(1-x) = pi/sin(pi x)` costs no accuracy against the direct
branch.

### Bessel

| function / domain | max abs | max rel | max ULP |
| --- | --- | --- | --- |
| `j0` (A&S 9.4.1/9.4.3), [-15, 15] | 3.90e-08 | 8.50e-07 | 5,615,164,118 |
| `j1`, [-15, 15] | 7.63e-09 | 3.08e-07 | 2,477,114,337 |
| `y0`, [0.1, 15] | 2.10e-08 | 1.35e-06 | 7,509,339,843 |
| `y1` (fitted near branch), [0.1, 15] | 1.61e-08 | 4.45e-07 | 3,819,678,784 |

Read the absolute column here. All four functions oscillate through zero
repeatedly on these domains, so relative error blows up near each root
regardless of implementation quality — which is exactly why A&S quotes
these bounds in absolute terms. The absolute numbers confirm them, across
both the near and far branches and their blend.

`y1`'s near-branch polynomial is the one coefficient set in `numax`
fit from scratch rather than transcribed, and at 1.61e-08 it is the *most*
accurate of the four.

### Lambert W

| function / domain | max abs | max rel | max ULP |
| --- | --- | --- | --- |
| `lambertw` (20 Halley), [-1/e, 20] | 3.83e-09 | 3.83e-09 | 34,492,154 |
| `lambertw_m1` (blended seed), [-1/e, -1e-12] | 3.83e-09 | 3.83e-09 | 17,243,253 |

Both branches hit their worst case at the branch point `-1/e` itself, where
the function has infinite derivative and the seed is at its least accurate.
That the two agree to three digits there, from completely different seeds,
is the evidence that `lambertw_m1`'s empirically chosen blend threshold is
in the right place.

### Elliptic integrals

| function / domain | max abs | max rel | max ULP |
| --- | --- | --- | --- |
| `elliptic_k` (A&S 17.3.34), m in [0, 0.999] | 1.36e-08 | 8.42e-09 | 61,135,317 |
| `elliptic_e` (A&S 17.3.36), m in [0, 1] | 1.57e-08 | 1.52e-08 | 70,859,860 |

`elliptic_e`'s bound matters more than most rows here, because its `b4`
coefficient is misdigitized in every OCR'd copy of the A&S table found
while implementing it and had to be recovered by fitting against an
independent AGM reference. A wrong `b4` produced ~3e-04 error; 1.57e-08 is
consistent with the documented ~2e-08, so the recovered value is right.

### Incomplete gamma

| function / domain | max abs | max rel | max ULP |
| --- | --- | --- | --- |
| `gammainc` a=0.5 (100-term series), x in [1e-3, 30] | 5.16e-10 | 5.71e-10 | 4,649,911 |
| `gammainc` a=1.0 | 7.56e-10 | 1.01e-09 | 6,807,862 |
| `gammainc` a=2.5 | 8.97e-10 | 2.60e-09 | 12,505,441 |
| `gammainc` a=10.0 | 2.44e-09 | 9.59e-09 | 58,324,960 |

The fixed 100-term series holds up across two orders of magnitude in `x`
and a factor of 20 in the shape parameter, degrading gently rather than
falling off a cliff. This is the function every gamma-family CDF in
`numax.stats` routes through, plus `poisson_cdf`.

### Incomplete beta

| function / domain | max abs | max rel | max ULP |
| --- | --- | --- | --- |
| `betainc` a=b=0.5 (100-iter Lentz), x in [0.001, 0.999] | 1.03e-10 | 6.40e-10 | 4,341,821 |
| `betainc` a=2, b=3 | 8.90e-10 | 3.56e-09 | 23,244,236 |
| `betainc` a=5, b=1.5 | 3.02e-09 | 9.74e-09 | 54,429,486 |
| `betainc` a=b=20 | 8.84e-09 | 2.79e-08 | 260,782,525 |

Accuracy degrades monotonically with parameter magnitude, losing about an
order of magnitude from `a=b=0.5` to `a=b=20`. That is consistent with two
different causes — the continued fraction needing more than its fixed 100
iterations as `a+b` grows, or just the `exp`/`ln` floor compounding through
a longer computation — and no attempt has been made to distinguish them.
Worth knowing before relying on `betainc` at large parameters, which
includes `binomial_cdf` and `f_cdf` at large sample sizes.

### Orthogonal polynomials

| function / domain | max abs | max rel | max ULP |
| --- | --- | --- | --- |
| `legendre_p` n=4 (Bonnet), [-1, 1] | 5.55e-16 | 3.17e-14 | 253 |
| `legendre_p` n=12 | 4.00e-15 | 2.58e-14 | 194 |
| `chebyshev_t` n=4 | 2.22e-16 | 2.91e-15 | 14 |
| `chebyshev_t` n=10 | 6.11e-16 | 6.44e-15 | 31 |
| `hermite_h` n=6, [-3, 3] | 3.64e-12 | 3.48e-16 | 3 |
| `laguerre_l` n=6, [0, 12] | 1.42e-14 | 6.21e-15 | 55 |

These are the only functions in `numax` with no approximation error to
document — the three-term recurrences evaluate the polynomial exactly, and
what is left is rounding. They are also the only rows here not built on
`exp`/`ln`, which is why they are the only ones anywhere near float64
resolution. Accuracy holding from n=4 to n=12 is the useful part: the
recurrences are numerically stable in the direction they are being run.

## What this does not cover

- **Only `Plain[float64, 1]`.** `Compensated` should do better on
  cancellation-heavy inputs and `Decimal` worse (it floors rather than
  rounds); neither is measured here. The per-conformer differences are
  covered by identity-based tests in `tests/` instead.
- **Only one dimension per function.** The two-argument functions are
  sampled at four fixed parameter values each, not over a 2D grid.
- **Nothing composed.** `numax.stats`, `numax.linalg`,
  `numax.integrate`, and `numax.fft` are validated against closed-form identities
  in `tests/` rather than against a reference table, since for most of them
  a closed-form identity is the stronger check.
