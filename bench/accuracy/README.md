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
same Mojo version should reproduce them exactly, ULP counts included --
`numax` supplies its own `exp`, `ln` and `erf` (below), so no row depends
on a platform libm.

## Read this first: the primitives

The primitives are printed first because everything below is built on
them. At float64, `Plain`'s `exp`, `ln` and `erf` are numax's own --
[`numax/core/libm.mojo`](../../numax/core/libm.mojo), Sun's fdlibm
algorithms transcribed to SIMD with masks in place of branches -- and
`erfc`, `sin` and `sqrt` are `std.math`'s. All of them read at one ulp:

| primitive | max abs | max rel | max ULP |
| --- | --- | --- | --- |
| `Plain.exp`, [-10, 10] | 5.55e-17 | 1.51e-16 | 1 |
| `Plain.ln`, [1e-6, 1e6] | 0 | 0 | 0 |
| `Plain.sin`, [-8, 8] | 1.11e-16 | 1.55e-16 | 1 |
| `Plain.sqrt`, [1e-8, 1e8] | 0 | 0 | 0 |

This harness is what found the need for `libm`. At the pinned release,
Mojo's `std.math` `exp`, `log` and `erf` are not correctly rounded at
float64: `exp(1.0)` came back as `2.718281828459813` against a true
`2.718281828459045`, wrong from the 13th significant digit. Measured on a
grid, `exp` was off by 105,000 ULP (`1.2e-11` relative at `|x| = 30`; it
reduces with `ln 2` rounded to float32, so the error grows with `x`),
`log` by 9,300,000 ULP (an *absolute* floor of about `2e-10`, which is
`8e-10` relative on `ln 0.8`), and `erf` by 196,000,000 ULP (`2.3e-8`,
float32 polynomial coefficients), while `erfc`, `sin`, `cos` and `sqrt`
were within 3 ULP. Every row in this file above `1e-13` was one of those
three seen through an algorithm that was itself at `1e-14`: `erf` read
`2.2e-8`, `lgamma` `3.4e-9`, the incomplete gamma and beta `1e-8`,
`erfinv` `6e-7`. Replacing the three collapsed all of it in one run, and
the rows now read the algorithms. The A&S polynomial families (`j0`..`y1`,
the elliptic integrals, `default_erf_approx`) did not move, because their
error is the polynomial's and that is the point of measuring them.

## Which column to read

No single metric is honest for every function here.

- **ULP** is right for anything that should be nearly exact -- `sqrt`,
  `sin`, the polynomial recurrences, and now most of the special functions.
  It is useless for a deliberately low-order approximation, where it reads
  in the billions and says only "not bit-exact", which was never the claim.
- **Max relative error** is right for those approximations, and is what
  their docstrings should be checked against -- except where the true value
  passes through zero, which makes relative error meaningless through no
  fault of the implementation.
- **Max absolute error** covers that case, and is how most of the
  literature's own bounds are quoted.

## Results

### Error function

| function / domain | max abs | max rel | max ULP |
| --- | --- | --- | --- |
| `erf` (Plain, `libm`), [-3, 3] | 1.11e-16 | 2.14e-16 | 1 |
| `erf` (Plain, `libm`), [1e-8, 1e-2] | 4.34e-19 | 2.10e-16 | 1 |
| `default_erf_approx` (A&S 7.1.26), [-3, 3] | 1.38e-07 | 4.35e-07 | 3,530,608,388 |
| `erfc` (Plain → `std.math`), [1, 6] | 2.78e-17 | 5.69e-16 | 4 |
| `erfinv` (guess + 3 Newton), [-0.999, 0.999] | 4.44e-16 | 1.41e-15 | 12 |
| `erfinv` (guess + 3 Newton), 1 - [1e-12, 1e-1] | 1.78e-15 | 8.71e-16 | 6 |

`erfinv` is three Newton steps against the library's own `erf`/`erfc`, so
its rows are `erf`'s accuracy seen through the derivative `sqrt(pi)/2 *
exp(x^2)`, which is 24 at `y = 0.99` and 59 at the region split near `y =
0.996`: one ulp in, twelve out. Before `libm` the same two rows read
`4.8e-7` and `1.2e-6`, `std.math.erf`'s `2.2e-8` multiplied by those same
factors, which is how the mechanism was confirmed. `default_erf_approx`'s
1.38e-07 matches the ~1.5e-7 bound A&S 7.1.26 documents, so that
transcription is correct; it remains the fallback for conformers with no
better `erf` of their own (`Compensated`, `Decimal`).

### Combinatorics

| function / domain | max abs | max rel | max ULP |
| --- | --- | --- | --- |
| `factorial` (`gamma(n+1)`), [0.5, 20] | 3.05e+04 | 1.45e-14 | 119 |
| `comb(n, 3)` (`exp` of `lgamma` differences), n in [3, 60] | 8.11e-10 | 2.90e-14 | 256 |
| `poch(z, 2.5)` (`exp` of `lgamma` differences), z in [0.5, 20] | 2.09e-11 | 1.43e-14 | 111 |

Read the relative column. All three are `exp` of one or two `lgamma`s, so
they carry `lgamma`'s `2.8e-14` (below) through an exponential: the
3.05e+04 absolute on `factorial` is `20!`'s `2.4e+18` times `1.3e-14`, not
a defect of its own.

### Exponential and trigonometric integrals

| function / domain | max abs | max rel | max ULP |
| --- | --- | --- | --- |
| `exp1` (series \| fraction at 1.5), [0.05, 30] | 4.44e-16 | 1.11e-15 | 8 |
| `expi` (series \| asymptotic at 40), [0.05, 60] | 2.68e+08 | 4.76e-15 | 32 |
| `expi` (`-exp1(-x)`), -[0.05, 30] | 4.44e-16 | 1.11e-15 | 8 |
| `Si` (series \| `E1(ix)` fraction at 2), [0.05, 60] | 2.22e-16 | 2.02e-16 | 1 |
| `Ci` (series \| `E1(ix)` fraction at 2), [0.05, 60] | 3.33e-16 | 8.71e-16 | 6 |
| Fresnel `S` (series \| `erfc` fraction at 2), [0.01, 10] | 1.22e-15 | 2.45e-15 | 21 |
| Fresnel `C` (series \| `erfc` fraction at 2), [0.01, 10] | 1.33e-15 | 3.56e-15 | 21 |

All at the rounding floor. `expi`'s 2.68e+08 absolute is `Ei(60) ~ 2e+24`
at `4.8e-15` relative, the series' 120 terms of rounding. Before `libm`
the four functions that carry `gamma + ln x` in their series region
(`exp1`, `expi`, `Ci`, and `expn`, which shares `exp1`'s route) read
`3e-9` while `Si` and the Fresnel pair read `1e-15`; the split was exactly
which functions took a logarithm.

### Bessel of real order, Airy, Struve, Owen

| function / domain | max abs | max rel | max ULP |
| --- | --- | --- | --- |
| `jv(2.5, x)` (Temme, CF depth 260), [0.01, 150] | 1.11e-16 | 2.72e-15 | 19 |
| `yv(2.5, x)` (Temme series \| Steed CF2 at 2), [0.01, 150] | 2.91e-11 | 3.51e-13 | 1,851 |
| `ive(2.5, x)` (scaled, Wronskian with `kve`), [0.01, 150] | 4.16e-17 | 1.07e-15 | 6 |
| `kve(2.5, x)` (Temme series \| CF2 at 2), [0.01, 150] | 5.82e-11 | 1.22e-15 | 6 |
| `airy` `Ai` (`J`,`Y` \| series \| `I`,`K` at 1.5), [-40, 8] | 4.22e-15 | 3.54e-13 | 2,218 |
| `airy` `Bi` (`J`,`Y` \| series \| `I`,`K` at 1.5), [-40, 8] | 2.79e-09 | 1.70e-12 | 14,120 |
| `struve(1, x)` (Bessel series \| asymptotic at 40), [0.01, 100] | 6.66e-16 | 1.31e-15 | 8 |
| `owens_t(h, 0.7)` (GL64 \| GL64 in `u` at 2), h in [0, 6] | 2.78e-17 | 1.45e-15 | 12 |

Every algorithm in this block was transcribed from a Python prototype
that agreed with mpmath to `1e-14` or better over the same grids, and the
rows now sit where the prototype did. The two above `1e-14`, `yv` and
`Ai` at `3.5e-13`, both pass through Temme's `Y` series and the thirty
held-or-taken recurrence steps, the longest arithmetic chain in the
block; `Bi`'s `1.7e-12` is the `e^{zeta}` scaling of a sum of two `I`s at
`x = 7.9`, where `Bi` is `1.6e6`. The large absolute columns are large
values: `yv`'s and `kve`'s are `Y_{2.5}(0.01) = -1.1e7` and `K_{2.5}(0.01)
e^{0.01} = 1.6e5` at rounding. Before `libm` these eight rows read
`1.5e-10` to `1.6e-9`, all `std.math.log`'s absolute floor seen through
Temme's `d = -ln(x/2)`.

### Zeta and hypergeometric

| function / domain | max abs | max rel | max ULP |
| --- | --- | --- | --- |
| `zeta` (Euler-Maclaurin, N=10, M=8), [1.1, 30] | 1.78e-15 | 5.99e-16 | 3 |
| `zeta` (Euler-Maclaurin, N=10, M=8), [-2.5, 0.9] | 4.25e-13 | 8.26e-11 | 467,677 |
| `hyp1f1(1.5, 2.5, x)` (series \| Kummer), [-40, 40] | 1.10e+01 | 2.06e-15 | 11 |
| `hyp2f1(0.5, 1.5, 2.5, x)` (series \| Pfaff), [-5, 0.9] | 1.78e-15 | 1.26e-15 | 8 |

The two `zeta` rows are one formula measured on two sides of a
cancellation. Above the pole the answer is the sum's size and the error is
rounding. Below it, `zeta(-2.5) = 0.0085` is a difference of terms near
`300`, so half an ulp on each term is already `1e-13` of the answer and
ten such terms give the `8e-11`; the formula itself is at `1e-18` in
mpmath. The `hyp1f1` row's 11.0 absolute is `2e-15` relative on an answer
of `7e+11` at `x = 40`.

### Gamma family

| function / domain | max abs | max rel | max ULP |
| --- | --- | --- | --- |
| `gamma` (9-term Lanczos), [0.5, 8] | 1.73e-11 | 3.99e-15 | 22 |
| `gamma` (reflected), (-5, 0) off poles | 3.91e-14 | 5.60e-15 | 38 |
| `lgamma`, [0.5, 40] | 2.84e-14 | 2.84e-14 | 244 |
| `digamma` (`Dual` of `lgamma`), [0.2, 12] | 1.78e-15 | 2.60e-14 | 233 |
| `digamma` (reflected), (-5, 0) off poles | 9.06e-14 | 5.39e-14 | 243 |

The nine-term Lanczos sum is good to nearly full float64 in exact
arithmetic, and now measures that way: `gamma`'s 1.73e-11 absolute occurs
at `x = 7.9` where `gamma` is about 3300, so it is `4e-15` relative.
`lgamma` and `digamma` both have zeros inside these domains (`x = 1` and
`2`; `x = 1.46`), and their relative columns are the absolute error
divided by a value near zero there -- read the absolute columns, which
are `3e-14` on values of order `100`, two ulp. The reflected rows confirm
that extending these through `Gamma(x)Gamma(1-x) = pi/sin(pi x)` costs no
accuracy against the direct branch.

### Bessel

| function / domain | max abs | max rel | max ULP |
| --- | --- | --- | --- |
| `j0` (A&S 9.4.1/9.4.3), [-15, 15] | 3.90e-08 | 8.50e-07 | 5,615,164,118 |
| `j1`, [-15, 15] | 7.63e-09 | 3.08e-07 | 2,477,114,116 |
| `y0`, [0.1, 15] | 2.10e-08 | 1.35e-06 | 7,509,340,252 |
| `y1` (fitted near branch), [0.1, 15] | 1.61e-08 | 4.45e-07 | 3,819,678,784 |

Read the absolute column here. All four functions oscillate through zero
repeatedly on these domains, so relative error blows up near each root
regardless of implementation quality -- which is exactly why A&S quotes
these bounds in absolute terms. The absolute numbers confirm them, across
both the near and far branches and their blend. These are the polynomials'
own error, which is why the `libm` change did not move them; `jv(0, x)`
and `jv(1, x)` above are the full-precision route to the same functions.

`y1`'s near-branch polynomial is the one coefficient set in `numax`
fit from scratch rather than transcribed, and at 1.61e-08 it is the *most*
accurate of the four.

### Lambert W

| function / domain | max abs | max rel | max ULP |
| --- | --- | --- | --- |
| `lambertw` (20 Halley), [-1/e, 20] | 7.17e-13 | 7.17e-13 | 6,457 |
| `lambertw_m1` (blended seed), [-1/e, -1e-12] | 3.87e-13 | 3.87e-13 | 1,741 |

Both branches hit their worst case at the branch point `-1/e` itself, where
the function has infinite derivative and the seed is at its least accurate.
That the two agree there, from completely different seeds, is the
evidence that `lambertw_m1`'s empirically chosen blend threshold is in the
right place; before `libm` both read `3.8e-9`, the `ln` in the seed.

### Elliptic integrals

| function / domain | max abs | max rel | max ULP |
| --- | --- | --- | --- |
| `elliptic_k` (A&S 17.3.34), m in [0, 0.999] | 1.34e-08 | 8.42e-09 | 60,526,842 |
| `elliptic_e` (A&S 17.3.36), m in [0, 1] | 1.57e-08 | 1.52e-08 | 70,705,268 |

`elliptic_e`'s bound matters more than most rows here, because its `b4`
coefficient is misdigitized in every OCR'd copy of the A&S table found
while implementing it and had to be recovered by fitting against an
independent AGM reference. A wrong `b4` produced ~3e-04 error; 1.57e-08 is
consistent with the documented ~2e-08, so the recovered value is right.

### Incomplete gamma

| function / domain | max abs | max rel | max ULP |
| --- | --- | --- | --- |
| `gammainc` a=0.5 (100-term series), x in [1e-3, 30] | 1.33e-15 | 1.33e-15 | 12 |
| `gammainc` a=1.0 | 2.78e-15 | 2.78e-15 | 25 |
| `gammainc` a=2.5 | 2.33e-15 | 2.56e-15 | 21 |
| `gammainc` a=10.0 | 4.88e-15 | 6.20e-15 | 71 |

The fixed 100-term series holds up across two orders of magnitude in `x`
and a factor of 20 in the shape parameter, at rounding throughout. This is
the function every gamma-family CDF in `numax.stats` routes through, plus
`poisson_cdf`.

### Incomplete beta

| function / domain | max abs | max rel | max ULP |
| --- | --- | --- | --- |
| `betainc` a=b=0.5 (100-iter Lentz), x in [0.001, 0.999] | 1.44e-15 | 2.44e-15 | 20 |
| `betainc` a=2, b=3 | 1.33e-15 | 3.34e-15 | 24 |
| `betainc` a=5, b=1.5 | 2.11e-15 | 6.80e-15 | 38 |
| `betainc` a=b=20 | 9.55e-15 | 2.47e-14 | 174 |

Before `libm` these four rows read `6.4e-10` to `2.8e-8`, rising with
the parameters, and it was an open question whether the fixed 100 Lentz
iterations were running out or the `exp`/`ln` floor was compounding
through a longer computation. The fraction was not touched and the rows
fell to rounding, so it was the floor. A slope remains -- a factor of ten
from `a=b=0.5` to `a=b=20` -- and at `2.5e-14` it is not worth separating
the prefactor's three `lgamma`s from the longer fraction. `binomial_cdf`
and `f_cdf` route through it.

### Orthogonal polynomials

| function / domain | max abs | max rel | max ULP |
| --- | --- | --- | --- |
| `legendre_p` n=4 (Bonnet), [-1, 1] | 5.55e-16 | 3.17e-14 | 253 |
| `legendre_p` n=12 | 4.00e-15 | 2.58e-14 | 194 |
| `chebyshev_t` n=4 | 2.22e-16 | 2.91e-15 | 14 |
| `chebyshev_t` n=10 | 6.11e-16 | 6.44e-15 | 31 |
| `hermite_h` n=6, [-3, 3] | 3.64e-12 | 3.48e-16 | 3 |
| `laguerre_l` n=6, [0, 12] | 1.42e-14 | 6.21e-15 | 55 |

These have no approximation error to document -- the three-term
recurrences evaluate the polynomial exactly, and what is left is rounding.
Accuracy holding from n=4 to n=12 is the useful part: the recurrences are
numerically stable in the direction they are being run.

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
