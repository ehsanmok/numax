"""A dense strictly convex quadratic program on the host: Goldfarb and
Idnani's dual active-set method, the one `quadprog`'s `qpgen2` implements.

**Tier 2, private.** `min (1/2) x^T G x - a^T x` subject to `C x >= b`,
the first `meq` rows as equalities, for a symmetric positive definite
`G`. The method starts at the unconstrained minimizer `G^-1 a` and adds
violated constraints one at a time, keeping every iterate dual feasible;
the active set's normals `N` are held through `J^T N = [R; 0]` with `J =
L^-T` rotated by Givens as constraints enter and leave, so each step is
`O(n^2)`. It is what `minimize(method="SLSQP")` solves its subproblem
with, where `n` is the variable count and small.
"""

from std.math import sqrt as _sqrt


struct QPSolution(Movable):
    """The minimizer, the multipliers of every constraint (zero on the
    inactive ones), and whether the constraints were consistent."""

    var x: List[Float64]
    var multipliers: List[Float64]
    var feasible: Bool

    def __init__(
        out self,
        var x: List[Float64],
        var multipliers: List[Float64],
        feasible: Bool,
    ):
        self.x = x^
        self.multipliers = multipliers^
        self.feasible = feasible


def _givens(a: Float64, b: Float64) -> Tuple[Float64, Float64, Float64]:
    """`(c, s, h)` with `c a + s b = h` and `-s a + c b = 0`."""
    var h = _sqrt(a * a + b * b)
    if h == 0:
        return (1.0, 0.0, 0.0)
    return (a / h, b / h, h)


def _slack(
    p: Int, x: List[Float64], cmat: List[Float64], b: List[Float64], n: Int
) -> Float64:
    """`C_p x - b_p`."""
    var s = -b[p]
    for j in range(n):
        s += cmat[p * n + j] * x[j]
    return s


def solve_qp(
    g: List[Float64],
    a: List[Float64],
    cmat: List[Float64],
    b: List[Float64],
    meq: Int,
    n: Int,
) raises -> QPSolution:
    """Goldfarb-Idnani on the row-major `n x n` `g` and `m x n` `cmat`.

    Raises:
        If `g` is not positive definite.
    """
    var m = len(b)
    # G = L L^T, then J = L^-T (upper triangular).
    var l = List[Float64](length=n * n, fill=0.0)
    for i in range(n):
        for j in range(i + 1):
            var s = g[i * n + j]
            for k in range(j):
                s -= l[i * n + k] * l[j * n + k]
            if i == j:
                if s <= 0:
                    raise Error("solve_qp: G is not positive definite")
                l[i * n + i] = _sqrt(s)
            else:
                l[i * n + j] = s / l[j * n + j]
    # L^-1 by forward substitution, column by column; J = (L^-1)^T.
    var linv = List[Float64](length=n * n, fill=0.0)
    for col in range(n):
        for i in range(col, n):
            var s = 1.0 if i == col else 0.0
            for k in range(col, i):
                s -= l[i * n + k] * linv[k * n + col]
            linv[i * n + col] = s / l[i * n + i]
    var jm = List[Float64](length=n * n, fill=0.0)
    for i in range(n):
        for j in range(n):
            jm[i * n + j] = linv[j * n + i]
    # x = G^-1 a = J J^T a.
    var x = List[Float64](length=n, fill=0.0)
    var jta = List[Float64](length=n, fill=0.0)
    for j in range(n):
        var s = 0.0
        for i in range(n):
            s += jm[i * n + j] * a[i]
        jta[j] = s
    for i in range(n):
        var s = 0.0
        for j in range(n):
            s += jm[i * n + j] * jta[j]
        x[i] = s

    var r = List[Float64](length=n * n, fill=0.0)
    var active = List[Int]()
    var u = List[Float64]()
    var sign = List[Float64](length=m, fill=1.0)
    var is_active = List[Bool](length=m, fill=False)
    var added_eq = 0
    var scale = 1.0
    for e in range(len(b)):
        scale = max(scale, abs(b[e]))

    for _ in range(10 * (m + n) + 10):
        # Step 1: the next constraint -- every equality first, then the
        # most violated inequality.
        var p = -1
        if added_eq < meq:
            p = added_eq
            added_eq += 1
            if _slack(p, x, cmat, b, n) > 0:
                sign[p] = -1.0
        else:
            var worst = -1e-12 * scale
            for i in range(meq, m):
                if not is_active[i]:
                    var s = _slack(i, x, cmat, b, n)
                    if s < worst:
                        worst = s
                        p = i
        if p < 0:
            var multipliers = List[Float64](length=m, fill=0.0)
            for k in range(len(active)):
                multipliers[active[k]] = sign[active[k]] * u[k]
            return QPSolution(x^, multipliers^, True)
        var np = List[Float64](length=n, fill=0.0)
        for j in range(n):
            np[j] = sign[p] * cmat[p * n + j]
        var bp = sign[p] * b[p]
        var up = 0.0

        # Step 2: move toward satisfying constraint p.
        while True:
            var q = len(active)
            var d = List[Float64](length=n, fill=0.0)
            for j in range(n):
                var s = 0.0
                for i in range(n):
                    s += jm[i * n + j] * np[i]
                d[j] = s
            var z = List[Float64](length=n, fill=0.0)
            for i in range(n):
                var s = 0.0
                for j in range(q, n):
                    s += jm[i * n + j] * d[j]
                z[i] = s
            var rr = List[Float64](length=q, fill=0.0)
            var k = q - 1
            while k >= 0:
                var s = d[k]
                for j in range(k + 1, q):
                    s -= r[k * n + j] * rr[j]
                rr[k] = s / r[k * n + k]
                k -= 1
            var t1 = Float64.MAX
            var drop = -1
            for j in range(q):
                if active[j] >= meq and rr[j] > 0:
                    var ratio = u[j] / rr[j]
                    if ratio < t1:
                        t1 = ratio
                        drop = j
            var ztn = 0.0
            var znorm = 0.0
            for i in range(n):
                ztn += z[i] * np[i]
                znorm = max(znorm, abs(z[i]))
            var t2 = Float64.MAX
            var sp = -bp
            for j in range(n):
                sp += np[j] * x[j]
            if znorm > 1e-14 and abs(ztn) > 1e-300:
                t2 = -sp / ztn
            var t = min(t1, t2)
            if t == Float64.MAX:
                var multipliers = List[Float64](length=m, fill=0.0)
                return QPSolution(x^, multipliers^, False)
            for j in range(q):
                u[j] -= t * rr[j]
            up += t
            if t2 != Float64.MAX:
                for i in range(n):
                    x[i] += t * z[i]
            if t == t2:
                # Add p: rotate d[q+1:] into d[q], carrying J's columns.
                var i = n - 1
                while i > q:
                    var rot = _givens(d[i - 1], d[i])
                    d[i - 1] = rot[2]
                    d[i] = 0.0
                    for row in range(n):
                        var left = jm[row * n + i - 1]
                        var right = jm[row * n + i]
                        jm[row * n + i - 1] = rot[0] * left + rot[1] * right
                        jm[row * n + i] = -rot[1] * left + rot[0] * right
                    i -= 1
                for row in range(q + 1):
                    r[row * n + q] = d[row]
                active.append(p)
                u.append(up)
                is_active[p] = True
                break
            # Drop the blocking constraint and retriangularize.
            is_active[active[drop]] = False
            for col in range(drop, q - 1):
                for row in range(n):
                    r[row * n + col] = r[row * n + col + 1]
            for row in range(n):
                r[row * n + q - 1] = 0.0
            for j in range(drop, q - 1):
                var rot = _givens(r[j * n + j], r[(j + 1) * n + j])
                for col in range(j, q - 1):
                    var top = r[j * n + col]
                    var bottom = r[(j + 1) * n + col]
                    r[j * n + col] = rot[0] * top + rot[1] * bottom
                    r[(j + 1) * n + col] = -rot[1] * top + rot[0] * bottom
                for row in range(n):
                    var left = jm[row * n + j]
                    var right = jm[row * n + j + 1]
                    jm[row * n + j] = rot[0] * left + rot[1] * right
                    jm[row * n + j + 1] = -rot[1] * left + rot[0] * right
            _ = active.pop(drop)
            _ = u.pop(drop)
    raise Error("solve_qp: the active-set iteration did not terminate")
