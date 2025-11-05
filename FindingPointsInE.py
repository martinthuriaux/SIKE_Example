"""
FindPointsInE.py
================

This module handles:
- Finite field arithmetic in F_{p^2} where F_{p^2} = F_p(i) and i^2 = -1 (mod p)
- Utility functions for quadratic residues / square roots in F_p and F_{p^2}
- Supersingularity test for the SIKE base curve
- Brute-force enumeration of all points on
      E0 : y^2 = x^3 + 6x^2 + x
  over F_{p^2}

This file does NOT implement point addition / scalar multiplication.
Those live in EllipticCurveArithmetic.py.
"""

# --------------------------
# Basic finite field helpers
# --------------------------

def modp(x, p):
    """Reduce integer x modulo p."""
    return x % p


def add_fp2(a, b, p):
    """
    Add two F_{p^2} elements a=(a0,a1), b=(b0,b1).
    Returns ( (a0+b0) mod p, (a1+b1) mod p ).
    """
    return ((a[0] + b[0]) % p, (a[1] + b[1]) % p)


def sub_fp2(a, b, p):
    """
    Subtract two F_{p^2} elements a - b.
    a=(a0,a1), b=(b0,b1).
    Returns ( (a0-b0) mod p, (a1-b1) mod p ).
    """
    return ((a[0] - b[0]) % p, (a[1] - b[1]) % p)


def mul_fp2(a, b, p):
    """
    Multiply two F_{p^2} elements a=(a0,a1), b=(b0,b1),
    where i^2 = -1 mod p, i.e.
      (a0 + a1*i)*(b0 + b1*i)
    = (a0*b0 - a1*b1) + (a0*b1 + a1*b0)*i.
    """
    (a0, a1) = a
    (b0, b1) = b
    real = (a0*b0 - a1*b1) % p
    imag = (a0*b1 + a1*b0) % p
    return (real, imag)


def sqr_fp2(a, p):
    """
    Square an element a in F_{p^2}.
    Equivalent to mul_fp2(a, a, p).
    """
    return mul_fp2(a, a, p)


def pow_fp2(a, e, p):
    """
    Exponentiate an F_{p^2} element a to integer power e
    using square-and-multiply.
    """
    result = (1 % p, 0)  # multiplicative identity in F_{p^2}
    base = a
    ee = e
    while ee > 0:
        if ee & 1:
            result = mul_fp2(result, base, p)
        base = mul_fp2(base, base, p)
        ee >>= 1
    return result


def eq_fp2(a, b):
    """
    Return True iff a == b as F_{p^2} elements.
    """
    return a[0] == b[0] and a[1] == b[1]


def negate_fp2(a, p):
    """
    Additive inverse in F_{p^2}.
    -(a0 + a1*i) = (-a0 mod p, -a1 mod p).
    """
    return ((-a[0]) % p, (-a[1]) % p)


def inv_fp2(a, p):
    """
    Multiplicative inverse in F_{p^2}.
    If a = a0 + a1*i, then
        a^{-1} = (a0 - a1*i) / (a0^2 + a1^2) mod p.
    """
    a0, a1 = a
    denom = (a0*a0 + a1*a1) % p
    # denom is in F_p, so invert it mod p:
    denom_inv = pow(denom, -1, p)
    return ((a0 * denom_inv) % p,
            ((-a1) * denom_inv) % p)


def div_fp2(a, b, p):
    """
    Division in F_{p^2}: a / b = a * b^{-1}.
    """
    return mul_fp2(a, inv_fp2(b, p), p)


# -----------------------------------------
# Curve RHS (Montgomery model of SIKE base)
# -----------------------------------------

def curve_rhs_fp2(x, p, A):
    """
    Compute f(x) = x^3 + Ax^2 + x in F_{p^2}.

    This is the right-hand side of the SIKE base curve
        y^2 = x^3 + 6x^2 + x
    over F_{p^2}.

    x is an F_{p^2} element given as a pair (x0,x1).
    Returns an F_{p^2} element (f0,f1).
    """
    (re,im) = A
    coefficient = (re % p, im % p)
    x2 = mul_fp2(x, x, p)      # x^2
    x3 = mul_fp2(x2, x, p)     # x^3
    term6x2 = mul_fp2(coefficient, x2, p)  # 6*x^2
    rhs = add_fp2(add_fp2(x3, term6x2, p), x, p)
    return rhs


# -------------------------------------------------
# Square root helpers / quadratic residue helpers
# -------------------------------------------------

def is_square_fp(a, p):
    """
    Check if a (an integer mod p) is a quadratic residue in F_p.
    Uses Euler's criterion: a^((p-1)/2) == 1 mod p OR a == 0.
    """
    a = a % p
    if a == 0:
        return True
    return pow(a, (p - 1) // 2, p) == 1


def sqrt_fp_all(a, p):
    """
    Return all y in F_p such that y^2 = a (mod p).
    For small toy p this brute force is fine.
    """
    a = a % p
    sols = []
    for y in range(p):
        if (y*y) % p == a:
            sols.append(y % p)
    return sols


def sqrt_fp2_all(z, p):
    """
    Return all y in F_{p^2} such that y^2 = z.

    We brute force:
      y = (c,d) with c,d in 0..p-1
      check (c,d)^2 == z using sqr_fp2.

    This is fine for tiny p like 23.
    """
    sols = []
    (z0, z1) = z
    for c in range(p):
        for d in range(p):
            y = (c, d)
            y2 = sqr_fp2(y, p)
            if y2[0] == z0 and y2[1] == z1:
                sols.append(y)
    # de-duplicate +/-y (not strictly necessary, but nice)
    uniq = []
    for y in sols:
        if not any(eq_fp2(y, u) for u in uniq):
            uniq.append(y)
    return uniq


# ---------------------------------
# Supersingularity + point counting
# ---------------------------------

def count_points_over_fp(p):
    """
    Count all points on E0 over F_p:
        E0(F_p): y^2 = x^3 + 6x^2 + x  (coords in F_p)
    We brute force x in F_p, solve y^2 = f(x) in F_p,
    and add the point at infinity.

    Returns (count, list_of_points)
    Each affine point is given as plain integers (x,y),
    and we append the string "INF" for the point at infinity.
    """
    points = []

    for x in range(p):
        rhs = (x**3 + 6*(x**2) + x) % p  # f(x) in F_p

        if rhs == 0:
            # Only y=0
            points.append((x, 0))
        else:
            # If rhs is a quadratic residue mod p, include both roots
            if pow(rhs, (p - 1) // 2, p) == 1:
                for y in range(p):
                    if (y * y) % p == rhs:
                        points.append((x, y))

    # Include point at infinity
    points.append("INF")
    return len(points), points


def is_curve_supersingular(p):
    """
    Check supersingularity using the NIST/SIKE condition:
        E is supersingular <=> p | (p + 1 - #E(F_p))

    We:
      1. count #E(F_p),
      2. compute t = p + 1 - #E(F_p),
      3. check p | t.

    Returns (is_ss, numFp, trace_t)
    """
    numFp, _pts = count_points_over_fp(p)
    t = (p + 1) - numFp
    return (t % p == 0), numFp, t


def enumerate_points_over_fp2(p, A):
    """
    Enumerate ALL affine points on E0(F_{p^2}):

        y^2 = x^3 + 6x^2 + x

    where x,y are in F_{p^2} = F_p(i).

    Strategy:
      - Build a lookup of all possible y^2 for y in F_{p^2}.
      - For each x in F_{p^2}, compute f(x).
      - Match f(x) against that lookup.
      - Collect all (x,y) that satisfy y^2 = f(x).
      - Then add the point at infinity "INF".

    Returns (count, points)
    where each point is ((x0,x1),(y0,y1)) or "INF".
    """
    elems = [(a, b) for a in range(p) for b in range(p)]

    # Precompute y^2 for all y in F_{p^2} so we can reverse-map squares
    square_table = {}
    for y in elems:
        y2 = sqr_fp2(y, p)
        square_table.setdefault(y2, []).append(y)

    points = []
    for x in elems:
        rhs = curve_rhs_fp2(x, p, A)  # x^3 + 6x^2 + x in F_{p^2}
        if rhs in square_table:
            for y in square_table[rhs]:
                points.append((x, y))

    points.append("INF")
    return len(points), points


# ---------------------------------
# Demo / sanity check
# ---------------------------------

if __name__ == "__main__":
    # Pick a toy SIKE-style prime
    p = 11  # 23 = 2^3 * 3^1 - 1, good for our toy example
    A = (6 % p, 0 % p)
    print(f"Using p = {p}")

    # Check supersingularity
    ss, numFp, t = is_curve_supersingular(p)
    print(f"#E(F_p)        = {numFp}")
    print(f"trace t        = {t}")
    print(f"supersingular? = {ss} (True if p | (p+1-#E(F_p)))")

    # Enumerate all curve points over F_{p^2}
    numFp2, ptsFp2 = enumerate_points_over_fp2(p, A)
    print(f"#E(F_p^2)      = {numFp2}")
    print(f"(p+1)^2        = {(p+1)**2}  <-- should match #E(F_p^2)")

    print("6*8 ", mul_fp2((6,0),(8,0),11))

    # Uncomment to see them all (576 points for p=23):
    # for P in ptsFp2:
    #    print(P)
