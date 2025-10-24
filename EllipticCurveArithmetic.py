"""
EllipticCurveArithmetic.py
==========================

Elliptic curve group operations for the SIKE base curve

    E0 : y^2 = x^3 + 6x^2 + x

over F_{p^2} = F_p(i), i^2 = -1 mod p.

This module defines:
    - curve_rhs_montgomery(x, p)
    - point_add_montgomery(P, Q, p)
    - scalar_mul_montgomery(P, k, p)

It imports finite-field ops from FindPointsInE.py.
"""

from FindingPointsInE import (
    add_fp2, sub_fp2, mul_fp2, sqr_fp2,
    eq_fp2, negate_fp2, inv_fp2, div_fp2,
)

def curve_rhs_montgomery(x, p):
    """
    Compute f(x) = x^3 + 6x^2 + x in F_{p^2},
    where x is an F_{p^2} element (x0,x1).

    This is the right-hand side of
        y^2 = x^3 + 6x^2 + x.
    """
    six = (6 % p, 0)
    x2 = mul_fp2(x, x, p)       # x^2
    x3 = mul_fp2(x2, x, p)      # x^3
    term6x2 = mul_fp2(six, x2, p)
    rhs = add_fp2(add_fp2(x3, term6x2, p), x, p)
    return rhs


def point_add_montgomery(P, Q, p):
    """
    Elliptic curve addition / doubling in affine coordinates on

        y^2 = x^3 + 6x^2 + x.

    We treat the curve as a general long Weierstrass form:
        y^2 = x^3 + a2*x^2 + a4*x + a6
    with a2 = 6, a4 = 1, a6 = 0, and a1=a3=0.

    Formulas:

    If P != Q:
        λ = (y2 - y1) / (x2 - x1)
    If P == Q:
        λ = (3*x1^2 + 2*a2*x1 + a4) / (2*y1)

    Then
        x3 = λ^2 - a2 - x1 - x2
        y3 = -y1 + λ*(x1 - x3)

    Special cases:
        - None represents the point at infinity O.
        - If x1 == x2 and y2 == -y1, then P + Q = O.

    Returns:
        R = (x3, y3) in F_{p^2}, or None for infinity.
    """
    if P is None:
        return Q
    if Q is None:
        return P

    (x1, y1) = P
    (x2, y2) = Q

    a2 = (6 % p, 0)
    a4 = (1 % p, 0)
    # a6 = (0,0) not needed explicitly
    # a1 = a3 = 0 for this curve

    # Check if P == -Q -> point at infinity
    if eq_fp2(x1, x2) and eq_fp2(y1, negate_fp2(y2, p)):
        return None

    if not eq_fp2(x1, x2):
        # P != Q, use chord slope:
        num = sub_fp2(y2, y1, p)         # y2 - y1
        den = sub_fp2(x2, x1, p)         # x2 - x1
        lam = div_fp2(num, den, p)       # (y2-y1)/(x2-x1) in F_{p^2}
    else:
        # P == Q, use tangent slope:
        # λ = (3*x1^2 + 2*a2*x1 + a4)/(2*y1)
        three = (3 % p, 0)
        two   = (2 % p, 0)

        x1_sq = mul_fp2(x1, x1, p)                     # x1^2
        term1 = mul_fp2(three, x1_sq, p)               # 3*x1^2
        term2 = mul_fp2(mul_fp2(two, a2, p), x1, p)    # 2*a2*x1
        num = add_fp2(add_fp2(term1, term2, p), a4, p) # 3*x1^2 + 2*a2*x1 + a4
        den = mul_fp2(two, y1, p)                      # 2*y1
        lam = div_fp2(num, den, p)

    # x3 = λ^2 - a2 - x1 - x2
    lam_sq = sqr_fp2(lam, p)
    x3 = sub_fp2(
            sub_fp2(
                sub_fp2(lam_sq, a2, p),
                x1, p
            ),
            x2, p
        )

    # y3 = -y1 + λ*(x1 - x3)
    x1_minus_x3 = sub_fp2(x1, x3, p)
    lam_times = mul_fp2(lam, x1_minus_x3, p)
    y3 = sub_fp2(lam_times, y1, p)

    return (x3, y3)


def scalar_mul_montgomery(P, k, p):
    """
    Scalar multiply a point P by integer k on
        y^2 = x^3 + 6x^2 + x
    using double-and-add.

    None is used for the point at infinity O.

    Returns:
        [k]P, or None if k == 0 or P is None or [k]P == O.
    """
    if P is None or k == 0:
        return None

    R = None     # accumulator (starts at O)
    Q = P        # running multiple
    kk = k
    while kk > 0:
        if kk & 1:
            R = point_add_montgomery(R, Q, p)
        Q = point_add_montgomery(Q, Q, p)
        kk >>= 1
    return R


# Quick self-check for tiny p
if __name__ == "__main__":
    p = 23

    # Example point: pick an x in F_{p^2}, solve y^2 = f(x)
    x = (5, 0)  # x = 5 + 1*i
    rhs = curve_rhs_montgomery(x, p)

    # Let's try to find y with y^2 = rhs by brute forcing over F_{p^2}
    from FindingPointsInE import sqrt_fp2_all
    ys = sqrt_fp2_all(rhs, p)
    if ys:
        P = (x, ys[0])
        print("Sample point P =", P)
        print("Check 2P = ", scalar_mul_montgomery(P, 2, p))
        print("Check 3P = ", scalar_mul_montgomery(P, 3, p))
        print("Check 4P = ", scalar_mul_montgomery(P, 4, p))
        print("Check 6P = ", scalar_mul_montgomery(P, 6, p))
        print("Check 8P = ", scalar_mul_montgomery(P, 8, p))
        print("Check 9P = ", scalar_mul_montgomery(P, 9, p))
        print("Check 13P = ", scalar_mul_montgomery(P, 13, p))
    else:
        print("No point found for that test x.")
