from FindingPointsInE import (
    add_fp2, sub_fp2, mul_fp2, sqr_fp2,
    eq_fp2, negate_fp2, inv_fp2, div_fp2,
)

def curve_rhs_montgomery(x, p, A):
    """
    Compute f(x) = x^3 + 6x^2 + x in F_{p^2},
    where x is an F_{p^2} element (x0,x1).

    This is the right-hand side of
        y^2 = x^3 + 6x^2 + x.
    """
    (re,im) = A
    coefficient = (re % p, im % p)
    x2 = mul_fp2(x, x, p)       # x^2
    x3 = mul_fp2(x2, x, p)      # x^3
    term6x2 = mul_fp2(coefficient, x2, p)
    rhs = add_fp2(add_fp2(x3, term6x2, p), x, p)
    return rhs


# ---- Full-point doubling: E_A : y^2 = x^3 + A x^2 + x (B = 1) ----
def point_double_montgomery(P, p, A):
    """
    Closed-form affine doubling on a Montgomery curve with B = 1:

        (x2P, y2P) =
        ( (xP^2 - 1)^2 / (4 xP (xP^2 + A xP + 1)),
          yP * (xP^2 - 1) * (xP^4 + 2 A xP^3 + 6 xP^2 + 2 A xP + 1)
              / (8 xP^2 (xP^2 + A xP + 1)^2) )

    P: (xP, yP) in F_{p^2}^2 (NOT infinity)
    A: curve coefficient in F_{p^2}
    """
    (xP, yP) = P

    one  = (1 % p, 0)
    two  = (2 % p, 0)
    four = (4 % p, 0)
    six  = (6 % p, 0)
    eight= (8 % p, 0)

    xP2 = sqr_fp2(xP, p)            # xP^2
    xP3 = mul_fp2(xP2, xP, p)       # xP^3
    xP4 = sqr_fp2(xP2, p)           # xP^4

    # Common term: xP^2 + A xP + 1
    AxP  = mul_fp2(A, xP, p)
    S    = add_fp2(add_fp2(xP2, AxP, p), one, p)

    # x(2P)
    num_x = sqr_fp2(sub_fp2(xP2, one, p), p)                          # (xP^2 - 1)^2
    den_x = mul_fp2(mul_fp2(four, xP, p), S, p)                        # 4 xP (xP^2 + A xP + 1)
    x2P   = mul_fp2(num_x, inv_fp2(den_x, p), p)

    # y(2P)
    # poly = xP^4 + 2 A xP^3 + 6 xP^2 + 2 A xP + 1
    twoAxP3 = mul_fp2(two, mul_fp2(A, xP3, p), p)
    sixxP2  = mul_fp2(six, xP2, p)
    twoAxP  = mul_fp2(two, AxP, p)
    poly    = add_fp2(add_fp2(add_fp2(xP4, twoAxP3, p), sixxP2, p), add_fp2(twoAxP, one, p), p)

    num_y = mul_fp2(sub_fp2(xP2, one, p), poly, p)
    den_y = mul_fp2(mul_fp2(eight, sqr_fp2(xP, p), p), sqr_fp2(S, p), p)  # 8 xP^2 ( ... )^2
    frac  = mul_fp2(num_y, inv_fp2(den_y, p), p)
    y2P   = mul_fp2(yP, frac, p)

    return (x2P, y2P)


# ---- Addition with doubling fallback (B = 1) ----
def point_add_montgomery(P, Q, p, A, B=None):
    """
    Affine addition on y^2 = x^3 + A x^2 + x (B = 1).

    - If P = O or Q = O: returns the other.
    - If P = -Q: returns O (None).
    - If P = Q: uses the closed-form doubling above.
    - Else: generic chord-slope formula.
    """
    if P is None:
        return Q
    if Q is None:
        return P

    (x1, y1) = P
    (x2, y2) = Q

    # P = -Q  ->  infinity
    if eq_fp2(x1, x2) and eq_fp2(y1, negate_fp2(y2, p)):
        return None

    # P = Q -> closed-form doubling
    if eq_fp2(x1, x2) and eq_fp2(y1, y2):
        return point_double_montgomery(P, p, A)

    # Generic addition (P != Q and P != -Q):
    # λ = (yP - yQ)/(xP - xQ),  xR = λ^2 - (xP + xQ) - A,  yR = λ(xP - xR) - yP
    num = sub_fp2(y1, y2, p)
    den = sub_fp2(x1, x2, p)
    lam = div_fp2(num, den, p)

    lam2 = sqr_fp2(lam, p)
    x3   = sub_fp2(sub_fp2(lam2, add_fp2(x1, x2, p), p), A, p)
    y3   = sub_fp2(mul_fp2(lam, sub_fp2(x1, x3, p), p), y1, p)

    return (x3, y3)


def xDBL_xonly(xP, A, p):
    # returns x_[2]P on curve y^2 = x^3 + A x^2 + x
    xP2 = sqr_fp2(xP, p)                        # xP^2
    xP4 = sqr_fp2(xP2, p)                       # xP^4

    one  = (1 % p, 0)
    two  = (2 % p, 0)
    four = (4 % p, 0)

    num_part = sub_fp2(xP2, one, p)             # xP^2 - 1
    num_sq   = sqr_fp2(num_part, p)             # (xP^2 - 1)^2

    AxP   = mul_fp2(A, xP, p)                   # A*xP
    xP2_plus_AxP = add_fp2(xP2, AxP, p)         # xP^2 + A*xP
    denom_inner  = add_fp2(xP2_plus_AxP, one, p)# xP^2 + A*xP + 1
    four_xP      = mul_fp2(four, xP, p)         # 4*xP
    denom        = mul_fp2(four_xP, denom_inner, p)

    denom_inv = inv_fp2(denom, p)

    x2P = mul_fp2(num_sq, denom_inv, p)
    return x2P


def xTPL_xonly(xP, A, p):
    # returns x_[3]P on curve y^2 = x^3 + A x^2 + x
    xP2 = sqr_fp2(xP, p)                        # xP^2
    xP3 = mul_fp2(xP2, xP, p)                   # xP^3
    xP4 = sqr_fp2(xP2, p)                       # xP^4

    one  = (1 % p, 0)
    two  = (2 % p, 0)
    three= (3 % p, 0)
    four = (4 % p, 0)
    six  = (6 % p, 0)

    # numerator core: (xP^4 - 4 A xP - 6 xP^2 - 3)
    AxP     = mul_fp2(A, xP, p)                 # A*xP
    fourAxP = mul_fp2(four, AxP, p)             # 4*A*xP

    six_xP2 = mul_fp2(six, xP2, p)              # 6*xP^2

    tmp1 = sub_fp2(xP4, fourAxP, p)             # xP^4 - 4AxP
    tmp2 = sub_fp2(tmp1, six_xP2, p)            # xP^4 - 4AxP - 6xP^2
    tmp3 = sub_fp2(tmp2, three, p)              # xP^4 - 4AxP - 6xP^2 - 3

    num_sq = sqr_fp2(tmp3, p)                   # (...)^2
    num    = mul_fp2(num_sq, xP, p)             # (...)^2 * xP

    # denominator core: (4 A xP^3 + 3 xP^4 + 6 xP^2 - 1)^2
    fourAxP3 = mul_fp2(four, mul_fp2(A, xP3, p), p)  # 4*A*xP^3
    three_xP4= mul_fp2(three, xP4, p)                # 3*xP^4
    six_xP2  = mul_fp2(six, xP2, p)                  # 6*xP^2

    tmpd1 = add_fp2(fourAxP3, three_xP4, p)
    tmpd2 = add_fp2(tmpd1, six_xP2, p)
    tmpd3 = sub_fp2(tmpd2, one, p)                   # 4AxP^3 + 3xP^4 + 6xP^2 - 1
    den_sq = sqr_fp2(tmpd3, p)

    den_inv = inv_fp2(den_sq, p)

    x3P = mul_fp2(num, den_inv, p)
    return x3P

def scalar_mul_montgomery(P, k, p, A, B=None):
    """
    Scalar multiply a point P by integer k on the Montgomery curve

        B*y^2 = x^3 + A*x^2 + x

    using double-and-add in affine coordinates.

    Args:
        P: (x,y) in F_{p^2}^2 or None (infinity)
        k: nonnegative integer
        p: base prime
        A: F_{p^2} element (a0, a1) for the curve coefficient
        B: F_{p^2} element for B (default (1,0))

    Returns:
        [k]P in affine coordinates, or None for infinity.
    """
    if B is None:
        B = (1 % p, 0)

    if P is None or k == 0:
        return None

    R = None            # accumulator (∞)
    Q = P               # running multiple
    kk = k
    while kk > 0:
        if kk & 1:
            R = point_add_montgomery(R, Q, p, A, B)
        Q = point_add_montgomery(Q, Q, p, A, B)
        kk >>= 1
    return R

def point_sub_montgomery(P, Q, p, A, B=None):
    """
    Compute P - Q on the Montgomery curve

        B*y^2 = x^3 + A*x^2 + x

    using the generalized point_add_montgomery.

    Args:
        P, Q: points (x,y) in F_{p^2}^2 or None
        p: base prime
        A: F_{p^2} curve coefficient
        B: F_{p^2} (default (1,0))

    Returns:
        P - Q in affine coordinates, or None for infinity.
    """
    if B is None:
        B = (1 % p, 0)

    if Q is None:
        return P
    if P is None:
        (xQ, yQ) = Q
        # Return -Q
        return (xQ, negate_fp2(yQ, p))

    (xQ, yQ) = Q
    Qneg = (xQ, negate_fp2(yQ, p))
    return point_add_montgomery(P, Qneg, p, A, B)


# Quick self-check for tiny p
if __name__ == "__main__":
    p = 23

    A = (6 % p, 0)

    # Example point: pick an x in F_{p^2}, solve y^2 = f(x)
    x = (5, 0)  # x = 5 + 1*i
    rhs = curve_rhs_montgomery(x, p, A)

    # Let's try to find y with y^2 = rhs by brute forcing over F_{p^2}
    from FindingPointsInE import sqrt_fp2_all
    ys = sqrt_fp2_all(rhs, p)
    if ys:
        P = (x, ys[0])
        print("Sample point P =", P)
        for _ in range(1,24):
            print("Check", _,"P = ", scalar_mul_montgomery(P, _, p, A))
    else:
        print("No point found for that test x.")
