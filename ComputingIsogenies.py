"""
ComputingIsogenies.py (x-only SIKE style)
========================================

Implements x-only 2- and 3-isogenies, and cfpk (curve-from-public-key),
using only field operations already imported from FindingPointsInE.

Curve model we maintain everywhere:
    E_A : y^2 = x^3 + A x^2 + x     (i.e. Montgomery with B = 1)

All F_{p^2} elements are tuples (a0, a1) meaning a0 + a1*i (mod p).

Exports:
    compute_2_isogeny_xonly(A, K, p) -> (A_next, phi2_x)
    compute_3_isogeny_xonly(A, K, p) -> (A_next, phi3_x)
    cfpk(xP, xQ, xR, p)              -> A_from_pk
"""

from FindingPointsInE import (
    add_fp2,
    sub_fp2,
    mul_fp2,
    sqr_fp2,
    inv_fp2,
    eq_fp2,
)

# -------------------------------------------------
# minimal helper
# -------------------------------------------------

def fp2_const(c, p):
    """Embed Python int c as F_{p^2} element (c mod p, 0)."""
    return (c % p, 0)

def mul_many_fp2(vals, p):
    """Multiply a sequence of F_{p^2} elements together with mul_fp2."""
    acc = (1 % p, 0)
    for v in vals:
        acc = mul_fp2(acc, v, p)
    return acc


# =====================================================================
# 2-ISOGENY (x-only)
# =====================================================================
#
# SIKE formulas:
#   A' = 2 * (1 - 2*x2^2)
#   xφ₂(P) = (xP^2 * x2 - xP) / (xP - x2)
#
# These depend only on x2 (kernel x) and xP (input x). No y is needed.
#
def compute_2_isogeny_xonly(A, K, p):
    """
    Input:
        A : F_{p^2}  (current curve coefficient in y^2 = x^3 + A x^2 + x)
        K : (xK, yK) kernel generator for the 2-isogeny (we only use xK)
        p : base prime

    Output:
        (A_next, phi2_x) where:
            A_next = A' in F_{p^2}
            phi2_x is a function x_out = phi2_x(x_in),
                   or None if x_in hits the kernel (denominator 0).
    """
    (x2, _y2) = K

    one = fp2_const(1, p)
    two = fp2_const(2, p)

    x2_sq     = sqr_fp2(x2, p)                   # x2^2
    two_x2_sq = mul_fp2(two, x2_sq, p)           # 2*x2^2
    inner     = sub_fp2(one, two_x2_sq, p)       # 1 - 2*x2^2
    A_next    = mul_fp2(two, inner, p)           # 2*(1 - 2*x2^2)

    def phi2_x(xP):
        """
        xφ₂(P) = (xP^2*x2 - xP) / (xP - x2)
        Return None if xP == x2 (kernel -> infinity).
        """
        diff = sub_fp2(xP, x2, p)                # xP - x2
        if eq_fp2(diff, (0,0)):
            return None

        diff_inv = inv_fp2(diff, p)              # 1/(xP - x2)
        xP_sq    = sqr_fp2(xP, p)                # xP^2
        num_x    = sub_fp2(mul_fp2(xP_sq, x2, p), xP, p)  # xP^2*x2 - xP
        x_out    = mul_fp2(num_x, diff_inv, p)
        return x_out

    return A_next, phi2_x


# =====================================================================
# 3-ISOGENY (x-only)
# =====================================================================
#
# SIKE formulas:
#   A' = (A*x3 - 6*x3^2 + 6) * x3
#
#   xφ₃(P) = xP * (xP*x3 - 1)^2 / (xP - x3)^2
#
# Again: depends only on x3 (kernel x) and xP (input x). No y.
#
def compute_3_isogeny_xonly(A, K, p):
    """
    Input:
        A : F_{p^2}  (current curve coefficient)
        K : (x3, y3) kernel generator for 3-isogeny (we only use x3)
        p : base prime

    Output:
        (A_next, phi3_x) where:
            A_next = A' in F_{p^2}
            phi3_x is a function x_out = phi3_x(x_in),
                    or None if x_in == x3.
    """
    (x3, _y3) = K

    one = fp2_const(1, p)
    six = fp2_const(6, p)

    x3_sq    = sqr_fp2(x3, p)                       # x3^2
    Ax3      = mul_fp2(A, x3, p)                    # A*x3
    six_x3sq = mul_fp2(six, x3_sq, p)               # 6*x3^2
    tmp      = sub_fp2(Ax3, six_x3sq, p)            # A*x3 - 6*x3^2
    tmp      = add_fp2(tmp, six, p)                 # A*x3 - 6*x3^2 + 6
    A_next   = mul_fp2(tmp, x3, p)                  # ( ... )*x3

    def phi3_x(xP):
        """
        xφ₃(P) = xP * (xP*x3 - 1)^2 / (xP - x3)^2
        Return None if xP == x3 (kernel -> infinity).
        """
        diff = sub_fp2(xP, x3, p)                   # xP - x3
        if eq_fp2(diff, (0,0)):
            return None

        diff_sq     = sqr_fp2(diff, p)              # (xP - x3)^2
        diff_sq_inv = inv_fp2(diff_sq, p)

        xPx3        = mul_fp2(xP, x3, p)            # xP*x3
        xPx3_minus1 = sub_fp2(xPx3, one, p)         # xP*x3 - 1
        xPx3m1_sq   = sqr_fp2(xPx3_minus1, p)       # (xP*x3 - 1)^2

        num_x       = mul_fp2(xP, xPx3m1_sq, p)
        x_out       = mul_fp2(num_x, diff_sq_inv, p)
        return x_out

    return A_next, phi3_x


# =====================================================================
# cfpk : curve-from-public-key
# =====================================================================
#
# cfpk(xP, xQ, xR) reconstructs A from the public key (xP, xQ, xR),
# where:
#   xP = x(P)
#   xQ = x(Q)
#   xR = x(P-Q)
#
# Formula from the SIKE spec:
#
#   s   = 1 - xP*xQ - xP*xR - xQ*xR
#   num = s^2
#   den = 4 * xP * xQ * xR
#   A   = num/den  - xP - xQ - xR
#
# All ops are in F_{p^2}.
#
def cfpk(xP, xQ, xR, p):
    zero = (0,0)
    one  = fp2_const(1, p)
    four = fp2_const(4, p)

    # Step 1. reject if any input x is 0 per spec
    if eq_fp2(xP, zero) or eq_fp2(xQ, zero) or eq_fp2(xR, zero):
        raise ValueError("cfpk: invalid public key (one of xP,xQ,xR is 0)")

    # s = 1 - xP*xQ - xP*xR - xQ*xR
    xPxQ = mul_fp2(xP, xQ, p)
    xPxR = mul_fp2(xP, xR, p)
    xQxR = mul_fp2(xQ, xR, p)

    tmp1 = sub_fp2(one, xPxQ, p)        # 1 - xP*xQ
    tmp2 = sub_fp2(tmp1, xPxR, p)       # 1 - xP*xQ - xP*xR
    s    = sub_fp2(tmp2, xQxR, p)       # 1 - xP*xQ - xP*xR - xQ*xR

    num  = sqr_fp2(s, p)                # ( ... )^2

    den_part = mul_many_fp2([xP, xQ, xR], p)         # xP*xQ*xR
    den      = mul_fp2(four, den_part, p)            # 4*xP*xQ*xR
    den_inv  = inv_fp2(den, p)

    frac     = mul_fp2(num, den_inv, p)              # num/den

    sum_xs   = add_fp2(add_fp2(xP, xQ, p), xR, p)     # xP + xQ + xR
    A_val    = sub_fp2(frac, sum_xs, p)              # num/den - sum

    return A_val


# =====================================================================
# DEMO
# =====================================================================

if __name__ == "__main__":
    from PointGenerator import find_P2_Q2, find_P3_Q3
    from EllipticCurveArithmetic import point_add_montgomery
    from FindingPointsInE import negate_fp2

    p = 23
    A0 = (6 % p, 0)   # base curve y^2 = x^3 + 6x^2 + x, i.e. A = (6,0)

    print("=== DEMO (x-only isogenies + cfpk) ===")

    # helper: R = P - Q (so we can get x(P-Q) for cfpk)
    def point_sub(P, Q, p):
        (xQ, yQ) = Q
        Qneg = (xQ, negate_fp2(yQ, p))
        return point_add_montgomery(P, Qneg, p)

    # -------------------------
    # 2^e2 torsion side
    # -------------------------
    P2, Q2 = find_P2_Q2(p)
    R2 = point_sub(P2, Q2, p)

    xP2 = P2[0]
    xQ2 = Q2[0]
    xR2 = R2[0]

    print("\n[2^e2 torsion basis]")
    print("P2 =", P2)
    print("Q2 =", Q2)
    print("R2 =", R2)

    # Build 2-isogeny from P2
    A_after2, phi2_x = compute_2_isogeny_xonly(A0, P2, p)
    print("\n[2-isogeny from P2]")
    print("A' =", A_after2)

    # Push P2.x, Q2.x and R2.x through φ₂
    img_xP2 = phi2_x(xP2)
    img_xQ2 = phi2_x(xQ2)
    img_xR2 = phi2_x(xR2)
    print("phi2_x(xP2) =", img_xP2)
    print("phi2_x(xQ2) =", img_xQ2)
    print("phi2_x(xR2) =", img_xR2)

    # -------------------------
    # 3^e3 torsion side
    # -------------------------
    P3, Q3 = find_P3_Q3(p)
    R3 = point_sub(P3, Q3, p)

    xP3 = P3[0]
    xQ3 = Q3[0]
    xR3 = R3[0]

    print("\n[3^e3 torsion basis]")
    print("P3 =", P3)
    print("Q3 =", Q3)
    print("R3 =", R3)

    # Build 3-isogeny from P3
    A_after3, phi3_x = compute_3_isogeny_xonly(A0, P3, p)
    print("\n[3-isogeny from P3]")
    print("A' =", A_after3)

    img_xQ3 = phi3_x(xQ3)
    img_xR3 = phi3_x(xR3)
    print("phi3_x(xQ3) =", img_xQ3)
    print("phi3_x(xR3) =", img_xR3)

    print("\n=== done ===")
