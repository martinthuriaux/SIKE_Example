"""
PointGenerator.py
=================

This module generates the public generator points P2, Q2, P3, Q3
for the SIKE base curve

    E0 : y^2 = x^3 + 6x^2 + x   over F_{p^2} = F_p(i), i^2 = -1 (mod p),

following the procedure described in the SIKE spec (NIST submission, §1.3.3).

Dependencies:
- FindingPointsInE.py   (finite field ops, sqrt helpers, curve_rhs_fp2)
- EllipticCurveArithmetic.py (point addition, scalar multiply)
"""

from FindingPointsInE import (
    add_fp2, sub_fp2, mul_fp2, sqr_fp2, eq_fp2, negate_fp2,
    sqrt_fp2_all, is_square_fp, sqrt_fp_all,
    curve_rhs_fp2,
)
from EllipticCurveArithmetic import (
    curve_rhs_montgomery,
    scalar_mul_montgomery,
)

# ----------------------------------------------------------
# Helpers: exponent extraction, sqrt(2), order checking
# ----------------------------------------------------------

def get_sike_exponents(p):
    """
    For a SIKE-style prime p = 2^{e2} * 3^{e3} - 1,
    p + 1 = 2^{e2} * 3^{e3}.

    This returns (e2, e3).
    """
    n = p + 1
    e2 = 0
    while n % 2 == 0:
        n //= 2
        e2 += 1
    e3 = 0
    while n % 3 == 0:
        n //= 3
        e3 += 1
    if n != 1:
        raise ValueError(f"p+1 is not 2^e * 3^e for p={p}")
    return e2, e3


def find_sqrt2_fp2(p):
    """
    Find some sqrt2 in F_{p^2} such that sqrt2^2 = 2.

    We'll brute force using the same approach as sqrt_fp2_all:
    just ask for square roots of (2,0).
    """
    target = (2 % p, 0)
    roots = sqrt_fp2_all(target, p)
    if not roots:
        raise ValueError("No sqrt(2) in F_{p^2} ???")
    return roots[0]  # either root is fine; ±sqrt2 both work


def point_has_exact_order_power(P, base, exponent, p):
    """
    Check if point P has exact order base^exponent.

    Meaning:
      [base^exponent]P = O
    and if exponent > 0,
      [base^(exponent-1)]P != O.

    We ONLY need base=3 here (for P3,Q3), but we write it generally.
    """
    if P is None:
        return False

    full_pow = base ** exponent
    prev_pow = base ** (exponent - 1) if exponent > 0 else 1

    Pfull = scalar_mul_montgomery(P, full_pow, p)
    if Pfull is not None:
        return False  # didn't vanish at expected full order

    if exponent == 0:
        # order 1 == infinity
        return True

    Pprev = scalar_mul_montgomery(P, prev_pow, p)
    return (Pprev is not None)


# ----------------------------------------------------------
# 1. Find P2 and Q2  (2-power torsion basis)
# ----------------------------------------------------------

def find_P2_Q2(p):
    """
    Find P2 and Q2 in the 2^e2 torsion according to §1.3.3.

    From the spec:
      P2 = [3^{e3}] ( i + c, sqrt(f(i+c)) ),
         c = smallest nonnegative integer such that:
              P2 is on curve,
              and [2^{e2-1}] P2 = (-3 ± 2 sqrt(2), 0).

      Q2 = [3^{e3}] ( i + c, sqrt(f(i+c)) ),
         c = smallest nonnegative integer such that:
              Q2 is on curve,
              and [2^{e2-1}] Q2 = (0, 0).

    We implement this literally.
    """

    e2, e3 = get_sike_exponents(p)
    cofactor_3e3 = 3 ** e3          # multiply to wipe 3-part
    half_two_tors = 2 ** (e2 - 1)   # 2^{e2-1}
    zero = (0, 0)
    i_elem = (0, 1)                 # "i" in F_{p^2}

    # we need (-3 ± 2√2, 0)
    sqrt2 = find_sqrt2_fp2(p)
    minus3 = ((-3) % p, 0)
    twosqrt2 = mul_fp2((2 % p, 0), sqrt2, p)
    x_target_plus  = add_fp2(minus3, twosqrt2, p)   # -3 + 2√2
    x_target_minus = sub_fp2(minus3, twosqrt2, p)   # -3 - 2√2

    P2 = None
    Q2 = None

    for c in range(p):
        if P2 is not None and Q2 is not None:
            break

        # Build the candidate affine x = i + c = (c,1)
        x = (c % p, 1)

        # Compute f(x) = x^3 + 6x^2 + x
        rhs = curve_rhs_fp2(x, p)

        # Find all y in F_{p^2} with y^2 = rhs
        y_candidates = sqrt_fp2_all(rhs, p)
        for y in y_candidates:
            # Raw point (before projection)
            cand = (x, y)

            # Project away 3-part: multiply by 3^{e3}
            cand_proj = scalar_mul_montgomery(cand, cofactor_3e3, p)
            if cand_proj is None:
                # degenerate, skip
                continue

            # Now test where [2^{e2-1}] sends it
            test_pt = scalar_mul_montgomery(cand_proj, half_two_tors, p)

            if test_pt is None:
                # shouldn't normally happen for P2/Q2 because RHS is supposed to be 2-torsion, not infinity
                continue

            (xtest, ytest) = test_pt

            # condition for P2:
            #   [2^{e2-1}]P2 = (-3 ± 2√2, 0)
            if P2 is None and eq_fp2(ytest, zero) and (
                eq_fp2(xtest, x_target_plus) or eq_fp2(xtest, x_target_minus)
            ):
                P2 = cand_proj
                # don't break yet; we still maybe need Q2

            # condition for Q2:
            #   [2^{e2-1}]Q2 = (0, 0)
            if Q2 is None and eq_fp2(xtest, zero) and eq_fp2(ytest, zero):
                Q2 = cand_proj

    return P2, Q2


# ----------------------------------------------------------
# 2. Find P3 and Q3 (3-power torsion basis)
# ----------------------------------------------------------

def find_P3_Q3(p):
    """
    Find P3 and Q3 according to §1.3.3:

      P3 = [2^{e2-1}] ( c, sqrt(f(c)) ),
           where:
             - c is the smallest nonnegative int
               such that f(c) is a square in F_p
             - and P3 has exact order 3^{e3}.

      Q3 = [2^{e2-1}] ( c, sqrt(f(c)) ),
           where:
             - c is the smallest nonnegative int
               such that f(c) is NON-square in F_p
             - we choose the sqrt in F_{p^2} \ F_p
             - and Q3 has exact order 3^{e3}.

    Note: x = (c,0) is "just c" in F_p ⊂ F_{p^2}.
    """

    e2, e3 = get_sike_exponents(p)
    cofactor_2e2m1 = 2 ** (e2 - 1)   # multiply to wipe 2-part
    zero = (0, 0)

    P3 = None
    Q3 = None

    for c in range(p):
        if P3 is not None and Q3 is not None:
            break

        x = (c % p, 0)   # x in F_p ⊂ F_{p^2}

        rhs = curve_rhs_fp2(x, p)   # rhs = f(c) in F_{p^2}
        (rhs_re, rhs_im) = rhs

        # ---- candidate for P3 (square in F_p) ----
        if P3 is None and rhs_im == 0 and is_square_fp(rhs_re, p):
            # sqrt only in F_p
            ys = sqrt_fp_all(rhs_re, p)  # all y0 in F_p with y0^2 = rhs_re
            for y0 in ys:
                y = (y0 % p, 0)  # y in F_p^2 with imag=0
                cand = (x, y)

                # Project away the 2-part: multiply by 2^{e2-1}
                P3cand = scalar_mul_montgomery(cand, cofactor_2e2m1, p)

                # Check exact order 3^{e3}
                if point_has_exact_order_power(P3cand, 3, e3, p):
                    P3 = P3cand
                    break

        # ---- candidate for Q3 (non-square in F_p) ----
        if Q3 is None:
            # We want c such that f(c) is NON-square in F_p.
            # That means either:
            # - rhs_im != 0 (so f(c) not even in F_p), OR
            # - rhs_im == 0 but rhs_re is not a quadratic residue in F_p.
            need_nonsquare = (rhs_im != 0) or (not is_square_fp(rhs_re, p))

            if need_nonsquare:
                # We now find y in F_{p^2} such that y^2 = rhs.
                # BUT for Q3, we specifically want y NOT in F_p,
                # i.e. y.imag != 0.
                y_candidates = sqrt_fp2_all(rhs, p)

                for y in y_candidates:
                    if y[1] == 0:
                        # y in F_p, skip (we want extension-only)
                        continue

                    cand = (x, y)
                    Q3cand = scalar_mul_montgomery(cand, cofactor_2e2m1, p)

                    if point_has_exact_order_power(Q3cand, 3, e3, p):
                        Q3 = Q3cand
                        break

    return P3, Q3


# ----------------------------------------------------------
# 3. Top-level convenience
# ----------------------------------------------------------

def generate_public_basis_points(p):
    """
    Convenience wrapper.

    Returns:
        (P2, Q2, P3, Q3)
    where each point is in affine Montgomery coordinates:
        P = ((x0,x1), (y0,y1))
    using (a,b) ↔ a + b*i in F_{p^2}.
    """
    print(f"[+] Using p = {p}")
    e2, e3 = get_sike_exponents(p)
    print(f"[+] p+1 = 2^{e2} * 3^{e3}")

    P2, Q2 = find_P2_Q2(p)
    P3, Q3 = find_P3_Q3(p)

    print("P2 =", P2)
    print("Q2 =", Q2)
    print("P3 =", P3)
    print("Q3 =", Q3)

    return P2, Q2, P3, Q3


if __name__ == "__main__":
    # demo for p = 23 (toy SIKE-style prime)
    generate_public_basis_points(431)
