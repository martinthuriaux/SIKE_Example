# isoex.py
#
# Implements SIKE §1.3.7 "Establishing shared keys: isoex_ell"
# For now we just return the final A coefficient of the shared curve
# (which determines j), instead of hashing j into bytes.
from secrets import randbelow

from FindingPointsInE import (
    add_fp2,
    mul_fp2,
    sqrt_fp2_all,
    sub_fp2,
    sqr_fp2,
    inv_fp2,
    eq_fp2
)
from EllipticCurveArithmetic import (
    scalar_mul_montgomery,
    point_add_montgomery,
    point_sub_montgomery,
    xDBL_xonly,
    xTPL_xonly,
)
from ComputingIsogenies import (
    cfpk,                       # curve-from-public-key: A_from_pk = cfpk(xP,xQ,xR)
    compute_2_isogeny_xonly,
    compute_3_isogeny_xonly,
)

# ------------------------------------------------------------------
# Helpers we also used in keygen
# ------------------------------------------------------------------


def montgomery_rhs(x, A, p):
    # rhs = x^3 + A*x^2 + x  in F_{p^2}
    x2   = mul_fp2(x, x, p)          # x^2
    x3   = mul_fp2(x2, x, p)         # x^3
    Ax2  = mul_fp2(A, x2, p)         # A*x^2
    tmp  = add_fp2(x3, Ax2, p)       # x^3 + A*x^2
    rhs  = add_fp2(tmp, x, p)        # x^3 + A*x^2 + x
    return rhs

def j_invariant_from_A(A, p):
    A_sq = sqr_fp2(A, p)
    three = (3 % p, 0)
    four = (4 % p, 0)
    twofive6 = (256 % p, 0)
    num = sub_fp2(A_sq, three, p)
    num = mul_fp2(num, num, p)
    num = mul_fp2(num, sub_fp2(A_sq, three, p), p)
    num = mul_fp2(twofive6, num, p)
    den = sub_fp2(A_sq, four, p)
    j = mul_fp2(num, inv_fp2(den, p), p)
    return j

def build_kernel_generator_from_secret(sk_l, P_l, Q_l, p):
    """
    S_master = P_l + [sk_l] Q_l  as a full affine point (x,y).
    """
    kQ = scalar_mul_montgomery(Q_l, sk_l, p)
    S_point = point_add_montgomery(P_l, kQ, p)
    return S_point

def repeated_xmul_power(xS, A_current, e_l, k, p):
    """
    Compute x([e_l^k] * S_i) using ONLY the x-coordinate xS of S_i
    on the current curve E_{A_current}, by repeated x-only doubling/tripling.

    xS : F_{p^2}, x-coordinate of the "current secret point" S_i on E_{A_current}
    e_l: 2 or 3
    k  : nonnegative int
    """
    xQ = xS
    for _ in range(k):
        if e_l == 2:
            xQ = xDBL_xonly(xQ, A_current, p)
        elif e_l == 3:
            xQ = xTPL_xonly(xQ, A_current, p)
        else:
            raise ValueError("e_l must be 2 or 3")
    return xQ

def lift_x_to_point_on_curve(xP, A, p):
    """
    Given xP in F_{p^2}, find some affine point (xP, yP) on
    y^2 = x^3 + A x^2 + x. We only need ANY valid square root.

    Returns (xP, yP) or raises if no sqrt exists (shouldn't happen
    for torsion points in SIKE).
    """
    rhs = montgomery_rhs(xP, A, p)      # y^2
    roots = sqrt_fp2_all(rhs, p)        # list of possible y
    if not roots:
        raise RuntimeError("lift_x_to_point_on_curve: no sqrt for rhs; invalid public key?")
    yP = roots[0]                       # pick either root; sign doesn't matter
    return (xP, yP)

def lift_basis_from_pk(xP, xQ, xR, A, p):
    """
    Given a public key triple (xP, xQ, xR) for curve y^2 = x^3 + A x^2 + x,
    recover full points (P, Q) so that x(P - Q) == xR.
    This fixes the ±y ambiguity consistently with the pk.
    """
    rhsP = montgomery_rhs(xP, A, p)
    rhsQ = montgomery_rhs(xQ, A, p)
    yP_cands = sqrt_fp2_all(rhsP, p)
    yQ_cands = sqrt_fp2_all(rhsQ, p)

    for yP in yP_cands:
        for yQ in yQ_cands:
            P = (xP, yP)
            Q = (xQ, yQ)

            R1 = point_sub_montgomery(P, Q, p, A)   # P - Q
            if R1 is not None and eq_fp2(R1[0], xR):
                return P, Q

            R2 = point_sub_montgomery(Q, P, p, A)   # Q - P
            if R2 is not None and eq_fp2(R2[0], xR):
                # Swap to keep semantics "first is P, second is Q"
                return Q, P

    raise RuntimeError("lift_basis_from_pk: couldn't match xR with any sign choice")

# ------------------------------------------------------------------
# isoex_l
# ------------------------------------------------------------------

def isoex_l(
    p,
    l,        # 2 or 3
    e_l,      # exponent e_l
    sk_l,     # our secret scalar in that torsion
    pk_m,   # other side's public key tuple: (xP_m, xQ_m, xR_m)
):
    """
    Run SIKE isoex_ll (key agreement) for one side.

    Steps (per §1.3.7):
      1. Rebuild starting curve coefficient A_current from pk_m via cfpk.
      2. Form S_master = P_l + [sk_l]Q_l; let xS = x(S_master).
      3. For i in 0..e_l-1:
         a. k = e_l - i - 1
            xKer = x([l^k]*S_i) on E_i   (using repeated_xmul_power)
            Build isogeny φ_i : E_i -> E_{i+1} with kernel <[l^k]S_i>.
            This gives us A_next and an x-only map phi_x().
         b. Update curve A_current <- A_next.
         c. Update xS <- phi_x(xS).
            (If phi_x(xS) = None before the *very last* step, it's an error.
             On the last step it's allowed to die, because the kernel point
             itself maps to infinity.)
      4. Return final A_current. In SIKE you'd hash j(A_current); for now we
         just hand A_current back.

    Output:
      A_current (F_{p^2}), the final curve coefficient after our secret walk.
      This defines j-invariant and is our "shared secret" surrogate.
    """

    # 1. Recover starting curve A from their public key.
    (xP_m, xQ_m, xR_m) = pk_m
    A_current = cfpk(xP_m, xQ_m, xR_m, p)

    # 2. Build S_master on the *same base curve E0*. In SIKE the base curve
    #    is shared, so P_l/Q_l are on that same E0. Then set xS = x(S_master).
    # (We DON'T use xR_m here to make S_master; R is only used in cfpk.)

    P_other, Q_other = lift_basis_from_pk(xP_m, xQ_m, xR_m, A_current, p) 
    kQ = scalar_mul_montgomery(Q_other, sk_l, p, A_current)
    S_master = point_add_montgomery(P_other, kQ, p, A_current)

    if S_master is None:
        raise RuntimeError("isoex_l: degenerate secret (P + sk*Q = O)")
    xS = S_master[0]

    # 3. Walk the isogeny e_l times.
    for i in range(e_l):
        k = e_l - i - 1  # how much torsion remains this round

        # kernel x-coordinate for this step:
        xKer = repeated_xmul_power(xS, A_current, l, k, p)

        # build the isogeny from xKer:
        if l == 2:
            A_next, phi_x = compute_2_isogeny_xonly(A_current, xKer, p)
        elif l == 3:
            A_next, phi_x = compute_3_isogeny_xonly(A_current, xKer, p)
        else:
            raise ValueError("isoex_l only supports l=2 or l=3")

        # advance curve
        A_current = A_next

        # move our secret forward along φ_i
        xS_new = phi_x(xS)
        if xS_new is None:
            # allowed only at final round
            if i != e_l - 1:
                raise RuntimeError("isoex_l: secret kernel died too early")
            # last round -> it's fine, we won't use xS afterward
        else:
            xS = xS_new

    # 4. Instead of hashing j(A_current), we just return A_current itself.
    return A_current


# ------------------------------------------------------------------
# quick demo
# ------------------------------------------------------------------

if __name__ == "__main__":
    from PointGenerator import find_P2_Q2, find_P3_Q3
    from EllipticCurveArithmetic import point_sub_montgomery
    from isogen import compute_public_key_isogeny

    p = 71
    A = (6 % p, 0)  

    # torsion bases on the base curve
    P2, Q2 = find_P2_Q2(p,A)
    R2 = point_sub_montgomery(P2, Q2, p, A)

    P3, Q3 = find_P3_Q3(p,A)
    R3 = point_sub_montgomery(P3, Q3, p, A)

    # toy exponents
    e2 = 3   # 2^e2 subgroup
    e3 = 2   # 3^e3 subgroup

    # toy secrets
    sk2 = randbelow(pow(2, e2))
    print("sk2:", sk2)
    sk3 = randbelow(pow(3, e3))
    print("sk3:", sk3)

    # Build Alice's pk_2 (2-side public key): she pushes Bob's 3-torsion x-basis
    xP3, xQ3, xR3 = P3[0], Q3[0], R3[0]

    pk2 = compute_public_key_isogeny(
        p        = p,
        l      = 2,
        e_l    = e2,
        sk_l   = sk2,
        A_start  = A,
        P_l    = P2,
        Q_l    = Q2,
        xP_m = xP3,
        xQ_m = xQ3,
        xR_m = xR3,
    )


    # Build Bob's pk_3 (3-side public key): he pushes Alice's 2-torsion x-basis
    xP2, xQ2, xR2 = P2[0], Q2[0], R2[0]
    pk3 = compute_public_key_isogeny(
        p        = p,
        l      = 3,
        e_l    = e3,
        sk_l   = sk3,
        A_start  = A,
        P_l    = P3,
        Q_l    = Q3,
        xP_m = xP2,
        xQ_m = xQ2,
        xR_m = xR2,
    )

    print("[+] Alice's public key (pk2):", pk2)
    print("[+] Bob's public key   (pk3):", pk3)

    # Alice computes shared secret using her 2-side secret and Bob's pk3
    A_alice = isoex_l(
        p        = p,
        l      = 2,
        e_l    = e2,
        sk_l   = sk2,
        pk_m = pk3,
    )

    # Bob computes shared secret using his 3-side secret and Alice's pk2
    A_bob = isoex_l(
        p        = p,
        l      = 3,
        e_l    = e3,
        sk_l   = sk3,
        pk_m = pk2,
    )

    j_alice = j_invariant_from_A(A_alice, p)
    j_bob = j_invariant_from_A(A_bob, p)

    print("j_alice =", j_alice)
    print("j_bob   =", j_bob)