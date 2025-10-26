# isoex.py
#
# Implements SIKE §1.3.7 "Establishing shared keys: isoex_ell"
# For now we return the raw j-invariant in F_{p^2} instead of hashing it.

from FindingPointsInE import (
    add_fp2,
    sub_fp2,
    mul_fp2,
    sqr_fp2,
    inv_fp2,
    sqrt_fp2_all
)

from EllipticCurveArithmetic import (
    scalar_mul_montgomery,
    point_add_montgomery,
)

from ComputingIsogenies import (
    cfpk,                       # curve-from-public-key: returns A from (xP,xQ,xR)
    compute_2_isogeny_xonly,
    compute_3_isogeny_xonly,
)

# ---------------------------
# helper: build kernel S = P + [sk]Q
# ---------------------------

def build_kernel_generator_from_secret(sk_ell, P_ell, Q_ell, p):
    """
    S = P_ell + [sk_ell] Q_ell  (full point (x,y) in affine Montgomery coords)
    """
    kQ = scalar_mul_montgomery(Q_ell, sk_ell, p)
    S_point = point_add_montgomery(P_ell, kQ, p)
    return S_point  # (x,y) or None if degenerate


# ---------------------------
# helper: j-invariant for y^2 = x^3 + A x^2 + x  (B=1 normalized)
# j = 256 * (A^2 - 3)^3 / (A^2 - 4)
# Everything in F_{p^2}.
# ---------------------------

def j_invariant_from_A(A, p):
    A_sq      = sqr_fp2(A, p)                        # A^2
    three     = (3 % p, 0)
    four      = (4 % p, 0)
    twofive6  = (256 % p, 0)

    A_sq_minus_3 = sub_fp2(A_sq, three, p)           # A^2 - 3
    A_sq_minus_4 = sub_fp2(A_sq, four,  p)           # A^2 - 4

    num = A_sq_minus_3
    num = mul_fp2(num, A_sq_minus_3, p)              # (A^2-3)^2
    num = mul_fp2(num, A_sq_minus_3, p)              # (A^2-3)^3
    num = mul_fp2(twofive6, num, p)                  # 256*(A^2-3)^3

    den = A_sq_minus_4                               # (A^2-4)
    den_inv = inv_fp2(den, p)                        # 1/(A^2-4)

    j = mul_fp2(num, den_inv, p)
    return j  # F_{p^2} element (j0,j1)


# ---------------------------
# main isoex_ell routine
# ---------------------------

def isoex_ell(
    p,
    ell,        # 2 or 3 in SIKE
    e_ell,      # exponent for that torsion subgroup
    sk_ell,     # your secret scalar in that subgroup
    P_ell,      # your public basis point P_ell (full (x,y) on base curve)
    Q_ell,      # your public basis point Q_ell (full (x,y) on base curve)
    pk_other,   # other party's public key tuple: (xP_other, xQ_other, xR_other)
):
    """
    Implements SIKE §1.3.7 "isoex_ell".

    Input:
      - pk_other = (xP_other, xQ_other, xR_other) in F_{p^2}^3 (the other party's public key)
      - sk_ell   = your secret scalar in the ℓ^e_ell torsion
      - P_ell,Q_ell = your basis for that ℓ^e_ell torsion
      - ell, e_ell, p as usual

    Output:
      - j (the j-invariant in F_{p^2}) which plays role of shared secret
    """

    # Step 1. Reconstruct starting curve A_current from pk_other using cfpk
    (xP_other, xQ_other, xR_other) = pk_other
    A_current = cfpk(xP_other, xQ_other, xR_other, p)

    # Step 2. Build your kernel generator S = P_ell + [sk_ell]Q_ell
    S_point = build_kernel_generator_from_secret(sk_ell, P_ell, Q_ell, p)
    if S_point is None:
        raise RuntimeError("isoex_ell: secret produced identity (degenerate)")

    # Step 3. Walk e_ell steps of ℓ-isogenies, updating only the curve.
    for i in range(e_ell):
        if ell == 2:
            A_next, phi_x = compute_2_isogeny_xonly(A_current, S_point, p)
        elif ell == 3:
            A_next, phi_x = compute_3_isogeny_xonly(A_current, S_point, p)
        else:
            raise ValueError("isoex_ell currently only supports ell=2 or ell=3")

        A_current = A_next

        S_point = scalar_mul_montgomery(S_point, ell, p)


    # Step 4. Encode j(E_final). We'll just return j(A_current).
    jval = j_invariant_from_A(A_current, p)
    return jval


# ---------------------------
# quick demo
# ---------------------------

if __name__ == "__main__":
    from PointGenerator import find_P2_Q2, find_P3_Q3
    from EllipticCurveArithmetic import point_sub_montgomery

    p = 11
    A0 = (6 % p, 0)

    # Get torsion bases
    P2, Q2 = find_P2_Q2(p)
    R2 = point_sub_montgomery(P2, Q2, p)

    P3, Q3 = find_P3_Q3(p)
    R3 = point_sub_montgomery(P3, Q3, p)

    # Build Alice's public key (ℓ=2) by pushing Bob's 3-torsion through her 2-isogeny walk
    # and Bob's public key (ℓ=3) by pushing Alice's 2-torsion through his 3-isogeny walk.

    from IsogenyKeygen import compute_public_key_isogeny

    # toy exponents and secrets
    e2 = 2  # small exponent for 2-torsion in our toy prime
    e3 = 1   
    sk2 = 3  # Alice's secret in 2^e2 subgroup
    sk3 = 5  # Bob's secret in 3^e3 subgroup (pick a small nonzero int)

    # Alice's view: she needs Bob's basis x-only (3-torsion)
    xP3 = P3[0]; xQ3 = Q3[0]; xR3 = R3[0]
    pk2_xP, pk2_xQ, pk2_xR= compute_public_key_isogeny(
        p        = p,
        ell      = 2,
        e_ell    = e2,
        sk_ell   = sk2,
        A_start  = A0,
        P_ell    = P2,
        Q_ell    = Q2,
        xP_other = xP3,
        xQ_other = xQ3,
        xR_other = xR3,
    )
    pk2 = (pk2_xP, pk2_xQ, pk2_xR)

    # Bob's view: he needs Alice's basis x-only (2-torsion)
    xP2 = P2[0]; xQ2 = Q2[0]; xR2 = R2[0]
    pk3_xP, pk3_xQ, pk3_xR = compute_public_key_isogeny(
        p        = p,
        ell      = 3,
        e_ell    = e3,
        sk_ell   = sk3,
        A_start  = A0,
        P_ell    = P3,
        Q_ell    = Q3,
        xP_other = xP2,
        xQ_other = xQ2,
        xR_other = xR2,
    )
    pk3 = (pk3_xP, pk3_xQ, pk3_xR)

    print("Alice's public key (for Bob):", pk2)
    print("Bob's public key (for Alice):", pk3)

    # Now simulate shared secret derivation from each side:

    # Alice derives j from Bob's public key using her secret sk2
    j_alice = isoex_ell(
        p        = p,
        ell      = 2,
        e_ell    = e2,
        sk_ell   = sk2,
        P_ell    = P2,
        Q_ell    = Q2,
        pk_other = pk3,
    )

    # Bob derives j from Alice's public key using his secret sk3
    j_bob = isoex_ell(
        p        = p,
        ell      = 3,
        e_ell    = e3,
        sk_ell   = sk3,
        P_ell    = P3,
        Q_ell    = Q3,
        pk_other = pk2,
    )

    print("j_alice =", j_alice)
    print("j_bob   =", j_bob)

    if j_alice == j_bob:
        print("✅ Shared secret matches! SIKE-style exchange succeeded (toy p).")
    else:
        print("❌ Mismatch. Something in the walk / cfpk / isogeny formulas disagrees.")
