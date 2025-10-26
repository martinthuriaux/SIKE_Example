# Isogen.py

from PointGenerator import (
    find_P2_Q2,
    find_P3_Q3,
)

from FindingPointsInE import (
    negate_fp2,
)

from EllipticCurveArithmetic import (
    point_add_montgomery,
    scalar_mul_montgomery,
)

from ComputingIsogenies import (
    compute_2_isogeny_xonly,
    compute_3_isogeny_xonly,
)

def fp2_const(n, p):
    return (n % p, 0)


def point_sub_montgomery(P, Q, p):
    """
    Return P - Q using affine formulas.
    """
    (xQ, yQ) = Q
    Qneg = (xQ, negate_fp2(yQ, p))
    return point_add_montgomery(P, Qneg, p)


def build_kernel_generator_from_secret(p, sk_ell, P_ell, Q_ell):
    """
    S = P_ell + [sk_ell] Q_ell.
    This is a full point (x,y), not just x, because internally the
    next kernel isogeny step needs a full point on the current curve.
    """
    kQ = scalar_mul_montgomery(Q_ell, sk_ell, p)
    S_point = point_add_montgomery(P_ell, kQ, p)
    return S_point


def compute_public_key_isogeny(
    p,
    ell,         # small prime: 2 or 3
    e_ell,       # exponent e_ell s.t. subgroup has order ell^e_ell
    sk_ell,      # secret scalar
    A_start,     # starting A for E0: y^2 = x^3 + A_start*x^2 + x
    # basis for *our* ℓ^e_ell torsion, USED FOR KERNEL (full points)
    P_ell,
    Q_ell,
    # x-coordinates of the *other side's* torsion basis
    # (xP_other, xQ_other, xR_other) that we must transform
    xP_other,
    xQ_other,
    xR_other,
):
    """
    This is the SIKE keygen isogeny walk ("Compute a public key pk_ℓ").

    We:
      1. Form the kernel generator S = P_ell + sk_ell*Q_ell  (a full point).
      2. Repeat e_ell times:
         - Build the ℓ-isogeny φ_i from <S>.
         - Update (x1,x2,x3) <- φ_i(x1,x2,x3)   [x-only map]
         - Update S <- φ_i(S)                  [full point so next step works]
         - Update A_current <- A_next          [new curve coefficient]
      3. Return (x1, x2, x3) as the public key.

    Output:
        (x1, x2, x3)  -- these are F_{p^2} x-coordinates after the walk.
    """

    # (1) Build kernel generator S for the first step
    S_point = build_kernel_generator_from_secret(p, sk_ell, P_ell, Q_ell)
    if S_point is None:
        raise RuntimeError(
        f"degenerate kernel for ell={ell}: P + {sk_ell}*Q = O. "
        "Pick a different sk_ell."
    )

    # Initialize the transported x-basis (the "other party"'s basis)
    x1 = xP_other
    x2 = xQ_other
    x3 = xR_other

    # Keep track of current curve coefficient
    # Curve model: y^2 = x^3 + A_current*x^2 + x  (B = 1)
    A_current = A_start

    # Walk e_ell times
    for i in range(e_ell):

        # ------------------------------------------------
        # (a) Build the ℓ-isogeny from the current kernel S_point
        #     We will get:
        #         A_next : new curve coefficient
        #         phi_x  : x-only map x -> x'
        #
        #     NOTE: we're using the x-only isogenies you wrote:
        #       compute_2_isogeny_xonly(A_current, S_point, p)
        #       compute_3_isogeny_xonly(A_current, S_point, p)
        # ------------------------------------------------
        if ell == 2:
            A_next, phi_x = compute_2_isogeny_xonly(A_current, S_point, p)
        elif ell == 3:
            A_next, phi_x = compute_3_isogeny_xonly(A_current, S_point, p)
        else:
            raise ValueError(f"Unsupported ell={ell} (only 2 or 3 supported here)")

        # ------------------------------------------------
        # (b) Push the other side's basis x-coordinates through this step.
        #     Each xi is an x in F_{p^2}. phi_x returns the new x (or None if
        #     that x belonged to the kernel -- that should not happen for the
        #     other side's basis in a valid SIKE keygen).
        # ------------------------------------------------
        x1_new = phi_x(x1)
        x2_new = phi_x(x2)
        x3_new = phi_x(x3)

        # Sanity: none of these should die unless something degenerate
        if x1_new is None or x2_new is None or x3_new is None:
            # In real code you'd handle/restart, but here we just raise
            raise RuntimeError(
                f"x-only map killed a basis point at step {i}; "
                "this shouldn't happen in valid SIKE parameters."
            )

        A_current = A_next

        S_point = scalar_mul_montgomery(S_point, ell, p)

        x1, x2, x3 = x1_new, x2_new, x3_new

    # end for

    # After e_ell steps, x1,x2,x3 are your public key coordinates
    return (x1, x2, x3)


if __name__ == "__main__":
    #
    # Demo: pretend "Alice" is the 2-side sender.
    # She:
    #   - uses the 2^e2 torsion basis (P2,Q2),
    #   - picks a secret sk_2,
    #   - builds kernel S = P2 + sk_2*Q2,
    #   - walks e_2 steps of 2-isogenies,
    #   - transports Bob's 3-torsion basis x-coords through the walk,
    #   - and outputs a 3-tuple of x-coordinates as her "public key".
    #

    p = 23

    # starting curve: y^2 = x^3 + 6x^2 + x
    A_start = (6 % p, 0)

    # Toy exponents: in real SIKEp434, e2 is ~110. Here let's say small.
    e2 = 3
    e3 = 1  

    # Alice's secret scalar on the 2-side
    sk2 = 7

    # Bob's secret scalar on the 3-side
    sk3 = 7

    # Our 2^e2 torsion basis (full points with (x,y))
    P2, Q2 = find_P2_Q2(p)
    R2 = point_sub_montgomery(P2, Q2, p)  # R2 = P2 - Q2

    xP2 = P2[0]
    xQ2 = Q2[0]
    xR2 = R2[0]
    

    # Bob's 3^e3 torsion basis (full points)
    P3, Q3 = find_P3_Q3(p)
    R3 = point_sub_montgomery(P3, Q3, p)  # R3 = P3 - Q3

    # Extract just x-coordinates of Bob's basis
    xP3 = P3[0]
    xQ3 = Q3[0]
    xR3 = R3[0]

    print("[+] Running toy SIKE-style isogeny walk (Alice-side)...")

    pk2 = compute_public_key_isogeny(
        p=p,
        ell=2,            # Alice uses 2-isogeny steps
        e_ell=e2,         # walk length (toy)
        sk_ell=sk2,       # Alice's secret
        A_start=A_start,  # starting curve coeff A
        P_ell=P2,         # Alice's 2-torsion generator P2
        Q_ell=Q2,         # Alice's 2-torsion generator Q2
        xP_other=xP3,     # Bob's basis x(P3)
        xQ_other=xQ3,     # Bob's basis x(Q3)
        xR_other=xR3,     # Bob's basis x(P3-Q3)
    )

    print(f"[+] Public key pk_2 = {pk2}")

    pk3 = compute_public_key_isogeny(
        p=p,
        ell=3,            # Alice uses 2-isogeny steps
        e_ell=e3,         # walk length (toy)
        sk_ell=sk3,       # Alice's secret
        A_start=A_start,  # starting curve coeff A
        P_ell=P3,         # Alice's 2-torsion generator P2
        Q_ell=Q3,         # Alice's 2-torsion generator Q2
        xP_other=xP2,     # Bob's basis x(P3)
        xQ_other=xQ2,     # Bob's basis x(Q3)
        xR_other=xR2,     # Bob's basis x(P3-Q3)
    )

    print(f"[+] Public key pk_3 = {pk3}")
