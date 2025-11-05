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
    point_sub_montgomery,
    xDBL_xonly,
    xTPL_xonly
)

from ComputingIsogenies import (
    compute_2_isogeny_xonly,
    compute_3_isogeny_xonly,
)

def fp2_const(n, p):
    return (n % p, 0)

def repeated_xmul_power(xS, A_current, l, k, p):
    """
    Return x([l^k] * S) using ONLY xS and A_current, by repeated
    xDBL_xonly or xTPL_xonly.

    xS is x(S) on the current curve E_A_current.
    l is 2 or 3.
    k is a small nonnegative integer.
    """
    xQ = xS
    for _ in range(k):
        if l == 2:
            xQ = xDBL_xonly(xQ, A_current, p)
        elif l == 3:
            xQ = xTPL_xonly(xQ, A_current, p)
        else:
            raise ValueError("l must be 2 or 3")
    return xQ

def build_kernel_generator_from_secret(p, sk_l, P_l, Q_l, A):
    """
    S = P_l + [sk_l] Q_l.
    This is a full point (x,y), not just x, because internally the
    next kernel isogeny step needs a full point on the current curve.
    """
    kQ = scalar_mul_montgomery(Q_l, sk_l, p, A)
    print(f"Value of {sk_l} * {Q_l}):", kQ)

    S_point = point_add_montgomery(P_l, kQ, p, A)
    print(f"Value of {P_l} + {kQ}):", S_point)
    return S_point


def compute_public_key_isogeny(
    p,
    l,         
    e_l,       
    sk_l,      
    A_start,     
    P_l,
    Q_l,
    xP_m,
    xQ_m,
    xR_m,
):
    """
    SIKE keygen isogeny walk ("Compute a public key pk_ℓ").
    Corrected version: uses [ℓ^(e_l-i-1)]S for kernel, updates xS via φ_x.
    """

    # --- (1) Build master secret point S_master = P + sk*Q ---
    S_master = build_kernel_generator_from_secret(p, sk_l, P_l, Q_l, A_start)
    if S_master is None:
        raise RuntimeError("degenerate kernel: P + sk*Q = O")

    xS = S_master[0]  # only keep x, per spec
    print("Initial xS:", xS)

    A_current = A_start

    # other side's basis x-coords
    x1, x2, x3 = xP_m, xQ_m, xR_m

    # i from 0 .. e_l-1
    for i in range(e_l):
        # how much torsion to strip this round
        k = e_l - i - 1

        # kernel x = x( [l^k] * S_i )
        xKer = repeated_xmul_power(xS, A_current, l, k, p)
        print(f"Round {i}: xKer = {xKer}")

        # build isogeny from xKer
        if l == 2:
            A_next, phi_x = compute_2_isogeny_xonly(A_current, xKer, p)
        else:
            A_next, phi_x = compute_3_isogeny_xonly(A_current, xKer, p)

        # push other side's basis
        x1_new = phi_x(x1)
        x2_new = phi_x(x2)
        x3_new = phi_x(x3)

        print(f"Value of A at round {i}", A_next)
        print(f"In round {i}, f(x_1):", x1_new, "f(x_2):", x2_new, "f(x_3):", x3_new)

        if None in (x1_new, x2_new, x3_new):
            raise RuntimeError("basis point killed (shouldn't happen)")

        # move secret forward to next curve
        xS_new = phi_x(xS)
        print(f"In round {i}, f(xS):", xS_new)
        if xS_new is None:
            if i != e_l - 1:
                raise RuntimeError("secret died early")
            # last round is allowed to die
        else:
            xS = xS_new

        #error checking: print("xS: ", xS)

        # advance curve and transported basis
        A_current = A_next
        x1, x2, x3 = x1_new, x2_new, x3_new

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

    p = 431

    # starting curve: y^2 = x^3 + 6x^2 + x
    A_start = (423 % p, 329 % p)  

    # Toy exponents: in real SIKEp434, e2 is ~110. Here let's say small.
    e2 = 4
    e3 = 3  

    # Alice's secret scalar on the 2-side
    sk2 = 11

    # Bob's secret scalar on the 3-side
    sk3 = 2

    # Our 2^e2 torsion basis (full points with (x,y))
    P2, Q2 = find_P2_Q2(p, A_start)
    R2 = point_sub_montgomery(P2, Q2, p, A_start)  # R2 = P2 - Q2

    xP2 = P2[0]
    xQ2 = Q2[0]
    xR2 = R2[0]
    

    # Bob's 3^e3 torsion basis (full points)
    P3, Q3 = find_P3_Q3(p, A_start)
    R3 = point_sub_montgomery(P3, Q3, p, A_start)  # R3 = P3 - Q3

    # Extract just x-coordinates of Bob's basis
    xP3 = P3[0]
    xQ3 = Q3[0]
    xR3 = R3[0]

    pk2 = compute_public_key_isogeny(
        p=p,
        l=2,            # Alice uses 2-isogeny steps
        e_l=e2,         # walk length (toy)
        sk_l=sk2,       # Alice's secret
        A_start=A_start,  # starting curve coeff A
        P_l=P2,         # Alice's 2-torsion generator P2
        Q_l=Q2,         # Alice's 2-torsion generator Q2
        xP_m=xP3,     # Bob's basis x(P3)
        xQ_m=xQ3,     # Bob's basis x(Q3)
        xR_m=xR3,     # Bob's basis x(P3-Q3)
    )

    print(f"[+] Public key pk_2 = {pk2}")

    pk3 = compute_public_key_isogeny(
        p=p,
        l=3,            # Alice uses 2-isogeny steps
        e_l=e3,         # walk length (toy)
        sk_l=sk3,       # Alice's secret
        A_start=A_start,  # starting curve coeff A
        P_l=P3,         # Alice's 2-torsion generator P2
        Q_l=Q3,         # Alice's 2-torsion generator Q2
        xP_m=xP2,     # Bob's basis x(P3)
        xQ_m=xQ2,     # Bob's basis x(Q3)
        xR_m=xR2,     # Bob's basis x(P3-Q3)
    )

    print(f"[+] Public key pk_3 = {pk3}")


