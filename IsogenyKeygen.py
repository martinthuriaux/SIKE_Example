# Isogen.py

from PointGenerator import (
    find_P2_Q2, 
    find_P3_Q3
)

from FindingPointsInE import (
    add_fp2,
    mul_fp2,
    sqrt_fp2_all,
    sub_fp2,
)
from EllipticCurveArithmetic import (
    point_add_montgomery,
    scalar_mul_montgomery,
    point_sub_montgomery,
)
from ComputingIsogenies import (
    compute_2_isogeny,
    compute_3_isogeny,
    compute_4_isogeny,
)

def fp2_const(n, p):
    return (n % p, 0)


def rhs_on_curve_for_x(x, A, p):
    """
    RHS(x) = x^3 + A*x^2 + x in F_{p^2}
    for curve  y^2 = x^3 + A x^2 + x.
    """
    x2 = mul_fp2(x, x, p)        # x^2
    x3 = mul_fp2(x2, x, p)       # x^3
    Ax2 = mul_fp2(A, x2, p)      # A*x^2
    tmp = add_fp2(x3, Ax2, p)
    rhs = add_fp2(tmp, x, p)
    return rhs


def build_kernel_generator_from_secret(p, sk_ell, P_ell, Q_ell):
    """
    S_point = P_ell + [sk_ell]Q_ell.
    """
    kQ = scalar_mul_montgomery(Q_ell, sk_ell, p)
    S_point = point_add_montgomery(P_ell, kQ, p)
    return S_point


def apply_isogeny_to_x(phi, x_in, A_source, p):
    """
    Take an x-coordinate x_in on source curve E_source: y^2 = x^3 + A_source x^2 + x
    and return x_out = x( phi(P) ) where P is any point with x(P)=x_in.

    Steps:
      1. Solve y^2 = RHS_source(x_in).
      2. Build P=(x_in,y).
      3. Map P' = phi(P).
      4. Return x(P').
    """
    rhs_val = rhs_on_curve_for_x(x_in, A_source, p)

    y_candidates = sqrt_fp2_all(rhs_val, p)
    if not y_candidates:
        raise ValueError("apply_isogeny_to_x: x_in not on source curve")

    P_source = (x_in, y_candidates[0])

    P_mapped = phi(P_source)
    if P_mapped is None:
        # x_in was in the kernel. In valid SIKE keygen this shouldn't happen
        # for xP_other/xQ_other/xR_other at any intermediate step.
        raise RuntimeError("apply_isogeny_to_x: point killed by kernel unexpectedly")

    (x_out, _y_out) = P_mapped
    return x_out


def compute_public_key_isogeny(
    p,
    ell,         # small prime: 2 or 3 (or 4 if you're experimenting)
    e_ell,       # exponent e_ell s.t. subgroup has order ell^e_ell
    sk_ell,      # secret scalar
    A_start,     # starting A for E0: y^2 = x^3 + A_start x^2 + x
    # basis for *our* ℓ^e_ell torsion, used to build the kernel
    P_ell,
    Q_ell,
    # x-coordinates of the *other side's* torsion basis that we must transform
    xP_other,
    xQ_other,
    xR_other,
):
    """
    This is the SIKE keygen isogeny walk ("Compute a public key pk_ℓ").

    Output:
      (x1, x2, x3) = the public key pk_ℓ
    """

    # ------------------------------------------------
    # Step 1. Build kernel generator S = P_ell + [sk_ell] Q_ell
    #         and record its x-coordinate xS (mainly for debugging)
    # ------------------------------------------------
    S_point = build_kernel_generator_from_secret(p, sk_ell, P_ell, Q_ell)
    if S_point is None:
        raise RuntimeError("Secret produced identity; invalid")

    xS = S_point[0]  # not strictly needed for math, nice to keep around

    # ------------------------------------------------
    # Step 2. Initialize (x1, x2, x3) as the other party's basis x-coordinates
    # ------------------------------------------------
    x1 = xP_other
    x2 = xQ_other
    x3 = xR_other

    # Current curve is E_A_current : y^2 = x^3 + A_current x^2 + x
    A_current = A_start
    B_current = fp2_const(1, p)  # we always keep curve normalized to B=1

    # ------------------------------------------------
    # Step 3. For i = 0 .. e_ell-1 do:
    # ------------------------------------------------
    for i in range(e_ell):

        # Save the source curve coefficient before updating it.
        # We'll need this to interpret x1,x2,x3 as points on E_i.
        A_source = A_current
        B_source = B_current  # should just be 1, but kept for clarity

        # (a) Compute the ℓ-isogeny φᵢ : E_i -> E_{i+1} with kernel <S_point>.
        if ell == 2:
            A_next, phi = compute_2_isogeny(A_source, B_source, S_point, p)
        elif ell == 3:
            A_next, phi = compute_3_isogeny(A_source, B_source, S_point, p)
        elif ell == 4:
            A_next, phi = compute_4_isogeny(A_source, B_source, S_point, p)
        else:
            raise ValueError(f"Unsupported ell={ell}")

        # (d) BEFORE we update A_current, push φᵢ through x1,x2,x3
        #     using A_source.
        x1 = apply_isogeny_to_x(phi, x1, A_source, p)
        x2 = apply_isogeny_to_x(phi, x2, A_source, p)
        x3 = apply_isogeny_to_x(phi, x3, A_source, p)

        # (b) Update the curve to E_{i+1}:
        A_current = A_next
        B_current = fp2_const(1, p)   # after normalization in compute_*_isogeny

        # (c) Update S_point <- φᵢ(S_point)
        S_point = phi(S_point)
        if S_point is None:
            # At the last step this is fine: the kernel generator
            # is supposed to land at infinity.
            if i != e_ell - 1:
                raise RuntimeError("Kernel point died too early in isogeny walk")
            xS = None
        else:
            xS = S_point[0]  # update xS for completeness/debug

    # ------------------------------------------------
    # Step 4. Output public key pk_ell = (x1,x2,x3)
    # ------------------------------------------------
    return (x1, x2, x3)

if __name__ == "__main__":
    # Quick self-test for compute_public_key_isogeny()

    p = 23
    el = 2        # small prime: 2 or 3 (or 4 if you're experimenting)
    e_el = 2       # exponent e_ell s.t. subgroup has order ell^e_ell
    sk_el = 3     # secret scalar
    A_start = (6 % p, 0)     # starting A for E0: y^2 = x^3 + A_start x^2 + x
    # basis for *our* ℓ^e_ell torsion, used to build the kernel
    x_Pl, x_Ql =   P2, Q2 = find_P2_Q2(p)
    
    # x-coordinates of the *other side's* torsion basis that we must transform
    P3, Q3 = find_P3_Q3(p)     # full points ((x,y) pairs)
    R3 = point_sub_montgomery(P3, Q3, p)  # also a full point

    # Now extract *only* x-coordinates (each is an Fp² pair)
    x_Pm = P3[0]
    x_Qm = Q3[0]
    x_Rm = R3[0]

    print("[+] Running demo isogeny walk...")
    pk = compute_public_key_isogeny(
        p=p,
        ell=el,
        e_ell=e_el,
        sk_ell=sk_el,
        A_start=A_start,
        P_ell=x_Pl,
        Q_ell=x_Ql,
        xP_other=x_Pm,
        xQ_other=x_Qm,
        xR_other=x_Rm,
    )

    print(f"[+] Public key pk_ell = {pk}")
