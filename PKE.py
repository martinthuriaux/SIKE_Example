# sike_pke.py
#
# A direct implementation of Algorithm 1 (PKE = Gen, Enc, Dec)
# using your isoex_l() routine as the "key agreement" step.
#
# c0 is a 2-side public key (pk2) produced from the randomness sk2.
# c1 is m XOR F(j), where j is derived from isoex_2(pk3, sk2).
#
# Notes:
# - Message m is bytes. We hash j with SHAKE256 to the exact m-length.
# - Keys/ciphertexts are kept as tuples of F_{p^2} x-coordinates,
#   exactly like your other code (no serialization here).

from dataclasses import dataclass
from secrets import randbelow
from math import ceil
from typing import Optional


from PointGenerator import find_P2_Q2, find_P3_Q3
from EllipticCurveArithmetic import point_sub_montgomery
from isogen import compute_public_key_isogeny
from isoex import isoex_l, j_invariant_from_A


# ---------- small helpers ----------

# This function will output the XOR of two byte strings
def xor_bytes(a: bytes, b: bytes) -> bytes:
    return bytes(x ^ y for x, y in zip(a, b))

# This function will convert an integer to a big-endian byte string of given length
def int_to_be(x: int, length: int) -> bytes:
    return x.to_bytes(length, "big")

# This function will compute SHAKE256 hash of given input to specified output length
def shake256_bytes(msg: bytes, outlen: int) -> bytes:
    import hashlib
    return hashlib.shake_256(msg).digest(outlen)

#This functional will hash a j-invariant to bytes
def hash_j_to_bytes(j, p: int, outlen: int) -> bytes:
    """
    j is an element of F_{p^2} represented as a pair of ints (a,b).
    Serialize deterministically and SHAKE256-expand to outlen bytes.
    """
    a, b = j
    bytelen = ceil(p.bit_length() / 8)
    enc = int_to_be(a % p, bytelen) + int_to_be(b % p, bytelen)
    return shake256_bytes(enc, outlen)


# ---------- parameters container ----------

@dataclass
class SIKEParams:
    p: int             # the underlying prime
    A0: tuple          # base curve coefficient A in F_{p^2}, e.g. (A,0)
    e2: int            # 2^e2
    e3: int            # 3^e3

    def bases(self):
        P2, Q2 = find_P2_Q2(self.p, self.A0)
        P3, Q3 = find_P3_Q3(self.p, self.A0)
        R2 = point_sub_montgomery(P2, Q2, self.p, self.A0)
        R3 = point_sub_montgomery(P3, Q3, self.p, self.A0)
        return (P2, Q2, R2), (P3, Q3, R3)


# ---------- PKE: Gen, Enc, Dec ----------

def Gen(params: SIKEParams):
    """
    Output: (pk3, sk3)
    sk3 is uniform in [0, 3^e3 - 1].
    pk3 is Bob's 3-side public key, which *pushes the 2-basis*.
    """
    (P2, Q2, R2), (P3, Q3, _) = params.bases()
    xP2, xQ2, xR2 = P2[0], Q2[0], R2[0]

    # 1) sk_3 <–– Random in K_3 = [0, 3^e3)
    sk3 = randbelow(pow(3, params.e3))

    # 2) pk3 <–– isogen_3(sk_3)
    pk3 = compute_public_key_isogeny(
        p        = params.p,
        l        = 3,
        e_l      = params.e3,
        sk_l     = sk3,
        A_start  = params.A0,
        P_l      = P3,
        Q_l      = Q3,
        xP_m     = xP2,
        xQ_m     = xQ2,
        xR_m     = xR2,
    )

    # 3) return (pk3, sk3)
    return pk3, sk3


def Enc(params: SIKEParams, pk3, m: bytes, r: Optional[int] = None):
    (P2, Q2, _), (P3, Q3, R3) = params.bases()

    # 4) sk3 <–– Random in K_2 = [0, 2^e2)
    sk2 = randbelow(pow(2, params.e2)) if r is None else r

    xP3, xQ3, xR3 = P3[0], Q3[0], R3[0]

    # 5) c0 <–– isogen_2(sk_2)
    c0 = compute_public_key_isogeny(
        p       = params.p,
        l       = 2,
        e_l     = params.e2,
        sk_l    = sk2,
        A_start = params.A0,
        P_l     = P2,
        Q_l     = Q2,
        xP_m    = xP3,
        xQ_m    = xQ3,
        xR_m    = xR3,
    )

    # 6) J <–– isoex_2(pk_3, sk_2)
    A_shared = isoex_l(
        p    = params.p,
        l    = 2,
        e_l  = params.e2,
        sk_l = sk2,
        pk_m = pk3,
    )
    j = j_invariant_from_A(A_shared, params.p)

    # 7) H <–– F(j)
    h = hash_j_to_bytes(j, params.p, len(m))

    # c1 <–– h XOR m 
    c1 = xor_bytes(h, m)

    return c0, c1


def Dec(params: SIKEParams, sk3, ciphertext) -> bytes:
    """
    Input:  sk3 (from Gen) and ciphertext (c0, c1)
    Output: recovered message m (bytes)
    """
    c0, c1 = ciphertext

    # shared secret via isoex_3 with Alice's c0 (= pk2)
    print("p in Dec:", params.p)
    A_shared = isoex_l(
        p      = params.p,
        l      = 3,
        e_l    = params.e3,
        sk_l   = sk3,
        pk_m   = c0,
    )

    j = j_invariant_from_A(A_shared, params.p)
    h = hash_j_to_bytes(j, params.p, len(c1))

    m = xor_bytes(h, c1)
    return m


# ---------- quick sanity demo (matches your isoex demo) ----------

if __name__ == "__main__":
    # tiny toy params from your demo:
    p = 23
    A0 = (6 % p, 0)
    e2, e3 = 3, 1
    params = SIKEParams(p=p, A0=A0, e2=e2, e3=e3)

    # Bob generates (pk3, sk3)
    pk3, sk3 = Gen(params)

    # Alice encrypts
    msg = b"Here is symmetric key data"
    print("Message before encryption:", msg)
    c0, c1 = Enc(params, pk3, msg)
    print("Ciphertext post encryption:", (c0, c1))
    print("Public key to encrypt:", c0)
    print("Encrypted msg:", c1)

    # Bob decrypts
    dec = Dec(params, sk3, (c0, c1))
    print("Decrypted version of ciphertext:", dec)
    assert dec == msg
