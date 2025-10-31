# KEM.py
# SIKE Algorithm 2 (KeyGen, Encaps, Decaps) built on your PKE.
# No external encode helpers required.

from secrets import token_bytes
from hashlib import shake_256
from math import ceil
from typing import Optional, Tuple

from PKE import SIKEParams, Gen, Enc, Dec
from isogen import compute_public_key_isogeny


# ---------- minimal, canonical serialization (local to this file) ----------

Fp2 = Tuple[int, int]
Pk2 = Tuple[Fp2, Fp2, Fp2]

def _nbytes_p(p: int) -> int:
    return ceil(p.bit_length() / 8)

def _ser_fp2(x: Fp2, p: int) -> bytes:
    a, b = x
    n = _nbytes_p(p)
    return (a % p).to_bytes(n, "big") + (b % p).to_bytes(n, "big")

def _ser_pk(pk: Pk2, p: int) -> bytes:
    xP, xQ, xR = pk
    return _ser_fp2(xP, p) + _ser_fp2(xQ, p) + _ser_fp2(xR, p)

def _ser_ct(c0: Pk2, c1: bytes, p: int) -> bytes:
    return _ser_pk(c0, p) + c1


# ---------- KDF-like helpers G and H ----------

def _G(m_and_pk3: bytes, e2: int) -> int:
    """r in [0, 2^e2) derived from SHAKE-256(m || pk3)."""
    outlen = ceil(e2 / 8)
    r = int.from_bytes(shake_256(m_and_pk3).digest(outlen), "big")
    return r % (1 << e2)

def _H(data: bytes, outlen: int = 32) -> bytes:
    """K = SHAKE-256(data, outlen)."""
    return shake_256(data).digest(outlen)


# ---------- KEM ----------

def KeyGen(params: SIKEParams):
    """Return (s, sk3, pk3)."""
    pk3, sk3 = Gen(params)      # reuse your PKE.Gen
    s = token_bytes(32)         # 256-bit fallback secret
    return s, sk3, pk3

def Encaps(params: SIKEParams, pk3: Pk2):
    """Return ((c0, c1), K)."""
    m = token_bytes(32)                         # random n-bit message (here 256b)
    r = _G(m + _ser_pk(pk3, params.p), params.e2)
    c0, c1 = Enc(params, pk3, m, r)             # PKE.Enc with explicit randomness
    K = _H(m + _ser_ct(c0, c1, params.p))       # H(m || (c0,c1))
    return (c0, c1), K

def Decaps(params: SIKEParams, s: bytes, sk3: int, pk3: Pk2, ciphertext):
    """Return K."""
    c0, c1 = ciphertext

    # m' = PKE.Dec
    m_prime = Dec(params, sk3, (c0, c1))

    # r' = G(m' || pk3)
    r_prime = _G(m_prime + _ser_pk(pk3, params.p), params.e2)

    # c0' = isogen2(r')
    (P2, Q2, _), (P3, Q3, R3) = params.bases()
    xP3, xQ3, xR3 = P3[0], Q3[0], R3[0]
    c0_prime = compute_public_key_isogeny(
        p       = params.p,
        l       = 2,
        e_l     = params.e2,
        sk_l    = r_prime,
        A_start = params.A0,
        P_l     = P2,
        Q_l     = Q2,
        xP_m    = xP3,
        xQ_m    = xQ3,
        xR_m    = xR3,
    )

    ct_bytes = _ser_ct(c0, c1, params.p)
    if c0_prime == c0:
        K = _H(m_prime + ct_bytes)   # good ciphertext
    else:
        K = _H(s + ct_bytes)         # fallback per Algorithm 2
    return K


# ---------- demo ----------

if __name__ == "__main__":
    # same toy params as your PKE demo
    p = 2591
    A0 = (6 % p, 0)
    e2, e3 = 5, 4
    params = SIKEParams(p=p, A0=A0, e2=e2, e3=e3)

    print("[+] KEM demo start")
    s, sk3, pk3 = KeyGen(params)
    (c0, c1), K_enc = Encaps(params, pk3)
    K_dec = Decaps(params, s, sk3, pk3, (c0, c1))
    print("K_enc =", K_enc.hex())
    print("K_dec =", K_dec.hex())
    assert K_enc == K_dec, "key mismatch!"
    print("✅ keys match")
