"""Signed models: the plugin only uses a model this server signed (ECDSA P-256 over SHA-256).

What is signed (the plugin builds the same text: src/App/Cloud.cs Envelope):
    "SAMPLE MANAGER MODEL\n" + version + "\n" + scheme + "\n" + the model's text
The signature is the raw r || s (64 bytes, base64): what .NET's ECDsa.VerifyData expects.
The key: the env var MODEL_SIGNING_KEY, a PEM private key or its PKCS#8 DER in base64 (tools/make_key.py makes one).
"""
from __future__ import annotations

import base64

from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec, utils


def envelope(version: int, scheme: int, text: str) -> bytes:
    return f"SAMPLE MANAGER MODEL\n{version}\n{scheme}\n{text}".encode("utf-8")


def load_key(value: str) -> ec.EllipticCurvePrivateKey:
    value = value.strip()
    if value.startswith("-----BEGIN"):
        key = serialization.load_pem_private_key(value.replace("\\n", "\n").encode(), password=None)
    else:
        key = serialization.load_der_private_key(base64.b64decode(value), password=None)
    if not isinstance(key, ec.EllipticCurvePrivateKey) or key.curve.name != "secp256r1":
        raise ValueError("MODEL_SIGNING_KEY must be a P-256 (secp256r1) private key")
    return key


def sign(key: ec.EllipticCurvePrivateKey, version: int, scheme: int, text: str) -> str:
    der = key.sign(envelope(version, scheme, text), ec.ECDSA(hashes.SHA256()))
    r, s = utils.decode_dss_signature(der)
    return base64.b64encode(r.to_bytes(32, "big") + s.to_bytes(32, "big")).decode()


def verify(public: ec.EllipticCurvePublicKey, version: int, scheme: int, text: str, signature: str) -> bool:
    raw = base64.b64decode(signature)
    if len(raw) != 64:
        return False
    der = utils.encode_dss_signature(int.from_bytes(raw[:32], "big"), int.from_bytes(raw[32:], "big"))
    try:
        public.verify(der, envelope(version, scheme, text), ec.ECDSA(hashes.SHA256()))
        return True
    except Exception:
        return False


def public_text(key: ec.EllipticCurvePrivateKey) -> str:
    """The public key as the plugin keeps it (config Cloud.PublicKey): base64 of X || Y (32 bytes each)."""
    n = key.public_key().public_numbers()
    return base64.b64encode(n.x.to_bytes(32, "big") + n.y.to_bytes(32, "big")).decode()


def new_key() -> ec.EllipticCurvePrivateKey:
    return ec.generate_private_key(ec.SECP256R1())


def key_text(key: ec.EllipticCurvePrivateKey) -> str:
    """The private key as one line for an env var: PKCS#8 DER in base64."""
    der = key.private_bytes(serialization.Encoding.DER, serialization.PrivateFormat.PKCS8, serialization.NoEncryption())
    return base64.b64encode(der).decode()
